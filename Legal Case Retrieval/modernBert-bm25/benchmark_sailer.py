"""Real-base update-time and full-4096 memory benchmark; never saves weights.

Each profile runs in a fresh process so an OOM cannot contaminate another
measurement. Reports are outside the formal output directory by default.
"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from config import (PROJECT_DIR, REPO_ROOT, default_report_directory, load_config,
                    read_json, write_json)


def worker(args):
    import torch
    from transformers import AutoTokenizer, set_seed
    from fine_tune.data import prepare
    from fine_tune.gradcache import cached_backward, masked_inbatch_loss
    from fine_tune.modernbert_contrastive_model import ModernBERTContrastive
    from fine_tune.sailer import (LogicalBatchDataset, SailerCollator, logical_epoch_rows, prepare_label_mask)
    from fine_tune.fine_tune import BM25NegativeSamplingTrainer

    torch.set_num_threads(4)
    cfg = load_config(PROJECT_DIR / "configs/coliee2025_sailer_gradcache.json")
    cfg.update(gradcache_chunk_size=args.worker, deduplicate_inputs=not args.no_dedup,
               gradient_checkpointing=not args.without_checkpointing or args.no_dedup)
    bundle = prepare(cfg, write=False)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["model_revision"], local_files_only=True)
    manifest = prepare_label_mask(cfg, bundle, tokenizer)
    rows, _ = logical_epoch_rows(bundle["train_labels"], bundle["train_pools"], 0, cfg["sampler_seed"])
    dataset = LogicalBatchDataset(rows, cfg["document_dir"])
    collate = SailerCollator(tokenizer, cfg, manifest)
    batches = [collate([dataset[i]]) for i in range(3)]
    # Stress memory with 68 DIFFERENT real case token sequences at length4096.
    # These are deliberately not relevance-labelled training pairs; this phase
    # measures tensor-shape memory only, and produces no retrieval score.
    long_tokens, seen = [], set()
    for pid in bundle["manifest"]["candidate_ids"]:
        key = manifest["documents"][pid]["token_hash"]
        if key in seen:
            continue
        text = (Path(cfg["document_dir"]) / f"{pid}.txt").read_text().strip()
        tokens = tokenizer(text, truncation=True, max_length=4096)["input_ids"]
        if len(tokens) == 4096:
            long_tokens.append(tokens)
            seen.add(key)
            if len(long_tokens) == 68:
                break
    assert len(long_tokens) == 68
    all_ids = torch.tensor(long_tokens)
    stress = {"chunks": [{"input_ids": c, "attention_mask": torch.ones_like(c)} for c in all_ids.split(args.worker)],
              "allowed": torch.ones(4,64,dtype=torch.bool), "labels": torch.arange(4)*16,
              "representation_indices": torch.arange(68)}
    set_seed(cfg["seed"])
    model = ModernBERTContrastive.from_base(cfg).to("cuda").train()
    model.encoder.config.reference_compile = False
    owner = BM25NegativeSamplingTrainer.__new__(BM25NegativeSamplingTrainer)
    owner.model, owner.config, owner.optimizer = model, cfg, None
    optimizer = owner.create_optimizer()
    scaler = torch.amp.GradScaler("cuda")
    report = {"chunk": args.worker, "deduplicate_inputs": not args.no_dedup,
              "gradient_checkpointing": cfg["gradient_checkpointing"],
              "gpu": torch.cuda.get_device_name(),
              "model": cfg["model_name"], "max_length":4096, "status":"ok"}

    def step(batch):
        optimizer.zero_grad(set_to_none=True)
        chunks = [{k:v.cuda() for k,v in c.items()} for c in batch["chunks"]]
        mask, targets = batch["allowed"].cuda(), batch["labels"].cuda()
        mapping = batch["representation_indices"].cuda()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        loss = cached_backward(model.encode, chunks,
            lambda z: masked_inbatch_loss(z[mapping], model.temperature, mask, targets)[0],
            lambda value: scaler.scale(value).backward(),
            lambda: torch.autocast("cuda",dtype=torch.float16), checkpointing_model=model.encoder)
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        return {"seconds":time.perf_counter()-start, "loss":float(loss), "grad_norm":float(norm),
                "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30,
                "updates_applied":int(scaler.get_scale()>=before),
                "unique_documents":sum(len(c["input_ids"]) for c in chunks)}
    try:
        step(batches[0])  # warmup, excluded from speed comparison
        report["real_batches"] = [step(batches[i%3]) for i in range(1,args.steps+1)]
        report["median_update_seconds"] = statistics.median(x["seconds"] for x in report["real_batches"])
        report["full_4096_stress"] = step(stress)
        report["safe"] = report["full_4096_stress"]["peak_allocated_gib"] <= args.max_allocated_gib
    except torch.OutOfMemoryError as error:
        report.update(status="oom", safe=False, error=str(error).splitlines()[0])
    except RuntimeError as error:
        if "CUDA" not in str(error):
            raise
        report.update(status="cuda_error",safe=False,error=str(error).splitlines()[0])
    write_json(args.result, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ("real_batches","full_4096_stress")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", nargs="+",type=int,help="Default: 2,4 without checkpointing; otherwise 2,4,8,16")
    parser.add_argument("--steps",type=int,default=2)
    parser.add_argument("--max-allocated-gib",type=float,default=11.5)
    parser.add_argument("--without-checkpointing",action="store_true",help="Benchmark GradCache without additional activation checkpointing")
    parser.add_argument("--output",help="JSON report; default: reports/modernbert-bm25/benchmark/<UTC-run-id>/benchmark*.json")
    parser.add_argument("--worker",type=int,help=argparse.SUPPRESS)
    parser.add_argument("--no-dedup",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--result",help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        return worker(args)
    if args.chunks is None:
        args.chunks = [2,4] if args.without_checkpointing else [2,4,8,16]
    if args.output is None:
        filename = "benchmark_no_checkpoint.json" if args.without_checkpointing else "benchmark.json"
        args.output = str(default_report_directory("benchmark") / filename)
    if args.steps<1 or any(not 1<=c<=32 for c in args.chunks):
        parser.error("Use steps>=1 and chunks in [1,32]")
    output=Path(args.output)
    print(f"Report path: {output}", flush=True)
    profiles=[(2,True)]+[(c,False) for c in args.chunks]
    results=[]
    for chunk,no_dedup in profiles:
        name=f"chunk_{chunk}_{'original_inputs' if no_dedup else 'dedup'}"
        if args.without_checkpointing and no_dedup: name+="_checkpoint_baseline"
        if args.without_checkpointing and not no_dedup: name+="_no_checkpoint"
        path=output.parent/f"{name}.json"
        command=[sys.executable,"-u",str(Path(__file__).resolve()),"--worker",str(chunk),
                 "--steps",str(args.steps),"--max-allocated-gib",str(args.max_allocated_gib),"--result",str(path)]
        if no_dedup: command.append("--no-dedup")
        if args.without_checkpointing: command.append("--without-checkpointing")
        subprocess.run(command,check=True,cwd=REPO_ROOT)
        results.append(read_json(path))
        write_json(output,{"profiles":results,"note":"GPU update timing excludes data preparation/evaluation; full4096 phase is a memory stress test."})
    valid=[r for r in results if r["status"]=="ok" and r["safe"] and r["deduplicate_inputs"]]
    if not valid: raise RuntimeError("No safe deduplicated chunk profile")
    best=min(valid,key=lambda r:r["median_update_seconds"])
    write_json(output,{"profiles":results,"recommended_chunk":best["chunk"],
                       "note":"GPU update timing excludes data preparation/evaluation; full4096 phase is a memory stress test."})
    print("Recommended chunk:",best["chunk"],"median update seconds:",best["median_update_seconds"])


if __name__=="__main__":
    main()
