"""與 modernBert-fp/inference.py 對應：載入完整 checkpoint，編碼 processed 文件。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from config import DEFAULT_CONFIG, digest, load_config, read_json
from fine_tune.data import prepare, select_debug_data, sha256_file
from fine_tune.modernbert_contrastive_model import ModernBERTContrastive
from find_best_model import find_best_checkpoint


def evaluation_context(config, checkpoint, legacy=False):
    """拒絕混用正式/debug 資料範圍，並核對訓練時的資料快照。"""
    if str(checkpoint) == "best":
        checkpoint, _ = find_best_checkpoint(config["output_dir"])
    checkpoint = Path(checkpoint).resolve()
    source = checkpoint.parent / "run_config.json"
    if source.exists():
        if legacy:
            raise ValueError("--legacy is only for historical checkpoints without a run_config.json")
        if read_json(source) != config:
            raise ValueError("Checkpoint belongs to another run/config; use its own --run-dir")
    elif not legacy:
        raise ValueError("Missing checkpoint run_config.json; historical checkpoints require explicit --legacy")
    bundle = prepare(config, write=False)
    manifest = checkpoint.parent / "run_manifest.json"
    if manifest.exists() and read_json(manifest)["data_hash"] != bundle["manifest"]["data_hash"]:
        raise ValueError("Evaluation data hash changed since training")
    if config["mode"] == "debug":
        bundle = select_debug_data(config, bundle)
    weights = checkpoint / "model.safetensors"
    if not weights.exists():
        weights = checkpoint / "pytorch_model.bin"
    metadata = {"format_version": 2, "checkpoint": str(checkpoint), "weight_sha256": sha256_file(weights),
                "data_hash": bundle["manifest"]["data_hash"], "config_hash": digest(config)}
    return checkpoint, bundle, metadata


def load_model(config, checkpoint, legacy=False):
    if not torch.cuda.is_available():
        raise RuntimeError("Use CUDA + FP16 + Flash Attention 2 for checkpoint inference")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = config["model_name"] if legacy else str(checkpoint)
    kwargs = {"revision": config["model_revision"]} if legacy else {}
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=config["local_files_only"], **kwargs)
    model = ModernBERTContrastive.from_checkpoint(checkpoint, config["attention_implementation"], config if legacy else None)
    if config.get("training_method") == "sailer_gradcache":
        # HF ModernBertConfig drops reference_compile when serializing; keep
        # the training/standalone execution policy identical for this method.
        model.encoder.config.reference_compile = False
    # 與訓練共用 FP32 weights + FP16 autocast，不另轉成全 FP16 權重。
    return model.to("cuda", dtype=torch.float32).eval(), tokenizer


def encode_documents(model, tokenizer, config, ids, description, position=0):
    device = next(model.parameters()).device
    vectors = []
    size = config["retrieval_batch_size"]
    with tqdm(total=len(ids), desc=description, unit="doc", position=position,
              dynamic_ncols=True, disable=False) as bar:
        for start in range(0, len(ids), size):
            texts = [(Path(config["document_dir"]) / f"{pid}.txt").read_text(encoding="utf-8").strip()
                     for pid in ids[start:start + size]]
            tokens = tokenizer(texts, padding=True, truncation=True,
                               max_length=config["max_length"], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in tokens.items() if k in ("input_ids", "attention_mask")}
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=config["fp16"] and device.type == "cuda"):
                encoded = model.encode(inputs)
            if not torch.isfinite(encoded).all():
                raise FloatingPointError(f"Non-finite embeddings: documents={ids[start:start + size]}")
            vectors.append(encoded.float().cpu())
            bar.update(len(texts))
    return torch.cat(vectors)


def encode_corpus(model, tokenizer, config, relevance, candidate_ids, stage, position=0):
    candidates = sorted(candidate_ids)
    if not candidates or len(candidates) != len(set(candidates)) or not relevance:
        raise ValueError("Empty/duplicate candidates or empty evaluation queries")
    missing = {p for ps in relevance.values() for p in ps} - set(candidates)
    if missing:
        raise ValueError(f"Evaluation corpus omits positives: {sorted(missing)[:10]}")
    previous = model.training
    model.eval()
    try:
        with torch.inference_mode():
            vectors = encode_documents(model, tokenizer, config, candidates, f"{stage} | Encoding candidates", position)
            absent = sorted(set(relevance) - set(candidates))
            if absent:
                extra = encode_documents(model, tokenizer, config, absent, f"{stage} | Encoding queries", position)
                vectors = torch.cat([vectors, extra])
        # Query 與 candidate 都是同一份 processed，故重用同一向量。
        return {"candidate_ids": candidates, "vector_ids": candidates + absent, "vectors": vectors}
    finally:
        model.train(previous)


def add_evaluation_arguments(parser):
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--config")
    source.add_argument("--run-dir", help="Read this run's saved config; use this for debug checkpoints")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--legacy", action="store_true", help="Strictly load historical 0911 weights")
    parser.add_argument("--destination", help="Output prefix (without suffix)")


def evaluation_config(parser, args):
    if args.run_dir and args.output_dir:
        parser.error("--run-dir cannot be combined with --output-dir")
    return (load_config(Path(args.run_dir) / "run_config.json") if args.run_dir
            else load_config(args.config or DEFAULT_CONFIG, args.output_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_arguments(parser)
    args = parser.parse_args()
    config = evaluation_config(parser, args)
    checkpoint, bundle, metadata = evaluation_context(config, args.checkpoint, args.legacy)
    model, tokenizer = load_model(config, checkpoint, args.legacy)
    packet = encode_corpus(model, tokenizer, config, bundle["valid_labels"],
        bundle.get("candidate_ids", bundle["manifest"]["candidate_ids"]), checkpoint.name)
    packet["metadata"] = metadata
    prefix = Path(args.destination) if args.destination else Path(config["output_dir"]) / "standalone" / checkpoint.name
    destination = prefix.with_suffix(".embeddings.pt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(packet, destination)
    print(f"Embeddings saved: {destination}")


if __name__ == "__main__":
    main()
