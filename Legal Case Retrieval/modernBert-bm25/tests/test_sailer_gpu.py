"""Opt-in real ModernBERT-base gradient, 4096-token AMP and resume checks.

BM25_RUN_SAILER_GPU=1 python -m unittest discover -s .../tests -p test_sailer_gpu.py -v
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_DIR, REPO_ROOT, load_config, read_json, small_run_config, write_json


@unittest.skipUnless(os.getenv("BM25_RUN_SAILER_GPU") == "1", "Set BM25_RUN_SAILER_GPU=1 for real ModernBERT CUDA verification")
class SailerRealGPUTests(unittest.TestCase):
    def test_01_real_base_gradients_match_graph_reference(self):
        import torch
        from transformers import AutoTokenizer, set_seed
        from fine_tune.data import prepare
        from fine_tune.gradcache import cached_backward, masked_inbatch_loss
        from fine_tune.modernbert_contrastive_model import ModernBERTContrastive
        from fine_tune.sailer import prepare_label_mask, logical_epoch_rows, LogicalBatchDataset, SailerCollator

        cfg = load_config(PROJECT_DIR / "configs/coliee2025_sailer_gradcache.json")
        bundle = prepare(cfg, write=False)
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], revision=cfg["model_revision"], local_files_only=True)
        manifest = prepare_label_mask(cfg, bundle, tokenizer)
        rows, _ = logical_epoch_rows(bundle["train_labels"], bundle["train_pools"], 0, cfg["sampler_seed"])
        # Same real base, loss and 68 actual documents. Shorter input permits a
        # full graph reference on 16GB; the integration test below uses 4096.
        batch = SailerCollator(tokenizer, dict(cfg, max_length=128), manifest)([LogicalBatchDataset(rows, cfg["document_dir"])[0]])
        chunks = [{k: v.cuda() for k, v in c.items()} for c in batch["chunks"]]
        mask, targets = batch["allowed"].cuda(), batch["labels"].cuda()
        mapping = batch["representation_indices"].cuda()
        set_seed(cfg["seed"])
        model = ModernBERTContrastive.from_base(cfg).to("cuda").train()
        model.encoder.config.reference_compile = False
        torch.cuda.reset_peak_memory_stats()
        graph_chunks = []
        for chunk in chunks:
            # Independent AMP contexts, exactly as cache/replay. Sharing one
            # autocast weight cache across 34 graphs accumulates gradients on
            # the same FP16 cast before FP32 conversion, changing rounding.
            with torch.autocast("cuda", dtype=torch.float16):
                graph_chunks.append(model.encode(chunk).float())
        embeddings = torch.cat(graph_chunks)
        ref_loss, _ = masked_inbatch_loss(embeddings[mapping], model.temperature, mask, targets)
        (ref_loss * 128).backward()
        gradients = {name: p.grad.detach().float().cpu().clone() for name, p in model.named_parameters()}
        expected_loss = float(ref_loss.detach())
        del embeddings, ref_loss, graph_chunks
        model.zero_grad(set_to_none=True)
        actual_loss = cached_backward(model.encode, chunks,
            lambda z: masked_inbatch_loss(z[mapping], model.temperature, mask, targets)[0],
            lambda loss: (loss * 128).backward(), lambda: torch.autocast("cuda", dtype=torch.float16),
            checkpointing_model=model.encoder)
        numerator = denominator = 0.
        per_parameter = {}
        for name, p in model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            actual = p.grad.detach().float().cpu()
            self.assertTrue(torch.isfinite(actual).all(), name)
            expected = gradients[name]
            error = float((actual - expected).double().square().sum())
            norm = float(expected.double().square().sum())
            numerator += error
            denominator += norm
            per_parameter[name] = (error / max(norm, 1e-30)) ** .5
        relative_l2 = (numerator / denominator) ** .5
        print("Gradient comparison: relative L2=", relative_l2,
              "loss difference=", float(actual_loss) - expected_loss, flush=True)
        print("Largest parameter differences:", sorted(per_parameter.items(), key=lambda x: x[1], reverse=True)[:8], flush=True)
        self.assertLess(relative_l2, 1e-5)
        self.assertLess(max(per_parameter.values()), 1e-4)
        self.assertAlmostEqual(float(actual_loss), expected_loss, places=5)
        self.assertGreater(float(model.log_temperature.grad.abs()), 0)
        report = {"passed": True, "model": cfg["model_name"], "documents": 68, "queries": 4,
                  "score_shape": [4, 64], "reference_length": 128, "chunk_documents": cfg["gradcache_chunk_size"],
                  "loss": float(actual_loss), "relative_gradient_l2": relative_l2,
                  "per_parameter_relative_l2": per_parameter,
                  "parameters_compared": len(gradients), "amp_gradient_scale": 128}
        report["reference_compile"] = False
        write_json(Path(cfg["output_dir"]) / "_debug/gradient_equivalence.json", report)
        print("Real-base gradient equivalence:", relative_l2)
        del model, gradients, chunks
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def test_02_real_4096_training_resume_reload_and_amp(self):
        from safetensors import safe_open
        import torch
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        formal = load_config(PROJECT_DIR / "configs/coliee2025_sailer_gradcache.json")
        from config import debug_run_config
        cfg = debug_run_config(formal, train_pairs=9)
        cfg.update(debug_valid_queries=4, debug_candidates=256)
        root = Path(cfg["output_dir"])
        request = root / "requested_config.json"
        write_json(request, cfg)
        script = '''
import sys, random
import numpy as np
import torch
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from config import load_config, write_json
from fine_tune.fine_tune import run_training
from transformers import Trainer, TrainerCallback
original = Trainer._load_rng_state
def check_rng(self, checkpoint):
    original(self, checkpoint)
    stored = torch.load(Path(checkpoint) / 'rng_state.pth', map_location='cpu', weights_only=False)
    assert random.getstate() == stored['python']
    assert torch.equal(torch.get_rng_state(), stored['cpu'])
    assert torch.equal(torch.cuda.get_rng_state(), stored['cuda'])
    actual = np.random.get_state()
    assert actual[0] == stored['numpy'][0] and np.array_equal(actual[1],stored['numpy'][1]) and actual[2:] == stored['numpy'][2:]
    write_json(Path(checkpoint).parent/'rng_verified.json', {'python':True,'numpy':True,'cpu':True,'cuda':True})
Trainer._load_rng_state=check_rng
class Stop(TrainerCallback):
    def on_epoch_end(self,args,state,control,**kwargs):
        if state.epoch >= 1: control.should_training_stop=True
resume = sys.argv[3] if len(sys.argv)>3 else None
run_training(load_config(sys.argv[2]),resume,callbacks=[] if resume else [Stop()])
'''
        cmd = [sys.executable, "-u", "-c", script, str(PROJECT_DIR), str(request)]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)
        first = root / "checkpoint-3"
        self.assertEqual(read_json(first / "trainer_state.json")["epoch"], 1.)
        subprocess.run([*cmd, str(first)], check=True, cwd=REPO_ROOT)
        summary = read_json(root / "summary.json")
        self.assertEqual(summary["global_step"], 6)
        self.assertEqual(summary["observed_max_length"], 4096)
        self.assertGreater(summary["amp"]["updates"], 0)
        self.assertEqual([h["epoch"] for h in read_json(root / "metrics_history.json")], [0., 1., 2.])
        self.assertTrue(all(read_json(root / "rng_verified.json").values()))
        hashes, patience, scheduler, updates, peak = [], [], [], [], []
        for epoch, step in ((1, 3), (2, 6)):
            cp = root / f"checkpoint-{step}"
            audit = read_json(root / "sampling" / f"epoch_{epoch:02d}_train.audit.json")
            received = read_json(root / "sampling" / f"epoch_{epoch:02d}_received.json")
            self.assertEqual(audit["original_pairs"], 9)
            self.assertEqual(audit["padded_pairs"], 3)
            self.assertEqual(received["rows"], 12)
            self.assertTrue(received["matches_planned"])
            hashes.append(audit["sample_hash"])
            gc_state = read_json(cp / "gradcache_state.json")
            self.assertEqual(gc_state["logical_batch_size"], 4)
            self.assertIs(read_json(cp / "contrastive_config.json")["reference_compile"], False)
            state = read_json(cp / "trainer_state.json")
            patience.append(state["stateful_callbacks"]["EarlyStoppingCallback"]["attributes"]["early_stopping_patience_counter"])
            scheduler.append(torch.load(cp / "scheduler.pt", map_location="cpu", weights_only=True)["last_epoch"])
            optimizer = torch.load(cp / "optimizer.pt", map_location="cpu", weights_only=True)
            steps = {int(p["step"]) for p in optimizer["state"].values()}
            self.assertEqual(len(steps), 1)
            updates.append(steps.pop())
            peak.extend(h["gradcache/peak_allocated_gib"] for h in state["log_history"] if "gradcache/peak_allocated_gib" in h)
        self.assertEqual(scheduler, [3, 6])
        self.assertGreater(updates[1], updates[0])
        self.assertNotEqual(hashes[0], hashes[1])
        self.assertEqual(patience[0], 0)
        with safe_open(str(first / "model.safetensors"), framework="pt") as a, safe_open(str(root / "checkpoint-6/model.safetensors"), framework="pt") as b:
            for key in ("encoder.layers.0.attn.Wqkv.weight", "projector.0.weight", "log_temperature"):
                self.assertFalse(torch.equal(a.get_tensor(key), b.get_tensor(key)), key)
        events = EventAccumulator(str(root / "tb")).Reload()
        self.assertEqual([e.step for e in events.Scalars("eval/global_f1")], [0, 3, 6])
        subprocess.run([sys.executable, "-u", str(PROJECT_DIR / "similarity_and_rank.py"),
                        "--run-dir", str(root), "--checkpoint", "best"], check=True, cwd=REPO_ROOT)
        best = Path(summary["best_model_checkpoint"])
        epoch = int(read_json(best / "trainer_state.json")["epoch"])
        self.assertEqual(read_json(root / "rankings" / f"epoch_{epoch:02d}.json"),
                         read_json(root / "standalone" / f"{best.name}.json"))
        write_json(root / "sailer_integration_result.json", {"passed": True, "model": cfg["model_name"],
                   "max_length_observed": summary["observed_max_length"], "score_shape": [4,64],
                   "scheduler_steps": scheduler, "optimizer_updates": updates, "patience": patience,
                   "epoch_sample_hashes": hashes, "checkpoint_reload_identical": True,
                   "peak_allocated_gib": max(peak), "amp": summary["amp"], "rng_verified": True})
        print("SAILER integration report:", root / "sailer_integration_result.json")


if __name__ == "__main__":
    unittest.main()
