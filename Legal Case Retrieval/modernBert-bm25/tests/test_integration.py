"""選用的真實 ModernBERT-base 中斷／續訓測試；一般 debug 不執行子程序。"""
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PROJECT_DIR, REPO_ROOT, debug_run_config, load_config, read_json, write_json


def compare_fp_reference(config_path, checkpoint):
    """只擷取成功程式的模型 class，不執行其 .env/NVML/global 初始化。"""
    import ast
    import numpy as np
    import torch
    from torch import nn
    from typing import Dict, Optional
    from inference import load_model
    from fine_tune.data import ContrastiveCollator, ContrastiveDataset

    source = REPO_ROOT / "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
    tree = ast.parse(source.read_text())
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ModernBERTContrastive")
    namespace = {"nn": nn, "torch": torch, "np": np, "Dict": Dict, "Optional": Optional}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
    config = load_config(config_path)
    model, tokenizer = load_model(config, checkpoint)
    reference = namespace["ModernBERTContrastive"].__new__(namespace["ModernBERTContrastive"])
    nn.Module.__init__(reference)
    reference.encoder, reference.projector = model.encoder, model.projector
    reference.log_temperature = model.log_temperature
    reference.temperature_min, reference.temperature_max = model.temperature_min, model.temperature_max
    reference.print_gpu_status = lambda *args: None
    reference.eval()
    rows = read_json(Path(config["output_dir"]) / "sampling/epoch_01_train.json")[:1]
    batch = ContrastiveCollator(tokenizer)([ContrastiveDataset(rows, config["document_dir"])[0]])
    batch.pop("sample_ids")
    batch = {k: ({name: value.cuda() for name, value in part.items()} if isinstance(part, dict) else part.cuda())
             for k, part in batch.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        actual, expected = model(**batch), reference(**batch)
    torch.testing.assert_close(actual["logits"], expected["logits"], rtol=0, atol=0)
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=0, atol=0)
    return {"logits_identical": True, "loss_identical": True,
            "loss": float(actual["loss"]), "sequence_length": batch["anchor_input"]["input_ids"].shape[-1]}


@unittest.skipUnless(os.getenv("BM25_RUN_INTEGRATION") == "1", "Set BM25_RUN_INTEGRATION=1 for the real-base resume test")
class ModernBERTIntegrationTests(unittest.TestCase):
    def test_real_base_epoch_resampling_checkpoint_resume_and_reload(self):
        import torch
        from safetensors import safe_open
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

        config = dict(debug_run_config(load_config(), train_pairs=9), debug_valid_queries=4, debug_candidates=256)
        root = Path(config["output_dir"])
        requested = root / "requested_config.json"
        write_json(requested, config)
        script = '''
import sys
sys.path.insert(0, sys.argv[1])
from config import load_config
from fine_tune.fine_tune import run_training
from transformers import Trainer, TrainerCallback
import random
import numpy as np
import torch
from pathlib import Path
from config import write_json
original_restore_rng = Trainer._load_rng_state
def verified_restore_rng(self, checkpoint):
    original_restore_rng(self, checkpoint)
    saved = torch.load(Path(checkpoint) / "rng_state.pth", map_location="cpu", weights_only=False)
    assert random.getstate() == saved["python"]
    actual_numpy = np.random.get_state()
    assert actual_numpy[0] == saved["numpy"][0]
    assert np.array_equal(actual_numpy[1], saved["numpy"][1])
    assert actual_numpy[2:] == saved["numpy"][2:]
    assert torch.equal(torch.random.get_rng_state(), saved["cpu"])
    assert torch.equal(torch.cuda.get_rng_state(), saved["cuda"])
    write_json(Path(checkpoint).parent / "rng_restore_verified.json", {"python": True, "numpy": True, "cpu": True, "cuda": True})
Trainer._load_rng_state = verified_restore_rng
class InterruptForTest(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if state.epoch >= 1:
            control.should_training_stop = True
config = load_config(sys.argv[2])
resume = sys.argv[3] if len(sys.argv) > 3 else None
run_training(config, resume, callbacks=[] if resume else [InterruptForTest()])
'''
        command = [sys.executable, "-u", "-c", script, str(PROJECT_DIR), str(requested)]
        subprocess.run(command, check=True)
        first = root / "checkpoint-3"
        self.assertEqual(read_json(first / "trainer_state.json")["epoch"], 1.)
        subprocess.run([*command, str(first)], check=True)
        summary = read_json(root / "summary.json")
        rng = read_json(root / "rng_restore_verified.json")
        self.assertTrue(all(rng.values()))
        self.assertEqual(summary["global_step"], 6)
        self.assertEqual(summary["observed_max_length"], 4096)
        history = read_json(root / "metrics_history.json")
        self.assertEqual([row["epoch"] for row in history], [0., 1., 2.])
        hashes, scheduler_steps, applied_steps, patience = [], [], [], []
        for epoch, step in ((1, 3), (2, 6)):
            audit = read_json(root / "sampling" / f"epoch_{epoch:02d}_train.audit.json")
            received = read_json(root / "sampling" / f"epoch_{epoch:02d}_received.json")
            self.assertEqual(received, {"rows": 9, "sample_hash": audit["sample_hash"], "matches_planned": True})
            hashes.append(audit["sample_hash"])
            checkpoint = root / f"checkpoint-{step}"
            state = read_json(checkpoint / "trainer_state.json")
            monitor = read_json(checkpoint / "amp_monitor.json")
            optimizer = torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True)
            applied = int(next(iter(optimizer["state"].values()))["step"])
            applied_steps.append(applied)
            self.assertEqual(monitor["attempts"], step)
            self.assertEqual(monitor["updates"], applied)
            scheduler = torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=True)
            self.assertEqual(scheduler["base_lrs"], [5e-6, 5e-6, 5e-4])
            scheduler_steps.append(scheduler["last_epoch"])
            scaler = torch.load(checkpoint / "scaler.pt", map_location="cpu", weights_only=True)
            self.assertTrue(math.isfinite(scaler["scale"]) and scaler["scale"] > 0)
            patience.append(state["stateful_callbacks"]["EarlyStoppingCallback"]["attributes"]["early_stopping_patience_counter"])
        self.assertNotEqual(*hashes)
        self.assertEqual(scheduler_steps, [3, 6])
        self.assertGreater(applied_steps[1], applied_steps[0])
        self.assertEqual(patience[0], 0)
        self.assertEqual(patience[1], 0 if history[2]["eval_global_f1"] > history[1]["eval_global_f1"] else 1)
        with safe_open(str(first / "model.safetensors"), framework="pt") as a, \
             safe_open(str(root / "checkpoint-6/model.safetensors"), framework="pt") as b:
            for name in ("encoder.layers.0.attn.Wqkv.weight", "projector.0.weight", "log_temperature"):
                self.assertFalse(torch.equal(a.get_tensor(name), b.get_tensor(name)), name)
        events = EventAccumulator(str(root / "tb")).Reload()
        self.assertEqual([row.step for row in events.Scalars("train/loss")], list(range(1, 7)))
        self.assertEqual([row.step for row in events.Scalars("eval/global_f1")], [0, 3, 6])
        self.assertTrue(all(math.isfinite(row.value) for row in events.Scalars("train/loss")))
        subprocess.run([sys.executable, "-u", str(PROJECT_DIR / "similarity_and_rank.py"),
                        "--run-dir", str(root), "--checkpoint", "best"], check=True)
        best = Path(summary["best_model_checkpoint"])
        reloaded = read_json(root / "standalone" / f"{best.name}.json")
        trained_epoch = int(read_json(best / "trainer_state.json")["epoch"])
        training_result = read_json(root / "rankings" / f"epoch_{trained_epoch:02d}.json")
        self.assertEqual(reloaded, training_result)
        comparison_script = '''
import sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + "/tests")
from test_integration import compare_fp_reference
from config import write_json
write_json(sys.argv[4], compare_fp_reference(sys.argv[2], sys.argv[3]))
'''
        subprocess.run([sys.executable, "-u", "-c", comparison_script, str(PROJECT_DIR), str(requested),
                        str(best), str(root / "fp_reference_comparison.json")], check=True)
        write_json(root / "integration_result.json", {"passed": True, "train_pairs": 9,
            "epochs": [0, 1, 2], "scheduler_steps": scheduler_steps,
            "applied_steps": applied_steps, "patience": patience,
            "sample_hashes": hashes, "checkpoint_reload_identical": True,
            "observed_max_length": summary["observed_max_length"],
            "rng_restored": rng,
            "fp_reference": read_json(root / "fp_reference_comparison.json")})
        print(f"Integration report: {root / 'integration_result.json'}")


if __name__ == "__main__":
    unittest.main()
