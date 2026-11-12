"""Opt-in real CUDA fused-AdamW overflow recovery; no substitute neural model."""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@unittest.skipUnless(os.getenv("BM25_RUN_GPU_TESTS") == "1", "Set BM25_RUN_GPU_TESTS=1 for the CUDA AMP regression")
class CUDAAMPTests(unittest.TestCase):
    def test_actual_accelerate_fused_adamw_recovers_from_overflow(self):
        import torch
        from accelerate import Accelerator
        from transformers import TrainerControl, TrainerState
        from fine_tune.fine_tune import TrainingMonitor

        self.assertTrue(torch.cuda.is_available(), "CUDA regression requires a real GPU")
        accelerator = Accelerator(mixed_precision="fp16")
        parameter = torch.tensor(1.0, device="cuda", requires_grad=True)
        optimizer = accelerator.prepare_optimizer(torch.optim.AdamW([parameter], lr=0.01, weight_decay=0, fused=True))
        model = SimpleNamespace(parameters=lambda: iter([parameter]))
        state, control = TrainerState(), TrainerControl()
        with tempfile.TemporaryDirectory() as temporary:
            trainer = SimpleNamespace(accelerator=accelerator, run_dir=Path(temporary), config={"mode": "debug"})
            monitor = TrainingMonitor(trainer)
            monitor.on_train_begin(None, state, control, model=model)
            for overflow in (True, False):
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(parameter * 1.0)
                if overflow:
                    parameter.grad.fill_(float("inf"))
                accelerator.clip_grad_norm_([parameter], 1.0)
                monitor.on_pre_optimizer_step(None, state, control)
                optimizer.step()
                monitor.on_optimizer_step(None, state, control, model=model, optimizer=optimizer)
                if overflow:
                    self.assertEqual(parameter.item(), 1.0)
                    self.assertEqual(accelerator.scaler.get_scale(), 32768.0)
                    print("CUDA fused AdamW overflow: scale reduced; Accelerate step_was_skipped =", optimizer.step_was_skipped)
                state.global_step += 1
            self.assertLess(parameter.item(), 1.0)
            self.assertEqual((monitor.attempts, monitor.updates, monitor.skipped), (2, 1, 1))
            self.assertEqual(int(optimizer.state[parameter]["step"]), 1)


if __name__ == "__main__":
    unittest.main()
