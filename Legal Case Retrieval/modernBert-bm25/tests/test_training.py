"""訓練介面與 AMP 回歸；不建立 tiny neural model。"""
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import Trainer, TrainerControl, TrainerState

from config import debug_run_config, load_config, read_json, small_run_config, write_json
from fine_tune.fine_tune import BM25NegativeSamplingTrainer, EpochSamplingCallback, TrainingMonitor, main
from find_best_model import find_best_checkpoint
from similarity_and_rank import evaluate_checkpoint, top_predictions


class AMPTests(unittest.TestCase):
    def setUp(self):
        self.parameter = torch.tensor(1.0, requires_grad=True)
        self.optimizer = torch.optim.SGD([self.parameter], lr=0.1)
        self.scaler = torch.amp.GradScaler("cpu")
        self.trainer = SimpleNamespace(accelerator=SimpleNamespace(scaler=self.scaler))
        self.monitor = TrainingMonitor(self.trainer)
        self.model = SimpleNamespace(parameters=lambda: iter([self.parameter]))
        self.state, self.control = TrainerState(), TrainerControl()

    def step(self, overflow=False):
        self.optimizer.zero_grad(set_to_none=True)
        self.monitor.on_step_begin(None, self.state, self.control)
        self.scaler.scale(self.parameter).backward()
        if overflow:
            self.parameter.grad.fill_(float("inf"))
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_([self.parameter], 1.0)
        self.monitor.on_pre_optimizer_step(None, self.state, self.control)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.monitor.on_optimizer_step(None, self.state, self.control, model=self.model)
        self.state.global_step += 1

    def test_amp_overflow_is_not_intercepted_and_next_step_recovers(self):
        self.monitor.on_train_begin(None, self.state, self.control, model=self.model)
        self.step(True)
        self.assertEqual(self.parameter.item(), 1.0)
        self.assertEqual(self.scaler.get_scale(), 32768.)
        self.step(False)
        self.assertLess(self.parameter.item(), 1.0)
        self.assertEqual(self.monitor.snapshot(), {"attempts": 2, "updates": 1, "skipped": 1, "consecutive": 0})

    def test_eight_consecutive_overflows_stop(self):
        for _ in range(7):
            self.step(True)
        with self.assertRaisesRegex(FloatingPointError, "8 consecutive"):
            self.step(True)

    def test_monitor_restores_and_validates_counters(self):
        self.step(True)
        self.step(False)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "amp.json"
            write_json(path, self.monitor.snapshot())
            restored = TrainingMonitor(self.trainer)
            restored.restore(path)
            self.assertEqual(restored.snapshot(), self.monitor.snapshot())
            restored.on_train_begin(None, self.state, self.control, model=self.model)
            write_json(path, dict(self.monitor.snapshot(), attempts=999))
            with self.assertRaises(ValueError):
                restored.restore(path)

    def test_bad_weights_fail_before_training(self):
        with torch.no_grad():
            self.parameter.fill_(float("nan"))
        with self.assertRaisesRegex(FloatingPointError, "model weights"):
            self.monitor.on_train_begin(None, self.state, self.control, model=self.model)


class TrainingTests(unittest.TestCase):
    def test_loss_removes_metadata_and_reports_nonfinite_before_backward(self):
        owner = BM25NegativeSamplingTrainer.__new__(BM25NegativeSamplingTrainer)
        owner.state = TrainerState(global_step=12)
        for value in (1., float("nan")):
            inputs = {"sample_ids": [{"query_id": "000123"}],
                      "anchor_input": {"input_ids": torch.ones(1, 4096, dtype=torch.long)}}
            with patch.object(Trainer, "compute_loss", return_value=(torch.tensor(value), {})) as compute:
                if value == 1.:
                    self.assertEqual(owner.compute_loss(SimpleNamespace(training=True), inputs).item(), 1.)
                else:
                    with self.assertRaisesRegex(FloatingPointError, "Non-finite forward loss.*000123"):
                        owner.compute_loss(SimpleNamespace(training=True), inputs)
            self.assertNotIn("sample_ids", compute.call_args.args[1])
        self.assertEqual(owner.observed_max_length, 4096)

    def test_debug_flags_call_the_same_function_once_without_child_processes(self):
        with tempfile.TemporaryDirectory() as temporary, patch("fine_tune.fine_tune.run_training") as train:
            main(["--debug", "--debug-train-pairs", "9", "--output-dir", temporary])
            train.assert_called_once()
            self.assertEqual(train.call_args.args[0]["debug_train_pairs"], 9)
            self.assertEqual(train.call_args.args[0]["num_train_epochs"], 2)
        with patch("fine_tune.fine_tune.run_training") as train, patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                main(["--no-debug", "--debug-train-pairs", "9"])
            train.assert_not_called()

    def test_debug_resume_preserves_profile(self):
        with tempfile.TemporaryDirectory() as temporary:
            formal = dict(load_config(), output_dir=temporary)
            config = debug_run_config(formal, train_pairs=1400)
            write_json(Path(config["output_dir"]) / "run_config.json", config)
            checkpoint = Path(config["output_dir"]) / "checkpoint-350"
            self.assertEqual(debug_run_config(formal, checkpoint), config)
            with self.assertRaisesRegex(ValueError, "Cannot change"):
                debug_run_config(formal, checkpoint, train_pairs=8)

    def test_old_debug_resume_does_not_silently_expand_validation_scope(self):
        from fine_tune.data import select_debug_data
        with tempfile.TemporaryDirectory() as temporary:
            formal = dict(load_config(), output_dir=temporary)
            legacy = small_run_config(formal, Path(temporary) / "_debug/old", train_pairs=8)
            legacy.pop("debug_valid_queries")
            legacy.pop("debug_candidates")
            write_json(Path(legacy["output_dir"]) / "run_config.json", legacy)
            restored = debug_run_config(formal, Path(legacy["output_dir"]) / "checkpoint-2")
            self.assertEqual(restored, legacy)
            ids = [f"{i:06d}" for i in range(1, 2000)]
            bundle = {"manifest": {"candidate_ids": ids}, "train_labels": {"000001": ids[1:201]},
                      "valid_labels": {f"{i:06d}": [f"{i+100:06d}"] for i in range(500, 530)}}
            selected = select_debug_data(restored, bundle)
            self.assertEqual((len(selected["pairs"]), len(selected["valid_labels"]), len(selected["candidate_ids"])),
                             (8, 4, 256))

    def test_debug_scope_fields_are_validated_and_not_allowed_for_formal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            config = small_run_config(load_config(), temporary)
            for key, value in (("debug_valid_queries", 0), ("debug_candidates", "1000"),
                               ("debug_valid_queries", 337), ("debug_candidates", 7350)):
                write_json(path, dict(config, **{key: value}))
                with self.assertRaises(ValueError):
                    load_config(path)
            write_json(path, dict(load_config(), debug_candidates=1000))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_three_optimizer_groups_match_fp_learning_rates(self):
        owner = BM25NegativeSamplingTrainer.__new__(BM25NegativeSamplingTrainer)
        owner.optimizer = None
        owner.config = load_config()
        parameters = [(name, torch.tensor(1., requires_grad=True)) for name in
                      ("encoder.weight", "projector.0.bias", "log_temperature")]
        owner.model = SimpleNamespace(named_parameters=lambda: iter(parameters))
        groups = owner.create_optimizer().param_groups
        self.assertEqual([g["lr"] for g in groups], [5e-6, 5e-6, 5e-4])
        self.assertEqual([g["weight_decay"] for g in groups], [0., 0., 0.])
        self.assertEqual([len(g["params"]) for g in groups], [1, 1, 1])

    def test_worker_or_epoch_mismatch_is_not_silently_accepted(self):
        owner = SimpleNamespace(train_dataset=SimpleNamespace(rows=[{"query_id": "1", "positive_id": "2"}]),
                                epoch_seen_samples=[])
        with self.assertRaisesRegex(ValueError, "resampled"):
            EpochSamplingCallback(owner).on_epoch_end(None, TrainerState(epoch=1.), TrainerControl())

    def test_best_checkpoint_excludes_epoch0_and_nested_debug_and_ties_choose_first(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for step, epoch, f1 in ((0, 0, 1.), (2, 1, .2), (4, 2, .2), (6, 3, .1)):
                write_json(root / f"checkpoint-{step}" / "trainer_state.json", {"global_step": step,
                    "log_history": [{"step": step, "epoch": epoch, "eval_global_f1": f1}]})
            write_json(root / "_debug/x/checkpoint-99/trainer_state.json", {"global_step": 99,
                "log_history": [{"step": 99, "epoch": 1, "eval_global_f1": 1.}]})
            checkpoint, value = find_best_checkpoint(root)
            self.assertEqual(Path(checkpoint).name, "checkpoint-2")
            self.assertEqual(value, .2)

    def test_cached_embeddings_must_match_checkpoint_config_and_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "embeddings.pt"
            torch.save({"metadata": {"weight_sha256": "wrong"}}, path)
            bundle = {"manifest": {"candidate_ids": ["a"]}}
            with patch("similarity_and_rank.evaluation_context", return_value=(Path(temporary), bundle, {"weight_sha256": "correct"})):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    evaluate_checkpoint({"output_dir": temporary}, embeddings=path)

    def test_ranking_rejects_nonfinite_scores(self):
        with self.assertRaises(FloatingPointError):
            top_predictions("q", ["p"], [float("nan")])


if __name__ == "__main__":
    unittest.main()
