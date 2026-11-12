"""Behavior tests without constructing a substitute/tiny neural network."""
import os
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, Trainer, TrainerControl, TrainerState
from types import SimpleNamespace

from fine_tune.data import (audit_samples, build_pools, prepare, sample_epoch, sha256_file,
                            ContrastiveCollator, ContrastiveDataset, select_debug_data)
from fine_tune.fine_tune import ConsoleProgressCallback, main, validate_resume
from similarity_and_rank import evaluate_model_retrieval, score_predictions, top_predictions
from config import (DEFAULT_CONFIG, IMPLEMENTATION, debug_run_config, load_config,
                    read_json, small_run_config, write_json)


class NumericTokenizer:
    """Expose document identity in collator/worker tests, not a model."""
    def __call__(self, texts, **kwargs):
        tokens = torch.tensor([[int(text)] for text in texts])
        return {"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.positives = {"000001": ["000002", "000003"]}
        self.pools = {"000001": [str(i).zfill(6) for i in range(4, 100)]}

    def test_reproducible_epoch_and_order_independent(self):
        a = sample_epoch(self.positives, self.pools, 0)
        self.assertEqual(a, sample_epoch(self.positives, self.pools, 0))
        self.assertNotEqual(a, sample_epoch(self.positives, self.pools, 1))
        b = sample_epoch(self.positives, self.pools, 0, pairs=[("000001", "000003"), ("000001", "000002")])
        self.assertEqual(a, b[::-1])
        self.assertTrue(all(len(set(r["negative_ids"])) == 15 for r in a))

    def test_raw_top100_no_refill_and_deduplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bm25.tsv"
            lines = [f"1 Q0 {i} {i} {100-i} BM25" for i in range(1, 102)]
            lines.append("1 Q0 4 4 96 BM25")
            path.write_text("\n".join(lines))
            pool = build_pools(path, self.positives)["000001"]
            self.assertEqual(len(pool), 97)
            self.assertNotIn("000101", pool)
            self.assertFalse(set(pool) & {"000001", "000002", "000003"})

    def test_insufficient_pool_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bm25.tsv"
            path.write_text("1 Q0 2 1 1 BM25\n")
            with self.assertRaisesRegex(ValueError, "only 0"):
                build_pools(path, self.positives)

    def test_conflicting_raw_ranks_cannot_expand_top100(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bm25.tsv"
            path.write_text("1 Q0 4 1 1 BM25\n1 Q0 5 1 0.9 BM25\n")
            with self.assertRaisesRegex(ValueError, "Conflicting documents"):
                build_pools(path, self.positives)

    def test_audit_detects_each_leak(self):
        for bad_id in ("000001", "000002", "999999"):
            rows = sample_epoch(self.positives, self.pools, 0)
            rows[0]["negative_ids"][0] = bad_id
            with self.assertRaises(ValueError):
                audit_samples(rows, self.positives, self.pools)


class MetricsTests(unittest.TestCase):
    def test_micro_f1_dedup_and_missing_query(self):
        metrics, rows = score_predictions({"q1": ["a", "a", "b"], "q2": ["c"]}, {"q1": ["a", "x"]})
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 1, 2))
        self.assertAlmostEqual(metrics["f1"], 0.4)
        self.assertEqual(metrics["num_queries"], 2)
        self.assertEqual(metrics["missing_predictions"], 1)
        self.assertEqual(rows["q2"]["fn"], 1)

    def test_top5_excludes_self_and_resolves_ties(self):
        ids = ["q", "e", "d", "c", "b", "a", "z"]
        result = top_predictions("q", ids, [100, 1, 1, 1, 1, 1, -1], 5)
        self.assertEqual(result, ["a", "b", "c", "d", "e"])

    def test_evaluation_rejects_empty_or_incomplete_corpus_before_encoding(self):
        for relevance, candidates in (({}, ["p"]), ({"q": ["p"]}, []), ({"q": ["p"]}, ["other"])):
            with self.assertRaises(ValueError):
                evaluate_model_retrieval(None, None, {}, relevance, candidates, "unused")
        with self.assertRaisesRegex(ValueError, "counts disagree"):
            top_predictions("q", ["p", "r"], [0.9])


class DatasetTests(unittest.TestCase):
    def test_worker_dataset_is_refreshed_before_each_iterator(self):
        positives = {"000001": ["000002"]}
        pools = {"000001": [str(i).zfill(6) for i in range(3, 100)]}
        with tempfile.TemporaryDirectory() as temporary:
            for i in range(1, 100):
                (Path(temporary) / f"{i:06d}.txt").write_text(str(i))
            first = sample_epoch(positives, pools, 0)
            second = sample_epoch(positives, pools, 1)
            dataset = ContrastiveDataset(first, temporary)
            loader = DataLoader(dataset, batch_size=1, num_workers=2, persistent_workers=False,
                                collate_fn=ContrastiveCollator(NumericTokenizer()))
            one = next(iter(loader))
            dataset.rows = second
            two = next(iter(loader))
            self.assertEqual(one["negative_input"]["input_ids"].flatten().tolist(), [int(x) for x in first[0]["negative_ids"]])
            self.assertEqual(two["negative_input"]["input_ids"].flatten().tolist(), [int(x) for x in second[0]["negative_ids"]])
            self.assertEqual(one["anchor_input"]["input_ids"].item(), 1)
            self.assertEqual(one["positive_input"]["input_ids"].item(), 2)
            self.assertEqual(one["labels"].tolist(), [0])

    def test_collator_multiple_pairs_alignment(self):
        batch = [{"query_text": str(q), "positive_text": str(q + 10),
                  "negative_texts": [str(q * 100 + n) for n in range(15)]} for q in (1, 2)]
        out = ContrastiveCollator(NumericTokenizer())(batch)
        self.assertEqual(out["anchor_input"]["input_ids"].flatten().tolist(), [1, 2])
        self.assertEqual(out["positive_input"]["input_ids"].flatten().tolist(), [11, 12])
        self.assertEqual(out["negative_input"]["input_ids"].shape, (30, 1))
        self.assertEqual(out["labels"].tolist(), [0, 0])


class ProgressTests(unittest.TestCase):
    def test_single_training_bar_resume_epoch_and_optimizer_only_counts(self):
        callback = ConsoleProgressCallback()
        state = TrainerState(global_step=1356, max_steps=27120, epoch=1.0)
        args = SimpleNamespace(num_train_epochs=20)
        control = TrainerControl(should_log=True)
        with patch("fine_tune.fine_tune.tqdm") as progress:
            training_bar = progress.return_value
            callback.on_train_begin(args, state, control)
            callback.on_epoch_begin(args, state, control, train_dataloader=list(range(5)))
            # Four accumulated batches, then a final single-batch update.
            for _ in range(3):
                callback.on_substep_end(args, state, control)
            training_bar.update.assert_not_called()
            state.global_step += 1
            callback.on_step_end(args, state, control)
            state.global_step += 1
            callback.on_step_end(args, state, control)
            self.assertEqual([call.args[0] for call in training_bar.update.call_args_list], [1, 1])
            self.assertEqual(progress.call_args_list[0].kwargs["initial"], 1356)
            self.assertEqual(progress.call_args_list[0].kwargs["total"], 27120)
            self.assertEqual(progress.call_args_list[0].kwargs["desc"], "Training | epoch 2/20")
            self.assertTrue(all(call.kwargs["disable"] is False for call in progress.call_args_list))
            self.assertTrue(control.should_log)  # UI must not consume Trainer logging/control.
            callback.on_epoch_end(args, state, control)
            state.epoch = 2.0
            callback.on_epoch_begin(args, state, control, train_dataloader=list(range(5)))
            training_bar.set_description.assert_called_with("Training | epoch 3/20")
            self.assertEqual(progress.call_count, 1)  # No separate per-epoch batch bar.
            callback.on_train_end(args, state, control)
            training_bar.close.assert_called_once()
            callback.close()  # Cleanup is idempotent after a normal finish.

    def test_epoch_label_for_fresh_formal_debug_and_completed_run(self):
        for epoch, total, expected in ((None, 20, "1/20"), (0.0, 2, "1/2"),
                                        (1.0, 2, "2/2"), (20.0, 20, "20/20")):
            with self.subTest(epoch=epoch, total=total):
                description = ConsoleProgressCallback.training_description(
                    SimpleNamespace(num_train_epochs=total), TrainerState(epoch=epoch))
                self.assertEqual(description, f"Training | epoch {expected}")

    def test_epoch_zero_sampled_loss_is_visible_and_closed_before_encoding(self):
        callback = ConsoleProgressCallback()
        state, control = TrainerState(epoch=0.0), TrainerControl()
        with patch("fine_tune.fine_tune.tqdm") as progress:
            for _ in range(3):
                callback.on_prediction_step(None, state, control, eval_dataloader=[1, 2, 3])
            self.assertEqual(progress.call_count, 1)
            self.assertEqual(progress.call_args.kwargs["desc"], "Epoch 0 | Sampled valid loss")
            self.assertEqual(progress.call_args.kwargs["position"], 0)
            self.assertEqual(progress.return_value.update.call_count, 3)
            callback.close_prediction_bar()
            self.assertIsNone(callback.prediction_bar)
            callback.on_evaluate(None, state, control)
            progress.return_value.close.assert_called_once()

    def test_non_primary_process_does_not_create_bars(self):
        callback = ConsoleProgressCallback()
        state = TrainerState(is_world_process_zero=False)
        control = TrainerControl()
        with patch("fine_tune.fine_tune.tqdm") as progress:
            callback.on_train_begin(None, state, control)
            callback.on_epoch_begin(None, state, control, train_dataloader=[1])
            callback.on_prediction_step(None, state, control, eval_dataloader=[1])
            callback.on_step_end(None, state, control)
            callback.on_train_end(None, state, control)
            progress.assert_not_called()

    def test_retrieval_bars_count_documents_queries_and_keep_scores_unchanged(self):
        from functools import partial
        from tqdm.auto import tqdm as real_tqdm

        # Precomputed embeddings, not a substitute neural network/GPU smoke test.
        vectors = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        model = Mock(training=True)
        model.parameters.side_effect = lambda: iter([torch.tensor(0.)])
        model.encode.side_effect = lambda inputs: vectors[inputs["input_ids"].flatten() - 1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for i in range(1, 5):
                (root / f"{i}.txt").write_text(str(i))
            config = {"document_dir": str(root), "retrieval_batch_size": 2, "max_length": 4096,
                      "fp16": False, "retrieval_top_k": 1, "mode": "debug"}
            output = io.StringIO()
            with patch("inference.tqdm", side_effect=partial(real_tqdm, file=output, mininterval=0)) as progress, \
                 patch("similarity_and_rank.tqdm", side_effect=partial(real_tqdm, file=output, mininterval=0)):
                metrics = evaluate_model_retrieval(model, NumericTokenizer(), config,
                    {"1": ["2"], "4": ["3"]}, ["3", "1", "2"], root / "epoch_00")
            self.assertEqual(metrics["f1"], 1.0)
            self.assertEqual(model.encode.call_count, 3)  # Candidate batches 2+1; only absent q4 encoded.
            self.assertIn("Encoding candidates", output.getvalue())
            self.assertIn("3/3", output.getvalue())
            self.assertIn("Encoding queries", output.getvalue())
            self.assertIn("Ranking valid queries", output.getvalue())
            self.assertIn("2/2", output.getvalue())
            self.assertTrue(all(call.kwargs["disable"] is False for call in progress.call_args_list))
            model.train.assert_called_once_with(True)


class StateTests(unittest.TestCase):
    def test_monitoring_does_not_consume_training_loss_log(self):
        from fine_tune.fine_tune import BM25NegativeSamplingTrainer
        trainer = BM25NegativeSamplingTrainer.__new__(BM25NegativeSamplingTrainer)
        trainer.model = SimpleNamespace(temperature=torch.tensor(1.0))
        trainer.optimizer = None
        trainer.monitor_metrics = {"amp/skipped_updates": 2}
        with patch.object(Trainer, "log") as log:
            trainer.log({"loss": 2.0})
        values = log.call_args.args[0]
        self.assertEqual(values["loss"], 2.0)
        self.assertEqual(values["model/temperature"], 1.0)
        self.assertEqual(values["amp/skipped_updates"], 2)

    def test_option_a_and_patience_serialization(self):
        args = SimpleNamespace(metric_for_best_model="eval_global_f1", greater_is_better=True)
        callback = EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=0.0)
        state, control = TrainerState(), TrainerControl()
        callback.on_evaluate(args, state, control, {"eval_global_f1": 0.9})
        self.assertIsNone(state.best_metric)  # epoch0: evaluate() without best-model determination
        callback.on_evaluate(args, state, control, {"eval_global_f1": 0.1})
        self.assertEqual(callback.early_stopping_patience_counter, 0)
        state.best_metric = 0.1  # trainer's epoch1 _determine_best_metric
        for _ in range(2):
            callback.on_evaluate(args, state, control, {"eval_global_f1": 0.1})
        callback = EarlyStoppingCallback.from_state(callback.state())
        self.assertEqual(callback.early_stopping_patience_counter, 2)
        callback.on_evaluate(args, state, control, {"eval_global_f1": 0.2})
        self.assertEqual(callback.early_stopping_patience_counter, 0)
        state.best_metric = 0.2
        for _ in range(5):
            callback.on_evaluate(args, state, control, {"eval_global_f1": 0.19})
        self.assertTrue(control.should_training_stop)

    def test_resume_rejects_changed_data_config_and_incomplete_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"output_dir": str(root), "learning_rate": 5e-6, "num_train_epochs": 2}
            write_json(root / "run_config.json", config)
            write_json(root / "run_manifest.json", {"data_hash": "frozen", "implementation": IMPLEMENTATION})
            write_json(root / "metrics_history.json", [{"epoch": 0}])
            checkpoint = root / "checkpoint-2"
            checkpoint.mkdir()
            for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "model.safetensors", "scaler.pt",
                         "config.json", "contrastive_config.json", "tokenizer_config.json"):
                (checkpoint / name).touch()
            write_json(checkpoint / "amp_monitor.json", {"attempts": 2, "updates": 2, "skipped": 0, "consecutive": 0})
            write_json(checkpoint / "trainer_state.json", {"epoch": 1.0, "global_step": 2})
            self.assertEqual(validate_resume(config, {"data_hash": "frozen"}, checkpoint), checkpoint)
            with self.assertRaisesRegex(ValueError, "data hash"):
                validate_resume(config, {"data_hash": "changed"}, checkpoint)
            with self.assertRaisesRegex(ValueError, "configuration"):
                validate_resume(dict(config, learning_rate=1e-3), {"data_hash": "frozen"}, checkpoint)
            (root / "checkpoint-4").mkdir()
            with self.assertRaisesRegex(ValueError, "Later checkpoints"):
                validate_resume(config, {"data_hash": "frozen"}, checkpoint)
            write_json(checkpoint / "trainer_state.json", {"epoch": 1.5, "global_step": 2})
            with self.assertRaisesRegex(ValueError, "epoch-boundary"):
                validate_resume(config, {"data_hash": "frozen"}, checkpoint)

    def test_root_2026_environment_cannot_override_config(self):
        with patch.dict(os.environ, {"COLIEE_TASK1_YEAR": "2026", "TASK1_SCOPE_FILTER": "1",
                                     "TASK1_BASE_ENCODER_DIR": "/bad/FP", "LCR_QUERY_CANDIDATE_SCOPE_JSON": "/bad/scope"}):
            config = load_config()
            self.assertIn("/2025/processed", config["document_dir"])
            self.assertEqual(config["model_name"], "answerdotai/ModernBERT-base")
            self.assertTrue(config["output_dir"].endswith("modernBERT_bm25_top100_epoch_resample15_2025"))


class DebugTests(unittest.TestCase):
    def test_small_profile_preserves_model_and_formal_config(self):
        formal = load_config()
        debug = debug_run_config(formal)
        short = small_run_config(formal, debug["output_dir"])
        self.assertEqual(debug["num_train_epochs"], 2)
        self.assertEqual(formal["num_train_epochs"], 20)
        self.assertIn("/_debug/", debug["output_dir"])
        for key in ("model_name", "model_revision", "max_length", "negatives", "top_k", "learning_rate",
                    "temperature_lr", "fp16", "attention_implementation", "gradient_accumulation_steps"):
            self.assertEqual(debug[key], formal[key])
            self.assertEqual(debug[key], short[key])
        self.assertEqual(debug["dataloader_num_workers"], formal["dataloader_num_workers"])
        self.assertEqual((debug["debug_train_pairs"], debug["debug_valid_queries"], debug["debug_candidates"]), (100, 20, 1000))

    def test_flags_override_source_default_and_never_start_formal_when_debug(self):
        with tempfile.TemporaryDirectory() as temporary, patch("fine_tune.fine_tune.run_training") as train:
            main(["--debug", "--output-dir", temporary], debug=False)
            config = train.call_args.args[0]
            self.assertEqual(config["mode"], "debug")
            self.assertIn("/_debug/", config["output_dir"])
            self.assertEqual(train.call_count, 1)
            main(["--no-debug", "--output-dir", temporary], debug=True)
            config = train.call_args.args[0]
            self.assertEqual(config["mode"], "formal")
            self.assertEqual(config["num_train_epochs"], 20)
            self.assertEqual(config["output_dir"], str(Path(temporary).resolve()))

    def test_debug_resume_keeps_run_and_rejects_formal_checkpoint(self):
        formal = load_config()
        debug = debug_run_config(formal)
        checkpoint = Path(debug["output_dir"]) / "checkpoint-2"
        with tempfile.TemporaryDirectory() as temporary:
            formal = dict(formal, output_dir=temporary)
            debug = debug_run_config(formal)
            checkpoint = Path(debug["output_dir"]) / "checkpoint-2"
            write_json(checkpoint.parent / "run_config.json", debug)
            self.assertEqual(debug_run_config(formal, checkpoint), debug)
        with self.assertRaisesRegex(ValueError, "_debug"):
            debug_run_config(formal, Path(formal["output_dir"]) / "checkpoint-1356")

    def test_small_dataset_includes_all_selected_queries_positives(self):
        ids = [f"{i:06d}" for i in range(1, 2000)]
        valid = {f"{i:06d}": [f"{i+100:06d}", f"{i+200:06d}"] for i in range(500, 530)}
        bundle = {"manifest": {"candidate_ids": ids}, "train_labels": {"000001": ids[1:201]},
                  "valid_labels": valid}
        small = select_debug_data(load_config(), bundle)
        self.assertEqual(len(small["pairs"]), 100)
        self.assertEqual(len(small["valid_labels"]), 20)
        self.assertEqual(len(small["candidate_ids"]), 1000)
        required = set(small["valid_labels"]) | {p for ps in small["valid_labels"].values() for p in ps}
        self.assertTrue(required <= set(small["candidate_ids"]))
        self.assertEqual(len(bundle["valid_labels"]), 30)


class AuditTests(unittest.TestCase):
    def test_config_rejects_invalid_types_values_and_unknown_fields(self):
        source = read_json(DEFAULT_CONFIG)
        changes = [("fp16", "false"), ("mode", "typo"), ("learning_rate", 0), ("temperature_lr", "0.5"),
                   ("warmup_ratio", None), ("weight_decay", "zero"), ("max_length", 512),
                   ("retrieval_top_k", 5.0), ("negatives", 5), ("model_revision", ""),
                   ("document_dir", None), ("unexpected_typo", 1), ("gradient_accumulation_steps", 0)]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            for key, value in changes:
                with self.subTest(field=key, value=value):
                    write_json(path, dict(source, **{key: value}))
                    with self.assertRaises(ValueError):
                        load_config(path)
            incomplete = dict(source)
            incomplete.pop("valid_qids")
            write_json(path, incomplete)
            with self.assertRaises(ValueError):
                load_config(path)

    def test_short_run_config_cannot_be_used_as_a_formal_training_entry(self):
        with tempfile.TemporaryDirectory() as temporary, patch("fine_tune.fine_tune.run_training") as train:
            path = Path(temporary) / "config.json"
            write_json(path, dict(read_json(DEFAULT_CONFIG), mode="debug"))
            with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as error:
                main(["--config", str(path), "--no-debug"])
            self.assertEqual(error.exception.code, 2)
            train.assert_not_called()

    def test_checkpoint_scope_must_match_its_own_run(self):
        from inference import evaluation_context
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-2"
            config = small_run_config(load_config(), root)
            write_json(root / "run_config.json", config)
            with self.assertRaisesRegex(ValueError, "another run/config"):
                evaluation_context(load_config(), checkpoint)
            with self.assertRaisesRegex(ValueError, "historical"):
                evaluation_context(config, checkpoint, legacy=True)

    def test_preparation_refuses_to_overwrite_frozen_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            documents = root / "docs"
            documents.mkdir()
            for i in range(1, 25):
                (documents / f"{i:06d}.txt").write_text(str(i))
            config = dict(load_config(), document_dir=str(documents), output_dir=str(root / "run"),
                          expected_counts={"train_queries": 1, "valid_queries": 1, "train_pairs": 1, "valid_pairs": 1, "candidates": 24})
            for split, query, positive in (("train", 1, 2), ("valid", 3, 4)):
                qids, rels, bm25 = root / f"{split}.tsv", root / f"{split}.json", root / f"{split}_bm25.tsv"
                qids.write_text(f"{query:06d}\n")
                write_json(rels, {f"{query:06d}.txt": [f"{positive:06d}.txt"]})
                bm25.write_text("\n".join(f"{query} Q0 {i} {i} {100-i} BM25" for i in range(1, 25)))
                config.update({f"{split}_qids": str(qids), f"{split}_labels": str(rels), f"bm25_{split}": str(bm25)})
            prepare(config)
            files = sorted((root / "run/prepared").glob("*.json"))
            hashes = {str(p): sha256_file(p) for p in files}
            prepare(config)  # identical preparation remains idempotent
            with self.assertRaisesRegex(ValueError, "Prepared artifact changed"):
                prepare(dict(config, sampler_seed=config["sampler_seed"] + 1))
            self.assertEqual(hashes, {str(p): sha256_file(p) for p in files})

    def test_formal_remainder_batch_and_scheduler_step_budget(self):
        # Verify the installed Trainer counts the final single-row accumulation
        # group rather than silently dropping one pair or one update per epoch.
        loader = DataLoader(list(range(5421)), batch_size=1)
        owner = SimpleNamespace(num_examples=lambda dl: len(dl.dataset))
        args = SimpleNamespace(max_steps=-1, gradient_accumulation_steps=4, num_train_epochs=20)
        values = Trainer.set_initial_training_values(owner, args, loader, total_train_batch_size=4)
        self.assertEqual(values[1], 1356)
        self.assertEqual(values[-1], 27120)


if __name__ == "__main__":
    unittest.main()
