from __future__ import annotations

import sys
import math
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn

CHUNKAGG_ROOT = Path(__file__).resolve().parents[1]
LEGAL_RETRIEVAL_ROOT = CHUNKAGG_ROOT.parent
REPO_ROOT = LEGAL_RETRIEVAL_ROOT.parent
FINE_TUNE_ROOT = CHUNKAGG_ROOT / "fine_tune"
for path in (str(FINE_TUNE_ROOT), str(CHUNKAGG_ROOT), str(LEGAL_RETRIEVAL_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import fine_tune as training
import modernbert_contrastive_model as model_module
from content_utils import (
    filter_id_text_pairs,
    scan_eligible_document_ids,
)
from lcr.data import EmbeddingsData
from lcr.retrieval import _chunk_single_text, generate_similarity_artifacts
from lcr.similarity import rank_candidates_with_scores


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=8,
            num_attention_heads=2,
            use_cache=True,
        )
        self.embedding = nn.Embedding(64, 8)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    def enable_input_require_grads(self):
        return None

    def gradient_checkpointing_enable(self, **kwargs):
        return None

    def gradient_checkpointing_disable(self):
        return None


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = None
    cls_token_id = 101
    sep_token_id = 102

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_offsets_mapping=False,
    ):
        tokens = text.split()
        ids = list(range(1, len(tokens) + 1))
        result = {"input_ids": ids}
        if return_offsets_mapping:
            offsets = []
            cursor = 0
            for token in tokens:
                start = text.index(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end
            result["offset_mapping"] = offsets
        return result

    def build_inputs_with_special_tokens(self, token_ids):
        # Match the real ModernBERT PreTrainedTokenizerFast behavior: its
        # backend post-processor owns the template, so this method is a no-op.
        return list(token_ids)


def make_document(token_ids: list[list[int]]) -> dict[str, torch.Tensor]:
    batch_size = len(token_ids)
    input_ids = torch.zeros((batch_size, 1, 4), dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, values in enumerate(token_ids):
        input_ids[row, 0, : len(values)] = torch.tensor(values, dtype=torch.long)
        attention_mask[row, 0, : len(values)] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "chunk_mask": torch.ones((batch_size, 1), dtype=torch.long),
    }


class ChunkingTests(unittest.TestCase):
    def test_fast_tokenizer_noop_still_adds_cls_and_sep(self):
        encoded = _chunk_single_text(
            "alpha beta",
            FakeTokenizer(),
            max_length=8,
            max_chunks=1,
        )
        active_ids = encoded["input_ids"][0][encoded["attention_mask"][0].bool()]
        self.assertEqual(active_ids.tolist(), [101, 1, 2, 102])

    @unittest.skipUnless(
        (REPO_ROOT / "models/modernbert-caselaw-accsteps-fp/checkpoint-29000/tokenizer.json").exists(),
        "local continued-pretraining tokenizer checkpoint is unavailable",
    )
    def test_real_modernbert_fast_tokenizer_has_cls_and_sep(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            REPO_ROOT / "models/modernbert-caselaw-accsteps-fp/checkpoint-29000",
            trust_remote_code=True,
            local_files_only=True,
        )
        encoded = _chunk_single_text(
            "alpha beta",
            tokenizer,
            max_length=8,
            max_chunks=1,
        )
        active_ids = encoded["input_ids"][0][encoded["attention_mask"][0].bool()].tolist()
        self.assertEqual(active_ids[0], tokenizer.cls_token_id)
        self.assertEqual(active_ids[-1], tokenizer.sep_token_id)
        self.assertEqual(
            tokenizer.convert_ids_to_tokens(active_ids),
            ["[CLS]", "alpha", "Ġbeta", "[SEP]"],
        )

    def test_sentence_boundary_and_padding_mask_are_preserved(self):
        encoded = _chunk_single_text(
            "aa bb. cc dd ee.",
            FakeTokenizer(),
            max_length=5,
            max_chunks=3,
        )
        self.assertEqual(tuple(encoded["input_ids"].shape), (3, 5))
        self.assertEqual(encoded["chunk_mask"].tolist(), [1, 1, 0])
        # The first hard budget would include `cc`; sentence fallback stops at `bb.`.
        self.assertEqual(encoded["attention_mask"][0].sum().item(), 4)
        self.assertEqual(encoded["input_ids"][0, :4].tolist(), [101, 1, 2, 102])

    def test_overflowing_document_uses_first_middle_last_sentence_chunks(self):
        encoded = _chunk_single_text(
            "a b. c d. e f. g h. i j.",
            FakeTokenizer(),
            max_length=5,
            max_chunks=3,
            selection_strategy="head_middle_tail",
        )
        self.assertEqual(encoded["chunk_mask"].tolist(), [1, 1, 1])
        self.assertEqual(encoded["input_ids"][0, 1:3].tolist(), [1, 2])
        self.assertEqual(encoded["input_ids"][1, 1:3].tolist(), [5, 6])
        self.assertEqual(encoded["input_ids"][2, 1:3].tolist(), [9, 10])


class SamplingTests(unittest.TestCase):
    def test_callback_refreshes_model_negatives_before_every_epoch(self):
        trainer = SimpleNamespace(
            current_epoch=0,
            _prepared_epoch_index=None,
            update_frequency=999,
        )
        refreshed_epochs = []
        trainer.update_negative_samples = lambda: refreshed_epochs.append(
            trainer.current_epoch
        )
        callback = training.AdaptiveNegativeSamplingCallback(trainer)

        for epoch in (0.0, 1.0, 2.0, 3.0):
            callback.on_epoch_begin(
                args=SimpleNamespace(),
                state=SimpleNamespace(epoch=epoch),
                control=SimpleNamespace(),
            )

        self.assertEqual(refreshed_epochs, [1, 2, 3])

    def test_epoch_zero_refresh_uses_current_model_scores(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = object.__new__(training.AdaptiveNegativeSamplingTrainer)
            trainer.current_epoch = 0
            trainer.finetune_data_dir = temp_dir
            trainer.args = SimpleNamespace(seed=289, device=torch.device("cpu"))
            trainer.sampling_temperature = 0.2
            trainer.sampling_temperature_schedule = [(0, 0.2)]
            trainer.sampling_target_ess = 0.0
            trainer.max_negatives = 1
            trainer.train_qids = ["q"]
            trainer.positives = {"q": {"p"}}
            trainer.query_candidate_scope = {"q": ["q", "p", "n1", "n2"]}
            trainer.query_candidate_scope_path = Path(temp_dir) / "scope.json"
            trainer.candidate_dataset_path = temp_dir
            trainer.query_dataset_path = temp_dir
            trainer.retrieval_tokenizer = object()
            trainer.retrieval_batch_size = 1
            trainer.model = object()
            trainer.train_dataset = training.ContrastiveDataset(
                data=[],
                doc_folder=temp_dir,
                query_candidate_scope=trainer.query_candidate_scope,
            )
            logged = {}
            trainer.log = lambda metrics: logged.update(metrics)

            def fake_generate(model, tokenizer, device, **kwargs):
                self.assertIs(model, trainer.model)
                self.assertFalse(kwargs["persist_trec"])
                self.assertFalse(kwargs["retain_scores"])
                self.assertTrue(kwargs["exclude_self"])
                kwargs["score_consumer"](
                    "q",
                    {"p": 0.9, "n1": 0.8, "n2": 0.1},
                )
                return SimpleNamespace()

            with patch.object(training, "generate_similarity_artifacts", side_effect=fake_generate):
                trainer.update_negative_samples()

            self.assertEqual(len(trainer.train_dataset), 1)
            row = trainer.train_dataset.data[0]
            self.assertEqual(row["query_id"], "q")
            self.assertEqual(row["positive_id"], "p")
            self.assertEqual(len(row["negative_ids"]), 1)
            self.assertNotIn("p", row["negative_ids"])
            self.assertEqual(logged["sampling/source_is_model"], 1.0)
            self.assertTrue((Path(temp_dir) / "adaptive_negative_epoch0_train.json").exists())

    def test_contrastive_dataset_filters_rows_and_negatives_to_scope(self):
        dataset = training.ContrastiveDataset(
            data=[
                {"query_id": "q", "positive_id": "p", "negative_ids": ["n", "p2", "outside"]},
                {"query_id": "q", "positive_id": "p2", "negative_ids": ["n", "p"]},
                {"query_id": "q", "positive_id": "outside", "negative_ids": ["n"]},
            ],
            doc_folder="unused",
            query_candidate_scope={"q": ["p", "p2", "n"]},
        )
        self.assertEqual(len(dataset), 2)
        self.assertEqual([row["negative_ids"] for row in dataset.data], [["n"], ["n"]])

    def test_low_sampling_temperature_still_samples_without_replacement(self):
        negatives = [f"n{i}" for i in range(20)]
        selected = training._sample_negative_ids(
            negatives,
            np.linspace(-1.0, 1.0, num=len(negatives)),
            max_negatives=15,
            temperature=1e-6,
            rng=np.random.default_rng(5),
        )
        self.assertEqual(len(selected), 15)
        self.assertEqual(len(set(selected)), 15)

    def test_each_positive_is_resampled_and_scope_invariants_hold(self):
        similarities = {
            "q": {"p1": 0.9, "p2": 0.8, "n1": 0.7, "n2": 0.6, "n3": 0.5}
        }
        positives = {"q": {"p1", "p2", "outside"}}
        scope = {"q": ["p1", "p2", "n1", "n2", "n3"]}
        calls = []

        def fake_choice(candidates, *, size, replace, p):
            self.assertFalse(replace)
            calls.append(list(candidates))
            selected = candidates[:size] if len(calls) == 1 else candidates[-size:]
            return np.asarray(selected)

        with patch.object(training.np.random, "choice", side_effect=fake_choice):
            samples = training.generate_adaptive_negative_samples(
                similarities,
                positives,
                max_negatives=2,
                temperature=1.0,
                query_candidate_scope=scope,
            )

        self.assertEqual(len(samples), 2)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(samples[0]["negative_ids"], samples[1]["negative_ids"])
        all_positives = positives["q"]
        for sample in samples:
            self.assertIn(sample["positive_id"], {"p1", "p2"})
            self.assertEqual(len(sample["negative_ids"]), len(set(sample["negative_ids"])))
            self.assertTrue(set(sample["negative_ids"]).issubset(set(scope["q"])))
            self.assertFalse(set(sample["negative_ids"]) & all_positives)
            self.assertNotIn("q", sample["negative_ids"])

    def test_sampling_temperature_is_calibrated_to_target_ess(self):
        scores = np.linspace(-1.0, 1.0, num=500)
        probabilities, diagnostics = training._build_sampling_distribution(
            scores,
            temperature=1.0,
            target_ess=100.0,
        )
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=7)
        self.assertAlmostEqual(diagnostics["effective_sample_size"], 100.0, delta=0.5)
        self.assertGreater(diagnostics["top100_probability_mass"], 0.2)

    def test_target_ess_curriculum_selects_expected_epoch_values(self):
        trainer = object.__new__(training.AdaptiveNegativeSamplingTrainer)
        trainer.sampling_target_ess = 300.0
        trainer.sampling_target_ess_schedule = (
            training._parse_sampling_target_ess_schedule(
                "0:4000,2:2000,4:1000,6:500,8:300"
            )
        )
        self.assertEqual(trainer._sampling_target_ess_for_epoch(0), 4000.0)
        self.assertEqual(trainer._sampling_target_ess_for_epoch(3), 2000.0)
        self.assertEqual(trainer._sampling_target_ess_for_epoch(7), 500.0)
        self.assertEqual(trainer._sampling_target_ess_for_epoch(19), 300.0)

    def test_dataset_rejects_ineligible_queries_positives_and_negatives(self):
        dataset = training.ContrastiveDataset(
            data=[
                {"query_id": "q", "positive_id": "p", "negative_ids": ["n", "tiny"]},
                {"query_id": "tiny", "positive_id": "p", "negative_ids": ["n"]},
                {"query_id": "q", "positive_id": "tiny", "negative_ids": ["n"]},
            ],
            doc_folder="unused",
            query_candidate_scope={"q": ["p", "n", "tiny"], "tiny": ["p", "n"]},
            eligible_document_ids={"q", "p", "n"},
        )
        self.assertEqual(
            dataset.data,
            [{"query_id": "q", "positive_id": "p", "negative_ids": ["n"]}],
        )

    def test_late_cache_masks_self_all_positives_and_explicit_negatives(self):
        trainer = object.__new__(training.AdaptiveNegativeSamplingTrainer)
        trainer.negative_cache_size = 8
        trainer._negative_cache = OrderedDict([
            ("q", torch.tensor([1.0, 0.0])),
            ("p1", torch.tensor([0.9, 0.1])),
            ("p2", torch.tensor([0.8, 0.2])),
            ("n_explicit", torch.tensor([0.7, 0.3])),
            ("safe", torch.tensor([0.0, 1.0])),
            ("outside", torch.tensor([0.5, 0.5])),
        ])
        trainer.model = SimpleNamespace(hidden_dim=2)
        trainer.query_candidate_scope = {
            "q": ["p1", "p2", "n_explicit", "safe"]
        }
        trainer.positives = {"q": {"p1", "p2"}}
        vectors, mask = trainer._build_cached_negatives(
            anchor_ids=["q"],
            negative_id_rows=[["n_explicit"]],
            all_positive_id_rows=[["p1", "p2"]],
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        self.assertEqual(mask.tolist(), [[True]])
        torch.testing.assert_close(vectors[0, 0], torch.tensor([0.0, 1.0]))


class SimilarityStreamingTests(unittest.TestCase):
    def test_same_corpus_reuses_query_embedding_and_excludes_self(self):
        class FakeDocumentModel(nn.Module):
            supports_chunked_documents = True
            max_chunks = 1

            def __init__(self):
                super().__init__()
                self.calls = 0

            def encode(self, batch):
                self.calls += 1
                values = batch["input_ids"].float().sum(dim=(1, 2))
                return torch.stack([values, torch.ones_like(values)], dim=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir)
            (corpus / "q.txt").write_text("query text", encoding="utf-8")
            (corpus / "a.txt").write_text("candidate text", encoding="utf-8")
            model = FakeDocumentModel()
            artifacts = generate_similarity_artifacts(
                model,
                FakeTokenizer(),
                torch.device("cpu"),
                candidate_dir=corpus,
                query_dir=corpus,
                query_ids=["q"],
                trec_output_path=corpus / "unused.tsv",
                run_tag="test",
                batch_size=2,
                max_length=8,
                query_to_candidate_ids={"q": ["q", "a"]},
                persist_trec=False,
                exclude_self=True,
                autocast_dtype=None,
                reuse_query_embeddings=True,
                verbose=False,
            )

        self.assertEqual(model.calls, 1)
        self.assertEqual(list(artifacts.scores["q"]), ["a"])

    def test_aggregation_variants_share_one_encoder_pass(self):
        class FakeVariantModel(nn.Module):
            supports_chunked_documents = True
            max_chunks = 1

            def __init__(self):
                super().__init__()
                self.calls = 0

            def encode_variants(self, batch):
                self.calls += 1
                values = batch["input_ids"].float().sum(dim=(1, 2))
                fused = torch.stack([values, torch.ones_like(values)], dim=1)
                fused = torch.nn.functional.normalize(fused, dim=-1)
                return {"fused": fused, "mean": fused, "first": fused}

        consumed = set()
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir)
            (corpus / "q.txt").write_text("query", encoding="utf-8")
            (corpus / "a.txt").write_text("candidate", encoding="utf-8")
            model = FakeVariantModel()
            generate_similarity_artifacts(
                model,
                FakeTokenizer(),
                torch.device("cpu"),
                candidate_dir=corpus,
                query_dir=corpus,
                query_ids=["q"],
                trec_output_path=corpus / "unused.tsv",
                run_tag="test",
                batch_size=2,
                max_length=8,
                query_to_candidate_ids={"q": ["q", "a"]},
                persist_trec=False,
                retain_scores=False,
                exclude_self=True,
                autocast_dtype=None,
                reuse_query_embeddings=True,
                verbose=False,
                embedding_variants=("fused", "mean", "first"),
                variant_score_consumer=(
                    lambda variant, qid, scores: consumed.add((variant, qid))
                ),
            )
        self.assertEqual(model.calls, 1)
        self.assertEqual(
            consumed,
            {("fused", "q"), ("mean", "q"), ("first", "q")},
        )

    def test_eligible_allowlist_excludes_short_queries_and_candidates(self):
        class FakeDocumentModel(nn.Module):
            supports_chunked_documents = True
            max_chunks = 1

            def encode(self, batch):
                values = batch["input_ids"].float().sum(dim=(1, 2))
                return torch.stack([values, torch.ones_like(values)], dim=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir)
            for doc_id in ("q", "short_query", "a", "short_candidate"):
                (corpus / f"{doc_id}.txt").write_text(doc_id, encoding="utf-8")
            artifacts = generate_similarity_artifacts(
                FakeDocumentModel(),
                FakeTokenizer(),
                torch.device("cpu"),
                candidate_dir=corpus,
                query_dir=corpus,
                query_ids=["q", "short_query"],
                trec_output_path=corpus / "unused.tsv",
                run_tag="content_filter",
                batch_size=2,
                max_length=8,
                query_to_candidate_ids={
                    "q": ["q", "a", "short_candidate"],
                    "short_query": ["a"],
                },
                persist_trec=False,
                exclude_self=True,
                autocast_dtype=None,
                reuse_query_embeddings=True,
                eligible_document_ids={"q", "a"},
                verbose=False,
            )

        self.assertEqual(artifacts.candidate_ids, ["a", "q"])
        self.assertEqual(artifacts.query_ids, ["q"])
        self.assertIn("short_query", artifacts.missing_queries)
        self.assertEqual(set(artifacts.scores["q"]), {"a"})

    def test_single_chunk_same_corpus_does_not_reuse_without_opt_in(self):
        class FakeSingleTokenizer:
            def __call__(self, texts, **_kwargs):
                values = torch.tensor(
                    [[len(text.split())] for text in texts],
                    dtype=torch.long,
                )
                return {
                    "input_ids": values,
                    "attention_mask": torch.ones_like(values),
                }

        class FakeSingleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def encode(self, inputs):
                self.calls += 1
                values = inputs["input_ids"].float().squeeze(-1)
                return torch.stack([values, torch.ones_like(values)], dim=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir)
            (corpus / "q.txt").write_text("query text", encoding="utf-8")
            (corpus / "a.txt").write_text("candidate text", encoding="utf-8")
            model = FakeSingleModel()
            generate_similarity_artifacts(
                model,
                FakeSingleTokenizer(),
                torch.device("cpu"),
                candidate_dir=corpus,
                query_dir=corpus,
                query_ids=["q"],
                trec_output_path=corpus / "ranking.tsv",
                run_tag="single_chunk_compat",
                batch_size=2,
                max_length=8,
                autocast_dtype=None,
                verbose=False,
            )

        # One candidate batch plus a separate query batch preserves legacy behavior.
        self.assertEqual(model.calls, 2)

    def test_self_is_removed_before_topk(self):
        queries = EmbeddingsData(["q"], torch.tensor([[1.0, 0.0]]))
        candidates = EmbeddingsData(
            ["q", "a", "b"],
            torch.tensor([[1.0, 0.0], [0.9, 0.0], [0.8, 0.0]]),
        )
        lines, scores, missing = rank_candidates_with_scores(
            ["q"],
            queries,
            candidates,
            topk=2,
            query_to_candidate_ids={"q": ["q", "a", "b"]},
            exclude_self=True,
        )
        self.assertEqual(missing, [])
        self.assertEqual(list(scores["q"]), ["a", "b"])
        self.assertTrue(all(" Q0 q " not in line for line in lines))

    def test_score_consumer_can_stream_without_retaining_rankings(self):
        queries = EmbeddingsData(["q"], torch.tensor([[1.0, 0.0]]))
        candidates = EmbeddingsData(
            ["a", "b", "c"],
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.0]]),
        )
        consumed = {}
        lines, retained, missing = rank_candidates_with_scores(
            ["q"],
            queries,
            candidates,
            query_to_candidate_ids={"q": ["a", "c"]},
            include_trec_lines=False,
            retain_scores=False,
            score_consumer=lambda qid, scores: consumed.update({qid: scores}),
        )
        self.assertEqual(lines, [])
        self.assertEqual(retained, {})
        self.assertEqual(missing, [])
        self.assertEqual(set(consumed["q"]), {"a", "c"})

    def test_global_metric_matches_zero_padded_ranking_ids_to_integer_labels(self):
        def fake_generate(*args, **kwargs):
            self.assertTrue(kwargs["exclude_self"])
            self.assertEqual(
                kwargs["eligible_document_ids"],
                {"008447", "000123", "000001", "000002"},
            )
            consumer = kwargs["score_consumer"]
            consumer("008447", {"008447": 1.0, "000123": 0.9})
            consumer("000001", {"000002": 0.8})
            return SimpleNamespace()

        with (
            patch.object(
                training,
                "rel_file_convert",
                side_effect=[{8447: [123]}, {1: [2]}],
            ),
            patch.object(
                training,
                "load_query_ids_from_utils",
                side_effect=[["008447"], ["000001"]],
            ),
            patch.object(training, "generate_similarity_artifacts", side_effect=fake_generate),
        ):
            results = training.evaluate_model_retrieval(
                model=object(),
                tokenizer=object(),
                device=torch.device("cpu"),
                candidate_dataset_path="unused",
                query_dataset_path="unused",
                train_qid_path="train",
                valid_qid_path="valid",
                labels_path="labels",
                query_candidate_scope_path="scope",
                eligible_document_ids={"008447", "000123", "000001", "000002"},
            )

        self.assertEqual(results["train"]["f1"], 1.0)
        self.assertEqual(results["valid"]["f1"], 1.0)
        self.assertEqual(results["train"]["missing_pred_qids"], 0)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.encoder_patch = patch.object(
            model_module.ModernBertModel,
            "from_pretrained",
            side_effect=lambda *args, **kwargs: FakeEncoder(),
        )
        self.encoder_patch.start()

    def tearDown(self):
        self.encoder_patch.stop()

    def make_model(self):
        return model_module.ModernBERTContrastive(
            "fake",
            device="cpu",
            temperature=0.5,
            max_chunks=3,
            chunk_microbatch_size=2,
            fusion_dropout=0.0,
        )

    def test_zero_initialized_residual_preserves_masked_chunk_mean(self):
        model = self.make_model().eval()
        single = make_document([[1, 2, 3]])
        padded = {
            "input_ids": torch.cat(
                [single["input_ids"], torch.full((1, 2, 4), 17, dtype=torch.long)],
                dim=1,
            ),
            "attention_mask": torch.cat(
                [single["attention_mask"], torch.ones((1, 2, 4), dtype=torch.long)],
                dim=1,
            ),
            "chunk_mask": torch.tensor([[1, 0, 0]], dtype=torch.long),
        }
        with torch.no_grad():
            single_vector = model.encode(single)
            padded_vector = model.encode(padded)
        torch.testing.assert_close(single_vector, padded_vector, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(single_vector.norm(dim=-1), torch.ones(1))
        self.assertEqual(torch.count_nonzero(model.fusion_output.weight).item(), 0)

    def test_single_positive_forward_is_finite_and_backpropagates(self):
        model = self.make_model().train()
        output = model(
            anchor_input=make_document([[1, 2]]),
            positive_input=make_document([[1, 3]]),
            negative_input=make_document([[8, 9], [10, 11]]),
            negative_counts=torch.tensor([2]),
            labels=torch.tensor([0]),
        )
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(tuple(output["logits"].shape), (1, 3))
        self.assertIn("loss_evidence", output["_diagnostics"])
        output["loss"].backward()
        self.assertIsNotNone(model.projector.fc2.weight.grad)
        self.assertIsNotNone(model.fusion_output.weight.grad)

    def test_fusion_residual_is_bounded_and_differentiable(self):
        model = self.make_model()
        raw_residual = torch.full((2, model.hidden_dim), 100.0, requires_grad=True)
        bounded_residual, raw_norm = model._soft_cap_fusion_residual(raw_residual)
        bounded_norm = bounded_residual.norm(dim=-1)
        self.assertTrue(torch.all(raw_norm > 100.0))
        self.assertTrue(torch.all(bounded_norm < 0.5))
        self.assertTrue(torch.all(bounded_norm > 0.49))
        bounded_residual.sum().backward()
        self.assertTrue(torch.isfinite(raw_residual.grad).all())
        self.assertGreater(raw_residual.grad.abs().sum().item(), 0.0)

    def test_training_forward_rejects_empty_documents(self):
        model = self.make_model().train()
        empty_anchor = make_document([[1, 2]])
        empty_anchor["chunk_mask"].zero_()
        with self.assertRaisesRegex(
            ValueError,
            "anchor_input contains documents with no valid chunks",
        ):
            model(
                anchor_input=empty_anchor,
                positive_input=make_document([[1, 3]]),
                negative_input=make_document([[8, 9]]),
                negative_counts=torch.tensor([1]),
                labels=torch.tensor([0]),
            )

    def test_cached_negatives_extend_document_logits_only(self):
        model = self.make_model().train()
        output = model(
            anchor_input=make_document([[1, 2]]),
            positive_input=make_document([[1, 3]]),
            negative_input=make_document([[8, 9], [10, 11]]),
            negative_counts=torch.tensor([2]),
            labels=torch.tensor([0]),
            cached_negative_vectors=torch.nn.functional.normalize(
                torch.randn(1, 3, 8),
                dim=-1,
            ),
            cached_negative_mask=torch.tensor([[1, 0, 1]], dtype=torch.bool),
        )
        self.assertEqual(tuple(output["logits"].shape), (1, 5))
        self.assertEqual(
            output["_diagnostics"]["cached_negative_count"].item(),
            2.0,
        )

    def test_temperature_parameter_clamp_can_recover_from_boundary(self):
        model = self.make_model()
        with torch.no_grad():
            model.log_temperature.fill_(math.log(1e-5))
        model.clamp_temperature_parameter_()
        self.assertAlmostEqual(model.temperature.item(), 0.05, places=6)
        model.log_temperature.grad = torch.tensor(-1.0)
        optimizer = torch.optim.SGD([model.log_temperature], lr=0.01)
        optimizer.step()
        self.assertGreater(model.temperature.item(), 0.05)

    def test_chunk_buffer_accepts_autocast_projector_output(self):
        model = self.make_model().eval()
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            vector = model.encode(make_document([[1, 2, 3]]))
        self.assertTrue(torch.isfinite(vector).all())
        self.assertEqual(vector.dtype, torch.float32)

    def test_checkpoint_loader_rejects_legacy_architecture(self):
        model = self.make_model()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")
            self.make_model().load_checkpoint(checkpoint_dir)

            v4_state = dict(model.state_dict())
            v4_state["chunkagg_architecture_version"] = torch.tensor(4)
            torch.save(v4_state, checkpoint_dir / "pytorch_model.bin")
            with self.assertRaisesRegex(RuntimeError, "checkpoint=4, expected=5"):
                self.make_model().load_checkpoint(checkpoint_dir)

            legacy_state = dict(model.state_dict())
            legacy_state.pop("chunkagg_architecture_version")
            torch.save(legacy_state, checkpoint_dir / "pytorch_model.bin")
            with self.assertRaisesRegex(RuntimeError, "Incompatible chunkAgg checkpoint"):
                self.make_model().load_checkpoint(checkpoint_dir)


class ContentFilterTests(unittest.TestCase):
    def test_scan_and_pair_filter_use_non_whitespace_character_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            corpus = Path(temp_dir)
            (corpus / "empty.txt").write_text(" \n\t", encoding="utf-8")
            (corpus / "short.txt").write_text("ab ", encoding="utf-8")
            (corpus / "valid.txt").write_text("a b c", encoding="utf-8")
            eligible, rejected = scan_eligible_document_ids(
                corpus,
                minimum_characters=3,
            )
        self.assertEqual(eligible, {"valid"})
        self.assertEqual(rejected, {"empty": 0, "short": 2})

        ids, texts, rejected_pairs = filter_id_text_pairs(
            ["empty", "valid"],
            [" \n", "a b c"],
            minimum_characters=3,
        )
        self.assertEqual(ids, ["valid"])
        self.assertEqual(texts, ["a b c"])
        self.assertEqual(rejected_pairs, {"empty": 0})


if __name__ == "__main__":
    unittest.main()
