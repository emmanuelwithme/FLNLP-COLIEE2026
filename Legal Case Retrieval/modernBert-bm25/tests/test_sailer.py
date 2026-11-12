"""Mathematical loss/gradient and data-contract regressions (CPU)."""
import copy
from collections import Counter
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn

from config import PROJECT_DIR, load_config, write_json
from fine_tune.gradcache import cached_backward, masked_inbatch_loss
from fine_tune.sailer import (debug_pairs, label_mask, logical_epoch_rows, LogicalBatchDataset,
                             SailerCollator)
from test_flow import NumericTokenizer


class PaddedNumericTokenizer(NumericTokenizer):
    def pad(self, values, **kwargs):
        return {k: torch.as_tensor(v) for k, v in values.items()}


def mask_fixture():
    rows, docs, labels = [], {}, {}
    for i in range(4):
        q, p = str(1000 + i), str(2000 + i)
        ns = [str(3000 + i * 20 + j) for j in range(15)]
        rows.append({"query_id": q, "positive_id": p, "negative_ids": ns})
        labels[q] = [p, str(6000 + i)]
        for pid in [q, p, *ns, str(6000 + i)]:
            docs[pid] = {"valid": True, "token_hash": pid}
    return rows, {"documents": docs, "known_positives": labels}


class SailerTests(unittest.TestCase):
    def test_mask_targets_all_known_positives_self_aliases_duplicates_invalid(self):
        rows, manifest = mask_fixture()
        # Candidate columns belong to a different query, testing cross-query masking.
        rows[1]["negative_ids"][:7] = ["6000", "1000", "2000", "alias_gold", "alias_self", "bad", "3000"]
        manifest["documents"]["alias_gold"] = {"valid": True, "token_hash": "6000"}
        manifest["documents"]["alias_self"] = {"valid": True, "token_hash": "1000"}
        manifest["documents"]["bad"] = {"valid": False, "token_hash": "bad"}
        allowed, targets, stats = label_mask(rows, manifest)
        self.assertEqual(allowed.shape, (4, 64))
        self.assertEqual(targets.tolist(), [0, 16, 32, 48])
        self.assertTrue(allowed[range(4), targets].all())
        self.assertFalse(allowed[0, 17:24].any())
        self.assertTrue(allowed[0, 1])  # original unique representative of 3000
        self.assertGreater(stats["known_positive_or_alias"], 0)
        self.assertGreater(stats["self_or_alias"], 0)
        self.assertGreater(stats["duplicate"], 0)
        self.assertGreater(stats["invalid"], 0)

    def test_target_is_retained_even_when_duplicate_occurs_in_earlier_group(self):
        rows, manifest = mask_fixture()
        rows[0]["negative_ids"][0] = rows[2]["positive_id"]
        allowed, targets, _ = label_mask(rows, manifest)
        self.assertFalse(allowed[2, 1])
        self.assertTrue(allowed[2, targets[2]])
        # This is a legal cross-query negative for query 0 (unless itself labelled).
        self.assertTrue(allowed[0, 1])
        self.assertFalse(allowed[0, 32])  # same ID appears twice, retain one

    def test_all_four_queries_share_the_same_sixty_four_passages(self):
        torch.manual_seed(9)
        reps = torch.randn(68, 12, requires_grad=True)
        allowed = torch.ones(4, 64, dtype=torch.bool)
        targets = torch.arange(4) * 16
        loss, scores = masked_inbatch_loss(reps, torch.tensor(.5), allowed, targets)
        expected = nn.functional.cross_entropy(
            nn.functional.normalize(reps[:4], dim=-1) @ nn.functional.normalize(reps[4:], dim=-1).T / .5, targets)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(reps.grad[20].abs().sum(), 0)
        self.assertEqual(scores.shape, (4, 64))

    def test_masked_columns_have_zero_embedding_gradient(self):
        reps = torch.randn(68, 7, requires_grad=True)
        allowed = torch.ones(4, 64, dtype=torch.bool)
        allowed[:, 5] = False
        loss, _ = masked_inbatch_loss(reps, torch.tensor(.5), allowed, torch.arange(4) * 16)
        loss.backward()
        torch.testing.assert_close(reps.grad[4 + 5], torch.zeros(7), rtol=0, atol=0)

    def test_cache_matches_full_graph_every_gradient_temperature_and_rng_with_dropout(self):
        # Small algebraic fixture, not a replacement for real ModernBERT GPU tests.
        torch.manual_seed(10)
        module = nn.Sequential(nn.Linear(9, 13), nn.Dropout(.3), nn.Linear(13, 7))
        cached = copy.deepcopy(module)
        t1 = nn.Parameter(torch.tensor(-.4))
        t2 = nn.Parameter(t1.detach().clone())
        x = torch.randn(68, 9)
        chunks = [{"x": c} for c in x.split(3)]
        allowed = torch.ones(4, 64, dtype=torch.bool)
        allowed[0, 16] = False
        targets = torch.arange(4) * 16
        rng = torch.get_rng_state()
        # Same chunk order/shapes so dropout consumes identical random draws.
        ref = torch.cat([module(c["x"]) for c in chunks])
        expected, _ = masked_inbatch_loss(ref, t1.exp(), allowed, targets)
        expected.backward()
        end_rng = torch.get_rng_state()
        torch.set_rng_state(rng)
        actual = cached_backward(lambda c: cached(c["x"]), chunks,
            lambda z: masked_inbatch_loss(z, t2.exp(), allowed, targets)[0], lambda loss: loss.backward())
        torch.testing.assert_close(actual, expected.detach(), rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.equal(end_rng, torch.get_rng_state()))
        for a, b in zip(module.parameters(), cached.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-5, atol=1e-6)
        torch.testing.assert_close(t1.grad, t2.grad)

    def test_scaled_cache_does_not_scale_temperature_or_encoder_twice(self):
        torch.manual_seed(4)
        linear = nn.Linear(5, 8)
        reference = copy.deepcopy(linear)
        t = nn.Parameter(torch.tensor(.2))
        rt = nn.Parameter(t.detach().clone())
        x = torch.randn(68, 5)
        targets = torch.arange(4) * 16
        mask = torch.ones(4, 64, dtype=torch.bool)
        loss, _ = masked_inbatch_loss(reference(x), rt.exp(), mask, targets)
        (loss * 128).backward()
        cached_backward(lambda c: linear(c["x"]), [{"x": c} for c in x.split(2)],
                        lambda z: masked_inbatch_loss(z, t.exp(), mask, targets)[0],
                        lambda value: (value * 128).backward())
        for a, b in zip(linear.parameters(), reference.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=1e-4)
        torch.testing.assert_close(t.grad, rt.grad)

    def test_all_pairs_preserved_distinct_queries_and_deterministic_audited_padding(self):
        labels = {f"{i:06d}": [str(10000 + i * 10 + j) for j in range(i % 5 + 1)] for i in range(1, 35)}
        pools = {q: [str(50000 + j) for j in range(90)] for q in labels}
        rows, audit = logical_epoch_rows(labels, pools, 0, 289)
        again, _ = logical_epoch_rows(labels, pools, 0, 289)
        second, _ = logical_epoch_rows(labels, pools, 1, 289)
        self.assertEqual(rows, again)
        self.assertNotEqual(rows, second)
        counts = Counter((r["query_id"], r["positive_id"]) for r in rows)
        self.assertEqual(set(counts), {(q, p) for q, ps in labels.items() for p in ps})
        self.assertEqual(sum(counts.values()) - len(counts), audit["padded_pairs"])
        self.assertLess(audit["padded_pairs"], 4)
        for start in range(0, len(rows), 4):
            self.assertEqual(len({r["query_id"] for r in rows[start:start + 4]}), 4)

    def test_debug_spreads_pairs_across_different_queries(self):
        labels = {str(i): [str(100 + j) for j in range(30)] for i in range(20)}
        self.assertEqual(len(set(q for q, _ in debug_pairs(labels, 8))), 8)

    def test_multiple_workers_receive_fresh_logical_batches_and_grouped_tokens(self):
        from torch.utils.data import DataLoader
        labels = {str(i).zfill(6): [str(i + 20).zfill(6)] for i in range(1, 9)}
        pools = {q: [str(i).zfill(6) for i in range(100, 180)] for q in labels}
        documents = {str(i).zfill(6): {"valid": True, "token_hash": str(i)} for i in range(1, 180)}
        manifest = {"documents": documents, "known_positives": labels}
        config = {"max_length": 4096, "gradcache_chunk_size": 2}
        with tempfile.TemporaryDirectory() as tmp:
            for pid in documents:
                (Path(tmp) / f"{pid}.txt").write_text(str(int(pid)))
            first, _ = logical_epoch_rows(labels, pools, 0, 289)
            second, _ = logical_epoch_rows(labels, pools, 1, 289)
            dataset = LogicalBatchDataset(first, tmp)
            loader = DataLoader(dataset, batch_size=1, num_workers=2, persistent_workers=False,
                                collate_fn=SailerCollator(PaddedNumericTokenizer(), config, manifest))
            one = list(loader)
            dataset.rows = second
            two = list(loader)
            for batches, planned in ((one, first), (two, second)):
                received = [r for b in batches for r in b["sample_ids"]]
                self.assertEqual(received, planned)
                for b in batches:
                    ids = [r["query_id"] for r in b["sample_ids"]] + [p for r in b["sample_ids"] for p in [r["positive_id"], *r["negative_ids"]]]
                    self.assertEqual(torch.cat([c["input_ids"] for c in b["chunks"]]).flatten().tolist(), [int(p) for p in ids])

    def test_config_rejects_accumulation_and_unsafe_chunk_profiles(self):
        config = load_config(PROJECT_DIR / "configs/coliee2025_sailer_gradcache.json")
        self.assertEqual(config["logical_batch_size"], 4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for bad in ({"gradient_accumulation_steps": 4}, {"logical_batch_size": 3},
                        {"gradcache_chunk_size": 0}, {"gradcache_chunk_size": 17}, {"deduplicate_inputs": "true"}):
                write_json(path, dict(config, **bad))
                with self.assertRaises(ValueError):
                    load_config(path)

    def test_deduplicated_forward_mapping_preserves_all_68_positions(self):
        rows, manifest = mask_fixture()
        rows[1]["negative_ids"][0] = rows[0]["negative_ids"][0]
        alias, original = rows[2]["negative_ids"][0], rows[0]["negative_ids"][1]
        manifest["documents"][alias]["token_hash"] = manifest["documents"][original]["token_hash"]
        with tempfile.TemporaryDirectory() as tmp:
            for pid in manifest["documents"]:
                (Path(tmp)/f"{pid}.txt").write_text(original if pid==alias else pid)
            item=LogicalBatchDataset(rows,tmp)[0]
            out=SailerCollator(PaddedNumericTokenizer(),{"max_length":4096,"gradcache_chunk_size":8,
                               "deduplicate_inputs":True},manifest)([item])
            encoded=torch.cat([c["input_ids"] for c in out["chunks"]])
            self.assertEqual(len(encoded),66)
            self.assertEqual(encoded[out["representation_indices"]].flatten().tolist(),[int(t) for t in item["texts"]])
            self.assertEqual(out["allowed"].shape,(4,64))

    def test_deduplication_sums_query_and_passage_gradients_without_changing_loss(self):
        torch.manual_seed(1)
        unique=torch.randn(50,9)
        indices=torch.cat([torch.arange(50),torch.arange(18)])
        original=nn.Linear(9,7)
        shared=copy.deepcopy(original)
        allowed=torch.ones(4,64,dtype=torch.bool)
        targets=torch.arange(4)*16
        full_loss,_=masked_inbatch_loss(original(unique[indices]),torch.tensor(.5),allowed,targets)
        full_loss.backward()
        cache_loss=cached_backward(lambda c:shared(c["x"]),[{"x":c} for c in unique.split(8)],
            lambda reps:masked_inbatch_loss(reps[indices],torch.tensor(.5),allowed,targets)[0],lambda loss:loss.backward())
        torch.testing.assert_close(full_loss,cache_loss)
        for a,b in zip(original.parameters(),shared.parameters()):
            torch.testing.assert_close(a.grad,b.grad,rtol=3e-5,atol=1e-6)

    def test_checkpoint_persists_compile_policy_omitted_by_transformers_config(self):
        from transformers import ModernBertConfig
        from unittest.mock import patch
        from config import read_json
        from fine_tune.modernbert_contrastive_model import ModernBERTContrastive

        def encoder(config):
            stub = nn.Module()
            stub.config = config
            return stub

        config = ModernBertConfig(reference_compile=False)
        self.assertNotIn("reference_compile", config.to_dict())
        # Serialization-only encoder stub; no replacement training model.
        model = ModernBERTContrastive(encoder(config))
        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp)
            self.assertIs(read_json(Path(temp) / "contrastive_config.json")["reference_compile"], False)
            with patch("fine_tune.modernbert_contrastive_model.ModernBertModel", side_effect=encoder):
                restored = ModernBERTContrastive.from_checkpoint(temp)
            self.assertIs(restored.encoder.config.reference_compile, False)
            for a, b in zip(model.parameters(), restored.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
