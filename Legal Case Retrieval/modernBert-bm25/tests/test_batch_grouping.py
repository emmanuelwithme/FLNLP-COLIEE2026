"""Grouping contracts and adversarial cases; no CUDA/model is constructed."""
from collections import Counter
import copy
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (PROJECT_DIR, digest, implementation_id, load_config, read_json,
                    write_json, SAILER_IMPLEMENTATION, SAILER_LEGACY_IMPLEMENTATION)
from fine_tune.batch_grouping import (_repair_distinct_queries, shuffle_repair_rows,
                                     SHUFFLE_REPAIR, LEGACY_DEGREE)


def rows_for_counts(counts):
    return [{"query_id": f"{q:06d}", "positive_id": f"{10000 + q * 1000 + p:06d}",
             "negative_ids": [str(900000 + k) for k in range(15)], "sentinel": [q, p]}
            for q, count in enumerate(counts) for p in range(count)]


def assert_contract(test, original, result, audit):
    expected = Counter((r["query_id"], r["positive_id"]) for r in original)
    expected.update((r["query_id"], r["positive_id"]) for r in audit["padding_pairs"])
    test.assertEqual(Counter((r["query_id"], r["positive_id"]) for r in result), expected)
    lookup = {(r["query_id"], r["positive_id"]): r for r in original}
    for r in result:
        test.assertEqual(r, lookup[(r["query_id"], r["positive_id"])])
    test.assertEqual(len(result), (len(original) + 3) // 4 * 4)
    test.assertEqual(audit["padded_pairs"], (-len(original)) % 4)
    test.assertLessEqual(audit["repair_swaps"], audit["initial_duplicate_slots"])
    for i in range(0, len(result), 4):
        test.assertEqual(len({r["query_id"] for r in result[i:i + 4]}), 4)


class BatchGroupingTests(unittest.TestCase):
    def test_all_rows_payloads_padding_and_order_independence(self):
        original = rows_for_counts([5, 8, 4, 3, 2, 7])
        untouched = copy.deepcopy(original)
        result, audit = shuffle_repair_rows(original, 0, 289)
        assert_contract(self, original, result, audit)
        self.assertEqual(original, untouched)
        shuffled = list(original)
        random.Random(5).shuffle(shuffled)
        self.assertEqual((result, audit), shuffle_repair_rows(shuffled, 0, 289))
        self.assertNotEqual(result, shuffle_repair_rows(original, 1, 289)[0])

    def test_padding_three_distinct_queries_when_capacity_allows(self):
        original = rows_for_counts([4, 4, 4, 4, 4, 4, 5])
        result, audit = shuffle_repair_rows(original, 1)
        assert_contract(self, original, result, audit)
        self.assertEqual(audit["padded_pairs"], 3)
        self.assertEqual(audit["padding_distinct_queries"], 3)

    def test_near_capacity_padding_may_need_one_query_three_times(self):
        original = rows_for_counts([4, 4, 4, 1])
        result, audit = shuffle_repair_rows(original, 0)
        assert_contract(self, original, result, audit)
        self.assertEqual(audit["padding_distinct_queries"], 1)
        self.assertEqual({r["query_id"] for r in audit["padding_pairs"]}, {"000003"})

    def test_global_conflict_repair_handles_two_invalid_batches(self):
        # AABC can only send A to BBCC. Candidate B/C is already in the
        # source, so requiring both batches to become valid in one swap fails.
        letters = "AABCBBCCABCDADDD"
        rows = [{"query_id": q, "positive_id": str(i)} for i, q in enumerate(letters)]
        before = Counter((r["query_id"], r["positive_id"]) for r in rows)
        audit = _repair_distinct_queries(rows, random.Random(1), random_attempts=0)
        self.assertGreater(audit["fallback_searches"], 0)
        self.assertEqual(Counter((r["query_id"], r["positive_id"]) for r in rows), before)
        self.assertTrue(all(len({r["query_id"] for r in rows[i:i + 4]}) == 4 for i in range(0, len(rows), 4)))

    def test_four_queries_at_full_capacity_terminate(self):
        for count in (1, 2, 10, 100):
            original = rows_for_counts([count] * 4)
            result, audit = shuffle_repair_rows(original, count)
            assert_contract(self, original, result, audit)

    def test_impossible_inputs_fail_without_dropping_pairs(self):
        for counts in ([], [3], [3, 3, 3], [10, 1, 1, 1]):
            with self.assertRaises(ValueError):
                shuffle_repair_rows(rows_for_counts(counts), 0)
        for epoch, seed in ((-1, 289), (True, 289), (0, -1), (0, False)):
            with self.assertRaises(ValueError):
                shuffle_repair_rows(rows_for_counts([1] * 4), epoch, seed)

    def test_500_random_feasible_distributions(self):
        rng = random.Random(9001)
        for trial in range(500):
            qcount, batches = rng.randint(4, 40), rng.randint(2, 40)
            sequence = [q for _ in range(batches) for q in rng.sample(range(qcount), 4)]
            for _ in range(rng.randrange(4)):
                sequence.pop(rng.randrange(len(sequence)))
            counts = Counter(sequence)
            original = rows_for_counts([counts[q] for q in range(qcount)])
            result, audit = shuffle_repair_rows(original, trial, 123)
            assert_contract(self, original, result, audit)

    def test_negative_sampling_is_identical_before_and_after_grouping(self):
        from fine_tune.data import sample_epoch
        from fine_tune.sailer import logical_epoch_rows
        labels = {f"{q:06d}": [f"{q + 100:06d}", f"{q + 200:06d}"] for q in range(1, 12)}
        pools = {q: [f"{n:06d}" for n in range(1000, 1095)] for q in labels}
        original = sample_epoch(labels, pools, 3, 289)
        result, audit = logical_epoch_rows(labels, pools, 3, 289, strategy=SHUFFLE_REPAIR)
        assert_contract(self, original, result, audit)
        legacy, _ = logical_epoch_rows(labels, pools, 3, 289, strategy=LEGACY_DEGREE)
        lookup = {(r["query_id"], r["positive_id"]): r["negative_ids"] for r in original}
        for r in result + legacy:
            self.assertEqual(r["negative_ids"], lookup[r["query_id"], r["positive_id"]])

    def test_group_receipt_accepts_batch_shuffle_and_rejects_changed_companions(self):
        from fine_tune.sailer import LabelMaskAudit
        original = rows_for_counts([2] * 8)
        rows, _ = shuffle_repair_rows(original, 0)
        hashes = [digest(rows[i:i + 4]) for i in range(0, len(rows), 4)]
        with tempfile.TemporaryDirectory() as temp:
            owner = SimpleNamespace(train_dataset=SimpleNamespace(rows=rows),
                epoch_seen_groups=hashes[::-1], epoch_mask_stats=Counter(), run_dir=Path(temp), log=lambda value: None)
            LabelMaskAudit(owner).on_epoch_end(None, SimpleNamespace(epoch=1.), None)
            self.assertTrue(read_json(Path(temp)/"sampling/epoch_01_groups_received.json")["matches_planned"])
            owner.epoch_seen_groups = hashes[:-1] + ["incorrect_companions"]
            with self.assertRaisesRegex(ValueError, "Logical groups"):
                LabelMaskAudit(owner).on_epoch_end(None, SimpleNamespace(epoch=1.), None)

    def test_configuration_versions_keep_old_runs_and_new_output_separate(self):
        new = load_config(PROJECT_DIR/"configs/coliee2025_sailer_gradcache.json")
        old = load_config(PROJECT_DIR/"configs/coliee2025_sailer_gradcache_legacy.json")
        self.assertNotEqual(new["output_dir"], old["output_dir"])
        self.assertEqual(implementation_id(new), SAILER_IMPLEMENTATION)
        self.assertEqual(implementation_id(old), SAILER_LEGACY_IMPLEMENTATION)
        self.assertNotIn("batch_grouping", old)  # preserve exact old config equality
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp)/"config.json"
            write_json(p, dict(new, batch_grouping="typo"))
            with self.assertRaisesRegex(ValueError, "batch_grouping"):
                load_config(p)

    def test_grouping_does_not_consume_global_rng_state(self):
        random.seed(1729)
        before = random.getstate()
        shuffle_repair_rows(rows_for_counts([5] * 8), 0)
        self.assertEqual(before, random.getstate())

    def test_sampler_epoch_boundary_resume_matches_uninterrupted_samples(self):
        from fine_tune.sailer import SailerGradCacheTrainer
        labels = {f"{q:06d}": [f"{q+100:06d}", f"{q+200:06d}"] for q in range(1,10)}
        pools = {q:[f"{i:06d}" for i in range(1000,1099)] for q in labels}
        with tempfile.TemporaryDirectory() as temp:
            def owner():
                obj=SailerGradCacheTrainer.__new__(SailerGradCacheTrainer)
                obj.config={"output_dir":temp,"sampler_seed":289,"batch_grouping":SHUFFLE_REPAIR}
                obj.bundle={"train_labels":labels,"train_pools":pools}
                obj.run_dir=Path(temp)
                obj.train_dataset=SimpleNamespace(rows=[])
                obj.label_manifest={"hash":"frozen-mask"}
                obj.log=lambda values:None
                return obj
            uninterrupted=owner()
            uninterrupted.resample_epoch(0)
            first=copy.deepcopy(uninterrupted.train_dataset.rows)
            uninterrupted.resample_epoch(1)
            second=copy.deepcopy(uninterrupted.train_dataset.rows)
            resumed=owner()
            resumed.resample_epoch(1)
            self.assertEqual(second,resumed.train_dataset.rows)
            self.assertNotEqual(first,second)
            self.assertEqual(resumed.epoch_seen_groups,[])
            self.assertEqual(read_json(Path(temp)/"sampling/epoch_02_train.audit.json")["policy"],SHUFFLE_REPAIR)

    def test_resume_rejects_grouping_state_mismatch(self):
        from fine_tune.fine_tune import validate_resume
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cp=root/"checkpoint-2";cp.mkdir()
            config={"output_dir":temp,"num_train_epochs":2,"training_method":"sailer_gradcache",
                    "batch_grouping":SHUFFLE_REPAIR}
            write_json(root/"run_config.json",config)
            write_json(root/"run_manifest.json",{"implementation":implementation_id(config),"data_hash":"data"})
            for name in ("trainer_state.json","optimizer.pt","scheduler.pt","scaler.pt","rng_state.pth",
                         "model.safetensors","config.json","contrastive_config.json","tokenizer_config.json","amp_monitor.json"):
                (cp/name).touch()
            write_json(cp/"gradcache_state.json",{"batch_grouping":LEGACY_DEGREE})
            with self.assertRaisesRegex(ValueError,"grouping strategy"):
                validate_resume(config,{"data_hash":"data"},cp)


if __name__ == "__main__":
    unittest.main()
