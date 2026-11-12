"""Large CPU-only grouping validation on real IDs and adversarial cases.

Never writes into a training run. Grouping-only trials use real positive-pair
IDs; the separate full-pipeline phase also draws and validates all negatives.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import statistics
import sys
import time

from config import (PROJECT_DIR, default_report_directory, digest, load_config,
                    read_json, write_json)
from fine_tune.batch_grouping import shuffle_repair_rows, legacy_degree_rows, _repair_distinct_queries


def summarize(values):
    values = sorted(values)
    return {"mean": statistics.mean(values), "min": values[0], "max": values[-1],
            "p05": values[int((len(values) - 1) * .05)], "median": statistics.median(values),
            "p95": values[int((len(values) - 1) * .95)]}


def group_view(rows):
    groups = [tuple(sorted(r["query_id"] for r in rows[i:i + 4])) for i in range(0, len(rows), 4)]
    peers = defaultdict(set)
    pairs, triples = set(), set()
    top_pair_count = 0
    for group in groups:
        if len(set(group)) != 4:
            raise AssertionError("Query collision")
        pairs.update(itertools.combinations(group, 2))
        triples.update(itertools.combinations(group, 3))
        for q in group:
            peers[q].update(set(group) - {q})
        top_pair_count += "037853" in group and "099187" in group
    counts = Counter(groups)
    metrics = {"distinct_groups": len(counts), "repeated_group_fraction": (len(groups) - len(counts)) / len(groups),
               "mean_distinct_partners": statistics.mean(map(len, peers.values())),
               "top_query_distinct_partners": len(peers["099187"]),
               "selected_query_pair_cooccurrences": top_pair_count}
    return groups, peers, (pairs, triples, set(groups)), metrics


def overlap(previous, groups):
    pairs, triples, quads = previous
    n = len(groups)
    return {"at_least_2_fraction": sum(any(c in pairs for c in itertools.combinations(g, 2)) for g in groups) / n,
            "at_least_3_fraction": sum(any(c in triples for c in itertools.combinations(g, 3)) for g in groups) / n,
            "at_least_4_fraction": sum(g in quads for g in groups) / n}


def check_pairs(original_counts, rows, audit):
    expected = original_counts.copy()
    expected.update((r["query_id"], r["positive_id"]) for r in audit["padding_pairs"])
    if Counter((r["query_id"], r["positive_id"]) for r in rows) != expected:
        raise AssertionError("Missing/extra original pair")


def evaluate_policy(original, seeds, epochs, packer, output):
    original_counts = Counter((r["query_id"], r["positive_id"]) for r in original)
    epoch_records, adjacent_records, runs = [], [], []
    positions = defaultdict(lambda: [0] * 4)
    start = time.perf_counter()
    for run in range(seeds):
        seed = 289 + run * 1009
        previous = None
        union = defaultdict(set)
        run_top_pair = 0
        for epoch in range(epochs):
            rows, audit = packer(original, epoch, seed)
            check_pairs(original_counts, rows, audit)
            if audit.get("policy") == "shuffle_repair_v1":
                assert audit["padding_distinct_queries"] == 3
            groups, peers, indices, metrics = group_view(rows)
            for q, others in peers.items():
                union[q].update(others)
            for i, row in enumerate(rows):
                positions[row["query_id"]][i % 4] += 1
            run_top_pair += metrics["selected_query_pair_cooccurrences"]
            epoch_records.append({"seed": seed, "epoch": epoch + 1, **metrics,
                                  "padded_pairs": audit["padded_pairs"],
                                  "repair_swaps": audit.get("repair_swaps", 0),
                                  "initial_conflicts": audit.get("initial_duplicate_slots", 0),
                                  "fallback_searches": audit.get("fallback_searches", 0),
                                  "grouping_hash": audit["grouping_hash"] if "grouping_hash" in audit else digest(groups)})
            if previous is not None:
                adjacent_records.append({"seed": seed, "epoch": epoch + 1, **overlap(previous, groups)})
            previous = indices
        runs.append({"seed": seed, "mean_unique_partners_across_run": statistics.mean(map(len, union.values())),
                     "top_query_unique_partners_across_run": len(union["099187"]),
                     "selected_query_pair_cooccurrences_across_run": run_top_pair})
        if (run + 1) % 10 == 0 or run + 1 == seeds:
            print(f"{output.stem}: {run+1}/{seeds} seeds, {(run+1)*epochs} epochs", flush=True)
    def fields(records, excluded):
        return {k: summarize([r[k] for r in records]) for k in records[0] if k not in excluded}
    result = {"seeds": seeds, "epochs_per_seed": epochs, "total_epochs": seeds * epochs,
              "logical_batches": len(epoch_records) * ((len(original) + 3) // 4),
              "pair_instances": len(epoch_records) * ((len(original) + 3) // 4 * 4),
              "all_invariants_passed": True, "seconds": time.perf_counter() - start,
              "within_epoch": fields(epoch_records, {"seed", "epoch", "grouping_hash"}),
              "adjacent_epochs": fields(adjacent_records, {"seed", "epoch"}),
              "across_run": fields(runs, {"seed"}),
              "max_absolute_query_position_share_deviation_from_quarter": max(
                  abs(value / sum(counts) - .25) for counts in positions.values() for value in counts)}
    write_json(output, {**result, "epoch_records": epoch_records, "adjacent_epoch_records": adjacent_records,
                        "runs": runs, "query_position_counts": dict(positions)})
    return result


def stress_test(count):
    rng = random.Random(278912)
    repairs = fallbacks = forced_fallback_cases = rejected = 0
    start = time.perf_counter()
    for trial in range(count):
        qcount = 4 if trial % 4 == 0 else rng.randint(5, 128)
        batches = rng.randint(2, 256)
        sequence = []
        for _ in range(batches):
            group = ([0] + rng.sample(range(1, qcount), 3)) if trial % 4 == 1 else rng.sample(range(qcount), 4)
            sequence.extend(group)
        for _ in range(rng.randrange(4)):
            sequence.pop(rng.randrange(len(sequence)))
        if len(set(sequence)) < 4:
            sequence = [0, 1, 2, 3]
        original = [{"query_id": str(q), "positive_id": str(i), "payload": [q, i]} for i, q in enumerate(sequence)]
        original_counts = Counter((r["query_id"], r["positive_id"]) for r in original)
        rows, audit = shuffle_repair_rows(original, trial, 289)
        check_pairs(original_counts, rows, audit)
        assert all(len({r["query_id"] for r in rows[i:i+4]}) == 4 for i in range(0,len(rows),4))
        assert audit["repair_swaps"] <= audit["initial_duplicate_slots"]
        repairs += audit["repair_swaps"]
        fallbacks += audit["fallback_searches"]
        if trial % 10 == 0:
            concentrated = sorted(rows, key=lambda r: r["query_id"])
            before = Counter(map(id, concentrated))
            extra = _repair_distinct_queries(concentrated, rng, random_attempts=0)
            assert Counter(map(id, concentrated)) == before
            forced_fallback_cases += 1
            fallbacks += extra["fallback_searches"]
        # Deliberately impossible majority query: verify explicit rejection.
        impossible = [{"query_id": "0", "positive_id": str(i)} for i in range(9)]
        impossible += [{"query_id": str(q), "positive_id": f"other{q}"} for q in [1,2,3]]
        try:
            shuffle_repair_rows(impossible, trial)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError("Impossible input was accepted")
    return {"feasible_cases": count, "impossible_cases_correctly_rejected": rejected,
            "forced_exhaustive_search_cases": forced_fallback_cases, "total_repair_swaps": repairs,
            "fallback_searches_including_forced": fallbacks, "all_invariants_passed": True,
            "seconds": time.perf_counter() - start}


def full_negative_pipeline(config, labels, trials):
    import torch
    torch.set_num_threads(1)
    from fine_tune.data import build_pools, sample_epoch, audit_samples
    pools = build_pools(config["bm25_train"], labels, 100, 15)
    snapshots = []
    start = time.perf_counter()
    for trial in range(trials):
        seed, epoch = 289 + (trial // 20) * 1009, trial % 20
        original = sample_epoch(labels, pools, epoch, seed)
        frozen = {(r["query_id"], r["positive_id"]): tuple(r["negative_ids"]) for r in original}
        rows, audit = shuffle_repair_rows(original, epoch, seed)
        assert audit["padded_pairs"] == audit["padding_distinct_queries"] == 3
        audit_samples(rows, labels, pools)
        check_pairs(Counter(frozen.keys()), rows, audit)
        assert all(tuple(r["negative_ids"]) == frozen[r["query_id"], r["positive_id"]] for r in rows)
        assert (rows, audit) == shuffle_repair_rows(original, epoch, seed)
        snapshots.append({"seed": seed, "epoch": epoch+1, "sample_hash": digest(rows), **audit})
        if (trial + 1) % 20 == 0:
            print(f"Full negatives: {trial+1}/{trials} epochs", flush=True)
    return {"epochs": trials, "original_pairs": len(frozen) * trials,
            "original_negative_slots": len(frozen) * trials * 15,
            "all_invariants_and_reproducibility_passed": True,
            "seconds": time.perf_counter() - start, "snapshots": snapshots}


def paired_comparison(new_path, legacy_path):
    new, old = read_json(new_path), read_json(legacy_path)
    shared = {r["seed"] for r in new["runs"]} & {r["seed"] for r in old["runs"]}
    result = {"matched_seeds": len(shared), "epochs_per_seed": new["epochs_per_seed"]}
    for section, keys in (("epoch_records", ["mean_distinct_partners", "repeated_group_fraction"]),
                          ("adjacent_epoch_records", ["at_least_2_fraction", "at_least_3_fraction", "at_least_4_fraction"])):
        for key in keys:
            means = []
            tables = []
            for policy in (new, old):
                grouped = defaultdict(list)
                for row in policy[section]:
                    if row["seed"] in shared: grouped[row["seed"]].append(row[key])
                tables.append({s:statistics.mean(v) for s,v in grouped.items()})
            differences = [tables[0][s]-tables[1][s] for s in sorted(shared)]
            result[key] = {"new_mean":statistics.mean(tables[0].values()),
                           "legacy_mean":statistics.mean(tables[1].values()),
                           "new_minus_legacy_per_seed":summarize(differences),
                           "seeds_new_greater":sum(x>0 for x in differences),
                           "seeds_new_lower":sum(x<0 for x in differences)}
    return result


def actual_mask_flow(config, labels, epochs):
    import torch
    torch.set_num_threads(1)
    from fine_tune.data import build_pools
    from fine_tune.sailer import logical_epoch_rows, label_mask
    from config import implementation_id
    legacy = load_config(PROJECT_DIR/"configs/coliee2025_sailer_gradcache_legacy.json")
    old_root = Path(legacy["output_dir"])
    manifest = read_json(old_root/"prepared/label_mask_manifest.json")
    assert manifest["hash"] == digest({k:v for k,v in manifest.items() if k!="hash"})
    pools = build_pools(config["bm25_train"], labels, 100, 15)
    documents = manifest["documents"]
    metrics = Counter()
    start = time.perf_counter()
    hashes = []
    for epoch in range(epochs):
        rows, audit = logical_epoch_rows(labels, pools, epoch, config["sampler_seed"], strategy="shuffle_repair_v1")
        hashes.append(audit["grouping_hash"])
        for i in range(0,len(rows),4):
            block = rows[i:i+4]
            allowed, targets, stats = label_mask(block,manifest)
            assert targets.tolist()==[0,16,32,48]
            passages = [p for r in block for p in [r["positive_id"],*r["negative_ids"]]]
            for row_index,flags in enumerate(allowed.tolist()):
                q=block[row_index]["query_id"]
                target=row_index*16
                assert flags[target] and passages[target]==block[row_index]["positive_id"]
                gold=set(manifest["known_positives"][q])
                gold_hashes={documents[p]["token_hash"] for p in gold}
                accepted=[]
                for column,flag in enumerate(flags):
                    if not flag:continue
                    pid=passages[column]
                    accepted.append(documents[pid]["token_hash"])
                    if column!=target:
                        assert pid!=q and pid not in gold and documents[pid]["valid"]
                        assert documents[pid]["token_hash"] not in gold_hashes
                        assert documents[pid]["token_hash"]!=documents[q]["token_hash"]
                assert len(set(accepted))==len(accepted) and len(accepted)>=2
            metrics.update(stats)
        if (epoch+1)%5==0:print(f"Real mask flow: {epoch+1}/{epochs} epochs",flush=True)
    saved_matches={}
    for path in sorted((old_root/"sampling").glob("epoch_*_train.json")):
        e=int(path.name.split("_")[1])-1
        expected,_=logical_epoch_rows(labels,pools,e,legacy["sampler_seed"],strategy="legacy_degree_v1")
        saved_matches[path.name]=read_json(path)==expected
    assert all(saved_matches.values())
    assert legacy==read_json(old_root/"run_config.json")
    assert implementation_id(legacy)==read_json(old_root/"run_manifest.json")["implementation"]
    return {"passed":True,"epochs":epochs,"logical_batches":metrics["logical_batches"],
            "score_cells_checked":metrics["logical_batches"]*4*64,"mask_stats":dict(metrics),
            "grouping_hashes":hashes,"legacy_saved_samples_unchanged":saved_matches,
            "legacy_saved_config_and_implementation_match":True,"seconds":time.perf_counter()-start,
            "scope":"CPU real data mask contracts; no model forward/backward or live training changes"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--legacy-seeds", type=int, default=100)
    parser.add_argument("--stress-cases", type=int, default=20000)
    parser.add_argument("--full-negative-epochs", type=int, default=200)
    parser.add_argument("--flow-only",action="store_true",help="CPU real-data LabelMask and legacy compatibility checks")
    parser.add_argument("--output-dir", help="Report directory; default: reports/modernbert-bm25/grouping/<UTC-run-id>")
    args = parser.parse_args()
    if min(args.seeds, args.legacy_seeds, args.stress_cases, args.full_negative_epochs) < 1 or args.epochs < 2:
        parser.error("Use positive trial counts and epochs>=2")
    config = load_config(PROJECT_DIR/"configs/coliee2025_sailer_gradcache.json")
    raw = read_json(config["train_labels"])
    normalize = lambda value: str(value).removesuffix(".txt").zfill(6)
    labels = {normalize(q): sorted({normalize(p) for p in ps}) for q, ps in raw.items()}
    original = [{"query_id": q, "positive_id": p} for q in sorted(labels) for p in labels[q]]
    assert len(labels) == 1341 and len(original) == 5421
    output = Path(args.output_dir) if args.output_dir else default_report_directory("grouping")
    output.mkdir(parents=True, exist_ok=True)
    print(f"Report directory: {output}", flush=True)
    if args.flow_only:
        result=actual_mask_flow(config,labels,args.epochs)
        write_json(output/"flow_validation.json",result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    start = time.perf_counter()
    result = {"scope": "CPU grouping statistics; no training/evaluation/model weights changed",
              "grouping_only_trials_use_real_pair_ids_without_negative_draws": True,
              "source_train_labels_sha256": hashlib.sha256(Path(config["train_labels"]).read_bytes()).hexdigest(),
              "source_batch_grouping_sha256": hashlib.sha256((PROJECT_DIR/"fine_tune/batch_grouping.py").read_bytes()).hexdigest(),
              "source_bm25_train_sha256": hashlib.sha256(Path(config["bm25_train"]).read_bytes()).hexdigest(),
              "python_version":sys.version.split()[0],
              "queries":len(labels), "pairs":len(original), "epochs_per_seed":args.epochs}
    result["new"] = evaluate_policy(original, args.seeds, args.epochs, shuffle_repair_rows, output/"new_policy.json")
    write_json(output/"summary.partial.json", result)
    result["legacy"] = evaluate_policy(original, args.legacy_seeds, args.epochs, legacy_degree_rows, output/"legacy_policy.json")
    result["paired_comparison"] = paired_comparison(output/"new_policy.json",output/"legacy_policy.json")
    result["stress"] = stress_test(args.stress_cases)
    write_json(output/"summary.partial.json", result)
    result["full_negatives"] = full_negative_pipeline(config, labels, args.full_negative_epochs)
    assert result["new"]["within_epoch"]["mean_distinct_partners"]["mean"] > result["legacy"]["within_epoch"]["mean_distinct_partners"]["mean"]
    assert result["new"]["adjacent_epochs"]["at_least_3_fraction"]["mean"] < result["legacy"]["adjacent_epochs"]["at_least_3_fraction"]["mean"]
    result["elapsed_seconds"] = time.perf_counter() - start
    result["passed"] = True
    write_json(output/"summary.json", result)
    (output/"summary.partial.json").unlink(missing_ok=True)
    print(json.dumps({k:v for k,v in result.items() if k not in ("new","legacy","full_negatives")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
