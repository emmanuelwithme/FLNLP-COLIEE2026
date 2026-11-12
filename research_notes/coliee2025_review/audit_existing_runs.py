"""Audit saved data/logs; no training, inference, .env loading, or dataset edits.

Default dependencies: Python standard library.
--plots additionally needs matplotlib; --check-batch needs torch/transformers.
Writes only derived evidence files beneath --output-dir.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
from typing import Dict, List


RUNS = {
    "static": "modernBERT_contrastive",
    "static_0911": "modernBERT_contrastive_0911",
    "static_0912": "modernBERT_contrastive_0912",
    "dynamic_2025": "modernBERT_contrastive_adaptive",
    "random_2026": "modernBERT_contrastive_random_fp_fp16_scopeFilteredRaw_plus0_emptyExclude_2026",
    "dynamic_2026": "modernBERT_contrastive_adaptive_fp_fp16_scopeFilteredRaw_plus0_emptyExclude_2026",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalized_labels(path):
    return {q.removesuffix(".txt"): {p.removesuffix(".txt") for p in ps}
            for q, ps in read_json(path).items()}


def read_bm25(path):
    ranks, counts = {}, Counter()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 6:
                continue
            qid, _, pid, rank, _, _ = fields
            qid, pid, rank = qid.zfill(6), pid.zfill(6), int(rank)
            counts[qid] += 1
            if rank <= 100:
                ranks.setdefault(qid, {})[pid] = rank
    return ranks, counts


def audit_static(task, split):
    data_path = task / "lht_process/modernBert/finetune_data" / f"contrastive_bm25_hard_negative_top100_random15_{split}.json"
    labels_path = task / f"task1_train_labels_2025_{split}.json"
    labels, rows = normalized_labels(labels_path), read_json(data_path)
    ranks, candidate_counts = read_bm25(task / f"lht_process/BM25/output_bm25_{split}.tsv")
    expected = {(q, p) for q, ps in labels.items() for p in ps}
    observed = Counter((r["query_id"], r["positive_id"]) for r in rows)
    docs = {p.stem for p in (task / "processed").glob("*.txt")}
    lengths, histogram, selected_by_query = Counter(), Counter(), {}
    self_rows, positive_rows, duplicate_rows, outside_rows, missing_ids = [], 0, 0, 0, set()
    for row in rows:
        qid, pid, negatives = row["query_id"], row["positive_id"], row["negative_ids"]
        lengths[len(negatives)] += 1
        if qid in negatives:
            self_rows.append(row)
        positive_rows += bool(set(negatives) & labels.get(qid, set()))
        duplicate_rows += len(set(negatives)) != len(negatives)
        outside_rows += any(n not in ranks.get(qid, {}) for n in negatives)
        missing_ids.update({qid, pid, *negatives} - docs)
        selected_by_query.setdefault(qid, set()).update(negatives)
        for negative in negatives:
            rank = ranks.get(qid, {}).get(negative)
            if rank is not None:
                bucket = "1-20" if rank <= 20 else "21-40" if rank <= 40 else "41-60" if rank <= 60 else "61-80" if rank <= 80 else "81-100"
                histogram[bucket] += 1
    pool_sizes = [len(set(ranks[q]) - labels[q]) for q in labels if q in ranks]
    expected_self = sum(15 / len(set(ranks[r["query_id"]]) - labels[r["query_id"]])
                        for r in rows if r["query_id"] in ranks[r["query_id"]])
    return {
        "data_path": str(data_path), "data_sha256": digest(data_path),
        "labels_sha256": digest(labels_path), "rows": len(rows), "queries": len({r['query_id'] for r in rows}),
        "expected_positive_pairs": len(expected), "missing_positive_pairs": len(expected - observed.keys()),
        "unexpected_positive_pairs": len(observed.keys() - expected),
        "duplicate_positive_pair_rows": sum(n - 1 for n in observed.values()),
        "negative_lengths": dict(lengths), "positive_leak_rows": positive_rows,
        "duplicate_negative_rows": duplicate_rows, "negative_outside_bm25_top100_rows": outside_rows,
        "missing_document_ids": sorted(missing_ids), "self_negative_rows": len(self_rows),
        "self_negative_fraction": len(self_rows) / len(rows), "self_negative_examples": self_rows[:3],
        "expected_self_rows_if_uniform_within_saved_pool": expected_self,
        "negative_bm25_rank_histogram": dict(histogram),
        "negative_pool_size_min_max": [min(pool_sizes), max(pool_sizes)],
        "unique_selected_per_query_mean": statistics.mean(map(len, selected_by_query.values())),
        "bm25_candidates_per_query_histogram": dict(Counter(candidate_counts.values())),
        "current_processed_document_count": len(docs),
    }


def audit_adaptive(task):
    labels = normalized_labels(task / "task1_train_labels_2025_train.json")
    paths = sorted((task / "lht_process/modernBert/finetune_data").glob("adaptive_negative_epoch*_train.json"),
                   key=lambda p: int(p.stem.split("epoch")[1].split("_")[0]))
    results = []
    for path in paths:
        rows = read_json(path)
        results.append({
            "file": path.name, "rows": len(rows),
            "self_negative_rows": sum(r["query_id"] in r["negative_ids"] for r in rows),
            "positive_leak_rows": sum(bool(set(r["negative_ids"]) & labels.get(r["query_id"], set())) for r in rows),
        })
    return {"warning": "Shared epoch files lack a run manifest; cannot attribute all files to the thesis checkpoint.",
            "files": results}


def audit_run(root, directory):
    paths = list((root / "models" / directory).glob("checkpoint-*/trainer_state.json"))
    if not paths:
        return {"missing": directory}
    path = max(paths, key=lambda p: int(p.parent.name.split("-")[-1]))
    state = read_json(path)
    history = state["log_history"]
    train = [r for r in history if "loss" in r]
    evaluate = [r for r in history if "eval_loss" in r]
    nonfinite = [(r.get("step"), key) for r in history for key, value in r.items()
                 if isinstance(value, (float, int)) and not math.isfinite(value)]
    epoch_losses = {}
    for row in train:
        epoch_losses.setdefault(math.ceil(row.get("epoch", 0) - 1e-9), []).append(row["loss"])
    return {
        "state_path": str(path), "state_sha256": digest(path),
        "best_model_checkpoint": state.get("best_model_checkpoint"), "best_metric": state.get("best_metric"),
        "first_logged_train_loss": train[0]["loss"] if train else None,
        "last_logged_train_loss": train[-1]["loss"] if train else None,
        "logged_epoch_mean_loss": {epoch: statistics.mean(values) for epoch, values in epoch_losses.items()},
        "epoch_mean_note": "Unweighted mean of logged training-loss windows, not exact per-sample epoch loss.",
        "nonfinite_logged_values": nonfinite,
        "max_logged_grad_norm": max((r["grad_norm"] for r in history if "grad_norm" in r), default=None),
        "train_history": train, "eval_history": evaluate,
    }


def sampling_distribution(task, epoch, limit):
    """Describe the first N stored query pools; do not claim source-run provenance."""
    path = task / f"lht_process/modernBert/finetune_data/similarity_scores_epoch{epoch}.tsv"
    if not path.exists():
        return {"missing": str(path)}
    positives = normalized_labels(task / "task1_train_labels_2025_train.json")
    pools = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 6:
                continue
            qid, _, pid, _, score, _ = fields
            if qid not in pools:
                if len(pools) >= limit:
                    break
                pools[qid] = []
            if pid != qid and pid not in positives.get(qid, set()):
                pools[qid].append(float(score))
    diagnostics = []
    for qid, scores in pools.items():
        if len(scores) < 2:
            continue
        maximum, n = max(scores), len(scores)
        weights = [math.exp(score - maximum) for score in scores]  # Ts=1.0
        total = sum(weights)
        probs = [w / total for w in weights]
        diagnostics.append({
            "query": qid, "pool_size": n, "score_min": min(scores), "score_max": maximum,
            "normalized_entropy": -sum(p * math.log(p) for p in probs) / math.log(n),
            "effective_sample_size_fraction": 1 / (n * sum(p * p for p in probs)),
            "total_variation_from_uniform": 0.5 * sum(abs(p - 1 / n) for p in probs),
            "top100_mass": sum(sorted(probs, reverse=True)[:100]), "uniform_top100_mass": min(100, n) / n,
        })
    fields = ["normalized_entropy", "effective_sample_size_fraction", "total_variation_from_uniform", "top100_mass", "uniform_top100_mass"]
    return {
        "source": str(path), "sampling_temperature_assumed": 1.0,
        "sampling_note": "First N contiguous query blocks, not a representative random query sample. Describes initial draw probabilities; sampling is without replacement.",
        "provenance_note": "Shared score files may have been overwritten; these values cannot establish the thesis run's distribution.",
        "queries_checked": len(diagnostics), "mean": {k: statistics.mean(r[k] for r in diagnostics) for k in fields},
        "per_query": diagnostics,
    }


def check_historical_batch(root, task):
    """Exercise only the preserved Dataset/Collator, on CPU, with a local tokenizer."""
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer

    source = subprocess.check_output(
        ["git", "show", "eb92236:Legal Case Retrieval/modernBert/fine_tune/fine_tune1.py"],
        cwd=root, text=True,
    )
    selected = [node for node in ast.parse(source).body
                if isinstance(node, ast.ClassDef) and node.name in {"ContrastiveDataset", "ContrastiveCollator"}]
    namespace = {"Dataset": Dataset, "torch": torch, "AutoTokenizer": AutoTokenizer,
                 "dataclass": dataclass, "json": json, "os": os, "List": List, "Dict": Dict}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "historical_static_dataset_collator", "exec"), namespace)
    path = task / "lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_train.json"
    dataset = namespace["ContrastiveDataset"](str(path), str(task / "processed"))
    tokenizer = AutoTokenizer.from_pretrained(root / "models/modernBERT_contrastive_0912/checkpoint-5424", local_files_only=True)
    samples = [dataset[0], dataset[1]]
    collator = namespace["ContrastiveCollator"](tokenizer, max_length=128)
    batch = collator(samples)
    texts = [s["query_text"] for s in samples] + [s["positive_text"] for s in samples] + [t for s in samples for t in s["negative_texts"]]
    expected = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    boundaries = {"anchor_input": (0, 2), "positive_input": (2, 4), "negative_input": (4, 34)}
    for role, (start, end) in boundaries.items():
        for key in ("input_ids", "attention_mask"):
            assert torch.equal(batch[role][key], expected[key][start:end]), (role, key)
    assert batch["labels"].tolist() == [0, 0]
    return {"source_commit": "eb92236", "samples_checked": 2, "negatives_per_sample": 15,
            "labels": batch["labels"].tolist(), "batch_mapping_matches_tokenized_input": True,
            "note": "CPU data-path test at max_length=128; no encoder, backpropagation, optimizer, or training. Historical training length was 4096."}


def plots(report, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key in ("static", "static_0911", "static_0912"):
        run = report["runs"][key]
        if "missing" in run:
            continue
        train, evaluation = run["train_history"], run["eval_history"]
        axes[0, 0].plot([r["epoch"] for r in train], [r["loss"] for r in train], label=key, alpha=0.8)
        axes[0, 1].plot([r["epoch"] for r in evaluation], [r["eval_loss"] for r in evaluation], marker="o", label=key)
        axes[1, 0].plot([r["epoch"] for r in evaluation], [r["eval_acc5"] for r in evaluation], marker="o", label=key)
    dynamic = report["runs"]["dynamic_2025"].get("eval_history", [])
    axes[1, 1].plot([r["epoch"] for r in dynamic], [r["eval_global_f1"] for r in dynamic], marker="o", label="dynamic_2025")
    titles = ["Static: logged training loss", "Static: sampled validation loss",
              "Static: top-5 accuracy among 16 documents", "Dynamic: full-corpus validation F1@5"]
    for ax, title in zip(axes.flat, titles):
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Saved COLIEE-2025 logs (no new training; static full-corpus F1 curves were not logged)")
    for suffix in ("png", "pdf"):
        fig.savefig(destination / f"coliee2025_saved_curves.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "evidence")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--check-batch", action="store_true")
    parser.add_argument("--score-query-limit", type=int, default=32)
    args = parser.parse_args()
    if args.score_query_limit < 1:
        parser.error("--score-query-limit must be positive")
    root, destination = args.repo_root.resolve(), args.output_dir.resolve()
    task = root / "coliee_dataset/task1/2025"
    report = {
        "scope": "Saved local evidence only; no causal proof and no guarantee of historical artifact provenance.",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "static_data": {split: audit_static(task, split) for split in ("train", "valid")},
        "adaptive_data": audit_adaptive(task),
        "runs": {name: audit_run(root, directory) for name, directory in RUNS.items()},
        "sampling_distribution": {str(epoch): sampling_distribution(task, epoch, args.score_query_limit) for epoch in (0, 5)},
    }
    if args.check_batch:
        report["batch_check"] = check_historical_batch(root, task)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.plots:
        plots(report, destination)
    for split, stats in report["static_data"].items():
        print(f"{split}: rows={stats['rows']} self_negatives={stats['self_negative_rows']} ({stats['self_negative_fraction']:.2%}), positive_leaks={stats['positive_leak_rows']}, missing_docs={len(stats['missing_document_ids'])}, outside_top100={stats['negative_outside_bm25_top100_rows']}")
    for key, run in report["runs"].items():
        print(f"{key}: best_metric={run.get('best_metric')} best_checkpoint={run.get('best_model_checkpoint')}")
    if args.check_batch:
        print("Historical Dataset/Collator CPU check passed; labels=[0,0], batch shapes=2/2/30.")
    for epoch, stats in report["sampling_distribution"].items():
        print(f"Stored scores epoch {epoch} (unverified provenance): {stats.get('mean')}")
    print(f"Evidence written to {destination}")


if __name__ == "__main__":
    main()
