#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_config import get_env, get_env_int, get_env_path, load_dotenv_if_present


def write_counter_csv(path: Path, counter: Counter) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["value", "count"])
        for key in sorted(counter.keys()):
            writer.writerow([key, counter[key]])


def iter_file_texts(files: Iterable[Path]) -> Iterable[str]:
    for path in files:
        yield path.read_text(encoding="utf-8").strip()


def compute_token_lengths(
    tokenizer,
    texts: Iterable[str],
    *,
    batch_size: int,
) -> List[int]:
    lengths: List[int] = []
    batch: List[str] = []
    for text in texts:
        batch.append(text)
        if len(batch) >= batch_size:
            lengths.extend(_tokenize_batch_lengths(tokenizer, batch))
            batch.clear()
    if batch:
        lengths.extend(_tokenize_batch_lengths(tokenizer, batch))
    return lengths


def _tokenize_batch_lengths(tokenizer, batch_texts: List[str]) -> List[int]:
    encoded = tokenizer(
        batch_texts,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_length=True,
    )
    if "length" in encoded:
        return [int(x) for x in encoded["length"]]
    return [len(ids) for ids in encoded["input_ids"]]


def save_histogram(
    values: List[int],
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    bins: int = 50,
    xlim: tuple[float, float] | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, color="#2E86DE", alpha=0.85, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    if xlim is not None:
        plt.xlim(*xlim)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_bar_from_counter(
    counter: Counter,
    out_path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = sorted(counter.keys())
    ys = [counter[x] for x in xs]
    plt.figure(figsize=(10, 6))
    plt.bar(xs, ys, color="#27AE60", alpha=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    load_dotenv_if_present(REPO_ROOT)
    prepared_dir = get_env_path("COLIEE_TASK2_PREPARED_DIR", required=True)

    parser = argparse.ArgumentParser(
        description="Compute task2 statistics with answerdotai/ModernBERT-base tokenizer."
    )
    parser.add_argument("--prepared-dir", type=Path, default=prepared_dir)
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default=get_env("TASK2_STATS_TOKENIZER_NAME", required=True),
    )
    parser.add_argument("--batch-size", type=int, default=get_env_int("TASK2_STATS_BATCH_SIZE", required=True))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <prepared-dir>/stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_dir = args.prepared_dir.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir else (prepared_dir / "stats"))
    output_dir.mkdir(parents=True, exist_ok=True)

    year = get_env("COLIEE_TASK2_YEAR", required=True)
    labels_path = prepared_dir / f"task2_train_labels_{year}_flat.json"
    query_dir = prepared_dir / "processed_queries"
    candidate_dir = prepared_dir / "processed_candidates"
    for path in [labels_path, query_dir, candidate_dir]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    relevant_counts = [len(v) for v in labels.values()]
    relevant_counter = Counter(relevant_counts)

    print("Loading tokenizer:", args.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    query_files = sorted(query_dir.glob("*.txt"))
    candidate_files = sorted(candidate_dir.glob("*.txt"))
    query_token_lengths = compute_token_lengths(
        tokenizer, iter_file_texts(query_files), batch_size=args.batch_size
    )
    candidate_token_lengths = compute_token_lengths(
        tokenizer, iter_file_texts(candidate_files), batch_size=args.batch_size
    )
    query_token_counter = Counter(query_token_lengths)
    candidate_token_counter = Counter(candidate_token_lengths)

    write_counter_csv(output_dir / "relevant_count_distribution.csv", relevant_counter)
    write_counter_csv(output_dir / "query_token_length_distribution.csv", query_token_counter)
    write_counter_csv(output_dir / "candidate_token_length_distribution.csv", candidate_token_counter)

    save_bar_from_counter(
        relevant_counter,
        output_dir / "relevant_count_distribution.png",
        title="Relevant Count Distribution per Query",
        xlabel="Relevant count per query",
    )
    save_histogram(
        query_token_lengths,
        output_dir / "query_token_length_hist.png",
        title="Query Token Length Distribution",
        xlabel="Token count",
        bins=50,
    )
    save_histogram(
        candidate_token_lengths,
        output_dir / "candidate_token_length_hist.png",
        title="Candidate Token Length Distribution",
        xlabel="Token count",
        bins=60,
        xlim=(
            max(0.0, float(np.mean(candidate_token_lengths) - 5.0 * np.std(candidate_token_lengths))),
            float(np.mean(candidate_token_lengths) + 5.0 * np.std(candidate_token_lengths)),
        ),
    )

    summary = {
        "num_queries": len(relevant_counts),
        "num_candidates": len(candidate_token_lengths),
        "avg_relevant_per_query": float(np.mean(relevant_counts)) if relevant_counts else 0.0,
        "avg_query_tokens": float(np.mean(query_token_lengths)) if query_token_lengths else 0.0,
        "avg_candidate_tokens": float(np.mean(candidate_token_lengths)) if candidate_token_lengths else 0.0,
        "max_query_tokens": int(max(query_token_lengths)) if query_token_lengths else 0,
        "max_candidate_tokens": int(max(candidate_token_lengths)) if candidate_token_lengths else 0,
        "min_query_tokens": int(min(query_token_lengths)) if query_token_lengths else 0,
        "min_candidate_tokens": int(min(candidate_token_lengths)) if candidate_token_lengths else 0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Statistics saved to:", output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
