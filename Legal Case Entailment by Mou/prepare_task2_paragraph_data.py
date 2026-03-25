#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_config import get_env, get_env_float, get_env_int, get_env_path, load_dotenv_if_present


def normalize_numeric_stem(raw_id: object, width: int = 3) -> str:
    stem = Path(str(raw_id)).stem.strip()
    if not stem.isdigit():
        raise ValueError(f"Expected numeric id, got: {raw_id}")
    return str(int(stem)).zfill(width)


def compose_candidate_id(case_id: str, paragraph_id: str) -> str:
    return f"{normalize_numeric_stem(case_id, 3)}{normalize_numeric_stem(paragraph_id, 3)}"


def clear_txt_files(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for file_path in directory.glob("*.txt"):
        if file_path.is_file():
            file_path.unlink()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_qid_tsv(path: Path, qid_keys: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [Path(qid).stem for qid in qid_keys]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def split_labels(
    labels: Dict[str, List[str]],
    *,
    train_ratio: float,
    split_seed: int,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str], List[str]]:
    qids = list(labels.keys())
    rng = random.Random(split_seed)
    rng.shuffle(qids)
    train_size = int(len(qids) * train_ratio)
    train_keys = qids[:train_size]
    valid_keys = qids[train_size:]
    train_labels = {qid: labels[qid] for qid in train_keys}
    valid_labels = {qid: labels[qid] for qid in valid_keys}
    return train_labels, valid_labels, train_keys, valid_keys


def build_contrastive_samples(
    labels: Dict[str, List[str]],
    query_candidates_map: Dict[str, List[str]],
    *,
    max_negatives: int,
    random_seed: int,
) -> Tuple[List[Dict[str, object]], int]:
    rng = random.Random(random_seed)
    samples: List[Dict[str, object]] = []
    skipped = 0

    for qid_key, positives in labels.items():
        qid = Path(qid_key).stem
        candidate_ids = query_candidates_map.get(qid, [])
        if not candidate_ids:
            skipped += len(positives)
            continue

        positive_ids = sorted({Path(pid).stem for pid in positives})
        negative_pool = [cid for cid in candidate_ids if cid not in positive_ids]
        if not negative_pool:
            skipped += len(positive_ids)
            continue

        for positive_id in positive_ids:
            if len(negative_pool) >= max_negatives:
                negative_ids = rng.sample(negative_pool, max_negatives)
            else:
                negative_ids = [rng.choice(negative_pool) for _ in range(max_negatives)]
            samples.append(
                {
                    "query_id": qid,
                    "positive_id": positive_id,
                    "negative_ids": negative_ids,
                }
            )
    return samples, skipped


def prepare_dataset(
    *,
    cases_dir: Path,
    labels_path: Path,
    output_dir: Path,
    train_ratio: float,
    split_seed: int,
    negative_seed: int,
    max_negatives: int,
) -> None:
    if not labels_path.is_file():
        raise FileNotFoundError(f"Label file not found: {labels_path}")
    if not cases_dir.is_dir():
        raise FileNotFoundError(f"Cases directory not found: {cases_dir}")

    raw_labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(raw_labels, dict):
        raise ValueError(f"Label file must be JSON object: {labels_path}")

    query_dir = output_dir / "processed_queries"
    candidate_dir = output_dir / "processed_candidates"
    finetune_dir = output_dir / "finetune_data"
    clear_txt_files(query_dir)
    clear_txt_files(candidate_dir)
    finetune_dir.mkdir(parents=True, exist_ok=True)

    flat_labels: Dict[str, List[str]] = {}
    query_candidates_map: Dict[str, List[str]] = {}
    stats = {
        "total_label_queries": len(raw_labels),
        "kept_queries": 0,
        "skipped_missing_query_file": 0,
        "skipped_empty_query_text": 0,
        "skipped_no_candidates": 0,
        "skipped_no_positive_in_candidates": 0,
        "kept_candidates": 0,
        "kept_positive_pairs": 0,
    }

    for raw_qid, raw_positive_list in sorted(raw_labels.items()):
        try:
            qid = normalize_numeric_stem(raw_qid, 3)
        except ValueError:
            stats["skipped_missing_query_file"] += 1
            continue

        case_dir = cases_dir / qid
        query_path = case_dir / "entailed_fragment.txt"
        paragraph_dir = case_dir / "paragraphs"

        if not query_path.is_file():
            stats["skipped_missing_query_file"] += 1
            continue
        if not paragraph_dir.is_dir():
            stats["skipped_no_candidates"] += 1
            continue

        paragraph_files = sorted(p for p in paragraph_dir.glob("*.txt") if p.is_file())
        if not paragraph_files:
            stats["skipped_no_candidates"] += 1
            continue

        query_text = query_path.read_text(encoding="utf-8").strip()
        if not query_text:
            stats["skipped_empty_query_text"] += 1
            continue

        candidate_ids: List[str] = []
        paragraph_stems: List[str] = []
        for paragraph_file in paragraph_files:
            try:
                paragraph_stem = normalize_numeric_stem(paragraph_file.stem, 3)
            except ValueError:
                continue
            candidate_id = compose_candidate_id(qid, paragraph_stem)
            candidate_text = paragraph_file.read_text(encoding="utf-8").strip()
            (candidate_dir / f"{candidate_id}.txt").write_text(candidate_text, encoding="utf-8")
            candidate_ids.append(candidate_id)
            paragraph_stems.append(paragraph_stem)

        if not candidate_ids:
            stats["skipped_no_candidates"] += 1
            continue

        paragraph_set = set(paragraph_stems)
        positive_stems = []
        for raw_positive in raw_positive_list:
            try:
                positive_stem = normalize_numeric_stem(raw_positive, 3)
            except ValueError:
                continue
            if positive_stem in paragraph_set:
                positive_stems.append(positive_stem)

        if not positive_stems:
            stats["skipped_no_positive_in_candidates"] += 1
            continue

        unique_positive_stems = sorted(set(positive_stems))
        flat_positive_ids = [
            f"{compose_candidate_id(qid, paragraph_stem)}.txt"
            for paragraph_stem in unique_positive_stems
        ]

        (query_dir / f"{qid}.txt").write_text(query_text, encoding="utf-8")
        flat_labels[f"{qid}.txt"] = flat_positive_ids
        query_candidates_map[qid] = candidate_ids

        stats["kept_queries"] += 1
        stats["kept_candidates"] += len(candidate_ids)
        stats["kept_positive_pairs"] += len(flat_positive_ids)

    labels_stem = labels_path.stem
    write_json(output_dir / f"{labels_stem}_flat.json", flat_labels)
    write_json(output_dir / "query_candidates_map.json", query_candidates_map)

    train_labels, valid_labels, train_keys, valid_keys = split_labels(
        flat_labels, train_ratio=train_ratio, split_seed=split_seed
    )
    write_json(output_dir / f"{labels_stem}_flat_train.json", train_labels)
    write_json(output_dir / f"{labels_stem}_flat_valid.json", valid_labels)
    write_qid_tsv(output_dir / "train_qid.tsv", train_keys)
    write_qid_tsv(output_dir / "valid_qid.tsv", valid_keys)

    train_samples, train_skipped = build_contrastive_samples(
        train_labels,
        query_candidates_map,
        max_negatives=max_negatives,
        random_seed=negative_seed,
    )
    valid_samples, valid_skipped = build_contrastive_samples(
        valid_labels,
        query_candidates_map,
        max_negatives=max_negatives,
        random_seed=negative_seed,
    )
    write_json(finetune_dir / "contrastive_task2_random15_train.json", train_samples)
    write_json(finetune_dir / "contrastive_task2_random15_valid.json", valid_samples)

    stats.update(
        {
            "train_queries": len(train_labels),
            "valid_queries": len(valid_labels),
            "contrastive_train_samples": len(train_samples),
            "contrastive_valid_samples": len(valid_samples),
            "contrastive_train_skipped_no_negative": train_skipped,
            "contrastive_valid_skipped_no_negative": valid_skipped,
            "train_ratio": train_ratio,
            "split_seed": split_seed,
            "negative_seed": negative_seed,
            "max_negatives": max_negatives,
        }
    )
    write_json(output_dir / "prepare_stats.json", stats)

    print("Task2 paragraph data prepared.")
    print(f"output_dir: {output_dir}")
    print(f"kept_queries: {stats['kept_queries']} / {stats['total_label_queries']}")
    print(f"train/valid queries: {stats['train_queries']} / {stats['valid_queries']}")
    print(
        "contrastive train/valid samples: "
        f"{stats['contrastive_train_samples']} / {stats['contrastive_valid_samples']}"
    )


def parse_args() -> argparse.Namespace:
    load_dotenv_if_present()
    default_input_dir = get_env_path("COLIEE_TASK2_DIR", required=True)
    default_output_dir = get_env_path("COLIEE_TASK2_PREPARED_DIR", required=True)
    labels_file = get_env("TASK2_LABELS_FILENAME", required=True)
    parser = argparse.ArgumentParser(
        description="Prepare COLIEE task2 paragraph-level data for ModernBERT contrastive fine-tuning."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Path to the task2 train-files directory.",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default=labels_file,
        help="Label JSON filename under --input-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Prepared output directory.",
    )
    parser.add_argument("--train-ratio", type=float, default=get_env_float("TASK2_PREPARE_TRAIN_RATIO", required=True))
    parser.add_argument("--split-seed", type=int, default=get_env_int("TASK2_PREPARE_SPLIT_SEED", required=True))
    parser.add_argument("--negative-seed", type=int, default=get_env_int("TASK2_PREPARE_NEGATIVE_SEED", required=True))
    parser.add_argument("--max-negatives", type=int, default=get_env_int("TASK2_PREPARE_MAX_NEGATIVES", required=True))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_dataset(
        cases_dir=args.input_dir / "cases",
        labels_path=args.input_dir / args.labels_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
        negative_seed=args.negative_seed,
        max_negatives=args.max_negatives,
    )


if __name__ == "__main__":
    main()
