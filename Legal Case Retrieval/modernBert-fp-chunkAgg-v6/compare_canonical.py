from __future__ import annotations

import argparse
import gc
import heapq
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Type

import torch
from transformers import AutoTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from content_utils import scan_eligible_document_ids
from env_utils import load_chunkagg_v6_dotenv

_LOADED_DOTENV_PATH = load_chunkagg_v6_dotenv(__file__)

from lcr.data import load_query_ids, normalize_case_id
from lcr.device import get_device
from lcr.metrics import my_classification_report, rel_file_to_dict
from lcr.retrieval import generate_similarity_artifacts
from lcr.task1_paths import get_env_path, get_task1_dir, get_task1_year
from repo_config import resolve_repo_path

FINE_TUNE_DIR = Path(__file__).resolve().parent / "fine_tune"
if str(FINE_TUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINE_TUNE_DIR))

from modernbert_contrastive_model import (  # noqa: E402
    CHUNK_MICROBATCH_SIZE,
    MAX_DOCUMENT_CHUNKS,
    ModernBERTContrastive as V6Model,
)
from inference import find_best_checkpoint  # noqa: E402


def _load_class(module_path: Path, class_name: str, module_name: str) -> Type:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _autocast_dtype() -> torch.dtype | None:
    if os.getenv("TASK1_CHUNKAGG_V6_BF16", "0") == "1":
        return torch.bfloat16
    if os.getenv("TASK1_CHUNKAGG_V6_FP16", "1") == "1":
        return torch.float16
    return None


def _canonical_metrics(
    *,
    model,
    tokenizer,
    device: torch.device,
    query_ids: list[str],
    labels: Dict,
    processed_dir: Path,
    scope_path: Path,
    eligible_ids: set[str],
    run_tag: str,
) -> Dict[str, float | int]:
    answers: Dict[int, list[str]] = {}

    def consume_scores(qid: str, scores: Dict[str, float]) -> None:
        normalized_qid = normalize_case_id(qid)
        top_items = heapq.nlargest(
            5,
            (
                (normalize_case_id(doc_id), float(score))
                for doc_id, score in scores.items()
                if normalize_case_id(doc_id) != normalized_qid
            ),
            key=lambda item: item[1],
        )
        answers[int(normalized_qid)] = [doc_id for doc_id, _ in top_items]

    generate_similarity_artifacts(
        model,
        tokenizer,
        device,
        candidate_dir=processed_dir,
        query_dir=processed_dir,
        query_ids=query_ids,
        trec_output_path=Path(os.devnull),
        run_tag=run_tag,
        batch_size=int(os.getenv("TASK1_CHUNKAGG_V6_RETRIEVAL_BATCH_SIZE", "2")),
        max_length=4096,
        query_candidate_scope_path=scope_path,
        fallback_to_all_candidates_if_scope_missing=False,
        persist_trec=False,
        retain_scores=False,
        score_consumer=consume_scores,
        exclude_self=True,
        autocast_dtype=_autocast_dtype(),
        chunk_selection_strategy="head_middle_tail",
        reuse_query_embeddings=True,
        eligible_document_ids=eligible_ids,
    )

    gold_rows = []
    answer_rows = []
    missing = 0
    for qid, positive_ids in labels.items():
        gold_rows.append([int(item) for item in positive_ids])
        predictions = answers.get(int(qid), [])
        if not predictions:
            missing += 1
        answer_rows.append([int(item) for item in predictions])
    f1, precision, recall = my_classification_report(gold_rows, answer_rows)
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "queries": len(gold_rows),
        "gold_pairs": sum(len(row) for row in gold_rows),
        "missing_prediction_queries": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Canonical same-corpus/scope/content comparison for baseline and chunkAgg v6"
    )
    parser.add_argument("--baseline-checkpoint", default=os.environ["TASK1_CHUNKAGG_V6_BASELINE_CHECKPOINT"])
    parser.add_argument("--v6-checkpoint", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    task1_dir = Path(get_task1_dir())
    task1_year = get_task1_year()
    processed_dir = task1_dir / "processed"
    scope_path = get_env_path("TASK1_CHUNKAGG_V6_SCOPE_PATH", required=True)
    assert scope_path is not None
    valid_qid_path = task1_dir / "valid_qid.tsv"
    labels_path = task1_dir / f"task1_train_labels_{task1_year}.json"
    query_ids = load_query_ids(valid_qid_path)
    labels = rel_file_to_dict(labels_path, valid_qid_path)
    eligible_ids, _ = scan_eligible_document_ids(processed_dir, minimum_characters=100)
    if len(eligible_ids) != 7703 or len(labels) != 401 or sum(map(len, labels.values())) != 1656:
        raise RuntimeError(
            "canonical data contract drift: "
            f"eligible={len(eligible_ids)}, valid_qids={len(labels)}, "
            f"gold_pairs={sum(map(len, labels.values()))}"
        )

    baseline_checkpoint = resolve_repo_path(args.baseline_checkpoint)
    assert baseline_checkpoint is not None
    output_root = get_env_path("TASK1_CHUNKAGG_V6_OUTPUT_DIR", required=True)
    assert output_root is not None
    v6_checkpoint = (
        resolve_repo_path(args.v6_checkpoint)
        if args.v6_checkpoint
        else Path(find_best_checkpoint(output_root, "eval_global_f1", "max")[0]).resolve()
    )
    assert v6_checkpoint is not None
    base_encoder = get_env_path("TASK1_CHUNKAGG_V6_BASE_ENCODER_DIR", required=True)
    assert base_encoder is not None
    for checkpoint in (baseline_checkpoint, v6_checkpoint, base_encoder):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

    device = get_device()
    print(f"🔹 Canonical denominator: {len(labels)} queries / {sum(map(len, labels.values()))} gold pairs")

    baseline_class = _load_class(
        PACKAGE_ROOT / "modernBert-fp" / "fine_tune" / "modernbert_contrastive_model.py",
        "ModernBERTContrastive",
        "chunkagg_v6_baseline_model",
    )
    baseline_tokenizer = AutoTokenizer.from_pretrained(
        baseline_checkpoint, trust_remote_code=True
    )
    baseline_model = baseline_class.from_pretrained(
        str(baseline_checkpoint),
        encoder_model_name_or_path=str(base_encoder),
        encoder_kwargs={"torch_dtype": torch.float32, "trust_remote_code": True},
    ).to(device).eval()
    baseline_metrics = _canonical_metrics(
        model=baseline_model,
        tokenizer=baseline_tokenizer,
        device=device,
        query_ids=query_ids,
        labels=labels,
        processed_dir=processed_dir,
        scope_path=scope_path,
        eligible_ids=eligible_ids,
        run_tag="single_chunk_canonical",
    )
    del baseline_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    v6_tokenizer = AutoTokenizer.from_pretrained(v6_checkpoint, trust_remote_code=True)
    v6_model = V6Model(
        str(base_encoder),
        device=device,
        temperature=float(os.getenv("TASK1_CHUNKAGG_V6_INIT_TEMPERATURE", "0.55555")),
        max_chunks=MAX_DOCUMENT_CHUNKS,
        chunk_microbatch_size=CHUNK_MICROBATCH_SIZE,
        temperature_min=float(os.getenv("TASK1_CHUNKAGG_V6_TEMPERATURE_MIN", "0.05")),
        temperature_max=float(os.getenv("TASK1_CHUNKAGG_V6_TEMPERATURE_MAX", "2.0")),
        evidence_loss_weight=float(os.getenv("TASK1_CHUNKAGG_V6_EVIDENCE_LOSS_WEIGHT", "0.2")),
        evidence_pool_temperature=float(os.getenv("TASK1_CHUNKAGG_V6_EVIDENCE_POOL_TEMPERATURE", "0.1")),
        fusion_residual_max_norm=float(os.getenv("TASK1_CHUNKAGG_V6_FUSION_RESIDUAL_MAX_NORM", "0.5")),
        fusion_residual_penalty_weight=float(os.getenv("TASK1_CHUNKAGG_V6_FUSION_RESIDUAL_PENALTY_WEIGHT", "0.01")),
    )
    v6_model.load_checkpoint(v6_checkpoint)
    v6_model = v6_model.to(device).eval()
    v6_metrics = _canonical_metrics(
        model=v6_model,
        tokenizer=v6_tokenizer,
        device=device,
        query_ids=query_ids,
        labels=labels,
        processed_dir=processed_dir,
        scope_path=scope_path,
        eligible_ids=eligible_ids,
        run_tag="chunkagg_v6_canonical",
    )
    del v6_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report = {
        "contract": {
            "processed": str(processed_dir),
            "scope": str(scope_path),
            "minimum_non_whitespace_characters": 100,
            "valid_queries": 401,
            "valid_gold_pairs": 1656,
            "top_k": 5,
        },
        "historical_baseline_f1_reference": 0.164518,
        "baseline": {"checkpoint": str(baseline_checkpoint), **baseline_metrics},
        "chunkagg_v6": {"checkpoint": str(v6_checkpoint), **v6_metrics},
        "f1_delta_v6_minus_baseline": v6_metrics["f1"] - baseline_metrics["f1"],
    }
    results_dir = get_env_path("TASK1_CHUNKAGG_V6_RESULTS_DIR", required=True)
    assert results_dir is not None
    output_path = (
        resolve_repo_path(args.output)
        if args.output
        else results_dir / "canonical_comparison.json"
    )
    assert output_path is not None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"✅ Canonical comparison saved to {output_path}")


if __name__ == "__main__":
    main()
