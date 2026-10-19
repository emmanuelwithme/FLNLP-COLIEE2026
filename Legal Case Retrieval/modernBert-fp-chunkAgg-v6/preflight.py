from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

V6_ROOT = Path(__file__).resolve().parent
FINE_TUNE_ROOT = V6_ROOT / "fine_tune"
PACKAGE_ROOT = V6_ROOT.parent
for path in (V6_ROOT, FINE_TUNE_ROOT, PACKAGE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from env_utils import load_chunkagg_v6_dotenv

_LOADED_DOTENV_PATH = load_chunkagg_v6_dotenv(__file__)

import fine_tune as training  # noqa: E402
from content_utils import scan_eligible_document_ids  # noqa: E402
from lcr.data import load_query_candidate_scope, normalize_case_id  # noqa: E402
from lcr.device import get_device  # noqa: E402
from lcr.retrieval import _chunk_single_text  # noqa: E402
from lcr.task1_paths import get_env_path, get_task1_dir, get_task1_year  # noqa: E402
from modernbert_contrastive_model import ModernBERTContrastive  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _tensor_shapes(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {key: _tensor_shapes(item) for key, item in value.items()}
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    return {"type": type(value).__name__}


def _static_preflight() -> tuple[dict, dict, set[str], Path]:
    task1_dir = Path(get_task1_dir())
    year = get_task1_year()
    processed_dir = task1_dir / "processed"
    train_qid_path = task1_dir / "train_qid.tsv"
    valid_qid_path = task1_dir / "valid_qid.tsv"
    train_labels_path = task1_dir / f"task1_train_labels_{year}_train.json"
    all_labels_path = task1_dir / f"task1_train_labels_{year}.json"
    scope_path = get_env_path("TASK1_CHUNKAGG_V6_SCOPE_PATH", required=True)
    assert scope_path is not None

    eligible_ids, rejected = scan_eligible_document_ids(
        processed_dir,
        minimum_characters=100,
    )
    scope = load_query_candidate_scope(scope_path)
    stats = training.audit_v6_data_contract(
        eligible_document_ids=eligible_ids,
        train_qid_path=train_qid_path,
        valid_qid_path=valid_qid_path,
        positive_train_json_path=train_labels_path,
        labels_path=all_labels_path,
        query_candidate_scope=scope,
        enforce_expected_counts=True,
    )
    print("\n[1/6] Static corpus/split/scope contract")
    print(json.dumps({**stats, "rejected_documents": rejected}, indent=2, ensure_ascii=False))
    _require(stats["outside_scope_positive_pairs"] > 0, "scope-external positives were not retained")
    return stats, scope, eligible_ids, processed_dir


def _sampling_preflight(
    scope: dict,
    eligible_ids: set[str],
    processed_dir: Path,
) -> list[dict]:
    task1_dir = Path(get_task1_dir())
    positives = training.read_positive_pairs_from_json(
        str(task1_dir / f"task1_train_labels_{get_task1_year()}_train.json")
    )
    selected_qid = None
    selected_positive_ids = None
    selected_pool = None
    for qid, positive_ids in positives.items():
        qid = normalize_case_id(qid)
        eligible_positives = sorted(set(positive_ids) & eligible_ids)
        pool = sorted((set(scope.get(qid, [])) & eligible_ids) - set(positive_ids) - {qid})
        if qid in eligible_ids and len(eligible_positives) >= 2 and len(pool) >= 15:
            selected_qid = qid
            selected_positive_ids = eligible_positives
            selected_pool = pool
            break
    _require(selected_qid is not None, "could not find a multi-positive preflight query")
    assert selected_positive_ids is not None and selected_pool is not None

    scores = np.linspace(-0.75, 0.95, num=len(selected_pool), dtype=np.float64)
    probabilities = training._softmax_sampling_probabilities(scores.copy(), 1.0)
    expected = np.exp(scores - scores.max())
    expected /= expected.sum()
    np.testing.assert_allclose(probabilities, expected, rtol=1e-12, atol=1e-15)

    similarities = dict(zip(selected_pool, scores.tolist()))
    rows, query_stats = training._generate_query_adaptive_samples(
        selected_qid,
        similarities,
        set(selected_positive_ids),
        max_negatives=15,
        temperature=1.0,
        query_candidate_scope=scope,
        eligible_document_ids=eligible_ids,
        rng=np.random.default_rng(42),
    )
    _require(len(rows) == len(selected_positive_ids), "one row per positive was not generated")
    _require(all(len(row["negative_ids"]) == 15 for row in rows), "row does not contain 15 negatives")
    _require(all(len(set(row["negative_ids"])) == 15 for row in rows), "duplicate within one row")
    _require(rows[0]["negative_ids"] != rows[1]["negative_ids"], "positives reused one fixed draw")

    print("\n[2/6] Full-scope sampling input")
    print(json.dumps({
        "query_id": selected_qid,
        "positive_ids": selected_positive_ids,
        "negative_pool_size": len(selected_pool),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "sampling_temperature": 1.0,
    }, indent=2))
    print("\n[3/6] Sampled epoch rows (first two)")
    print(json.dumps({"stats": query_stats, "rows": rows[:2]}, indent=2, ensure_ascii=False))

    # Use the shortest scoped negatives for the forward smoke test. The sampler
    # itself was validated above over the complete pool.
    shortest_negatives = sorted(
        selected_pool,
        key=lambda doc_id: (processed_dir / f"{doc_id}.txt").stat().st_size,
    )[:15]
    # Include every positive in the temporary dataset so its false-negative
    # exclusion metadata matches the formal epoch dataset. The independent
    # probability draws were already validated and printed above.
    return [
        {
            "query_id": selected_qid,
            "positive_id": positive_id,
            "negative_ids": shortest_negatives,
        }
        for positive_id in selected_positive_ids
    ]


def _dataset_preflight(
    rows: list[dict],
    scope: dict,
    eligible_ids: set[str],
    processed_dir: Path,
):
    dataset = training.ContrastiveDataset(
        doc_folder=str(processed_dir),
        data=rows,
        query_candidate_scope=scope,
        eligible_document_ids=eligible_ids,
        minimum_content_characters=100,
        required_negative_count=15,
    )
    item = dataset[0]
    print("\n[4/6] Dataset item")
    print(json.dumps({
        "query_id": item["query_id"],
        "query_characters": len(item["query_text"]),
        "positive_id": item["positive_id"],
        "positive_characters": len(item["positive_text"]),
        "negative_ids": item["negative_ids"],
        "negative_text_count": len(item["negative_texts"]),
        "all_positive_ids": item["all_positive_ids"],
    }, indent=2, ensure_ascii=False))
    return dataset, item


def _model_preflight(dataset, item, *, backward: bool) -> None:
    device = get_device()
    base_encoder = get_env_path("TASK1_CHUNKAGG_V6_BASE_ENCODER_DIR", required=True)
    assert base_encoder is not None
    tokenizer = AutoTokenizer.from_pretrained(base_encoder, trust_remote_code=True)
    tokenizer.model_max_length = 1_000_000_000

    canonical_chunks = _chunk_single_text(
        item["query_text"],
        tokenizer,
        max_length=4096,
        max_chunks=3,
        selection_strategy="head_middle_tail",
    )
    _require(tuple(canonical_chunks["input_ids"].shape) == (3, 4096), "canonical chunk shape mismatch")
    print("\n[5/6] Canonical 4096-token chunk structure")
    print(json.dumps(_tensor_shapes(canonical_chunks), indent=2))
    print(f"chunk_mask={canonical_chunks['chunk_mask'].tolist()}")

    # A shorter token dimension keeps the architecture/backward smoke test fast;
    # the canonical 3x4096 chunking contract is validated immediately above.
    collator = training.ContrastiveCollator(
        tokenizer,
        max_length=128,
        max_chunks=3,
        chunk_cache_size=0,
        chunk_selection_strategy="head_middle_tail",
    )
    batch = collator([item])
    print("\n[6/6] Collator/model input structures")
    print(json.dumps(_tensor_shapes(batch), indent=2))
    _require(batch["negative_counts"].tolist() == [15], "expected 15 negatives")

    model = ModernBERTContrastive(
        str(base_encoder),
        device=device,
        temperature=0.55555,
        max_chunks=3,
        chunk_microbatch_size=1,
        temperature_min=0.05,
        temperature_max=2.0,
        evidence_loss_weight=0.2,
        evidence_pool_temperature=0.1,
        fusion_residual_max_norm=0.5,
        fusion_residual_penalty_weight=0.01,
    ).to(device)
    _require(model.ARCHITECTURE_VERSION == 6, "checkpoint contract is not v6")
    model.train()
    model_inputs = {
        key: ({name: tensor.to(device) for name, tensor in value.items()} if isinstance(value, dict) else value.to(device))
        for key, value in batch.items()
        if key in {"anchor_input", "positive_input", "negative_input", "labels", "negative_counts"}
    }
    autocast = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if device.type == "cuda" and os.getenv("TASK1_CHUNKAGG_V6_FP16", "1") == "1"
        else contextlib.nullcontext()
    )
    with autocast:
        outputs = model(**model_inputs)
    loss = outputs["loss"]
    _require(torch.isfinite(loss).item(), "model loss is not finite")
    if backward:
        loss.backward()
        gradient_groups = {
            "encoder": any(p.grad is not None for p in model.encoder.parameters()),
            "projector": any(p.grad is not None for p in model.projector.parameters()),
            "fusion": any(p.grad is not None for _, p in model.chunk_aggregation_named_parameters()),
            "temperature": model.log_temperature.grad is not None,
        }
        _require(all(gradient_groups.values()), f"missing gradient group: {gradient_groups}")
    else:
        gradient_groups = {"backward_skipped": True}
    print(json.dumps({
        "architecture_version": model.ARCHITECTURE_VERSION,
        "logits_shape": list(outputs["logits"].shape),
        "loss": float(loss.detach().cpu()),
        "temperature": float(model.temperature.detach().cpu()),
        "gradients": gradient_groups,
    }, indent=2))
    _require(tuple(outputs["logits"].shape) == (1, 16), "expected 16-way logits")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and print every chunkAgg v6 data/model stage")
    parser.add_argument("--skip-model", action="store_true", help="run only static data and sampling checks")
    parser.add_argument("--skip-backward", action="store_true", help="run model forward without backward")
    args = parser.parse_args()

    stats, scope, eligible_ids, processed_dir = _static_preflight()
    rows = _sampling_preflight(scope, eligible_ids, processed_dir)
    dataset, item = _dataset_preflight(rows, scope, eligible_ids, processed_dir)
    if not args.skip_model:
        _model_preflight(dataset, item, backward=not args.skip_backward)
    print("\n✅ chunkAgg v6 preflight passed")


if __name__ == "__main__":
    main()
