from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch


CHUNKAGG_ROOT = Path(__file__).resolve().parent
FINE_TUNE_ROOT = CHUNKAGG_ROOT / "fine_tune"
LEGAL_RETRIEVAL_ROOT = CHUNKAGG_ROOT.parent
for import_path in (FINE_TUNE_ROOT, CHUNKAGG_ROOT, LEGAL_RETRIEVAL_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import fine_tune as training
from lcr.data import load_query_candidate_scope, normalize_case_id
from modernbert_contrastive_model import ModernBERTContrastive
from transformers import AutoTokenizer


def _stage(number: int, title: str) -> None:
    print(f"\n{'=' * 18} Stage {number}: {title} {'=' * 18}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _normalized_ids(values: Iterable[str]) -> list[str]:
    return [normalize_case_id(value) for value in values]


def _resolve_paths() -> Dict[str, Path]:
    task_dir = Path(training.TASK1_DIR)
    year = training.TASK1_YEAR
    paths = {
        "base_encoder": training.get_env_path(
            "TASK1_CHUNKAGG_BASE_ENCODER_DIR",
            required=True,
        ),
        "candidate_dir": task_dir / "processed",
        "train_qids": task_dir / "train_qid.tsv",
        "valid_qids": task_dir / "valid_qid.tsv",
        "train_labels": task_dir / f"task1_train_labels_{year}_train.json",
        "all_labels": task_dir / f"task1_train_labels_{year}.json",
        "scope": training.get_env_path("TASK1_CHUNKAGG_SCOPE_PATH", required=True),
        "valid_contrastive": training.get_env_path(
            "TASK1_CHUNKAGG_VALID_JSON_PATH",
            required=True,
        ),
        "output": training.get_env_path("TASK1_CHUNKAGG_OUTPUT_DIR", required=True),
        "finetune_data": training.get_env_path(
            "TASK1_CHUNKAGG_FINETUNE_DATA_DIR",
            required=True,
        ),
    }
    for name, path in paths.items():
        _require(path is not None, f"unresolved path: {name}")
        paths[name] = Path(path).resolve()
    return paths


def _check_paths(paths: Dict[str, Path]) -> None:
    _stage(1, "resolved configuration and files")
    required_inputs = (
        "base_encoder",
        "candidate_dir",
        "train_qids",
        "valid_qids",
        "train_labels",
        "all_labels",
        "scope",
        "valid_contrastive",
    )
    for name in required_inputs:
        path = paths[name]
        _require(path.exists(), f"required input does not exist: {path}")
        print(f"{name:19s}: {path}")

    output_path = paths["output"]
    data_path = paths["finetune_data"]
    checkpoints = sorted(output_path.glob("checkpoint-*")) if output_path.exists() else []
    event_files = (
        sorted((output_path / "tb").glob("**/events.out.tfevents.*"))
        if output_path.exists()
        else []
    )
    adaptive_files = (
        sorted(data_path.glob("adaptive_negative_epoch*_train.json"))
        if data_path.exists()
        else []
    )
    print(f"output              : {output_path}")
    print(f"finetune_data       : {data_path}")
    print(
        "existing artifacts  : "
        f"checkpoints={len(checkpoints)}, tensorboard_events={len(event_files)}, "
        f"adaptive_json={len(adaptive_files)}"
    )
    print(
        "training contract    : "
        f"chunks={training.MAX_DOCUMENT_CHUNKS}, "
        f"chunk_length={training.DOCUMENT_CHUNK_LENGTH}, "
        f"selection={training.CHUNK_SELECTION_STRATEGY}, "
        f"max_negatives={training.MAX_NEGATIVES}, "
        f"fp16={training.ENABLE_FP16}, bf16={training.ENABLE_BF16}"
    )


def _audit_labels_and_scope(paths: Dict[str, Path]):
    _stage(2, "labels, scope, and global metric denominator")
    candidate_ids = {path.stem for path in paths["candidate_dir"].glob("*.txt")}
    eligible_candidate_ids, rejected_content_lengths = training.scan_eligible_document_ids(
        paths["candidate_dir"],
        minimum_characters=training.MIN_DOCUMENT_CONTENT_CHARS,
    )
    train_qids = _normalized_ids(training.load_query_ids_from_utils(paths["train_qids"]))
    valid_qids = _normalized_ids(training.load_query_ids_from_utils(paths["valid_qids"]))
    train_positives = training.read_positive_pairs_from_json(paths["train_labels"])
    all_positives = training.read_positive_pairs_from_json(paths["all_labels"])
    scope = load_query_candidate_scope(paths["scope"])

    _require(len(train_qids) == len(set(train_qids)), "duplicate train qids")
    _require(len(valid_qids) == len(set(valid_qids)), "duplicate valid qids")
    _require(not (set(train_qids) & set(valid_qids)), "train/valid qids overlap")

    missing_query_files = [qid for qid in [*train_qids, *valid_qids] if qid not in candidate_ids]
    missing_scope = [qid for qid in [*train_qids, *valid_qids] if qid not in scope]
    scope_candidate_ids = {
        normalize_case_id(doc_id)
        for qid in [*train_qids, *valid_qids]
        for doc_id in scope.get(qid, [])
    }
    missing_scoped_files = sorted(scope_candidate_ids - candidate_ids)

    train_positive_pairs = [
        (qid, positive_id)
        for qid in train_qids
        for positive_id in train_positives.get(qid, set())
    ]
    out_of_scope_train_positives = [
        (qid, positive_id)
        for qid, positive_id in train_positive_pairs
        if positive_id not in set(scope.get(qid, []))
    ]
    missing_positive_files = [
        (qid, positive_id)
        for qid, positive_id in train_positive_pairs
        if positive_id not in candidate_ids
    ]
    content_excluded_train_qids = [
        qid for qid in train_qids if qid not in eligible_candidate_ids
    ]
    content_excluded_valid_qids = [
        qid for qid in valid_qids if qid not in eligible_candidate_ids
    ]
    content_excluded_positive_pairs = [
        (qid, positive_id)
        for qid, positive_id in train_positive_pairs
        if positive_id not in eligible_candidate_ids
    ]
    empty_negative_pools = []
    trainable_positive_pairs = 0
    in_scope_positives_without_negatives = 0
    scoped_pool_sizes = []
    for qid in train_qids:
        if qid not in eligible_candidate_ids:
            continue
        allowed = {normalize_case_id(doc_id) for doc_id in scope.get(qid, [])}
        in_scope_positives = (
            train_positives.get(qid, set()) & allowed & eligible_candidate_ids
        )
        negatives = allowed - train_positives.get(qid, set()) - {qid}
        negatives &= eligible_candidate_ids
        scoped_pool_sizes.append(len(negatives))
        if not negatives:
            empty_negative_pools.append(qid)
            in_scope_positives_without_negatives += len(in_scope_positives)
        else:
            trainable_positive_pairs += len(in_scope_positives)

    train_rel = training.rel_file_convert(paths["all_labels"], paths["train_qids"])
    valid_rel = training.rel_file_convert(paths["all_labels"], paths["valid_qids"])
    print(
        f"candidate files      : {len(candidate_ids)}\n"
        f"content-eligible     : {len(eligible_candidate_ids)} "
        f"(minimum={training.MIN_DOCUMENT_CONTENT_CHARS} non-whitespace chars)\n"
        f"content-rejected IDs : "
        f"{training.summarize_rejected_documents(rejected_content_lengths) or 'none'}\n"
        f"scope query entries  : {len(scope)}\n"
        f"train qids/pairs     : {len(train_qids)} / {len(train_positive_pairs)}\n"
        f"scope-trainable pairs: {trainable_positive_pairs}\n"
        f"valid qids           : {len(valid_qids)}\n"
        f"global denominators  : train={len(train_rel)}, valid={len(valid_rel)}\n"
        f"negative pool size   : min={min(scoped_pool_sizes)}, "
        f"median={float(np.median(scoped_pool_sizes)):.1f}, max={max(scoped_pool_sizes)}"
    )
    print(
        "alignment errors     : "
        f"missing_query_files={len(missing_query_files)}, "
        f"missing_scope={len(missing_scope)}, "
        f"missing_scoped_files={len(missing_scoped_files)}, "
        f"train_positive_excluded_by_scope={len(out_of_scope_train_positives)}, "
        f"missing_positive_files={len(missing_positive_files)}, "
        f"content_excluded_train_qids={len(content_excluded_train_qids)}, "
        f"content_excluded_valid_qids={len(content_excluded_valid_qids)}, "
        f"content_excluded_train_positives={len(content_excluded_positive_pairs)}, "
        f"empty_negative_pools={len(empty_negative_pools)}, "
        f"in_scope_positives_without_negatives={in_scope_positives_without_negatives}"
    )

    _require(not missing_query_files, f"queries without text files: {missing_query_files[:5]}")
    _require(not missing_scope, f"queries without scope: {missing_scope[:5]}")
    _require(not missing_scoped_files, f"scoped candidates without files: {missing_scoped_files[:5]}")
    _require(
        not missing_positive_files,
        f"train positives without files: {missing_positive_files[:5]}",
    )
    _require(trainable_positive_pairs > 0, "scope produced zero trainable positive pairs")
    _require(
        in_scope_positives_without_negatives == 0,
        "scope contains positives that cannot form a contrastive row because the "
        f"negative pool is empty: {empty_negative_pools[:5]}",
    )
    _require(len(train_rel) == len(train_qids), "train metric denominator lost qids")
    _require(len(valid_rel) == len(valid_qids), "valid metric denominator lost qids")
    for qid in [*train_qids, *valid_qids]:
        _require(qid in all_positives, f"qid has no global label entry: {qid}")

    trainable_qids = [qid for qid in train_qids if qid in eligible_candidate_ids]
    return (
        eligible_candidate_ids,
        trainable_qids,
        valid_qids,
        train_positives,
        scope,
    )


def _audit_valid_dataset(
    paths: Dict[str, Path],
    scope: Dict[str, Sequence[str]],
    valid_qids: Sequence[str],
    eligible_document_ids: set[str],
) -> None:
    _stage(3, "validation contrastive rows")
    raw_rows = json.loads(paths["valid_contrastive"].read_text(encoding="utf-8"))
    dataset = training.ContrastiveDataset(
        data=raw_rows,
        doc_folder=str(paths["candidate_dir"]),
        query_candidate_scope=scope,
        eligible_document_ids=eligible_document_ids,
        minimum_content_characters=training.MIN_DOCUMENT_CONTENT_CHARS,
    )
    valid_qid_set = set(valid_qids)
    row_query_ids = {normalize_case_id(row["query_id"]) for row in dataset.data}
    for row in dataset.data:
        qid = normalize_case_id(row["query_id"])
        positive_id = normalize_case_id(row["positive_id"])
        negatives = _normalized_ids(row["negative_ids"])
        allowed = {normalize_case_id(doc_id) for doc_id in scope[qid]}
        _require(qid in valid_qid_set, f"validation row uses non-valid qid: {qid}")
        _require(positive_id in allowed, f"validation positive outside scope: {qid}/{positive_id}")
        _require(len(negatives) == len(set(negatives)), "duplicate validation negatives")
        _require(set(negatives) <= allowed, f"validation negative outside scope: {qid}")
        _require(qid not in negatives and positive_id not in negatives, "validation false negative")

    print(
        f"raw rows             : {len(raw_rows)}\n"
        f"scope-valid rows     : {len(dataset)}\n"
        f"represented qids     : {len(row_query_ids)}/{len(valid_qids)}\n"
        f"missing contrastive qids (allowed; global F1 still counts them): "
        f"{len(valid_qid_set - row_query_ids)}"
    )
    _require(len(dataset) > 0, "validation contrastive dataset is empty")


def _build_scope_contract_samples(
    candidate_ids: set[str],
    train_qids: Sequence[str],
    positives: Dict[str, set[str]],
    scope: Dict[str, Sequence[str]],
):
    _stage(4, "model-only dynamic-negative contract")
    base_temperature = float(os.getenv("TASK1_CHUNKAGG_SAMPLING_TEMPERATURE", "0.15"))
    schedule = training._parse_sampling_temperature_schedule(
        os.getenv(
            "TASK1_CHUNKAGG_SAMPLING_TEMPERATURE_SCHEDULE",
            "0:0.20,3:0.15,5:0.10",
        )
    )
    epoch0_temperature = schedule[0][1]
    target_ess_schedule = training._parse_sampling_target_ess_schedule(
        os.getenv(
            "TASK1_CHUNKAGG_SAMPLING_TARGET_ESS_SCHEDULE",
            training.DEFAULT_SAMPLING_TARGET_ESS_SCHEDULE,
        )
    )
    target_ess = target_ess_schedule[0][1]
    max_negatives = training.MAX_NEGATIVES
    samples = []
    skipped_out_of_scope = 0
    skipped_empty_pool = 0
    for qid in train_qids:
        normalized_qid = normalize_case_id(qid)
        allowed = {
            normalize_case_id(doc_id)
            for doc_id in scope.get(normalized_qid, [])
        } & candidate_ids
        all_positives = {
            normalize_case_id(doc_id)
            for doc_id in positives.get(normalized_qid, set())
        }
        trainable_positives = sorted(all_positives & allowed)
        skipped_out_of_scope += len(all_positives - allowed)
        negative_pool = sorted(allowed - all_positives - {normalized_qid})
        if not negative_pool:
            skipped_empty_pool += len(trainable_positives)
            continue

        # These deterministic rows exist only to inspect Dataset/collator/model
        # contracts. Production rows are sampled from current-model scores.
        structural_negatives = negative_pool[:max_negatives]
        for positive_id in trainable_positives:
            samples.append(
                {
                    "query_id": normalized_qid,
                    "positive_id": positive_id,
                    "negative_ids": structural_negatives,
                }
            )

    _require(samples, "scope contract generated zero inspectable rows")

    negative_counts = []
    for row in samples:
        qid = normalize_case_id(row["query_id"])
        positive_id = normalize_case_id(row["positive_id"])
        negatives = _normalized_ids(row["negative_ids"])
        allowed = {normalize_case_id(doc_id) for doc_id in scope[qid]}
        all_positives = positives[qid]
        _require(positive_id in all_positives, f"unknown sampled positive: {qid}/{positive_id}")
        _require(positive_id in allowed, f"sampled positive outside scope: {qid}/{positive_id}")
        _require(len(negatives) == len(set(negatives)), f"duplicate negatives: {qid}")
        _require(set(negatives) <= allowed, f"negative outside scope: {qid}")
        _require(not (set(negatives) & all_positives), f"positive leaked into negatives: {qid}")
        _require(qid not in negatives, f"self leaked into negatives: {qid}")
        negative_counts.append(len(negatives))

    print(
        "production source    : current model at every epoch, including epoch 0\n"
        "epoch transition     : epoch N refresh runs before training epoch N, using "
        "the weights left by epoch N-1\n"
        "forbidden sources    : BM25, pre-generated negatives, random mixture, top-k cutoff\n"
        f"inspect-only rows     : {len(samples)} (never saved or used for training)\n"
        f"negative counts       : min={min(negative_counts)}, "
        f"median={float(np.median(negative_counts)):.1f}, max={max(negative_counts)}\n"
        f"sampling fallback T   : {base_temperature}\n"
        f"epoch-0 schedule T    : {epoch0_temperature}\n"
        f"target ESS curriculum : {target_ess_schedule}\n"
        f"epoch-0 target ESS    : {target_ess:.1f}\n"
        f"skipped positives     : outside_scope={skipped_out_of_scope}, "
        f"empty_negative_pool={skipped_empty_pool}"
    )
    return samples


def _choose_representative_sample(samples: Sequence[Dict], candidate_dir: Path) -> int:
    def representative_size(row: Dict) -> int:
        ids = [row["query_id"], row["positive_id"], *row["negative_ids"][:2]]
        sizes = [(candidate_dir / f"{normalize_case_id(doc_id)}.txt").stat().st_size for doc_id in ids]
        return max(sizes)

    candidates = []
    for index, row in enumerate(samples):
        size = representative_size(row)
        if 30_000 <= size <= 500_000:
            candidates.append((size, index))
    if candidates:
        target_size = 120_000
        return min(candidates, key=lambda item: abs(item[0] - target_size))[1]
    return max(range(len(samples)), key=lambda index: representative_size(samples[index]))


def _tensor_summary(name: str, tensor: torch.Tensor) -> str:
    return f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}"


def _inspect_document_batch(
    role: str,
    batch: Dict[str, torch.Tensor],
    tokenizer,
) -> None:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    chunk_mask = batch["chunk_mask"].bool()
    _require(input_ids.shape == attention_mask.shape, f"{role}: ids/mask shape mismatch")
    _require(input_ids.shape[:2] == chunk_mask.shape, f"{role}: chunk mask shape mismatch")
    _require(input_ids.shape[-1] == training.DOCUMENT_CHUNK_LENGTH, f"{role}: wrong length")

    active_lengths = attention_mask.sum(dim=-1)
    valid_chunks = 0
    for document_index in range(input_ids.size(0)):
        for chunk_index in range(input_ids.size(1)):
            is_valid = bool(chunk_mask[document_index, chunk_index])
            active_length = int(active_lengths[document_index, chunk_index])
            if is_valid:
                valid_chunks += 1
                _require(active_length >= 2, f"{role}: valid chunk lacks special tokens")
                active_ids = input_ids[document_index, chunk_index, :active_length]
                _require(
                    int(active_ids[0]) == int(tokenizer.cls_token_id),
                    f"{role}: chunk does not start with CLS",
                )
                _require(
                    int(active_ids[-1]) == int(tokenizer.sep_token_id),
                    f"{role}: chunk does not end with SEP",
                )
            else:
                _require(active_length == 0, f"{role}: padded chunk has active tokens")

    first_lengths = active_lengths[0].tolist() if input_ids.size(0) else []
    print(_tensor_summary(f"{role}.input_ids", input_ids))
    print(_tensor_summary(f"{role}.attention_mask", attention_mask))
    print(_tensor_summary(f"{role}.chunk_mask", batch["chunk_mask"]))
    print(
        f"{role}.contract: documents={input_ids.size(0)}, valid_chunks={valid_chunks}, "
        f"first_document_active_lengths={first_lengths}, CLS/SEP=verified"
    )
    if input_ids.size(0):
        for chunk_index in range(input_ids.size(1)):
            active_length = int(active_lengths[0, chunk_index])
            if active_length == 0:
                print(f"  {role}[0].chunk{chunk_index + 1}: PAD (chunk_mask=0)")
                continue
            active_ids = input_ids[0, chunk_index, :active_length]
            tail_tokens = tokenizer.convert_ids_to_tokens(active_ids[-8:].tolist())
            print(
                f"  {role}[0].chunk{chunk_index + 1}: active_tokens={active_length}, "
                f"first={tokenizer.convert_ids_to_tokens([int(active_ids[0])])[0]}, "
                f"last={tokenizer.convert_ids_to_tokens([int(active_ids[-1])])[0]}, "
                f"tail={tail_tokens}"
            )


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in batch.items()}


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float | None:
    squared_norm = 0.0
    found = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        found = True
        _require(torch.isfinite(parameter.grad).all().item(), "non-finite gradient")
        squared_norm += float(parameter.grad.float().square().sum().item())
    return math.sqrt(squared_norm) if found else None


def _inspect_real_pipeline(
    paths: Dict[str, Path],
    scope,
    samples: Sequence[Dict],
    eligible_document_ids: set[str],
) -> None:
    _stage(5, "Dataset -> collator -> 4096-token chunk tensors")
    dataset = training.ContrastiveDataset(
        data=list(samples),
        doc_folder=str(paths["candidate_dir"]),
        query_candidate_scope=scope,
        eligible_document_ids=eligible_document_ids,
        minimum_content_characters=training.MIN_DOCUMENT_CONTENT_CHARS,
    )
    sample_index = _choose_representative_sample(dataset.data, paths["candidate_dir"])
    record = dataset.data[sample_index]
    item = dataset[sample_index]
    print(
        "adaptive JSON record : "
        f"keys={list(record)}, query_id={record['query_id']}, "
        f"positive_id={record['positive_id']}, negatives={len(record['negative_ids'])}"
    )
    print(
        "Dataset.__getitem__  : "
        f"keys={list(item)}, query_chars={len(item['query_text'])}, "
        f"positive_chars={len(item['positive_text'])}, "
        f"negative_texts={len(item['negative_texts'])}, "
        f"all_positive_ids={item['all_positive_ids']}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        paths["base_encoder"],
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.model_max_length = 1_000_000_000
    collator = training.ContrastiveCollator(
        tokenizer=tokenizer,
        max_length=training.DOCUMENT_CHUNK_LENGTH,
        max_chunks=training.MAX_DOCUMENT_CHUNKS,
        chunk_cache_size=0,
        chunk_selection_strategy=training.CHUNK_SELECTION_STRATEGY,
    )
    collated = collator([item])
    print(
        "collator top-level   : "
        f"keys={list(collated)}, labels={collated['labels'].tolist()}, "
        f"negative_counts={collated['negative_counts'].tolist()}"
    )
    for role in ("anchor_input", "positive_input", "negative_input"):
        _inspect_document_batch(role, collated[role], tokenizer)

    _stage(6, "real v5 encoder and bounded fusion outputs")
    device = training.get_device()
    model = ModernBERTContrastive(
        str(paths["base_encoder"]),
        device=device,
        temperature=training.INIT_TEMPERATURE,
        max_chunks=training.MAX_DOCUMENT_CHUNKS,
        chunk_microbatch_size=training.CHUNK_MICROBATCH_SIZE,
        temperature_min=training.TEMPERATURE_MIN,
        temperature_max=training.TEMPERATURE_MAX,
        evidence_loss_weight=training.EVIDENCE_LOSS_WEIGHT,
        evidence_pool_temperature=training.EVIDENCE_POOL_TEMPERATURE,
        fusion_residual_max_norm=training.FUSION_RESIDUAL_MAX_NORM,
        fusion_residual_penalty_weight=training.FUSION_RESIDUAL_PENALTY_WEIGHT,
    ).to(device)
    _require(model.ARCHITECTURE_VERSION == 5, "model is not architecture v5")
    full_anchor = _to_device(collated["anchor_input"], device)
    autocast_context = (
        torch.amp.autocast("cuda", dtype=training.RETRIEVAL_AUTOCAST_DTYPE)
        if device.type == "cuda" and training.RETRIEVAL_AUTOCAST_DTYPE is not None
        else contextlib.nullcontext()
    )
    model.eval()
    with torch.no_grad(), autocast_context:
        encoded = model._encode_document_outputs(full_anchor, return_attention=True)

    for key in ("fused", "mean", "first", "chunks", "chunk_mask"):
        print(_tensor_summary(f"model.{key}", encoded[key]))
    for key in (
        "fused",
        "mean",
        "first",
        "chunks",
        "chunk_mean_pre_norm",
        "fusion_raw_residual_norm",
        "fusion_residual_norm",
        "fusion_residual_cap_utilization",
        "base_fused_cosine",
        "fusion_attention_entropy",
    ):
        _require(torch.isfinite(encoded[key]).all().item(), f"non-finite model output: {key}")
    for variant in ("fused", "mean", "first"):
        norms = encoded[variant].float().norm(dim=-1)
        _require(torch.allclose(norms, torch.ones_like(norms), atol=2e-4), f"{variant} is not normalized")
        print(f"model.{variant}.norm: {norms.tolist()}")
    print(
        f"fusion residual norm : {encoded['fusion_residual_norm'].tolist()}\n"
        f"fusion residual cap  : {float(model.fusion_residual_max_norm.item()):.4f}\n"
        f"fused-mean max diff  : "
        f"{float((encoded['fused'] - encoded['mean']).abs().max().item()):.8f}\n"
        f"attention entropy    : {encoded['fusion_attention_entropy'].tolist()}"
    )
    _require(
        float(encoded["fusion_residual_norm"].max().item()) == 0.0,
        "zero-initialized fusion residual is not zero",
    )

    _stage(7, "initial-model scores -> probabilities -> sampled negatives")
    query_id = normalize_case_id(item["query_id"])
    forbidden_ids = {
        normalize_case_id(doc_id)
        for doc_id in item["all_positive_ids"]
    } | {query_id}
    probe_negative_ids = sorted(
        (
            {
                normalize_case_id(doc_id)
                for doc_id in scope.get(query_id, [])
            }
            & eligible_document_ids
        )
        - forbidden_ids
    )[: max(training.MAX_NEGATIVES + 1, 24)]
    _require(
        len(probe_negative_ids) > training.MAX_NEGATIVES,
        "model-score probe needs a negative pool larger than MAX_NEGATIVES",
    )
    probe_item = dict(item)
    probe_item["negative_ids"] = probe_negative_ids
    probe_item["negative_texts"] = [
        dataset.load_text(doc_id) for doc_id in probe_negative_ids
    ]
    probe_collator = training.ContrastiveCollator(
        tokenizer=tokenizer,
        max_length=128,
        max_chunks=training.MAX_DOCUMENT_CHUNKS,
        chunk_cache_size=0,
        chunk_selection_strategy=training.CHUNK_SELECTION_STRATEGY,
    )
    probe_batch = probe_collator([probe_item])
    probe_anchor = _to_device(probe_batch["anchor_input"], device)
    probe_negatives = _to_device(probe_batch["negative_input"], device)
    probe_autocast = (
        torch.amp.autocast("cuda", dtype=training.RETRIEVAL_AUTOCAST_DTYPE)
        if device.type == "cuda" and training.RETRIEVAL_AUTOCAST_DTYPE is not None
        else contextlib.nullcontext()
    )
    model.eval()
    with torch.no_grad(), probe_autocast:
        probe_query_vector = model._encode_document_outputs(probe_anchor)["fused"]
        probe_negative_vectors = model._encode_document_outputs(probe_negatives)["fused"]
        probe_scores = torch.matmul(
            probe_negative_vectors.float(),
            probe_query_vector[0].float(),
        )
    _require(torch.isfinite(probe_scores).all().item(), "non-finite initial-model scores")

    schedule = training._parse_sampling_temperature_schedule(
        os.getenv(
            "TASK1_CHUNKAGG_SAMPLING_TEMPERATURE_SCHEDULE",
            "0:0.20,3:0.15,5:0.10",
        )
    )
    fallback_temperature = schedule[0][1]
    target_ess_schedule = training._parse_sampling_target_ess_schedule(
        os.getenv(
            "TASK1_CHUNKAGG_SAMPLING_TARGET_ESS_SCHEDULE",
            training.DEFAULT_SAMPLING_TARGET_ESS_SCHEDULE,
        )
    )
    target_ess = target_ess_schedule[0][1]
    score_array = probe_scores.detach().cpu().numpy().astype(np.float64)
    probabilities, sampling_diagnostics = training._build_sampling_distribution(
        score_array,
        temperature=fallback_temperature,
        target_ess=target_ess,
    )
    sampled_ids = training._sample_negative_ids(
        probe_negative_ids,
        score_array,
        max_negatives=training.MAX_NEGATIVES,
        temperature=fallback_temperature,
        target_ess=target_ess,
        rng=np.random.default_rng(289),
    )
    _require(np.isclose(probabilities.sum(), 1.0), "sampling probabilities do not sum to 1")
    _require(len(sampled_ids) == training.MAX_NEGATIVES, "wrong sampled negative count")
    _require(len(sampled_ids) == len(set(sampled_ids)), "model-sampled negatives repeat")
    _require(not (set(sampled_ids) & forbidden_ids), "positive/self leaked into model samples")
    _require(
        set(sampled_ids) <= set(probe_negative_ids),
        "model sample escaped the scoped probe pool",
    )
    print(
        f"query/pool/sampled    : {query_id} / {len(probe_negative_ids)} / "
        f"{len(sampled_ids)}\n"
        f"score tensor          : shape={tuple(probe_scores.shape)}, "
        f"min={float(probe_scores.min().item()):.6f}, "
        f"max={float(probe_scores.max().item()):.6f}\n"
        f"probability sum       : {float(probabilities.sum()):.8f}\n"
        f"fallback/calibrated T : {fallback_temperature:.6g} / "
        f"{sampling_diagnostics['temperature']:.6g}\n"
        f"target/actual ESS     : {target_ess:.1f} / "
        f"{sampling_diagnostics['effective_sample_size']:.2f}\n"
        f"sampled ids           : {sampled_ids}\n"
        "scope/positive/self  : verified"
    )
    del probe_batch, probe_anchor, probe_negatives, probe_query_vector
    del probe_negative_vectors, probe_scores

    _stage(8, "real InfoNCE forward and backward")
    short_item = dict(item)
    short_item["negative_ids"] = item["negative_ids"][:1]
    short_item["negative_texts"] = item["negative_texts"][:1]
    short_collator = training.ContrastiveCollator(
        tokenizer=tokenizer,
        max_length=128,
        max_chunks=training.MAX_DOCUMENT_CHUNKS,
        chunk_cache_size=0,
        chunk_selection_strategy=training.CHUNK_SELECTION_STRATEGY,
    )
    short_batch = short_collator([short_item])
    model_inputs = {
        key: _to_device(short_batch[key], device)
        for key in ("anchor_input", "positive_input", "negative_input")
    }
    model_inputs["labels"] = short_batch["labels"].to(device)
    model_inputs["negative_counts"] = short_batch["negative_counts"].to(device)
    model.zero_grad(set_to_none=True)
    model.train()
    output = model(**model_inputs)
    loss = output["loss"]
    _require(torch.isfinite(loss).item(), "non-finite real-model loss")
    loss.backward()

    gradient_norms = {
        "encoder": _gradient_norm(model.encoder.parameters()),
        "projector": _gradient_norm(model.projector.parameters()),
        "fusion_output": _gradient_norm(model.fusion_output.parameters()),
        "chunk_fusion": _gradient_norm(model.chunk_fusion.parameters()),
        "temperature": _gradient_norm([model.log_temperature]),
    }
    for group_name in ("encoder", "projector", "fusion_output", "temperature"):
        norm = gradient_norms[group_name]
        _require(norm is not None and norm > 0, f"missing/zero gradient: {group_name}")
    diagnostics = output["_diagnostics"]
    print(
        f"forward input shapes : anchor={tuple(model_inputs['anchor_input']['input_ids'].shape)}, "
        f"positive={tuple(model_inputs['positive_input']['input_ids'].shape)}, "
        f"negative={tuple(model_inputs['negative_input']['input_ids'].shape)}\n"
        f"logits/labels        : shape={tuple(output['logits'].shape)} / "
        f"{model_inputs['labels'].tolist()}\n"
        f"total loss           : {float(loss.detach().cpu().item()):.6f}\n"
        f"document/evidence    : "
        f"{float(diagnostics['loss_document'].cpu().item()):.6f} / "
        f"{float(diagnostics['loss_evidence'].cpu().item()):.6f}\n"
        f"temperature          : {float(model.temperature.detach().cpu().item()):.6f}\n"
        f"gradient norms       : {gradient_norms}"
    )
    model.zero_grad(set_to_none=True)
    del output, model_inputs, encoded, full_anchor, model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    print("chunkAgg v5 read-only preflight: no checkpoint, TensorBoard event, or training JSON is written")
    paths = _resolve_paths()
    _check_paths(paths)
    candidate_ids, train_qids, valid_qids, positives, scope = _audit_labels_and_scope(paths)
    _audit_valid_dataset(paths, scope, valid_qids, candidate_ids)
    samples = _build_scope_contract_samples(candidate_ids, train_qids, positives, scope)
    _inspect_real_pipeline(paths, scope, samples, candidate_ids)
    _stage(9, "result")
    print(
        "PASS: content filtering, scope, model-only dynamic negatives, chunk tensors, "
        "v5 bounded fusion, InfoNCE, and gradients are consistent."
    )
    print("The normal fine_tune.py entry point can start a fresh chunkAgg v5 training run.")


if __name__ == "__main__":
    main()
