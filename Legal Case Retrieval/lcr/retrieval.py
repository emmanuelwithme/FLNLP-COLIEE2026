from __future__ import annotations

import contextlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Dict, List, Mapping, Optional, Sequence, overload

import torch
from tqdm import tqdm

from .data import EmbeddingsData, resolve_query_candidate_scope
from .embeddings import generate_embeddings
from .similarity import rank_candidates_with_scores

MAX_DOCUMENT_CHUNKS = 3


@dataclass
class SimilarityArtifacts:
    scores: Dict[str, Dict[str, float]]
    trec_path: Path
    candidate_ids: List[str]
    query_ids: List[str]
    missing_queries: List[str]

    @property
    def candidate_count(self) -> int:
        return len(self.candidate_ids)

    @property
    def query_count(self) -> int:
        return len(self.query_ids)


def _resolve_pad_token_id(tokenizer) -> int:
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        return int(tokenizer.eos_token_id)
    if tokenizer.sep_token_id is not None:
        return int(tokenizer.sep_token_id)
    return 0


class _TextFileSequence(Sequence[str]):
    """Read document text lazily so a full corpus is not held in RAM."""

    def __init__(self, paths: Sequence[Path]):
        self.paths = list(paths)

    def __len__(self) -> int:
        return len(self.paths)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> List[str]: ...

    def __getitem__(self, index: int | slice) -> str | List[str]:
        if isinstance(index, slice):
            return [path.read_text(encoding="utf-8").strip() for path in self.paths[index]]
        return self.paths[index].read_text(encoding="utf-8").strip()


def _tokenize_with_offsets(text: str, tokenizer) -> tuple[List[int], List[tuple[int, int]]]:
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_offsets_mapping=True,
    )
    input_ids = list(tokenized["input_ids"])
    offsets = [tuple(item) for item in tokenized["offset_mapping"]]
    if len(input_ids) != len(offsets):
        raise ValueError("input_ids and offset_mapping length mismatch")
    return input_ids, offsets


def _add_single_sequence_special_tokens(
    token_ids: Sequence[int],
    tokenizer,
    *,
    expected_special_tokens: int,
) -> List[int]:
    """Add the tokenizer's single-sequence template without losing fast-tokenizer specials."""
    raw_ids = [int(token_id) for token_id in token_ids]
    built_ids = [
        int(token_id)
        for token_id in tokenizer.build_inputs_with_special_tokens(raw_ids)
    ]
    if len(built_ids) == len(raw_ids) + expected_special_tokens:
        return built_ids

    # Some PreTrainedTokenizerFast implementations keep the template only in
    # the Rust backend, so build_inputs_with_special_tokens() is a no-op.
    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    if expected_special_tokens == 2 and cls_token_id is not None and sep_token_id is not None:
        return [int(cls_token_id), *raw_ids, int(sep_token_id)]

    raise ValueError(
        "Tokenizer did not add the expected single-sequence special tokens: "
        f"raw={len(raw_ids)}, built={len(built_ids)}, expected_special={expected_special_tokens}"
    )


def _looks_like_sentence_boundary(text: str, char_end: int, next_char_start: int) -> bool:
    left_text = text[:char_end].rstrip()
    if not left_text:
        return False
    right_text = text[next_char_start:]
    next_nonspace_match = re.search(r"\S", right_text)
    next_nonspace_char = next_nonspace_match.group(0) if next_nonspace_match else ""

    if re.search(r'[.!?;][\'")\]]*$', left_text) is None:
        gap_text = text[char_end:next_char_start]
        return "\n\n" in gap_text

    boundary_token = re.search(r'([A-Za-z](?:\.[A-Za-z])+\.|[A-Za-z]+\.?)$', left_text)
    if boundary_token:
        token_text = boundary_token.group(1)
        lowered = token_text.lower()
        common_abbrev = {
            "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "no.", "art.", "sec.",
            "para.", "paras.", "fig.", "eq.", "e.g.", "i.e.", "etc.", "v.", "vs.", "inc.",
            "ltd.", "co.", "corp.", "u.s.", "u.k.", "ca.", "nos.",
        }
        if lowered in common_abbrev:
            return False
        if re.fullmatch(r'[A-Z]\.', token_text) and next_nonspace_char.isupper():
            return False
        if re.fullmatch(r'(?:[A-Za-z]\.){2,}', token_text):
            return False

    return True


def _chunk_single_text(
    text: str,
    tokenizer,
    *,
    max_length: int = 4096,
    max_chunks: int = MAX_DOCUMENT_CHUNKS,
    selection_strategy: str = "head",
) -> Dict[str, torch.Tensor]:
    """
    將單一文件先按句界切成 chunks，再選出最多 `max_chunks` 個。

    `head_middle_tail` 會在超長文件保留首、中、尾的覆蓋；`head`
    保留舊行為，只取前 `max_chunks` 個。
    """
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be > 0, got {max_chunks}")
    if selection_strategy not in {"head", "head_middle_tail"}:
        raise ValueError(
            "selection_strategy must be 'head' or 'head_middle_tail', got "
            f"{selection_strategy!r}"
        )

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if max_length <= special_tokens:
        raise ValueError(
            f"max_length must be larger than num_special_tokens_to_add ({special_tokens}), got {max_length}"
        )

    chunk_token_budget = max_length - special_tokens
    all_raw_chunks: List[List[int]] = []
    token_ids, offsets = _tokenize_with_offsets(text, tokenizer)

    cursor = 0
    total_tokens = len(token_ids)
    while cursor < total_tokens:
        tentative_end = min(cursor + chunk_token_budget, total_tokens)
        chosen_end = tentative_end

        if tentative_end < total_tokens:
            # 中文註解：優先往回找最近的句尾；找不到時才在 token budget 處硬切。
            for candidate_end in range(tentative_end, cursor, -1):
                char_end = offsets[candidate_end - 1][1]
                next_char_start = offsets[candidate_end][0] if candidate_end < total_tokens else len(text)
                if _looks_like_sentence_boundary(text, char_end, next_char_start):
                    chosen_end = candidate_end
                    break

        if chosen_end <= cursor:
            chosen_end = tentative_end
        all_raw_chunks.append(token_ids[cursor:chosen_end])
        cursor = chosen_end

    if len(all_raw_chunks) <= max_chunks or selection_strategy == "head":
        raw_chunks = all_raw_chunks[:max_chunks]
    elif max_chunks == 1:
        raw_chunks = [all_raw_chunks[0]]
    else:
        # Evenly cover the complete document. For three slots this is exactly
        # first/middle/last, while preserving deterministic sentence chunks.
        selected_indices = [
            round(slot * (len(all_raw_chunks) - 1) / (max_chunks - 1))
            for slot in range(max_chunks)
        ]
        raw_chunks = [all_raw_chunks[index] for index in selected_indices]

    pad_token_id = _resolve_pad_token_id(tokenizer)
    chunk_input_ids: List[torch.Tensor] = []
    chunk_attention_masks: List[torch.Tensor] = []
    chunk_mask_values: List[int] = []

    for raw_chunk in raw_chunks:
        chunk_ids = _add_single_sequence_special_tokens(
            raw_chunk,
            tokenizer,
            expected_special_tokens=special_tokens,
        )
        if len(chunk_ids) == 0:
            continue
        if len(chunk_ids) > max_length:
            raise ValueError(
                "Chunk plus special tokens exceeds max_length; refusing to truncate "
                f"a boundary token: chunk={len(raw_chunk)}, encoded={len(chunk_ids)}, "
                f"max_length={max_length}"
            )
        chunk_attention = [1] * len(chunk_ids)

        pad_len = max_length - len(chunk_ids)
        chunk_ids = chunk_ids + [pad_token_id] * pad_len
        chunk_attention = chunk_attention + [0] * pad_len

        chunk_input_ids.append(torch.tensor(chunk_ids, dtype=torch.long))
        chunk_attention_masks.append(torch.tensor(chunk_attention, dtype=torch.long))
        chunk_mask_values.append(1)

    while len(chunk_input_ids) < max_chunks:
        chunk_input_ids.append(torch.full((max_length,), pad_token_id, dtype=torch.long))
        chunk_attention_masks.append(torch.zeros(max_length, dtype=torch.long))
        chunk_mask_values.append(0)

    return {
        "input_ids": torch.stack(chunk_input_ids, dim=0),
        "attention_mask": torch.stack(chunk_attention_masks, dim=0),
        "chunk_mask": torch.tensor(chunk_mask_values, dtype=torch.long),
    }


def _build_document_batch(
    texts: Sequence[str],
    tokenizer,
    *,
    max_length: int = 4096,
    max_chunks: int = MAX_DOCUMENT_CHUNKS,
    chunk_selection_strategy: str = "head",
    device: torch.device | str | None = None,
) -> Dict[str, torch.Tensor]:
    """中文註解：將多篇文本整理成 model.encode 可直接吃的 [B, C, L] batch。"""
    if len(texts) == 0:
        batch = {
            "input_ids": torch.empty((0, max_chunks, max_length), dtype=torch.long),
            "attention_mask": torch.empty((0, max_chunks, max_length), dtype=torch.long),
            "chunk_mask": torch.empty((0, max_chunks), dtype=torch.long),
        }
    else:
        chunked_documents = [
            _chunk_single_text(
                text,
                tokenizer,
                max_length=max_length,
                max_chunks=max_chunks,
                selection_strategy=chunk_selection_strategy,
            )
            for text in texts
        ]
        batch = {
            key: torch.stack([doc[key] for doc in chunked_documents], dim=0)
            for key in ("input_ids", "attention_mask", "chunk_mask")
        }

    if device is not None:
        device_obj = torch.device(device)
        batch = {key: value.to(device_obj) for key, value in batch.items()}
    return batch


def _should_use_chunked_encoding(model) -> bool:
    return bool(
        getattr(model, "supports_chunked_documents", False)
        or hasattr(model, "encode_document")
    )


def _generate_document_embeddings(
    texts: Sequence[str],
    tokenizer,
    model,
    *,
    batch_size: int = 1,
    max_length: int = 4096,
    max_chunks: int = MAX_DOCUMENT_CHUNKS,
    chunk_selection_strategy: str = "head",
    device: torch.device | str | None = None,
    show_progress: bool = True,
    progress_desc: str | None = None,
    autocast_dtype: torch.dtype | None = torch.float16,
) -> torch.Tensor:
    """中文註解：使用與訓練一致的 chunk batch，直接呼叫 `model.encode(document_batch)`。"""
    if len(texts) == 0:
        return torch.empty((0, 0))

    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    device_obj = torch.device(device) if device is not None else torch.device("cpu")
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc=progress_desc or "document encoding",
            total=(len(texts) + batch_size - 1) // batch_size,
        )

    chunks: List[torch.Tensor] = []
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for start in iterator:
            batch_texts = texts[start : start + batch_size]
            document_batch = _build_document_batch(
                batch_texts,
                tokenizer,
                max_length=max_length,
                max_chunks=max_chunks,
                chunk_selection_strategy=chunk_selection_strategy,
                device=device_obj,
            )
            with torch.no_grad():
                autocast_ctx = (
                    torch.amp.autocast("cuda", dtype=autocast_dtype)
                    if device_obj.type == "cuda" and autocast_dtype is not None
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    batch_embeddings = model.encode(document_batch)

            if not isinstance(batch_embeddings, torch.Tensor):
                batch_embeddings = torch.as_tensor(batch_embeddings)
            chunks.append(batch_embeddings.detach().cpu())
    finally:
        if was_training:
            model.train()

    return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 0))


def _generate_document_embedding_variants(
    texts: Sequence[str],
    tokenizer,
    model,
    *,
    variants: Sequence[str],
    batch_size: int = 1,
    max_length: int = 4096,
    max_chunks: int = MAX_DOCUMENT_CHUNKS,
    chunk_selection_strategy: str = "head",
    device: torch.device | str | None = None,
    show_progress: bool = True,
    progress_desc: str | None = None,
    autocast_dtype: torch.dtype | None = torch.float16,
) -> Dict[str, torch.Tensor]:
    """Encode multiple aggregation variants while running the encoder once."""
    requested = tuple(dict.fromkeys(variants))
    if not requested:
        raise ValueError("at least one embedding variant is required")
    if not hasattr(model, "encode_variants"):
        raise TypeError("model does not implement encode_variants")
    if len(texts) == 0:
        return {name: torch.empty((0, 0)) for name in requested}

    device_obj = torch.device(device) if device is not None else torch.device("cpu")
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc=progress_desc or "document encoding",
            total=(len(texts) + batch_size - 1) // batch_size,
        )

    collected: Dict[str, List[torch.Tensor]] = {name: [] for name in requested}
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        for start in iterator:
            document_batch = _build_document_batch(
                texts[start : start + batch_size],
                tokenizer,
                max_length=max_length,
                max_chunks=max_chunks,
                chunk_selection_strategy=chunk_selection_strategy,
                device=device_obj,
            )
            with torch.no_grad():
                autocast_ctx = (
                    torch.amp.autocast("cuda", dtype=autocast_dtype)
                    if device_obj.type == "cuda" and autocast_dtype is not None
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    batch_variants = model.encode_variants(document_batch)
            missing = set(requested) - set(batch_variants)
            if missing:
                raise KeyError(f"model did not return embedding variants: {sorted(missing)}")
            for name in requested:
                collected[name].append(batch_variants[name].detach().cpu())
    finally:
        if was_training:
            model.train()

    return {
        name: torch.cat(parts, dim=0) if parts else torch.empty((0, 0))
        for name, parts in collected.items()
    }


def generate_similarity_artifacts(
    model,
    tokenizer,
    device: torch.device,
    *,
    candidate_dir: str | Path,
    query_dir: str | Path,
    query_ids: Sequence[str],
    trec_output_path: str | Path,
    run_tag: str,
    batch_size: int = 1,
    max_length: int = 4096,
    quick_test: bool = False,
    candidate_files_override: Optional[Sequence[str]] = None,
    candidate_limit: int = 20,
    query_limit: int = 5,
    verbose: bool = True,
    query_to_candidate_ids: Mapping[str, Sequence[str]] | None = None,
    query_candidate_scope_path: str | Path | None = None,
    fallback_to_all_candidates_if_scope_missing: bool = False,
    persist_trec: bool = True,
    retain_scores: bool = True,
    score_consumer: Callable[[str, Dict[str, float]], None] | None = None,
    exclude_self: bool = False,
    autocast_dtype: torch.dtype | None = torch.float16,
    chunk_selection_strategy: str = "head",
    reuse_query_embeddings: bool = False,
    embedding_variants: Sequence[str] | None = None,
    variant_score_consumer: Callable[[str, str, Dict[str, float]], None] | None = None,
    eligible_document_ids: Collection[str] | None = None,
) -> SimilarityArtifacts:
    """
    Produce embeddings for the provided candidates and queries, compute
    dot-product similarities, and persist TREC-formatted rankings.
    """
    candidate_dir = Path(candidate_dir)
    query_dir = Path(query_dir)
    trec_output_path = Path(trec_output_path)
    resolved_scope, scope_source = resolve_query_candidate_scope(
        query_to_candidate_ids=query_to_candidate_ids,
        query_candidate_scope_path=query_candidate_scope_path,
    )
    eligible_id_set = (
        {str(doc_id).split(".")[0] for doc_id in eligible_document_ids}
        if eligible_document_ids is not None
        else None
    )

    if candidate_files_override:
        candidate_files = [
            candidate_dir / fname for fname in candidate_files_override
            if (candidate_dir / fname).is_file()
        ]
    else:
        candidate_files = sorted(candidate_dir.glob("*.txt"))
        if quick_test and candidate_files:
            k = min(candidate_limit, len(candidate_files))
            candidate_files = random.sample(candidate_files, k)
    candidate_files = sorted(candidate_files)
    excluded_candidate_ids: List[str] = []
    if eligible_id_set is not None:
        excluded_candidate_ids = [
            path.stem for path in candidate_files if path.stem not in eligible_id_set
        ]
        candidate_files = [
            path for path in candidate_files if path.stem in eligible_id_set
        ]

    candidate_ids = [path.stem for path in candidate_files]
    candidate_texts = _TextFileSequence(candidate_files)

    incoming_qids = list(query_ids)
    if quick_test and len(incoming_qids) > query_limit:
        incoming_qids = random.sample(incoming_qids, query_limit)

    query_paths: List[Path] = []
    actual_query_ids: List[str] = []
    missing_files: List[str] = []
    excluded_query_ids: List[str] = []

    for qid in incoming_qids:
        qid_str = str(qid).split(".")[0]
        candidates = [query_dir / f"{qid_str}.txt", query_dir / f"{qid_str.zfill(6)}.txt"]
        actual_path = next((path for path in candidates if path.exists()), None)
        if actual_path and (
            eligible_id_set is None or actual_path.stem in eligible_id_set
        ):
            query_paths.append(actual_path)
            actual_query_ids.append(qid_str)
        elif actual_path:
            excluded_query_ids.append(qid_str)
            missing_files.append(qid_str)
        else:
            missing_files.append(qid_str)

    if verbose:
        print(f"🔹 Queries found: {len(actual_query_ids)}/{len(incoming_qids)} in {query_dir}")
        if excluded_candidate_ids:
            print(
                "🔹 Content filter excluded "
                f"{len(excluded_candidate_ids)} candidate documents: "
                f"{excluded_candidate_ids[:10]}"
            )
        if excluded_query_ids:
            print(
                "🔹 Content filter excluded "
                f"{len(excluded_query_ids)} query documents: {excluded_query_ids[:10]}"
            )
        if not actual_query_ids and incoming_qids:
            print(f"⚠️ None of the query files were found. Example missing IDs: {missing_files[:5]}")
        if resolved_scope is not None:
            source_text = scope_source or "provided mapping"
            print(f"🔹 Query-specific candidate scope enabled from: {source_text}")
            unscoped = [qid for qid in actual_query_ids if qid not in resolved_scope]
            if unscoped:
                fallback_text = (
                    "fallback to all candidates"
                    if fallback_to_all_candidates_if_scope_missing
                    else "empty candidate list"
                )
                print(
                    f"⚠️ Scope missing {len(unscoped)} queries ({fallback_text}). "
                    f"Example: {unscoped[:5]}"
                )

    use_chunked_encoding = _should_use_chunked_encoding(model)
    if verbose:
        model_max_chunks = int(getattr(model, "max_chunks", MAX_DOCUMENT_CHUNKS))
        encode_mode = (
            f"{model_max_chunks}-chunk document encoder"
            if use_chunked_encoding
            else "single-chunk fallback"
        )
        print(f"🔹 Retrieval encode mode: {encode_mode}")
        print("🔹 Generating candidate embeddings...")

    requested_variants = tuple(dict.fromkeys(embedding_variants or ()))
    if requested_variants and "fused" not in requested_variants:
        raise ValueError("embedding_variants must include 'fused' as the primary ranking")
    if requested_variants and not use_chunked_encoding:
        raise ValueError("embedding_variants require a chunk-aware model")

    if use_chunked_encoding:
        model_max_chunks = int(getattr(model, "max_chunks", MAX_DOCUMENT_CHUNKS))

        def encode_texts(texts: Sequence[str], *, desc: str) -> torch.Tensor:
            return _generate_document_embeddings(
                texts,
                tokenizer,
                model,
                batch_size=batch_size,
                max_length=max_length,
                max_chunks=model_max_chunks,
                chunk_selection_strategy=chunk_selection_strategy,
                device=device,
                show_progress=verbose,
                progress_desc=desc,
                autocast_dtype=autocast_dtype,
            )

        def encode_text_variants(
            texts: Sequence[str],
            *,
            desc: str,
        ) -> Dict[str, torch.Tensor]:
            return _generate_document_embedding_variants(
                texts,
                tokenizer,
                model,
                variants=requested_variants,
                batch_size=batch_size,
                max_length=max_length,
                max_chunks=model_max_chunks,
                chunk_selection_strategy=chunk_selection_strategy,
                device=device,
                show_progress=verbose,
                progress_desc=desc,
                autocast_dtype=autocast_dtype,
            )
    else:
        def encode_batch(inputs):
            if device.type == "cuda" and autocast_dtype is not None:
                with torch.amp.autocast("cuda", dtype=autocast_dtype):
                    return model.encode(inputs)
            return model.encode(inputs)

        def encode_texts(texts: Sequence[str], *, desc: str) -> torch.Tensor:
            return generate_embeddings(
                texts,
                tokenizer,
                encode_batch=encode_batch,
                batch_size=batch_size,
                max_length=max_length,
                device=device,
                show_progress=verbose,
                progress_desc=desc,
            )

    if requested_variants:
        candidate_variants = encode_text_variants(
            candidate_texts,
            desc="Candidate embeddings",
        )
    else:
        candidate_variants = {
            "fused": encode_texts(candidate_texts, desc="Candidate embeddings")
        }
    if verbose:
        print("🔹 Generating query embeddings...")

    same_corpus = candidate_dir.resolve() == query_dir.resolve()
    can_reuse_query_embeddings = reuse_query_embeddings and same_corpus
    candidate_index = {doc_id: idx for idx, doc_id in enumerate(candidate_ids)}
    reused_vectors_by_variant: Dict[str, List[torch.Tensor | None]] = {
        name: [None] * len(actual_query_ids) for name in candidate_variants
    }
    missing_query_positions: List[int] = []
    missing_query_paths: List[Path] = []
    if can_reuse_query_embeddings:
        for position, query_path in enumerate(query_paths):
            candidate_position = candidate_index.get(query_path.stem)
            if candidate_position is None:
                missing_query_positions.append(position)
                missing_query_paths.append(query_path)
            else:
                for name, embeddings in candidate_variants.items():
                    reused_vectors_by_variant[name][position] = embeddings[candidate_position]
    else:
        missing_query_positions = list(range(len(query_paths)))
        missing_query_paths = list(query_paths)

    if requested_variants:
        newly_encoded_variants = encode_text_variants(
            _TextFileSequence(missing_query_paths),
            desc="Query embeddings",
        )
    else:
        newly_encoded_variants = {
            "fused": encode_texts(
                _TextFileSequence(missing_query_paths),
                desc="Query embeddings",
            )
        }
    for name, newly_encoded in newly_encoded_variants.items():
        for encoded_position, query_position in enumerate(missing_query_positions):
            reused_vectors_by_variant[name][query_position] = newly_encoded[encoded_position]

    query_variants: Dict[str, torch.Tensor] = {}
    for name, reused_vectors in reused_vectors_by_variant.items():
        candidate_embeddings = candidate_variants[name]
        if reused_vectors:
            if any(vector is None for vector in reused_vectors):
                raise RuntimeError(
                    f"Failed to construct all query embeddings for variant {name}"
                )
            query_variants[name] = torch.stack(
                [vector for vector in reused_vectors if vector is not None]
            )
        else:
            embedding_dim = (
                candidate_embeddings.shape[1]
                if candidate_embeddings.ndim == 2
                else 0
            )
            query_variants[name] = candidate_embeddings.new_empty((0, embedding_dim))

    if verbose and can_reuse_query_embeddings:
        reused_count = len(actual_query_ids) - len(missing_query_positions)
        print(f"🔹 Reused {reused_count}/{len(actual_query_ids)} query embeddings from candidate corpus")

    lines: List[str] = []
    scores_dict: Dict[str, Dict[str, float]] = {}
    missing_from_scores: List[str] = []
    for variant_name, candidate_embeddings_cpu in candidate_variants.items():
        query_embeddings_cpu = query_variants[variant_name]
        candidate_embeddings = (
            candidate_embeddings_cpu.to(device)
            if device.type == "cuda"
            else candidate_embeddings_cpu
        )
        query_embeddings = (
            query_embeddings_cpu.to(device)
            if device.type == "cuda"
            else query_embeddings_cpu
        )

        def consume_variant(qid: str, scores: Dict[str, float]) -> None:
            if variant_name == "fused" and score_consumer is not None:
                score_consumer(qid, scores)
            if variant_score_consumer is not None:
                variant_score_consumer(variant_name, qid, scores)

        variant_lines, variant_scores, variant_missing = rank_candidates_with_scores(
            query_ids=actual_query_ids,
            query_embeddings=EmbeddingsData(actual_query_ids, query_embeddings),
            candidate_embeddings=EmbeddingsData(candidate_ids, candidate_embeddings),
            metric="dot",
            run_tag=(run_tag if variant_name == "fused" else f"{run_tag}_{variant_name}"),
            query_to_candidate_ids=resolved_scope,
            fallback_to_all_candidates_if_scope_missing=fallback_to_all_candidates_if_scope_missing,
            include_trec_lines=persist_trec and variant_name == "fused",
            retain_scores=retain_scores and variant_name == "fused",
            score_consumer=(
                consume_variant
                if score_consumer is not None or variant_score_consumer is not None
                else None
            ),
            exclude_self=exclude_self,
        )
        if variant_name == "fused":
            lines = variant_lines
            scores_dict = variant_scores
            missing_from_scores = variant_missing
        del candidate_embeddings, query_embeddings

    combined_missing = sorted(set(missing_files + missing_from_scores))
    if verbose and combined_missing:
        print(f"⚠️ Missing embeddings for {len(combined_missing)} queries: {combined_missing}")

    if persist_trec:
        trec_output_path.parent.mkdir(parents=True, exist_ok=True)
        trec_output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if verbose:
        if persist_trec:
            print(f"✅ Saved similarity scores to {trec_output_path}")
        elif retain_scores:
            print("✅ Similarity scores retained in memory")
        elif score_consumer is not None or variant_score_consumer is not None:
            print("✅ Similarity scores streamed to consumer (not retained or written)")
        else:
            print("✅ Similarity scoring completed (not retained or written)")
        if quick_test:
            print(
                f"[QUICK_TEST] Using {len(candidate_ids)} candidates and {len(actual_query_ids)} queries"
            )

    return SimilarityArtifacts(
        scores=scores_dict,
        trec_path=trec_output_path,
        candidate_ids=candidate_ids,
        query_ids=actual_query_ids,
        missing_queries=combined_missing,
    )
