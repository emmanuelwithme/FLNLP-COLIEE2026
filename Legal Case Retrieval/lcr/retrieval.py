from __future__ import annotations

import contextlib
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

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


def _split_text_into_sentence_like_units(text: str) -> List[str]:
    """
    中文註解：用輕量規則做 sentence-like split，優先保留句尾標點。
    若文件沒有明顯句號，則退回以換行段落切分。
    """
    normalized_text = (text or "").strip()
    if not normalized_text:
        return [""]

    sentence_matches = re.findall(r".*?(?:[.!?;]+(?:['\")\\]]+)?|\n{2,}|$)", normalized_text, flags=re.S)
    sentences = [segment.strip() for segment in sentence_matches if segment and segment.strip()]
    if sentences:
        return sentences

    paragraphs = [segment.strip() for segment in normalized_text.splitlines() if segment.strip()]
    return paragraphs or [normalized_text]


def _tokenize_without_special_tokens(text: str, tokenizer) -> List[int]:
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    return list(tokenized["input_ids"])


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
) -> Dict[str, torch.Tensor]:
    """
    將單一文件切成最多 `max_chunks` 個 chunk。
    每個 chunk 各自加入 tokenizer special tokens，超出部分直接截斷。
    """
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be > 0, got {max_chunks}")

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if max_length <= special_tokens:
        raise ValueError(
            f"max_length must be larger than num_special_tokens_to_add ({special_tokens}), got {max_length}"
        )

    chunk_token_budget = max_length - special_tokens
    raw_chunks: List[List[int]] = []
    token_ids, offsets = _tokenize_with_offsets(text, tokenizer)
    token_ids = token_ids[: max_chunks * chunk_token_budget]
    offsets = offsets[: len(token_ids)]

    cursor = 0
    total_tokens = len(token_ids)
    while cursor < total_tokens and len(raw_chunks) < max_chunks:
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
        raw_chunks.append(token_ids[cursor:chosen_end])
        cursor = chosen_end

    pad_token_id = _resolve_pad_token_id(tokenizer)
    chunk_input_ids: List[torch.Tensor] = []
    chunk_attention_masks: List[torch.Tensor] = []
    chunk_mask_values: List[int] = []

    for raw_chunk in raw_chunks:
        chunk_ids = tokenizer.build_inputs_with_special_tokens(raw_chunk)
        if len(chunk_ids) == 0:
            # 中文註解：ModernBERT tokenizer 對空字串不一定會自動補 special tokens；
            # 此時直接把該 chunk 視為無效，避免後續 flash attention 吃到全 0 mask。
            continue
        chunk_attention = [1] * len(chunk_ids)
        if len(chunk_ids) > max_length:
            chunk_ids = chunk_ids[:max_length]
            chunk_attention = chunk_attention[:max_length]

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
    device: torch.device | str | None = None,
    show_progress: bool = True,
    progress_desc: str | None = None,
) -> torch.Tensor:
    """中文註解：使用與訓練一致的 chunk batch，直接呼叫 `model.encode(document_batch)`。"""
    if len(texts) == 0:
        return torch.empty((0, 0))

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
                device=device_obj,
            )
            with torch.no_grad():
                autocast_ctx = (
                    torch.amp.autocast("cuda", dtype=torch.float16)
                    if device_obj.type == "cuda"
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

    candidate_ids = [path.stem for path in candidate_files]
    candidate_texts = [path.read_text(encoding="utf-8").strip() for path in candidate_files]

    incoming_qids = list(query_ids)
    if quick_test and len(incoming_qids) > query_limit:
        incoming_qids = random.sample(incoming_qids, query_limit)

    query_texts: List[str] = []
    actual_query_ids: List[str] = []
    missing_files: List[str] = []

    for qid in incoming_qids:
        qid_str = str(qid).split(".")[0]
        candidates = [query_dir / f"{qid_str}.txt", query_dir / f"{qid_str.zfill(6)}.txt"]
        actual_path = next((path for path in candidates if path.exists()), None)
        if actual_path:
            query_texts.append(actual_path.read_text(encoding="utf-8").strip())
            actual_query_ids.append(qid_str)
        else:
            missing_files.append(qid_str)

    if verbose:
        print(f"🔹 Queries found: {len(actual_query_ids)}/{len(incoming_qids)} in {query_dir}")
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
        encode_mode = "3-chunk document encoder" if use_chunked_encoding else "single-chunk fallback"
        print(f"🔹 Retrieval encode mode: {encode_mode}")
        print("🔹 Generating candidate embeddings...")

    if use_chunked_encoding:
        candidate_embeddings = _generate_document_embeddings(
            candidate_texts,
            tokenizer,
            model,
            batch_size=batch_size,
            max_length=max_length,
            max_chunks=MAX_DOCUMENT_CHUNKS,
            device=device,
            show_progress=verbose,
            progress_desc="Candidate embeddings",
        )
        if verbose:
            print("🔹 Generating query embeddings...")
        query_embeddings = _generate_document_embeddings(
            query_texts,
            tokenizer,
            model,
            batch_size=batch_size,
            max_length=max_length,
            max_chunks=MAX_DOCUMENT_CHUNKS,
            device=device,
            show_progress=verbose,
            progress_desc="Query embeddings",
        )
    else:
        def encode_batch(inputs):
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    return model.encode(inputs)
            return model.encode(inputs)

        candidate_embeddings = generate_embeddings(
            candidate_texts,
            tokenizer,
            encode_batch=encode_batch,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            show_progress=verbose,
            progress_desc="Candidate embeddings",
        )
        if verbose:
            print("🔹 Generating query embeddings...")
        query_embeddings = generate_embeddings(
            query_texts,
            tokenizer,
            encode_batch=encode_batch,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            show_progress=verbose,
            progress_desc="Query embeddings",
        )

    if device.type == "cuda":
        candidate_embeddings = candidate_embeddings.to(device)
        query_embeddings = query_embeddings.to(device)

    candidate_data = EmbeddingsData(candidate_ids, candidate_embeddings)
    query_data = EmbeddingsData(actual_query_ids, query_embeddings)

    lines, scores_dict, missing_from_scores = rank_candidates_with_scores(
        query_ids=actual_query_ids,
        query_embeddings=query_data,
        candidate_embeddings=candidate_data,
        metric="dot",
        run_tag=run_tag,
        query_to_candidate_ids=resolved_scope,
        fallback_to_all_candidates_if_scope_missing=fallback_to_all_candidates_if_scope_missing,
    )

    combined_missing = sorted(set(missing_files + missing_from_scores))
    if verbose and combined_missing:
        print(f"⚠️ Missing embeddings for {len(combined_missing)} queries: {combined_missing}")

    trec_output_path.parent.mkdir(parents=True, exist_ok=True)
    trec_output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    if verbose:
        print(f"✅ Saved similarity scores to {trec_output_path}")
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
