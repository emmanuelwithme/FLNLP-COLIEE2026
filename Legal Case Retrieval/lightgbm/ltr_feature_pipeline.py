from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import random
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
from transformers import AutoTokenizer

# Make `lcr` package importable.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.data import EmbeddingsData
from lcr.device import get_device
from lcr.task1_paths import (
    get_task1_base_encoder_dir,
    get_task1_dir,
    get_task1_model_name,
    get_task1_model_root_dir,
    get_task1_year,
    resolve_repo_path,
)
from cutoff_postprocess import build_cutoff_config, run_cutoff_postprocess, run_fixed_topk_postprocess

# Reuse existing ModernBERT contrastive model class (core inference logic unchanged).
FP_FINE_TUNE_DIR = PACKAGE_ROOT / "modernBert-fp" / "fine_tune"
if str(FP_FINE_TUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FP_FINE_TUNE_DIR))
from modernbert_contrastive_model import ModernBERTContrastive


PLACEHOLDER_TOKENS = (
    "CITATION_SUPPRESSED",
    "REFERENCE_SUPPRESSED",
    "FRAGMENT_SUPPRESSED",
)

BASE_FEATURE_COLUMNS = [
    "query_id",
    "candidate_id",
    "bm25_score",
    "qld_score",
    "bm25_ngram_score",
    "dense_score",
    "bm25_rank",
    "dense_rank",
    "query_length",
    "doc_length",
    "len_ratio",
    "len_diff",
    "query_citation_num",
    "query_reference_num",
    "query_fragment_num",
    "doc_citation_num",
    "doc_reference_num",
    "doc_fragment_num",
    "query_citation_ratio",
    "query_reference_ratio",
    "query_fragment_ratio",
    "doc_citation_ratio",
    "doc_reference_ratio",
    "doc_fragment_ratio",
    "query_year",
    "doc_year",
    "year_diff",
    "chunk_sim_max",
    "chunk_sim_mean",
    "chunk_sim_top2_mean",
]

MAX_CHUNKS = 3


def normalize_case_id(raw_id: object) -> str:
    case_id = str(raw_id).strip()
    if case_id.endswith(".txt"):
        case_id = case_id[:-4]
    if case_id.isdigit():
        case_id = case_id.zfill(6)
    return case_id


def _read_case_text_from_directory(case_id: str, directory: Path) -> str:
    path = directory / f"{case_id}.txt"
    if not path.exists() and case_id.isdigit():
        path = directory / f"{case_id.zfill(6)}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


@dataclass(frozen=True)
class PrecomputedCaseFeatures:
    case_id: str
    raw_text: str
    clean_text: str
    citation_num: int
    reference_num: int
    fragment_num: int
    year: int
    year_source: str


def _precompute_case_features_worker(
    case_id: str,
    raw_dir: str,
    clean_dir: str,
) -> PrecomputedCaseFeatures:
    normalized_id = normalize_case_id(case_id)
    raw_text = _read_case_text_from_directory(normalized_id, Path(raw_dir))
    clean_text = _read_case_text_from_directory(normalized_id, Path(clean_dir))
    if clean_text:
        clean_text = build_clean_text(clean_text)
    else:
        clean_text = build_clean_text(raw_text)

    placeholders = count_placeholders(raw_text)
    year = extract_case_year(raw_text, metadata=None)
    return PrecomputedCaseFeatures(
        case_id=normalized_id,
        raw_text=raw_text,
        clean_text=clean_text,
        citation_num=int(placeholders["CITATION_SUPPRESSED"]),
        reference_num=int(placeholders["REFERENCE_SUPPRESSED"]),
        fragment_num=int(placeholders["FRAGMENT_SUPPRESSED"]),
        year=int(year.year),
        year_source=str(year.source),
    )


def load_qids(path: Path) -> list[str]:
    qids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid = normalize_case_id(line.split()[0])
        if not qid or qid in seen:
            continue
        seen.add(qid)
        qids.append(qid)
    return qids


def load_scope(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Scope JSON must be an object: {path}")

    scope: dict[str, list[str]] = {}
    for raw_qid, raw_candidates in payload.items():
        qid = normalize_case_id(raw_qid)
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
            raise ValueError(f"Scope candidates for `{qid}` must be a sequence.")
        seen: set[str] = set()
        candidates: list[str] = []
        for raw_doc in raw_candidates:
            docid = normalize_case_id(raw_doc)
            if not docid or docid in seen:
                continue
            seen.add(docid)
            candidates.append(docid)
        scope[qid] = candidates
    return scope


def load_labels(path: Path) -> dict[str, set[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Label JSON must be an object: {path}")

    labels: dict[str, set[str]] = {}
    for raw_qid, raw_docs in payload.items():
        qid = normalize_case_id(raw_qid)
        if isinstance(raw_docs, Sequence) and not isinstance(raw_docs, (str, bytes)):
            labels[qid] = {normalize_case_id(docid) for docid in raw_docs}
        else:
            labels[qid] = set()
    return labels


def count_placeholders(text: str) -> dict[str, int]:
    """
    Count placeholder tokens from raw text only.
    """
    counts: dict[str, int] = {}
    for token in PLACEHOLDER_TOKENS:
        counts[token] = len(re.findall(rf"\b{re.escape(token)}\b", text or ""))
    return counts


def build_clean_text(text: str) -> str:
    """
    Keep lexical text cleanup consistent for query/candidate:
    flatten whitespace and remove suppressed placeholders.
    """
    text = (text or "").replace("\t", " ").replace("\r", " ").replace("\n", " ")
    for token in PLACEHOLDER_TOKENS:
        text = text.replace(f"<{token}>", " ")
        text = text.replace(token, " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_lexical_length(text: str, analyzer: Analyzer) -> int:
    tokens = analyzer.analyze(text or "")
    return int(len(tokens))


@dataclass(frozen=True)
class YearExtractionResult:
    year: int
    source: str


def _extract_year_candidates(text: str) -> list[int]:
    years = [int(y) for y in re.findall(r"\b(18\d{2}|19\d{2}|20\d{2}|2100)\b", text or "")]
    return [y for y in years if 1800 <= y <= 2100]


def _extract_year_from_date_like_text(text: str) -> int | None:
    years = _extract_year_candidates(text)
    if not years:
        return None
    # Date fields can include multiple dates; use the latest explicit year there.
    return max(years)


def extract_case_year(raw_text: str, metadata: Mapping[str, object] | None = None) -> YearExtractionResult:
    """
    Year extraction priority:
    1) Explicit date fields (metadata/header)
    2) Neutral citation year
    3) Front-of-document regex scan
    4) Conservative fallback (global scan)
    Missing => year = -1
    """
    raw_text = raw_text or ""
    metadata = metadata or {}

    # 1) metadata explicit date-like fields
    explicit_keys = {
        "date",
        "date_of_judgment",
        "judgment_rendered",
        "judgment rendered",
        "released",
        "hearing_date",
        "heard",
        "decision_date",
        "judgment_date",
    }
    for raw_key, raw_value in metadata.items():
        key = str(raw_key).strip().lower().replace("-", "_")
        if key in explicit_keys or any(k in key for k in ["judgment", "released", "decision_date", "hearing"]):
            year = _extract_year_from_date_like_text(str(raw_value))
            if year is not None:
                return YearExtractionResult(year=year, source=f"metadata:{raw_key}")

    # 1) header explicit date-like fields in raw text
    header_slice = "\n".join(raw_text.splitlines()[:120])
    explicit_patterns = [
        r"(?im)^\s*Date\s*:\s*(.+)$",
        r"(?im)^\s*Date of Judgment\s*:\s*(.+)$",
        r"(?im)^\s*Judgment Rendered\s*:\s*(.+)$",
        r"(?im)^\s*Judgment rendered\s*:\s*(.+)$",
        r"(?im)^\s*Released\s*:\s*(.+)$",
        r"(?im)^\s*Heard(?:\s*:\s*|\s*/\s*Judgment rendered\s*:\s*)(.+)$",
    ]
    for pat in explicit_patterns:
        for match in re.finditer(pat, header_slice):
            year = _extract_year_from_date_like_text(match.group(1))
            if year is not None:
                return YearExtractionResult(year=year, source=f"header_explicit:{pat}")

    # 2) neutral citation year
    neutral_patterns = [
        r"\[(18\d{2}|19\d{2}|20\d{2}|2100)\]\s*[A-Z]{1,8}[A-Za-z\-]*\s*\d+",
        r"\b(18\d{2}|19\d{2}|20\d{2}|2100)\s+(?:FC|FCA|SCC|QCCA|BCCA|ONCA|ABCA|SKCA|MBCA|NBCA|NSCA|PECA|CanLII)\b",
        r"\((?:[^\n\)]*?;\s*)?(18\d{2}|19\d{2}|20\d{2}|2100)\s+[A-Z]{2,10}\s+\d+\)",
    ]
    for pat in neutral_patterns:
        for match in re.finditer(pat, header_slice):
            year = _extract_year_from_date_like_text(match.group(0))
            if year is not None:
                return YearExtractionResult(year=year, source=f"neutral_citation:{pat}")

    # 3) front-of-document regex scan
    front_slice = raw_text[:12000]
    front_years = _extract_year_candidates(front_slice)
    if front_years:
        # Pick the most frequent year in the front slice; tie-break by larger year.
        counts: dict[int, int] = {}
        for y in front_years:
            counts[y] = counts.get(y, 0) + 1
        best_year = sorted(counts.items(), key=lambda item: (item[1], item[0]), reverse=True)[0][0]
        return YearExtractionResult(year=best_year, source="front_regex")

    # 4) conservative fallback
    all_years = _extract_year_candidates(raw_text)
    if all_years:
        return YearExtractionResult(year=max(all_years), source="fallback_global_max")

    return YearExtractionResult(year=-1, source="missing")


def _split_text_into_sentence_like_units(text: str) -> List[str]:
    normalized_text = (text or "").strip()
    if not normalized_text:
        return [""]

    sentence_matches = re.findall(r".*?(?:[.!?;]+(?:['\")\]]+)?|\n{2,}|$)", normalized_text, flags=re.S)
    sentences = [segment.strip() for segment in sentence_matches if segment and segment.strip()]
    if sentences:
        return sentences

    paragraphs = [segment.strip() for segment in normalized_text.splitlines() if segment.strip()]
    return paragraphs or [normalized_text]


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
        if re.fullmatch(r"[A-Z]\.", token_text) and next_nonspace_char.isupper():
            return False
        if re.fullmatch(r"(?:[A-Za-z]\.){2,}", token_text):
            return False

    return True


def split_into_up_to_3_sentence_end_chunks(
    text: str,
    tokenizer,
    max_tokens: int = 4096,
    max_chunks: int = 3,
) -> list[str]:
    """
    Chunking rule from chunkAgg main-branch logic:
    - single chunk if <= max_tokens (after special token budget)
    - otherwise backtrack to nearest sentence boundary
    - keep up to first 3 chunks
    """
    if max_chunks <= 0:
        raise ValueError(f"max_chunks must be > 0, got {max_chunks}")

    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    if max_tokens <= special_tokens:
        raise ValueError(
            f"max_tokens must be larger than special tokens ({special_tokens}), got {max_tokens}"
        )

    budget = max_tokens - special_tokens
    input_ids, offsets = _tokenize_with_offsets(text, tokenizer)
    input_ids = input_ids[: max_chunks * budget]
    offsets = offsets[: len(input_ids)]

    if not input_ids:
        return [""]

    raw_chunks: list[tuple[int, int]] = []
    cursor = 0
    total_tokens = len(input_ids)

    while cursor < total_tokens and len(raw_chunks) < max_chunks:
        tentative_end = min(cursor + budget, total_tokens)
        chosen_end = tentative_end

        if tentative_end < total_tokens:
            for candidate_end in range(tentative_end, cursor, -1):
                char_end = offsets[candidate_end - 1][1]
                next_char_start = offsets[candidate_end][0] if candidate_end < total_tokens else len(text)
                if _looks_like_sentence_boundary(text, char_end, next_char_start):
                    chosen_end = candidate_end
                    break

        if chosen_end <= cursor:
            chosen_end = tentative_end

        raw_chunks.append((cursor, chosen_end))
        cursor = chosen_end

    out: list[str] = []
    for start_idx, end_idx in raw_chunks:
        if end_idx <= start_idx:
            continue
        char_start = offsets[start_idx][0]
        char_end = offsets[end_idx - 1][1]
        chunk_text = text[char_start:char_end].strip()
        if chunk_text:
            out.append(chunk_text)

    if not out:
        fallback_ids = input_ids[:budget]
        chunk_text = tokenizer.decode(fallback_ids, skip_special_tokens=True).strip()
        out = [chunk_text or (text or "")[:2000]]

    return out[:max_chunks]


class DenseEncoder:
    def __init__(
        self,
        model_root_dir: Path,
        base_encoder_dir: Path,
        device: torch.device,
        logger: logging.Logger,
        inference_batch_size: int = 8,
    ):
        self.model_root_dir = model_root_dir
        self.base_encoder_dir = base_encoder_dir
        self.device = device
        self.logger = logger
        self.inference_batch_size = max(int(inference_batch_size), 1)
        self._best_checkpoint: Path | None = None
        self._tokenizer = None
        self._model = None

    @staticmethod
    def _find_best_checkpoint(checkpoint_root: Path, metric: str = "eval_global_f1", mode: str = "max") -> Path:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be min or max")
        best_path: Path | None = None
        best_value: float | None = None
        larger_is_better = mode == "max"

        for folder in checkpoint_root.iterdir():
            if not folder.is_dir() or not folder.name.startswith("checkpoint-"):
                continue
            state_path = folder / "trainer_state.json"
            if not state_path.is_file():
                continue
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            step = state.get("global_step")
            if step is None:
                continue

            value = None
            for record in state.get("log_history", []):
                if record.get("step") == step and metric in record:
                    value = record[metric]
                    break
            if value is None:
                continue

            if best_value is None:
                best_value = float(value)
                best_path = folder
                continue

            if larger_is_better and float(value) > best_value:
                best_value = float(value)
                best_path = folder
            if (not larger_is_better) and float(value) < best_value:
                best_value = float(value)
                best_path = folder

        if best_path is None:
            raise FileNotFoundError(
                f"No checkpoint with metric `{metric}` found under {checkpoint_root}"
            )
        return best_path

    @property
    def best_checkpoint(self) -> Path:
        if self._best_checkpoint is None:
            self._best_checkpoint = self._find_best_checkpoint(self.model_root_dir)
            self.logger.info("Using dense checkpoint: %s", self._best_checkpoint)
        return self._best_checkpoint

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.best_checkpoint), trust_remote_code=True
            )
            # Avoid tokenizer hard-stop for long raw docs.
            self._tokenizer.model_max_length = 10**9
        return self._tokenizer

    def _load_model(self):
        if self._model is not None:
            return self._model

        encoder_kwargs = {
            "device_map": {"": str(self.device)},
            "torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32,
            "trust_remote_code": True,
        }
        if self.device.type == "cuda":
            encoder_kwargs["attn_implementation"] = "flash_attention_2"

        model = ModernBERTContrastive.from_pretrained(
            str(self.best_checkpoint),
            encoder_model_name_or_path=str(self.base_encoder_dir),
            encoder_kwargs=encoder_kwargs,
        )
        model = model.to(self.device)
        if self.device.type == "cuda":
            model = model.half()
        model = model.eval()
        self._model = model
        return model

    def encode_texts(self, texts: Sequence[str], max_length: int = 4096) -> torch.Tensor:
        model = self._load_model()
        tokenizer = self.tokenizer

        vectors: list[torch.Tensor] = []
        normalized_texts = [text or "" for text in texts]
        start = 0
        batch_size = min(max(self.inference_batch_size, 1), max(len(normalized_texts), 1))

        while start < len(normalized_texts):
            current_batch_size = min(batch_size, len(normalized_texts) - start)
            while current_batch_size > 0:
                batch_texts = normalized_texts[start : start + current_batch_size]
                try:
                    inputs = tokenizer(
                        batch_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                        pad_to_multiple_of=8 if self.device.type == "cuda" else None,
                    )
                    if self.device.type == "cuda":
                        inputs = {
                            k: v.pin_memory().to(self.device, non_blocking=True)
                            for k, v in inputs.items()
                        }
                    else:
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    with torch.inference_mode():
                        with (
                            torch.amp.autocast("cuda", dtype=torch.float16)
                            if self.device.type == "cuda"
                            else contextlib.nullcontext()
                        ):
                            emb = model.encode(inputs)
                    vectors.append(emb.detach().cpu())
                    start += current_batch_size
                    batch_size = max(current_batch_size, 1)
                    break
                except torch.cuda.OutOfMemoryError:
                    if self.device.type != "cuda" or current_batch_size <= 1:
                        raise
                    reduced_batch_size = max(current_batch_size // 2, 1)
                    self.logger.warning(
                        "Dense encoding OOM at batch_size=%d, retry with batch_size=%d",
                        current_batch_size,
                        reduced_batch_size,
                    )
                    torch.cuda.empty_cache()
                    current_batch_size = reduced_batch_size

        return torch.cat(vectors, dim=0) if vectors else torch.empty((0, 0))

    def encode_chunks(self, chunks: Sequence[str], max_length: int = 4096) -> torch.Tensor:
        return self.encode_texts(chunks, max_length=max_length)


class LexicalScorer:
    def __init__(
        self,
        name: str,
        index_path: Path | None,
        logger: logging.Logger,
        mode: str = "bm25",
        k1: float = 3.0,
        b: float = 1.0,
        mu: float = 1000.0,
        default_score: float = 0.0,
        max_query_terms: int = 900,
        max_query_chars: int = 10000,
        batch_search_max_threads: int = 4,
        batch_search_max_queries: int = 8,
        batch_search_max_total_hits: int = 20000,
        batch_search_max_k: int = 2500,
    ):
        self.name = name
        self.logger = logger
        self.default_score = float(default_score)
        self.max_query_terms = int(max_query_terms)
        self.max_query_chars = int(max_query_chars)
        self.batch_search_max_threads = max(int(batch_search_max_threads), 1)
        self.batch_search_max_queries = max(int(batch_search_max_queries), 1)
        self.batch_search_max_total_hits = max(int(batch_search_max_total_hits), 1)
        self.batch_search_max_k = max(int(batch_search_max_k), 1)
        self._cache: dict[str, dict[str, float]] = {}

        self.searcher: LuceneSearcher | None = None
        self.num_docs = 0
        if index_path and index_path.exists():
            self.searcher = LuceneSearcher(str(index_path))
            if mode == "bm25":
                self.searcher.set_bm25(k1=k1, b=b)
            elif mode == "qld":
                self.searcher.set_qld(mu=mu)
            else:
                raise ValueError(f"Unsupported lexical mode: {mode}")
            self.num_docs = int(self.searcher.num_docs)
            self.logger.info("[%s] index loaded: %s (num_docs=%d)", self.name, index_path, self.num_docs)
        else:
            self.logger.warning("[%s] index not found, using fallback score %.4f", self.name, self.default_score)

    @staticmethod
    def _hits_to_score_map(hits: Sequence[object]) -> dict[str, float]:
        score_map: dict[str, float] = {}
        for hit in hits:
            score_map[normalize_case_id(hit.docid)] = float(hit.score)
        return score_map

    def _truncate_query(self, text: str) -> str:
        text = text or ""
        if self.max_query_chars > 0 and len(text) > self.max_query_chars:
            text = text[: self.max_query_chars]
        if self.max_query_terms > 0:
            terms = text.split()
            if len(terms) > self.max_query_terms:
                text = " ".join(terms[: self.max_query_terms])
        return text

    def _search(self, qid: str, query_text: str, k: int) -> dict[str, float]:
        if self.searcher is None:
            return {}

        safe_query = self._truncate_query(query_text)
        if not safe_query:
            return {}

        k = min(max(int(k), 1), max(self.num_docs, 1))
        try:
            hits = self.searcher.search(safe_query, k=k)
        except Exception as exc:
            self.logger.warning("[%s] search failed qid=%s: %s", self.name, qid, exc)
            return {}

        return self._hits_to_score_map(hits)

    def _should_use_batch(self, chunk: Sequence[tuple[str, str, int]], k: int) -> bool:
        return (
            len(chunk) <= self.batch_search_max_queries
            and k <= self.batch_search_max_k
            and (len(chunk) * k) <= self.batch_search_max_total_hits
        )

    def prefetch_scores(
        self,
        query_specs: Sequence[tuple[str, str, int]],
        threads: int = 1,
        batch_size: int = 32,
    ) -> None:
        if self.searcher is None or not query_specs:
            return

        missing: list[tuple[str, str, int]] = []
        for qid, query_text, candidate_count in query_specs:
            norm_qid = normalize_case_id(qid)
            if norm_qid in self._cache:
                continue
            safe_query = self._truncate_query(query_text)
            if not safe_query:
                self._cache[norm_qid] = {}
                continue
            missing.append((norm_qid, safe_query, int(max(candidate_count, 1))))

        if not missing:
            return

        safe_threads = min(max(int(threads), 1), self.batch_search_max_threads)
        safe_batch_size = min(max(int(batch_size), 1), self.batch_search_max_queries)

        if safe_threads <= 1:
            for qid, query_text, candidate_count in missing:
                self._cache[qid] = self._search(qid, query_text, k=candidate_count)
            return

        for start in range(0, len(missing), safe_batch_size):
            chunk = missing[start : start + safe_batch_size]
            cursor = 0
            while cursor < len(chunk):
                sub_chunk_size = min(safe_batch_size, len(chunk) - cursor)
                sub_chunk = chunk[cursor : cursor + sub_chunk_size]
                k = min(max(item[2] for item in sub_chunk), max(self.num_docs, 1))

                while sub_chunk_size > 1 and not self._should_use_batch(sub_chunk, k):
                    sub_chunk_size = max(sub_chunk_size // 2, 1)
                    sub_chunk = chunk[cursor : cursor + sub_chunk_size]
                    k = min(max(item[2] for item in sub_chunk), max(self.num_docs, 1))

                if sub_chunk_size == 1 and not self._should_use_batch(sub_chunk, k):
                    qid, query_text, candidate_count = sub_chunk[0]
                    self._cache[qid] = self._search(qid, query_text, k=candidate_count)
                    cursor += 1
                    continue

                qids = [item[0] for item in sub_chunk]
                queries = [item[1] for item in sub_chunk]
                try:
                    batch_hits = self.searcher.batch_search(queries, qids, k=k, threads=safe_threads)
                except Exception as exc:
                    self.logger.warning(
                        "[%s] batch_search failed for %d queries, fallback single search: %s",
                        self.name,
                        len(sub_chunk),
                        exc,
                    )
                    for qid, query_text, candidate_count in sub_chunk:
                        self._cache[qid] = self._search(qid, query_text, k=candidate_count)
                    cursor += len(sub_chunk)
                    continue

                for qid, _, _ in sub_chunk:
                    self._cache[qid] = self._hits_to_score_map(batch_hits.get(qid, []))
                cursor += len(sub_chunk)

    def score_candidates(
        self,
        query_id: str,
        query_text: str,
        candidate_ids: Sequence[str],
    ) -> dict[str, float]:
        norm_qid = normalize_case_id(query_id)
        if norm_qid not in self._cache:
            self._cache[norm_qid] = self._search(norm_qid, query_text, k=len(candidate_ids))

        hit_map = self._cache[norm_qid]
        out: dict[str, float] = {}
        missing = 0
        for cid in candidate_ids:
            score = hit_map.get(cid)
            if score is None:
                score = self.default_score
                missing += 1
            out[cid] = float(score)

        if missing:
            self.logger.debug(
                "[%s] qid=%s missing=%d/%d -> fallback %.4f",
                self.name,
                norm_qid,
                missing,
                len(candidate_ids),
                self.default_score,
            )
        return out

    def release_scores(self, query_id: str) -> None:
        self._cache.pop(normalize_case_id(query_id), None)


def get_bm25_score(query_id: str, candidate_id: str, score_map: Mapping[str, float]) -> float:
    _ = query_id  # query-scoped map is already resolved by caller
    return float(score_map.get(candidate_id, 0.0))


def get_qld_score(query_id: str, candidate_id: str, score_map: Mapping[str, float]) -> float:
    _ = query_id
    return float(score_map.get(candidate_id, 0.0))


def get_bm25_ngram_score(query_id: str, candidate_id: str, score_map: Mapping[str, float]) -> float:
    _ = query_id
    return float(score_map.get(candidate_id, 0.0))


def encode_chunks(chunks: Sequence[str], dense_encoder: "DenseEncoder", max_length: int = 4096) -> torch.Tensor:
    return dense_encoder.encode_chunks(chunks, max_length=max_length)


def aggregate_chunk_similarities_batch(
    query_chunk_embs: torch.Tensor,
    doc_chunk_embs: torch.Tensor,
    doc_chunk_counts: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_docs = int(doc_chunk_embs.shape[0])
    if query_chunk_embs.numel() == 0 or doc_chunk_embs.numel() == 0 or num_docs == 0:
        empty = np.zeros(num_docs, dtype=np.float32)
        return empty, empty, empty

    compute_dtype = torch.float32
    query_chunk_embs = query_chunk_embs.to(device=device, dtype=compute_dtype, non_blocking=device.type == "cuda")
    doc_chunk_embs = doc_chunk_embs.to(device=device, dtype=compute_dtype, non_blocking=device.type == "cuda")
    doc_chunk_counts = doc_chunk_counts.to(device=device, dtype=torch.int64, non_blocking=device.type == "cuda")

    with torch.inference_mode():
        sims = torch.einsum("qd,nkd->nqk", query_chunk_embs, doc_chunk_embs)
        max_doc_chunks = int(doc_chunk_embs.shape[1])
        doc_mask = (
            torch.arange(max_doc_chunks, device=device).unsqueeze(0)
            < doc_chunk_counts.unsqueeze(1)
        )
        pair_mask = doc_mask.unsqueeze(1).expand(-1, query_chunk_embs.shape[0], -1)

        flat_sims = sims.reshape(num_docs, -1)
        flat_mask = pair_mask.reshape(num_docs, -1)
        neg_inf = torch.full_like(flat_sims, torch.finfo(flat_sims.dtype).min)
        masked_for_max = torch.where(flat_mask, flat_sims, neg_inf)
        masked_for_mean = torch.where(flat_mask, flat_sims, torch.zeros_like(flat_sims))

        sim_max = masked_for_max.amax(dim=1)
        valid_counts = flat_mask.sum(dim=1).clamp(min=1)
        sim_mean = masked_for_mean.sum(dim=1) / valid_counts

        topk_count = 2 if flat_sims.shape[1] > 1 else 1
        topk_vals = masked_for_max.topk(k=topk_count, dim=1).values
        if topk_count == 1:
            sim_top2_mean = topk_vals[:, 0]
        else:
            sim_top2_mean = torch.where(
                valid_counts > 1,
                topk_vals[:, :2].mean(dim=1),
                topk_vals[:, 0],
            )

    return (
        sim_max.detach().cpu().numpy().astype(np.float32, copy=False),
        sim_mean.detach().cpu().numpy().astype(np.float32, copy=False),
        sim_top2_mean.detach().cpu().numpy().astype(np.float32, copy=False),
    )


class CaseFeatureStore:
    def __init__(
        self,
        raw_dir: Path,
        clean_dir: Path,
        analyzer: Analyzer,
        dense_vectors: Mapping[str, torch.Tensor],
        dense_encoder: DenseEncoder,
        logger: logging.Logger,
    ):
        self.raw_dir = raw_dir
        self.clean_dir = clean_dir
        self.analyzer = analyzer
        self.dense_vectors = dense_vectors
        self.dense_encoder = dense_encoder
        self.logger = logger

        self._raw_cache: dict[str, str] = {}
        self._clean_cache: dict[str, str] = {}
        self._length_cache: dict[str, int] = {}
        self._placeholder_cache: dict[str, dict[str, int]] = {}
        self._year_cache: dict[str, YearExtractionResult] = {}
        self._chunk_cache: dict[str, torch.Tensor] = {}
        self._chunk_padded_cache: dict[str, torch.Tensor] = {}
        self._chunk_count_cache: dict[str, int] = {}
        self._year_log_rows: list[dict[str, object]] = []
        self.embedding_dim = 0
        self.embedding_dtype = torch.float16
        for value in self.dense_vectors.values():
            if value is None:
                continue
            self.embedding_dim = int(value.shape[-1])
            self.embedding_dtype = value.dtype
            break
        self.all_case_ids: list[str] = sorted(
            {
                normalize_case_id(path.stem)
                for path in self.clean_dir.glob("*.txt")
                if path.is_file()
            }
        )
        self._warmup_done: set[str] = set()

    def _zero_dense_vector(self) -> torch.Tensor:
        width = max(int(self.embedding_dim), 1)
        return torch.zeros((width,), dtype=self.embedding_dtype)

    def _store_chunk_embeddings(self, case_id: str, embs: torch.Tensor | None) -> None:
        if embs is None or embs.numel() == 0:
            embs = self._zero_dense_vector().unsqueeze(0)
        if embs.ndim == 1:
            embs = embs.unsqueeze(0)

        embs = embs.detach().cpu().contiguous()
        self.embedding_dim = max(int(self.embedding_dim), int(embs.shape[-1]))
        self.embedding_dtype = embs.dtype

        chunk_count = min(int(embs.shape[0]), MAX_CHUNKS)
        embs = embs[:chunk_count]
        self._chunk_cache[case_id] = embs
        self._chunk_count_cache[case_id] = chunk_count

        padded = torch.zeros((MAX_CHUNKS, self.embedding_dim), dtype=embs.dtype)
        padded[:chunk_count, : embs.shape[1]] = embs
        self._chunk_padded_cache[case_id] = padded.contiguous()

    def _read_case(self, case_id: str, directory: Path) -> str:
        return _read_case_text_from_directory(case_id, directory)

    def warmup_case_features(
        self,
        case_ids: Sequence[str],
        num_workers: int = 1,
        progress_desc: str | None = None,
    ) -> None:
        targets: list[str] = []
        for cid in case_ids:
            norm_cid = normalize_case_id(cid)
            if norm_cid:
                targets.append(norm_cid)
        if not targets:
            return

        pending: list[str] = []
        seen: set[str] = set()
        for cid in targets:
            if cid in seen or cid in self._warmup_done:
                continue
            seen.add(cid)
            pending.append(cid)

        if not pending:
            return

        desc = progress_desc or "case warmup"
        workers = max(int(num_workers), 1)
        self.logger.info("Precomputing %d cases (%s workers=%d)", len(pending), desc, workers)

        def _ingest_precomputed(item: PrecomputedCaseFeatures) -> None:
            case_id = item.case_id
            self._raw_cache[case_id] = item.raw_text
            self._clean_cache[case_id] = item.clean_text
            self._placeholder_cache[case_id] = {
                "CITATION_SUPPRESSED": int(item.citation_num),
                "REFERENCE_SUPPRESSED": int(item.reference_num),
                "FRAGMENT_SUPPRESSED": int(item.fragment_num),
            }
            year_result = YearExtractionResult(year=int(item.year), source=item.year_source)
            self._year_cache[case_id] = year_result
            self._year_log_rows.append(
                {
                    "case_id": case_id,
                    "year": year_result.year,
                    "source": year_result.source,
                }
            )
            self._length_cache[case_id] = compute_lexical_length(item.clean_text, self.analyzer)
            self._warmup_done.add(case_id)

        if workers <= 1:
            for cid in tqdm(pending, desc=f"[warmup] {desc}", unit="case", dynamic_ncols=True):
                item = _precompute_case_features_worker(cid, str(self.raw_dir), str(self.clean_dir))
                _ingest_precomputed(item)
            return

        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                iterator = pool.map(
                    _precompute_case_features_worker,
                    pending,
                    [str(self.raw_dir)] * len(pending),
                    [str(self.clean_dir)] * len(pending),
                    chunksize=16,
                )
                for item in tqdm(iterator, total=len(pending), desc=f"[warmup] {desc}", unit="case", dynamic_ncols=True):
                    _ingest_precomputed(item)
        except Exception as exc:
            self.logger.warning("Parallel warmup failed (%s); fallback to single-process warmup.", exc)
            for cid in tqdm(pending, desc=f"[warmup] {desc}/fallback", unit="case", dynamic_ncols=True):
                item = _precompute_case_features_worker(cid, str(self.raw_dir), str(self.clean_dir))
                _ingest_precomputed(item)

    def get_raw_text(self, case_id: str) -> str:
        if case_id not in self._raw_cache:
            self._raw_cache[case_id] = self._read_case(case_id, self.raw_dir)
        return self._raw_cache[case_id]

    def get_clean_text(self, case_id: str) -> str:
        if case_id in self._clean_cache:
            return self._clean_cache[case_id]

        clean_text = self._read_case(case_id, self.clean_dir)
        if clean_text:
            clean_text = build_clean_text(clean_text)
        else:
            # Fallback: derive lexical text from raw text if cleaned file missing.
            clean_text = build_clean_text(self.get_raw_text(case_id))
        self._clean_cache[case_id] = clean_text
        return clean_text

    def get_placeholders(self, case_id: str) -> dict[str, int]:
        if case_id not in self._placeholder_cache:
            self._placeholder_cache[case_id] = count_placeholders(self.get_raw_text(case_id))
        return self._placeholder_cache[case_id]

    def get_year(self, case_id: str) -> YearExtractionResult:
        if case_id in self._year_cache:
            return self._year_cache[case_id]
        result = extract_case_year(self.get_raw_text(case_id), metadata=None)
        self._year_cache[case_id] = result
        self._year_log_rows.append(
            {
                "case_id": case_id,
                "year": result.year,
                "source": result.source,
            }
        )
        self.logger.debug("year case=%s -> %s (%s)", case_id, result.year, result.source)
        return result

    def get_lexical_length(self, case_id: str) -> int:
        if case_id not in self._length_cache:
            self._length_cache[case_id] = compute_lexical_length(
                self.get_clean_text(case_id),
                self.analyzer,
            )
        return self._length_cache[case_id]

    def get_dense_vector(self, case_id: str) -> torch.Tensor | None:
        vec = self.dense_vectors.get(case_id)
        if vec is not None:
            return vec

        # Stable fallback: on-the-fly encode whole clean document.
        text = self.get_clean_text(case_id)
        if not text:
            return None
        emb = self.dense_encoder.encode_chunks([text], max_length=4096)
        if emb.numel() == 0:
            return None
        vec = emb[0].detach().cpu().contiguous()
        self.embedding_dim = max(int(self.embedding_dim), int(vec.shape[-1]))
        self.embedding_dtype = vec.dtype
        self.dense_vectors[case_id] = vec
        return vec

    def get_chunk_embeddings(self, case_id: str) -> torch.Tensor:
        if case_id in self._chunk_cache:
            return self._chunk_cache[case_id]

        clean_text = self.get_clean_text(case_id)
        full_vec = self.get_dense_vector(case_id)

        chunks = split_into_up_to_3_sentence_end_chunks(
            clean_text,
            tokenizer=self.dense_encoder.tokenizer,
            max_tokens=4096,
            max_chunks=MAX_CHUNKS,
        )

        if len(chunks) <= 1 and full_vec is not None:
            embs = full_vec.unsqueeze(0)
        else:
            embs = encode_chunks(chunks, self.dense_encoder, max_length=4096)
            if embs.numel() == 0:
                if full_vec is not None:
                    embs = full_vec.unsqueeze(0)
                else:
                    embs = self._zero_dense_vector().unsqueeze(0)

        self._store_chunk_embeddings(case_id, embs)
        return self._chunk_cache[case_id]

    def get_chunk_count(self, case_id: str) -> int:
        if case_id not in self._chunk_count_cache:
            _ = self.get_chunk_embeddings(case_id)
        return int(self._chunk_count_cache[case_id])

    def get_padded_chunk_embeddings(self, case_id: str) -> torch.Tensor:
        if case_id not in self._chunk_padded_cache:
            _ = self.get_chunk_embeddings(case_id)
        return self._chunk_padded_cache[case_id]

    def get_dense_batch(self, case_ids: Sequence[str]) -> torch.Tensor:
        vectors: list[torch.Tensor] = []
        for cid in case_ids:
            vec = self.get_dense_vector(cid)
            vectors.append(vec if vec is not None else self._zero_dense_vector())
        if not vectors:
            return torch.empty((0, max(int(self.embedding_dim), 1)), dtype=self.embedding_dtype)
        return torch.stack(vectors, dim=0).contiguous()

    def get_padded_chunk_batch(self, case_ids: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        chunks: list[torch.Tensor] = []
        counts: list[int] = []
        for cid in case_ids:
            chunks.append(self.get_padded_chunk_embeddings(cid))
            counts.append(self.get_chunk_count(cid))
        if not chunks:
            empty = torch.empty((0, MAX_CHUNKS, max(int(self.embedding_dim), 1)), dtype=self.embedding_dtype)
            return empty, torch.empty((0,), dtype=torch.int64)
        return torch.stack(chunks, dim=0).contiguous(), torch.as_tensor(counts, dtype=torch.int64)

    def warmup_chunk_embeddings(
        self,
        case_ids: Sequence[str],
        progress_desc: str | None = None,
        case_batch_size: int = 128,
    ) -> None:
        pending: list[str] = []
        seen: set[str] = set()
        for cid in case_ids:
            norm_cid = normalize_case_id(cid)
            if not norm_cid or norm_cid in seen or norm_cid in self._chunk_cache:
                continue
            seen.add(norm_cid)
            pending.append(norm_cid)

        if not pending:
            return

        desc = progress_desc or "chunk warmup"
        safe_case_batch_size = max(int(case_batch_size), 1)
        self.logger.info(
            "Precomputing chunk embeddings for %d cases (%s, case_batch_size=%d, dense_batch_size=%d)",
            len(pending),
            desc,
            safe_case_batch_size,
            self.dense_encoder.inference_batch_size,
        )

        text_batch: list[str] = []
        case_slices: list[tuple[str, int, int]] = []

        def flush_pending() -> None:
            nonlocal text_batch, case_slices
            if not text_batch:
                return
            encoded = self.dense_encoder.encode_texts(text_batch, max_length=4096)
            for case_id, start, end in case_slices:
                self._store_chunk_embeddings(case_id, encoded[start:end])
            text_batch = []
            case_slices = []

        for cid in tqdm(pending, desc=f"[chunk] {desc}", unit="case", dynamic_ncols=True):
            clean_text = self.get_clean_text(cid)
            full_vec = self.get_dense_vector(cid)
            chunks = split_into_up_to_3_sentence_end_chunks(
                clean_text,
                tokenizer=self.dense_encoder.tokenizer,
                max_tokens=4096,
                max_chunks=MAX_CHUNKS,
            )

            if len(chunks) <= 1 and full_vec is not None:
                self._store_chunk_embeddings(cid, full_vec.unsqueeze(0))
                continue

            if not chunks:
                self._store_chunk_embeddings(cid, full_vec.unsqueeze(0) if full_vec is not None else None)
                continue

            batch_start = len(text_batch)
            text_batch.extend(chunks)
            case_slices.append((cid, batch_start, len(text_batch)))

            if len(case_slices) >= safe_case_batch_size:
                flush_pending()

        flush_pending()

    def dump_year_log(self, path: Path) -> None:
        if not self._year_log_rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self._year_log_rows).drop_duplicates(subset=["case_id"]).to_csv(
            path,
            index=False,
        )


@dataclass
class SplitPaths:
    raw_dir: Path
    clean_dir: Path
    bm25_index: Path | None
    qld_index: Path | None
    bm25_ngram_index: Path | None
    qid_path: Path
    scope_path: Path
    labels_path: Path | None
    embeddings_path: Path


def _collect_scope_case_ids(
    query_ids: Sequence[str],
    scope: Mapping[str, Sequence[str]],
    max_queries: int = 0,
) -> list[str]:
    effective_query_ids = list(query_ids[:max_queries]) if max_queries > 0 else list(query_ids)
    out: list[str] = []
    seen: set[str] = set()

    for qid in effective_query_ids:
        norm_qid = normalize_case_id(qid)
        if norm_qid and norm_qid not in seen:
            seen.add(norm_qid)
            out.append(norm_qid)

        candidates = scope.get(norm_qid, scope.get(qid, []))
        for cid in candidates:
            norm_cid = normalize_case_id(cid)
            if norm_cid and norm_cid not in seen:
                seen.add(norm_cid)
                out.append(norm_cid)
    return out


def _compute_query_ranks(candidate_ids: Sequence[str], scores: np.ndarray) -> np.ndarray:
    if len(candidate_ids) == 0:
        return np.empty((0,), dtype=np.int32)
    candidate_array = np.asarray(candidate_ids, dtype=str)
    score_array = np.asarray(scores, dtype=np.float32)
    order = np.lexsort((candidate_array, -score_array))
    ranks = np.empty(len(candidate_ids), dtype=np.int32)
    ranks[order] = np.arange(1, len(candidate_ids) + 1, dtype=np.int32)
    return ranks


def _compute_query_candidate_similarity_features(
    query_id: str,
    candidate_ids: Sequence[str],
    store: CaseFeatureStore,
    feature_compute_device: torch.device,
    feature_score_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not candidate_ids:
        empty = np.zeros((0,), dtype=np.float32)
        return empty, empty, empty, empty

    q_vec = store.get_dense_vector(query_id)
    q_chunk_embs = store.get_chunk_embeddings(query_id)
    safe_batch_size = max(int(feature_score_batch_size), 1)

    dense_batches: list[np.ndarray] = []
    chunk_max_batches: list[np.ndarray] = []
    chunk_mean_batches: list[np.ndarray] = []
    chunk_top2_batches: list[np.ndarray] = []

    for start in range(0, len(candidate_ids), safe_batch_size):
        batch_ids = candidate_ids[start : start + safe_batch_size]
        candidate_dense = store.get_dense_batch(batch_ids)
        candidate_chunks, candidate_chunk_counts = store.get_padded_chunk_batch(batch_ids)

        if q_vec is None or candidate_dense.numel() == 0:
            dense_scores = np.zeros(len(batch_ids), dtype=np.float32)
        else:
            q_vec_device = q_vec.to(
                device=feature_compute_device,
                dtype=torch.float32,
                non_blocking=feature_compute_device.type == "cuda",
            )
            candidate_dense_device = candidate_dense.to(
                device=feature_compute_device,
                dtype=torch.float32,
                non_blocking=feature_compute_device.type == "cuda",
            )
            with torch.inference_mode():
                dense_scores = torch.matmul(candidate_dense_device, q_vec_device).detach().cpu().numpy()
            dense_scores = dense_scores.astype(np.float32, copy=False)

        chunk_max, chunk_mean, chunk_top2 = aggregate_chunk_similarities_batch(
            q_chunk_embs,
            candidate_chunks,
            candidate_chunk_counts,
            device=feature_compute_device,
        )

        dense_batches.append(dense_scores)
        chunk_max_batches.append(chunk_max)
        chunk_mean_batches.append(chunk_mean)
        chunk_top2_batches.append(chunk_top2)

    return (
        np.concatenate(dense_batches).astype(np.float32, copy=False),
        np.concatenate(chunk_max_batches).astype(np.float32, copy=False),
        np.concatenate(chunk_mean_batches).astype(np.float32, copy=False),
        np.concatenate(chunk_top2_batches).astype(np.float32, copy=False),
    )


def _build_split_rows(
    split_name: str,
    query_ids: Sequence[str],
    scope: Mapping[str, Sequence[str]],
    labels: Mapping[str, set[str]] | None,
    store: CaseFeatureStore,
    bm25_scorer: LexicalScorer,
    qld_scorer: LexicalScorer,
    bm25_ngram_scorer: LexicalScorer,
    output_csv: Path,
    logger: logging.Logger,
    max_queries: int = 0,
    num_workers: int = 1,
    lexical_prefetch_batch_size: int = 32,
    feature_compute_device: torch.device | None = None,
    feature_score_batch_size: int = 4096,
) -> list[int]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    include_label = labels is not None
    columns = BASE_FEATURE_COLUMNS + (["label"] if include_label else [])
    compute_device = feature_compute_device or store.dense_encoder.device

    group_sizes: list[int] = []
    written = 0
    effective_query_ids = list(query_ids[:max_queries]) if max_queries > 0 else list(query_ids)

    with output_csv.open("w", encoding="utf-8", newline="") as fout:
        fout.write(",".join(columns) + "\n")

        for idx, qid in enumerate(
            tqdm(
                effective_query_ids,
                desc=f"[{split_name}] queries",
                unit="query",
                dynamic_ncols=True,
            )
        ):
            qid = normalize_case_id(qid)
            if idx % max(int(lexical_prefetch_batch_size), 1) == 0:
                batch_qids = effective_query_ids[idx : idx + max(int(lexical_prefetch_batch_size), 1)]
                query_specs: list[tuple[str, str, int]] = []
                for pref_qid in batch_qids:
                    pref_qid = normalize_case_id(pref_qid)
                    pref_candidates = [
                        normalize_case_id(cid)
                        for cid in scope.get(pref_qid, scope.get(pref_qid.lstrip("0"), []))
                    ]
                    if not pref_candidates:
                        pref_candidates = list(store.all_case_ids)
                    query_specs.append(
                        (
                            pref_qid,
                            store.get_clean_text(pref_qid),
                            len(pref_candidates),
                        )
                    )

                bm25_scorer.prefetch_scores(
                    query_specs,
                    threads=max(int(num_workers), 1),
                    batch_size=max(int(lexical_prefetch_batch_size), 1),
                )
                qld_scorer.prefetch_scores(
                    query_specs,
                    threads=max(int(num_workers), 1),
                    batch_size=max(int(lexical_prefetch_batch_size), 1),
                )
                bm25_ngram_scorer.prefetch_scores(
                    query_specs,
                    threads=max(int(num_workers), 1),
                    batch_size=max(int(lexical_prefetch_batch_size), 1),
                )

            candidate_ids = [normalize_case_id(cid) for cid in scope.get(qid, scope.get(qid.lstrip("0"), []))]
            if not candidate_ids:
                candidate_ids = list(store.all_case_ids)
                logger.warning(
                    "[%s] query %s has empty scope; fallback to all candidates (%d)",
                    split_name,
                    qid,
                    len(candidate_ids),
                )
                if not candidate_ids:
                    continue

            q_clean = store.get_clean_text(qid)
            q_len = store.get_lexical_length(qid)
            q_place = store.get_placeholders(qid)
            q_year = store.get_year(qid).year

            bm25_scores = bm25_scorer.score_candidates(qid, q_clean, candidate_ids)
            qld_scores = qld_scorer.score_candidates(qid, q_clean, candidate_ids)
            bm25_ngram_scores = bm25_ngram_scorer.score_candidates(qid, q_clean, candidate_ids)

            dense_scores, chunk_max, chunk_mean, chunk_top2 = _compute_query_candidate_similarity_features(
                query_id=qid,
                candidate_ids=candidate_ids,
                store=store,
                feature_compute_device=compute_device,
                feature_score_batch_size=feature_score_batch_size,
            )

            candidate_count = len(candidate_ids)
            doc_lengths = np.empty(candidate_count, dtype=np.int32)
            doc_years = np.empty(candidate_count, dtype=np.int32)
            doc_citation_num = np.empty(candidate_count, dtype=np.int32)
            doc_reference_num = np.empty(candidate_count, dtype=np.int32)
            doc_fragment_num = np.empty(candidate_count, dtype=np.int32)
            for cand_idx, cid in enumerate(candidate_ids):
                doc_lengths[cand_idx] = int(store.get_lexical_length(cid))
                doc_years[cand_idx] = int(store.get_year(cid).year)
                placeholders = store.get_placeholders(cid)
                doc_citation_num[cand_idx] = int(placeholders["CITATION_SUPPRESSED"])
                doc_reference_num[cand_idx] = int(placeholders["REFERENCE_SUPPRESSED"])
                doc_fragment_num[cand_idx] = int(placeholders["FRAGMENT_SUPPRESSED"])

            query_length_arr = np.full(candidate_count, int(q_len), dtype=np.int32)
            query_year_arr = np.full(candidate_count, int(q_year), dtype=np.int32)
            query_citation_num_arr = np.full(candidate_count, int(q_place["CITATION_SUPPRESSED"]), dtype=np.int32)
            query_reference_num_arr = np.full(candidate_count, int(q_place["REFERENCE_SUPPRESSED"]), dtype=np.int32)
            query_fragment_num_arr = np.full(candidate_count, int(q_place["FRAGMENT_SUPPRESSED"]), dtype=np.int32)
            denom_doc_lengths = np.maximum(doc_lengths, 1)

            year_diff = np.where(
                (query_year_arr == -1) | (doc_years == -1),
                0,
                query_year_arr - doc_years,
            ).astype(np.int32, copy=False)
            bm25_score_arr = np.fromiter(
                (get_bm25_score(qid, cid, bm25_scores) for cid in candidate_ids),
                dtype=np.float32,
                count=candidate_count,
            )
            qld_score_arr = np.fromiter(
                (get_qld_score(qid, cid, qld_scores) for cid in candidate_ids),
                dtype=np.float32,
                count=candidate_count,
            )
            bm25_ngram_score_arr = np.fromiter(
                (get_bm25_ngram_score(qid, cid, bm25_ngram_scores) for cid in candidate_ids),
                dtype=np.float32,
                count=candidate_count,
            )

            q_df = pd.DataFrame(
                {
                    "query_id": np.full(candidate_count, qid, dtype=object),
                    "candidate_id": np.asarray(candidate_ids, dtype=object),
                    "bm25_score": bm25_score_arr,
                    "qld_score": qld_score_arr,
                    "bm25_ngram_score": bm25_ngram_score_arr,
                    "dense_score": dense_scores,
                    "bm25_rank": _compute_query_ranks(candidate_ids, bm25_score_arr),
                    "dense_rank": _compute_query_ranks(candidate_ids, dense_scores),
                    "query_length": query_length_arr,
                    "doc_length": doc_lengths,
                    "len_ratio": query_length_arr.astype(np.float32) / denom_doc_lengths.astype(np.float32),
                    "len_diff": np.abs(query_length_arr - doc_lengths).astype(np.int32, copy=False),
                    "query_citation_num": query_citation_num_arr,
                    "query_reference_num": query_reference_num_arr,
                    "query_fragment_num": query_fragment_num_arr,
                    "doc_citation_num": doc_citation_num,
                    "doc_reference_num": doc_reference_num,
                    "doc_fragment_num": doc_fragment_num,
                    "query_citation_ratio": query_citation_num_arr.astype(np.float32) / float(max(q_len, 1)),
                    "query_reference_ratio": query_reference_num_arr.astype(np.float32) / float(max(q_len, 1)),
                    "query_fragment_ratio": query_fragment_num_arr.astype(np.float32) / float(max(q_len, 1)),
                    "doc_citation_ratio": doc_citation_num.astype(np.float32) / denom_doc_lengths.astype(np.float32),
                    "doc_reference_ratio": doc_reference_num.astype(np.float32) / denom_doc_lengths.astype(np.float32),
                    "doc_fragment_ratio": doc_fragment_num.astype(np.float32) / denom_doc_lengths.astype(np.float32),
                    "query_year": query_year_arr,
                    "doc_year": doc_years,
                    "year_diff": year_diff,
                    "chunk_sim_max": chunk_max,
                    "chunk_sim_mean": chunk_mean,
                    "chunk_sim_top2_mean": chunk_top2,
                }
            )

            if include_label:
                relevant_docs = labels.get(qid, set())
                q_df["label"] = np.fromiter(
                    (int(cid in relevant_docs) for cid in candidate_ids),
                    dtype=np.int8,
                    count=candidate_count,
                )

            q_df = q_df[columns]
            q_df.to_csv(fout, index=False, header=False)
            written += candidate_count
            group_sizes.append(candidate_count)

            bm25_scorer.release_scores(qid)
            qld_scorer.release_scores(qid)
            bm25_ngram_scorer.release_scores(qid)
            if (idx + 1) % 20 == 0:
                logger.info(
                    "[%s] processed %d/%d queries, rows=%d",
                    split_name,
                    idx + 1,
                    len(effective_query_ids),
                    written,
                )

    logger.info("[%s] feature rows written: %d -> %s", split_name, written, output_csv)
    return group_sizes


def _load_feature_frame(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    dtype_map: dict[str, object] = {"query_id": str, "candidate_id": str}
    for column in header.columns:
        if column in {"query_id", "candidate_id"}:
            continue
        if column == "label":
            dtype_map[column] = np.int8
        elif column.endswith("_rank") or column.endswith("_length") or column.endswith("_num") or column.endswith("_year") or column == "year_diff":
            dtype_map[column] = np.int32
        else:
            dtype_map[column] = np.float32
    df = pd.read_csv(path, dtype=dtype_map)
    return df


def _compute_groups(df: pd.DataFrame) -> list[int]:
    return df.groupby("query_id", sort=False).size().tolist()


def _compute_topk_metrics(
    df_valid: pd.DataFrame,
    topk: int = 5,
    *,
    label_source_df: pd.DataFrame | None = None,
    all_query_ids: Sequence[str] | None = None,
) -> tuple[float, float, float]:
    ranked = df_valid.sort_values(
        ["query_id", "pred_score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    topk_df = ranked.groupby("query_id", sort=False).head(topk)

    tp = 0
    fp = 0
    fn = 0

    label_source = label_source_df if label_source_df is not None else df_valid

    label_map = (
        label_source[label_source["label"] > 0]
        .groupby("query_id", sort=False)["candidate_id"]
        .apply(set)
        .to_dict()
    )
    pred_map = topk_df.groupby("query_id", sort=False)["candidate_id"].apply(list).to_dict()

    if all_query_ids is None:
        all_qids = list(dict.fromkeys(label_source["query_id"].tolist()))
    else:
        all_qids = [str(qid) for qid in all_query_ids]
    for qid in all_qids:
        labels = label_map.get(qid, set())
        preds = pred_map.get(qid, [])

        for d in labels:
            if d in preds:
                tp += 1
            else:
                fn += 1
        for d in preds:
            if d not in labels:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def _apply_scope_post_filter(
    df_pred: pd.DataFrame,
    scope: Mapping[str, Sequence[str]],
    *,
    split_name: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    removed_rows = 0
    missing_scope_queries: list[str] = []
    empty_after_filter_queries: list[str] = []
    query_count = 0

    for raw_qid, qdf in df_pred.groupby("query_id", sort=False):
        query_count += 1
        qid = normalize_case_id(raw_qid)
        allowed = scope.get(qid, scope.get(qid.lstrip("0"), None))
        if allowed is None:
            missing_scope_queries.append(qid)
            filtered_qdf = qdf
        else:
            keep_mask = qdf["candidate_id"].isin(allowed)
            removed_rows += int((~keep_mask).sum())
            filtered_qdf = qdf.loc[keep_mask]
            if filtered_qdf.empty:
                empty_after_filter_queries.append(qid)

        if not filtered_qdf.empty:
            frames.append(filtered_qdf)

    if frames:
        filtered = pd.concat(frames, ignore_index=True)
        filtered = filtered.sort_values(
            ["query_id", "pred_score", "candidate_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        filtered["pred_rank"] = filtered.groupby("query_id", sort=False).cumcount() + 1
    else:
        filtered = df_pred.iloc[0:0].copy()

    stats = {
        "query_count": query_count,
        "removed_rows": removed_rows,
        "missing_scope_queries": missing_scope_queries,
        "empty_after_filter_queries": empty_after_filter_queries,
    }
    logger.info(
        "[%s] raw-scope post-filter: removed_rows=%d missing_scope_queries=%d empty_after_filter=%d",
        split_name,
        removed_rows,
        len(missing_scope_queries),
        len(empty_after_filter_queries),
    )
    if missing_scope_queries:
        logger.warning(
            "[%s] raw-scope post-filter missing scope for queries: %s",
            split_name,
            ", ".join(missing_scope_queries[:10]),
        )
    if empty_after_filter_queries:
        logger.warning(
            "[%s] raw-scope post-filter left zero predictions for queries: %s",
            split_name,
            ", ".join(empty_after_filter_queries[:10]),
        )
    return filtered, stats


def run_train_valid_test(
    train_feature_csv: Path,
    valid_feature_csv: Path,
    test_feature_csv: Path,
    output_dir: Path,
    logger: logging.Logger,
    num_workers: int,
    lgbm_device: str,
    valid_scope: Mapping[str, Sequence[str]],
    test_scope: Mapping[str, Sequence[str]],
) -> None:
    import lightgbm as lgb

    output_dir.mkdir(parents=True, exist_ok=True)

    df_train = _load_feature_frame(train_feature_csv)
    df_valid = _load_feature_frame(valid_feature_csv)
    df_test = _load_feature_frame(test_feature_csv)

    if "label" not in df_train.columns or "label" not in df_valid.columns:
        raise ValueError("Train/valid feature file must include `label` column")

    feature_cols = [
        c for c in BASE_FEATURE_COLUMNS
        if c not in {"query_id", "candidate_id"}
    ]

    X_train = df_train[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = df_train["label"].to_numpy(dtype=np.int32, copy=False)
    X_valid = df_valid[feature_cols].to_numpy(dtype=np.float32, copy=False)
    y_valid = df_valid["label"].to_numpy(dtype=np.int32, copy=False)
    X_test = df_test[feature_cols].to_numpy(dtype=np.float32, copy=False)

    group_train = _compute_groups(df_train)
    group_valid = _compute_groups(df_valid)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=max(int(num_workers), 1),
        device_type=lgbm_device,
        force_col_wise=(lgbm_device == "cpu"),
    )

    ranker.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[group_valid],
        eval_at=[5],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=20),
        ],
    )

    model_path = output_dir / "lgbm_ranker_scope_raw.txt"
    ranker.booster_.save_model(str(model_path))
    logger.info("Saved model: %s", model_path)

    # Validation prediction + metric
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRanker was fitted with feature names",
        )
        valid_scores = ranker.predict(X_valid)
    df_valid_out_raw = df_valid[["query_id", "candidate_id", "label"]].copy()
    df_valid_out_raw["pred_score"] = valid_scores
    df_valid_out_raw = df_valid_out_raw.sort_values(
        ["query_id", "pred_score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    df_valid_out_raw["pred_rank"] = df_valid_out_raw.groupby("query_id", sort=False).cumcount() + 1
    valid_query_ids = list(dict.fromkeys(df_valid_out_raw["query_id"].tolist()))

    valid_pred_raw_path = output_dir / "valid_predictions_raw.csv"
    df_valid_out_raw.to_csv(valid_pred_raw_path, index=False)
    raw_f1, raw_p, raw_r = _compute_topk_metrics(df_valid_out_raw, topk=5)
    logger.info("Valid raw top5: F1=%.6f Precision=%.6f Recall=%.6f", raw_f1, raw_p, raw_r)

    df_valid_out, _ = _apply_scope_post_filter(
        df_valid_out_raw,
        valid_scope,
        split_name="valid",
        logger=logger,
    )
    valid_pred_path = output_dir / "valid_predictions.csv"
    df_valid_out.to_csv(valid_pred_path, index=False)
    f1, p, r = _compute_topk_metrics(
        df_valid_out,
        topk=5,
        label_source_df=df_valid_out_raw,
        all_query_ids=valid_query_ids,
    )
    logger.info("Valid top5: F1=%.6f Precision=%.6f Recall=%.6f", f1, p, r)

    # Test prediction
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRanker was fitted with feature names",
        )
        test_scores = ranker.predict(X_test)
    df_test_out_raw = df_test[["query_id", "candidate_id"]].copy()
    df_test_out_raw["pred_score"] = test_scores
    df_test_out_raw = df_test_out_raw.sort_values(
        ["query_id", "pred_score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    df_test_out_raw["pred_rank"] = df_test_out_raw.groupby("query_id", sort=False).cumcount() + 1

    test_pred_raw_path = output_dir / "test_predictions_raw.csv"
    df_test_out_raw.to_csv(test_pred_raw_path, index=False)
    df_test_out, _ = _apply_scope_post_filter(
        df_test_out_raw,
        test_scope,
        split_name="test",
        logger=logger,
    )
    test_pred_path = output_dir / "test_predictions.csv"
    df_test_out.to_csv(test_pred_path, index=False)
    logger.info("Saved test predictions: %s", test_pred_path)


def parse_args() -> argparse.Namespace:
    task1_dir = Path(get_task1_dir())
    year = get_task1_year()
    model_name = get_task1_model_name()
    cpu_count = max(os.cpu_count() or 4, 4)
    has_cuda = torch.cuda.is_available()
    train_qid_default = resolve_repo_path(os.getenv("TASK1_TRAIN_QID_PATH")) or (task1_dir / "train_qid.tsv")
    valid_qid_default = resolve_repo_path(os.getenv("TASK1_VALID_QID_PATH")) or (task1_dir / "valid_qid.tsv")
    test_qid_default = resolve_repo_path(os.getenv("TASK1_TEST_QID_PATH")) or (task1_dir / "test_qid.tsv")
    train_valid_scope_default = resolve_repo_path(os.getenv("COLIEE_LTR_VALID_SCOPE_PATH")) or (
        task1_dir / "lht_process" / "scope_compare" / "query_candidate_scope_raw_plus0.json"
    )
    test_scope_default = resolve_repo_path(os.getenv("COLIEE_LTR_TEST_SCOPE_PATH")) or (
        task1_dir / "lht_process" / "modernBert" / "query_candidate_scope_test_raw.json"
    )
    train_valid_embeddings_default = resolve_repo_path(os.getenv("TASK1_CANDIDATE_EMBEDDINGS_OUTPUT")) or (
        task1_dir / "processed" / f"processed_document_{model_name}_embeddings.pkl"
    )
    test_embeddings_default = resolve_repo_path(os.getenv("TASK1_TEST_CANDIDATE_EMBEDDINGS_PATH")) or (
        task1_dir / "processed_test" / f"processed_test_document_{model_name}_embeddings.pkl"
    )
    model_root_dir_default = Path(get_task1_model_root_dir(scope_filter=True, quick_test=False))
    base_encoder_dir_default = Path(get_task1_base_encoder_dir())
    output_dir_default = resolve_repo_path(os.getenv("COLIEE_LTR_OUTPUT_DIR")) or (
        task1_dir / "lht_process" / "lightgbm_ltr_scope_raw"
    )

    parser = argparse.ArgumentParser(
        description="Build COLIEE Task1 LTR features (LightGBM Ranker) and run train/valid/test pipeline."
    )
    parser.add_argument("--task1-dir", type=Path, default=task1_dir)
    parser.add_argument("--year", type=str, default=year)

    parser.add_argument(
        "--train-labels",
        type=Path,
        default=task1_dir / f"task1_train_labels_{year}_train.json",
    )
    parser.add_argument(
        "--valid-labels",
        type=Path,
        default=task1_dir / f"task1_train_labels_{year}_valid.json",
    )

    parser.add_argument("--train-qid", type=Path, default=train_qid_default)
    parser.add_argument("--valid-qid", type=Path, default=valid_qid_default)
    parser.add_argument("--test-qid", type=Path, default=test_qid_default)

    parser.add_argument(
        "--train-valid-scope",
        type=Path,
        default=train_valid_scope_default,
    )
    parser.add_argument(
        "--test-scope",
        type=Path,
        default=test_scope_default,
    )

    parser.add_argument(
        "--train-valid-raw-dir",
        type=Path,
        default=task1_dir / f"task1_train_files_{year}",
    )
    parser.add_argument(
        "--train-valid-clean-dir",
        type=Path,
        default=task1_dir / "processed",
    )
    parser.add_argument(
        "--test-raw-dir",
        type=Path,
        default=task1_dir / f"task1_test_files_{year}",
    )
    parser.add_argument(
        "--test-clean-dir",
        type=Path,
        default=task1_dir / "processed_test",
    )

    parser.add_argument(
        "--train-valid-embeddings",
        type=Path,
        default=train_valid_embeddings_default,
    )
    parser.add_argument(
        "--test-embeddings",
        type=Path,
        default=test_embeddings_default,
    )

    parser.add_argument(
        "--bm25-index",
        type=Path,
        default=task1_dir / "lht_process" / "BM25" / "index",
    )
    parser.add_argument(
        "--qld-index",
        type=Path,
        default=task1_dir / "lht_process" / "BM25" / "index",
    )
    parser.add_argument(
        "--bm25-ngram-index",
        type=Path,
        default=task1_dir / "lht_process" / "BM25_ngram" / "index",
        help="Optional BM25 ngram index for train/valid. If missing => fallback score 0.",
    )
    parser.add_argument(
        "--bm25-test-index",
        type=Path,
        default=task1_dir / "lht_process" / "BM25_test" / "index",
    )
    parser.add_argument(
        "--qld-test-index",
        type=Path,
        default=task1_dir / "lht_process" / "BM25_test" / "index",
    )
    parser.add_argument(
        "--bm25-ngram-test-index",
        type=Path,
        default=task1_dir / "lht_process" / "BM25_ngram_test" / "index",
        help="Optional BM25 ngram index for test. If missing => fallback score 0.",
    )

    parser.add_argument(
        "--model-root-dir",
        type=Path,
        default=model_root_dir_default,
    )
    parser.add_argument(
        "--base-encoder-dir",
        type=Path,
        default=base_encoder_dir_default,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir_default,
    )

    parser.add_argument("--max-queries", type=int, default=0, help="Debug cap per split. 0 => full.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=cpu_count,
        help="CPU workers for warmup multiprocessing and lexical batch_search threads.",
    )
    parser.add_argument(
        "--lexical-prefetch-batch-size",
        type=int,
        default=64,
        help="Query batch size for lexical score prefetch.",
    )
    parser.add_argument(
        "--lexical-batch-max-threads",
        type=int,
        default=min(cpu_count, 12),
        help="Upper bound of Lucene batch_search threads to avoid JVM OOM.",
    )
    parser.add_argument(
        "--lexical-batch-max-queries",
        type=int,
        default=16,
        help="Upper bound of query count per Lucene batch_search call.",
    )
    parser.add_argument(
        "--lexical-batch-max-total-hits",
        type=int,
        default=120000,
        help="Upper bound of (queries * k) per batch_search; larger queries fallback to single search.",
    )
    parser.add_argument(
        "--lexical-batch-max-k",
        type=int,
        default=8000,
        help="Upper bound of k for using batch_search; larger k fallback to single search.",
    )
    parser.add_argument(
        "--dense-batch-size",
        type=int,
        default=24 if has_cuda else 8,
        help="Batch size for dense encoder inference (query/doc chunk encoding).",
    )
    parser.add_argument(
        "--chunk-warmup-case-batch-size",
        type=int,
        default=256,
        help="How many cases to queue before flushing batched chunk-embedding precompute.",
    )
    parser.add_argument(
        "--feature-score-batch-size",
        type=int,
        default=8192 if has_cuda else 2048,
        help="Candidate batch size for vectorized dense/chunk similarity feature computation.",
    )
    parser.add_argument(
        "--lgbm-device",
        type=str,
        default="cuda",
        choices=["auto", "cpu", "gpu", "cuda"],
        help="LightGBM training device. Default: cuda.",
    )
    parser.add_argument(
        "--skip-cutoff-search",
        action="store_true",
        help="Skip validation cutoff search / test cutoff application stage.",
    )
    parser.add_argument(
        "--skip-fixed-topk-export",
        action="store_true",
        help="Skip fixed top-k export for test predictions.",
    )
    parser.add_argument(
        "--fixed-topk-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for fixed top-k artifacts. Default: <output-dir>/fixed_top5",
    )
    parser.add_argument(
        "--fixed-topk-k",
        type=int,
        default=5,
        help="Fixed top-k cutoff to export for test predictions. Default: 5",
    )
    parser.add_argument(
        "--fixed-topk-keep-self",
        action="store_true",
        help="Disable self-removal in fixed top-k export.",
    )
    parser.add_argument(
        "--fixed-topk-no-submission",
        action="store_true",
        help="Skip writing fixed top-k submission files.",
    )
    parser.add_argument(
        "--fixed-topk-submission-run-tag",
        type=str,
        default="lgbm_top5",
        help="Run tag for fixed top-k submission output.",
    )
    parser.add_argument(
        "--fixed-topk-final-submission-path",
        type=Path,
        default=None,
        help="Optional copy target for the fixed top-k submission file.",
    )
    parser.add_argument(
        "--cutoff-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for cutoff search artifacts. Default: <output-dir>/cutoff_search",
    )
    parser.add_argument(
        "--cutoff-config-json",
        type=Path,
        default=None,
        help="Optional JSON config for cutoff search grids.",
    )
    parser.add_argument(
        "--cutoff-fixed-k-values",
        type=str,
        default=None,
        help="Comma-separated fixed top-k values. Default: 1,3,5,7,10",
    )
    parser.add_argument(
        "--cutoff-ratio-p-values",
        type=str,
        default=None,
        help="Comma-separated ratio cutoff p values. Default: 0.70,0.75,0.80,0.85,0.90,0.95",
    )
    parser.add_argument(
        "--cutoff-ratio-l-values",
        type=str,
        default=None,
        help="Comma-separated ratio cutoff l values. Default: 1,2,3",
    )
    parser.add_argument(
        "--cutoff-ratio-h-values",
        type=str,
        default=None,
        help="Comma-separated ratio cutoff h values. Default: 5,7,10",
    )
    parser.add_argument(
        "--cutoff-gap-n-values",
        type=str,
        default=None,
        help="Comma-separated largest-gap N values. Default: 10,20,30",
    )
    parser.add_argument(
        "--cutoff-gap-buffer-values",
        type=str,
        default=None,
        help="Comma-separated largest-gap buffer values. Default: 0,1,2",
    )
    parser.add_argument(
        "--cutoff-gap-l-values",
        type=str,
        default=None,
        help="Comma-separated largest-gap l values. Default: 1,2,3",
    )
    parser.add_argument(
        "--cutoff-gap-h-values",
        type=str,
        default=None,
        help="Comma-separated largest-gap h values. Default: 5,7,10",
    )
    parser.add_argument(
        "--cutoff-keep-self",
        action="store_true",
        help="Disable self-removal in cutoff post-process. Default removes qid==candidate_id.",
    )
    parser.add_argument(
        "--cutoff-no-submission",
        action="store_true",
        help="Skip writing cutoff submission files.",
    )
    parser.add_argument(
        "--cutoff-submission-run-tag",
        type=str,
        default="lgbm_cutoff",
        help="Run tag for cutoff submission output.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _can_use_lightgbm_device(device_type: str) -> tuple[bool, str]:
    import lightgbm as lgb

    X = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.2, 0.8],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.9, 0.1],
            [0.3, 0.7],
            [0.7, 0.3],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
    group = [4, 4]

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            probe = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=1,
                min_data_in_leaf=1,
                min_data_in_bin=1,
                n_jobs=1,
                device_type=device_type,
                verbosity=-1,
            )
            probe.fit(X, y, group=group)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def resolve_lgbm_device(requested: str, logger: logging.Logger) -> str:
    requested = (requested or "auto").strip().lower()
    if requested == "cpu":
        return "cpu"

    if requested in {"gpu", "cuda"}:
        ok, reason = _can_use_lightgbm_device(requested)
        if ok:
            logger.info("Using LightGBM device: %s", requested)
            return requested
        logger.warning("LightGBM device `%s` unavailable, fallback to CPU: %s", requested, reason)
        return "cpu"

    for candidate in ("cuda", "gpu"):
        ok, reason = _can_use_lightgbm_device(candidate)
        if ok:
            logger.info("Using LightGBM device: %s", candidate)
            return candidate
        logger.info("LightGBM device `%s` unavailable: %s", candidate, reason)

    logger.info("Using LightGBM device: cpu")
    return "cpu"


def setup_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("ltr_feature_pipeline")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def _resolve_optional_index(path: Path) -> Path | None:
    if path is None:
        return None
    text = str(path).strip()
    if text in {"", ".", "./"}:
        return None
    if not path.exists():
        return None
    return path


def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.num_workers = max(int(args.num_workers), 1)
    args.lexical_prefetch_batch_size = max(int(args.lexical_prefetch_batch_size), 1)
    args.lexical_batch_max_threads = max(int(args.lexical_batch_max_threads), 1)
    args.lexical_batch_max_queries = max(int(args.lexical_batch_max_queries), 1)
    args.lexical_batch_max_total_hits = max(int(args.lexical_batch_max_total_hits), 1)
    args.lexical_batch_max_k = max(int(args.lexical_batch_max_k), 1)
    args.dense_batch_size = max(int(args.dense_batch_size), 1)
    args.chunk_warmup_case_batch_size = max(int(args.chunk_warmup_case_batch_size), 1)
    args.feature_score_batch_size = max(int(args.feature_score_batch_size), 1)

    torch.set_num_threads(max(1, min(args.num_workers, os.cpu_count() or args.num_workers)))
    if args.num_workers > 1:
        torch.set_num_interop_threads(max(1, min(4, args.num_workers)))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = get_device()
    lgbm_device = resolve_lgbm_device(args.lgbm_device, logger)
    logger.info("Device: %s", device)
    logger.info(
        (
            "CPU acceleration config: num_workers=%d lexical_prefetch_batch_size=%d "
            "lexical_batch_max_threads=%d lexical_batch_max_queries=%d "
            "lexical_batch_max_total_hits=%d lexical_batch_max_k=%d dense_batch_size=%d "
            "chunk_warmup_case_batch_size=%d feature_score_batch_size=%d lgbm_device=%s"
        ),
        args.num_workers,
        args.lexical_prefetch_batch_size,
        args.lexical_batch_max_threads,
        args.lexical_batch_max_queries,
        args.lexical_batch_max_total_hits,
        args.lexical_batch_max_k,
        args.dense_batch_size,
        args.chunk_warmup_case_batch_size,
        args.feature_score_batch_size,
        lgbm_device,
    )

    analyzer = Analyzer(get_lucene_analyzer(language="en", stemming=True, stopwords=True))

    dense_encoder = DenseEncoder(
        model_root_dir=args.model_root_dir,
        base_encoder_dir=args.base_encoder_dir,
        device=device,
        logger=logger,
        inference_batch_size=args.dense_batch_size,
    )

    # Load split query ids + scope (scope filter all raw as requested).
    train_qids = load_qids(args.train_qid)
    valid_qids = load_qids(args.valid_qid)
    test_qids = load_qids(args.test_qid)

    train_valid_scope = load_scope(args.train_valid_scope)
    test_scope = load_scope(args.test_scope)

    train_labels = load_labels(args.train_labels) if args.train_labels.exists() else {}
    valid_labels = load_labels(args.valid_labels) if args.valid_labels.exists() else {}

    # Load dense embeddings.
    train_valid_emb = EmbeddingsData.load(args.train_valid_embeddings)
    test_emb = EmbeddingsData.load(args.test_embeddings)
    train_valid_dense = {
        normalize_case_id(k): v.detach().cpu().contiguous()
        for k, v in train_valid_emb.id2vec.items()
    }
    test_dense = {
        normalize_case_id(k): v.detach().cpu().contiguous()
        for k, v in test_emb.id2vec.items()
    }

    # Build stores.
    train_valid_store = CaseFeatureStore(
        raw_dir=args.train_valid_raw_dir,
        clean_dir=args.train_valid_clean_dir,
        analyzer=analyzer,
        dense_vectors=train_valid_dense,
        dense_encoder=dense_encoder,
        logger=logger,
    )
    test_store = CaseFeatureStore(
        raw_dir=args.test_raw_dir,
        clean_dir=args.test_clean_dir,
        analyzer=analyzer,
        dense_vectors=test_dense,
        dense_encoder=dense_encoder,
        logger=logger,
    )

    # Lexical scorers.
    bm25_scorer_train = LexicalScorer(
        name="bm25_train_valid",
        index_path=args.bm25_index,
        logger=logger,
        mode="bm25",
        batch_search_max_threads=args.lexical_batch_max_threads,
        batch_search_max_queries=args.lexical_batch_max_queries,
        batch_search_max_total_hits=args.lexical_batch_max_total_hits,
        batch_search_max_k=args.lexical_batch_max_k,
    )
    qld_scorer_train = LexicalScorer(
        name="qld_train_valid",
        index_path=args.qld_index,
        logger=logger,
        mode="qld",
        batch_search_max_threads=args.lexical_batch_max_threads,
        batch_search_max_queries=args.lexical_batch_max_queries,
        batch_search_max_total_hits=args.lexical_batch_max_total_hits,
        batch_search_max_k=args.lexical_batch_max_k,
    )
    bm25_ngram_scorer_train = LexicalScorer(
        name="bm25_ngram_train_valid",
        index_path=_resolve_optional_index(args.bm25_ngram_index),
        logger=logger,
        mode="bm25",
        k1=0.9,
        b=0.4,
        batch_search_max_threads=args.lexical_batch_max_threads,
        batch_search_max_queries=args.lexical_batch_max_queries,
        batch_search_max_total_hits=args.lexical_batch_max_total_hits,
        batch_search_max_k=args.lexical_batch_max_k,
    )

    bm25_scorer_test = LexicalScorer(
        name="bm25_test",
        index_path=args.bm25_test_index,
        logger=logger,
        mode="bm25",
        batch_search_max_threads=args.lexical_batch_max_threads,
        batch_search_max_queries=args.lexical_batch_max_queries,
        batch_search_max_total_hits=args.lexical_batch_max_total_hits,
        batch_search_max_k=args.lexical_batch_max_k,
    )
    qld_scorer_test = LexicalScorer(
        name="qld_test",
        index_path=args.qld_test_index,
        logger=logger,
        mode="qld",
        batch_search_max_threads=args.lexical_batch_max_threads,
        batch_search_max_queries=args.lexical_batch_max_queries,
        batch_search_max_total_hits=args.lexical_batch_max_total_hits,
        batch_search_max_k=args.lexical_batch_max_k,
    )
    bm25_ngram_scorer_test = LexicalScorer(
        name="bm25_ngram_test",
        index_path=_resolve_optional_index(args.bm25_ngram_test_index),
        logger=logger,
        mode="bm25",
        k1=0.9,
        b=0.4,
        batch_search_max_threads=args.lexical_batch_max_threads,
        batch_search_max_queries=args.lexical_batch_max_queries,
        batch_search_max_total_hits=args.lexical_batch_max_total_hits,
        batch_search_max_k=args.lexical_batch_max_k,
    )

    # CPU-parallel warmup for raw/clean/year/placeholder/length features.
    train_valid_case_ids = list(
        dict.fromkeys(
            _collect_scope_case_ids(
                query_ids=train_qids,
                scope=train_valid_scope,
                max_queries=args.max_queries,
            )
            + _collect_scope_case_ids(
                query_ids=valid_qids,
                scope=train_valid_scope,
                max_queries=args.max_queries,
            )
        )
    )
    test_case_ids = _collect_scope_case_ids(
        query_ids=test_qids,
        scope=test_scope,
        max_queries=args.max_queries,
    )

    train_valid_store.warmup_case_features(
        train_valid_case_ids,
        num_workers=args.num_workers,
        progress_desc="train+valid",
    )
    test_store.warmup_case_features(
        test_case_ids,
        num_workers=args.num_workers,
        progress_desc="test",
    )
    train_valid_store.warmup_chunk_embeddings(
        train_valid_case_ids,
        progress_desc="train+valid",
        case_batch_size=args.chunk_warmup_case_batch_size,
    )
    test_store.warmup_chunk_embeddings(
        test_case_ids,
        progress_desc="test",
        case_batch_size=args.chunk_warmup_case_batch_size,
    )

    train_feature_csv = args.output_dir / "train_features.csv"
    valid_feature_csv = args.output_dir / "valid_features.csv"
    test_feature_csv = args.output_dir / "test_features.csv"

    logger.info("Building train features...")
    _build_split_rows(
        split_name="train",
        query_ids=train_qids,
        scope=train_valid_scope,
        labels=train_labels,
        store=train_valid_store,
        bm25_scorer=bm25_scorer_train,
        qld_scorer=qld_scorer_train,
        bm25_ngram_scorer=bm25_ngram_scorer_train,
        output_csv=train_feature_csv,
        logger=logger,
        max_queries=args.max_queries,
        num_workers=args.num_workers,
        lexical_prefetch_batch_size=args.lexical_prefetch_batch_size,
        feature_compute_device=device,
        feature_score_batch_size=args.feature_score_batch_size,
    )

    logger.info("Building valid features...")
    _build_split_rows(
        split_name="valid",
        query_ids=valid_qids,
        scope=train_valid_scope,
        labels=valid_labels,
        store=train_valid_store,
        bm25_scorer=bm25_scorer_train,
        qld_scorer=qld_scorer_train,
        bm25_ngram_scorer=bm25_ngram_scorer_train,
        output_csv=valid_feature_csv,
        logger=logger,
        max_queries=args.max_queries,
        num_workers=args.num_workers,
        lexical_prefetch_batch_size=args.lexical_prefetch_batch_size,
        feature_compute_device=device,
        feature_score_batch_size=args.feature_score_batch_size,
    )

    logger.info("Building test features...")
    _build_split_rows(
        split_name="test",
        query_ids=test_qids,
        scope=test_scope,
        labels=None,
        store=test_store,
        bm25_scorer=bm25_scorer_test,
        qld_scorer=qld_scorer_test,
        bm25_ngram_scorer=bm25_ngram_scorer_test,
        output_csv=test_feature_csv,
        logger=logger,
        max_queries=args.max_queries,
        num_workers=args.num_workers,
        lexical_prefetch_batch_size=args.lexical_prefetch_batch_size,
        feature_compute_device=device,
        feature_score_batch_size=args.feature_score_batch_size,
    )

    train_valid_store.dump_year_log(args.output_dir / "year_extraction_train_valid.csv")
    test_store.dump_year_log(args.output_dir / "year_extraction_test.csv")

    logger.info("Train/valid/test LightGBM pipeline...")
    run_train_valid_test(
        train_feature_csv=train_feature_csv,
        valid_feature_csv=valid_feature_csv,
        test_feature_csv=test_feature_csv,
        output_dir=args.output_dir,
        logger=logger,
        num_workers=args.num_workers,
        lgbm_device=lgbm_device,
        valid_scope=train_valid_scope,
        test_scope=test_scope,
    )

    if not args.skip_fixed_topk_export:
        fixed_topk_output_dir = args.fixed_topk_output_dir or (args.output_dir / f"fixed_top{args.fixed_topk_k}")
        test_pred_input = args.output_dir / "test_predictions_raw.csv"
        if not test_pred_input.exists():
            test_pred_input = args.output_dir / "test_predictions.csv"

        logger.info("Running fixed top-%d export...", args.fixed_topk_k)
        fixed_topk_summary = run_fixed_topk_postprocess(
            test_predictions_path=test_pred_input,
            test_scope=test_scope,
            output_dir=fixed_topk_output_dir,
            logger=logger,
            k=args.fixed_topk_k,
            test_query_ids=test_qids,
            remove_self=not args.fixed_topk_keep_self,
            write_submission=not args.fixed_topk_no_submission,
            submission_run_tag=args.fixed_topk_submission_run_tag,
            final_submission_path=args.fixed_topk_final_submission_path,
        )
        logger.info(
            "Fixed top-%d export complete: summary=%s",
            args.fixed_topk_k,
            fixed_topk_output_dir / "fixed_topk_summary.json",
        )
        if "final_submission_path" in fixed_topk_summary:
            logger.info("Fixed top-%d copied submission: %s", args.fixed_topk_k, fixed_topk_summary["final_submission_path"])

    if not args.skip_cutoff_search:
        cutoff_output_dir = args.cutoff_output_dir or (args.output_dir / "cutoff_search")
        cutoff_config = build_cutoff_config(
            config_path=args.cutoff_config_json,
            fixed_k_values=args.cutoff_fixed_k_values,
            ratio_p_values=args.cutoff_ratio_p_values,
            ratio_l_values=args.cutoff_ratio_l_values,
            ratio_h_values=args.cutoff_ratio_h_values,
            gap_n_values=args.cutoff_gap_n_values,
            gap_buffer_values=args.cutoff_gap_buffer_values,
            gap_l_values=args.cutoff_gap_l_values,
            gap_h_values=args.cutoff_gap_h_values,
            remove_self=not args.cutoff_keep_self,
            write_submission=not args.cutoff_no_submission,
            submission_run_tag=args.cutoff_submission_run_tag,
        )
        valid_pred_input = args.output_dir / "valid_predictions_raw.csv"
        test_pred_input = args.output_dir / "test_predictions_raw.csv"
        if not valid_pred_input.exists():
            valid_pred_input = args.output_dir / "valid_predictions.csv"
        if not test_pred_input.exists():
            test_pred_input = args.output_dir / "test_predictions.csv"

        logger.info("Running cutoff postprocess search...")
        cutoff_summary = run_cutoff_postprocess(
            valid_predictions_path=valid_pred_input,
            test_predictions_path=test_pred_input,
            valid_scope=train_valid_scope,
            test_scope=test_scope,
            output_dir=cutoff_output_dir,
            logger=logger,
            config=cutoff_config,
            valid_query_ids=valid_qids,
            test_query_ids=test_qids,
        )
        logger.info(
            "Cutoff postprocess complete: best_mode=%s best_params=%s summary=%s",
            cutoff_summary["best_mode"],
            json.dumps(cutoff_summary["best_params"], sort_keys=True),
            cutoff_output_dir / "cutoff_summary.json",
        )

    logger.info("Done. Output directory: %s", args.output_dir)


if __name__ == "__main__":
    main()
