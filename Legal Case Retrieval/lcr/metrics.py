from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from .data import load_query_candidate_scope, normalize_query_candidate_scope


def classification_report(
    label_sets: Sequence[Iterable[int]],
    predicted_sets: Sequence[Iterable[int]],
) -> tuple[float, float, float]:
    """
    Compute F1 / precision / recall for multi-label retrieval results.

    The original implementation lived in ``utils/eval.py`` under the name
    ``my_classification_report``.  The behaviour is kept identical but now
    exposed under a clearer name.
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for labels, predicted in zip(label_sets, predicted_sets):
        labels = set(labels)
        predicted = list(predicted)
        for label in labels:
            if label in predicted:
                true_positive += 1
            else:
                false_negative += 1
        for answer in predicted:
            if answer not in labels:
                false_positive += 1

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) else 0.0

    return f1, precision, recall


# Backwards compatible alias
my_classification_report = classification_report


def trec_file_to_dict(trec_path: str | Path, topk: int, skip_self: bool = True) -> Dict[int, List[int]]:
    """
    Parse a TREC-formatted ranking file into a dictionary mapping query IDs to
    the top-k predicted document IDs.
    """
    path = Path(trec_path)
    trec_dict: Dict[int, List[int]] = {}
    with path.open("r", encoding="utf-8") as trec_file:
        for line in trec_file:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid = int(parts[0])
            pid = int(parts[2])

            if skip_self and pid == qid:
                continue
            entries = trec_dict.setdefault(qid, [])
            if len(entries) < topk:
                entries.append(pid)
    return trec_dict


def rel_file_to_dict(rel_path: str | Path, query_id_path: str | Path) -> Dict[int, List[int]]:
    """
    Convert the organiser-provided JSON relevance labels into a dictionary
    filtered by the queries listed in ``query_id_path``.
    """
    query_ids = {int(qid.split()[0]) for qid in Path(query_id_path).read_text(encoding="utf-8").splitlines() if qid.strip()}

    with Path(rel_path).open("r", encoding="utf-8") as rel_file:
        label_dict = json.load(rel_file)

    rel_dict: Dict[int, List[int]] = {}
    for qid_str, label_list in label_dict.items():
        qid = int(qid_str.split(".")[0])
        if qid not in query_ids:
            continue
        unique_labels = {label.split(".")[0] for label in label_list}
        rel_dict[qid] = [int(pid) for pid in unique_labels]
    return rel_dict


def random_guess_predictions(
    rel_dict: Dict[int, List[int]],
    *,
    topk: int = 5,
    seed: int = 42,
    query_candidate_scope: Mapping[object, Sequence[object]] | None = None,
    query_candidate_scope_path: str | Path | None = None,
    zero_pad_width: int = 6,
) -> tuple[List[List[int]], Dict[str, int]]:
    """
    Generate deterministic random-guess predictions for each query.

    When ``query_candidate_scope`` or ``query_candidate_scope_path`` is
    provided, sampling is restricted to that query's allowed candidates.  This
    is useful for scope-filtered baselines such as year-filtered judgment
    pools.
    """
    rng = random.Random(seed)
    resolved_scope: dict[str, List[str]] | None = None
    if query_candidate_scope is not None:
        resolved_scope = normalize_query_candidate_scope(query_candidate_scope)
    elif query_candidate_scope_path is not None:
        resolved_scope = load_query_candidate_scope(query_candidate_scope_path)

    predicted_sets: List[List[int]] = []
    stats = {
        "missing_scope_queries": 0,
        "empty_candidate_pool_queries": 0,
    }

    if resolved_scope is None:
        all_doc_ids = sorted({pid for pids in rel_dict.values() for pid in pids})
        if not all_doc_ids:
            return predicted_sets, stats

        for qid in rel_dict.keys():
            pool = [pid for pid in all_doc_ids if pid != qid]
            if len(pool) <= topk:
                predicted_sets.append(pool)
            else:
                predicted_sets.append(rng.sample(pool, topk))
        return predicted_sets, stats

    for qid in rel_dict.keys():
        raw_key = str(qid)
        padded_key = raw_key.zfill(zero_pad_width) if zero_pad_width > 0 and raw_key.isdigit() else raw_key
        raw_candidates = resolved_scope.get(padded_key)
        scope_missing = False
        if raw_candidates is None:
            raw_candidates = resolved_scope.get(raw_key)
        if raw_candidates is None:
            scope_missing = True
            stats["missing_scope_queries"] += 1
            raw_candidates = []

        pool: List[int] = []
        seen: set[int] = set()
        for raw_doc_id in raw_candidates:
            docid = str(raw_doc_id).replace(".txt", "").strip()
            if not docid.isdigit():
                continue
            pid = int(docid)
            if pid == qid or pid in seen:
                continue
            seen.add(pid)
            pool.append(pid)

        if not pool and not scope_missing:
            stats["empty_candidate_pool_queries"] += 1
        if len(pool) <= topk:
            predicted_sets.append(pool)
        else:
            predicted_sets.append(rng.sample(pool, topk))

    return predicted_sets, stats


def random_guess_baseline(
    rel_dict: Dict[int, List[int]],
    topk: int = 5,
    seed: int = 42,
    query_candidate_scope: Mapping[object, Sequence[object]] | None = None,
    query_candidate_scope_path: str | Path | None = None,
    zero_pad_width: int = 6,
) -> tuple[float, float, float]:
    """
    Sample ``topk`` documents at random (excluding the query id) and compute
    the resulting metrics.  Useful as a sanity-check baseline.

    If query-specific candidate scope is supplied, random sampling is limited
    to that scope instead of the global pool.
    """
    label_sets: List[List[int]] = []
    predicted_sets, _stats = random_guess_predictions(
        rel_dict,
        topk=topk,
        seed=seed,
        query_candidate_scope=query_candidate_scope,
        query_candidate_scope_path=query_candidate_scope_path,
        zero_pad_width=zero_pad_width,
    )

    for labels in rel_dict.values():
        label_sets.append([int(pid) for pid in labels])

    return classification_report(label_sets, predicted_sets)
