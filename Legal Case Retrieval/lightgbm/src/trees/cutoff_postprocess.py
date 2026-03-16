from __future__ import annotations

import argparse
import itertools
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

DEFAULT_FIXED_TOPK_VALUES = (1, 3, 5, 7, 10)
DEFAULT_RATIO_P_VALUES = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
DEFAULT_RATIO_L_VALUES = (1, 2, 3)
DEFAULT_RATIO_H_VALUES = (5, 7, 10)
DEFAULT_GAP_N_VALUES = (10, 20, 30)
DEFAULT_GAP_BUFFER_VALUES = (0, 1, 2)
DEFAULT_GAP_L_VALUES = (1, 2, 3)
DEFAULT_GAP_H_VALUES = (5, 7, 10)

NDCG_DISCOUNTS_10 = 1.0 / np.log2(np.arange(2, 12, dtype=np.float64))


@dataclass(frozen=True)
class FixedTopKConfig:
    k_values: tuple[int, ...] = DEFAULT_FIXED_TOPK_VALUES


@dataclass(frozen=True)
class RatioCutoffConfig:
    p_values: tuple[float, ...] = DEFAULT_RATIO_P_VALUES
    l_values: tuple[int, ...] = DEFAULT_RATIO_L_VALUES
    h_values: tuple[int, ...] = DEFAULT_RATIO_H_VALUES


@dataclass(frozen=True)
class LargestGapCutoffConfig:
    N_values: tuple[int, ...] = DEFAULT_GAP_N_VALUES
    buffer_values: tuple[int, ...] = DEFAULT_GAP_BUFFER_VALUES
    l_values: tuple[int, ...] = DEFAULT_GAP_L_VALUES
    h_values: tuple[int, ...] = DEFAULT_GAP_H_VALUES


@dataclass(frozen=True)
class CutoffSearchConfig:
    fixed_topk: FixedTopKConfig = field(default_factory=FixedTopKConfig)
    ratio_cutoff: RatioCutoffConfig = field(default_factory=RatioCutoffConfig)
    largest_gap_cutoff: LargestGapCutoffConfig = field(default_factory=LargestGapCutoffConfig)
    remove_self: bool = True
    write_submission: bool = True
    submission_run_tag: str = "lgbm_cutoff"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass(frozen=True)
class QueryRanking:
    query_id: str
    frame: pd.DataFrame
    candidate_ids: np.ndarray
    rank_scores: np.ndarray
    labels: np.ndarray | None
    total_relevant: int
    cumulative_relevant: np.ndarray | None

    @property
    def size(self) -> int:
        return int(self.rank_scores.shape[0])


@dataclass(frozen=True)
class ModeSearchResult:
    mode_name: str
    best_params: dict[str, Any]
    best_metrics: dict[str, Any]
    all_results: pd.DataFrame
    best_validation_predictions: pd.DataFrame
    best_validation_query_stats: pd.DataFrame
    best_test_predictions: pd.DataFrame | None = None
    best_test_query_stats: pd.DataFrame | None = None


def normalize_case_id(raw_id: object) -> str:
    case_id = str(raw_id).strip()
    if case_id.endswith(".txt"):
        case_id = case_id[:-4]
    if case_id.isdigit():
        case_id = case_id.zfill(6)
    return case_id


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


def _parse_int_list(raw_value: str | None, default: Sequence[int]) -> tuple[int, ...]:
    if not raw_value:
        return tuple(int(v) for v in default)
    values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return tuple(values)


def _parse_float_list(raw_value: str | None, default: Sequence[float]) -> tuple[float, ...]:
    if not raw_value:
        return tuple(float(v) for v in default)
    values = [float(part.strip()) for part in raw_value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return tuple(values)


def _load_config_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Cutoff config JSON must be an object: {path}")
    return payload


def build_cutoff_config(
    *,
    config_path: Path | None = None,
    fixed_k_values: str | None = None,
    ratio_p_values: str | None = None,
    ratio_l_values: str | None = None,
    ratio_h_values: str | None = None,
    gap_n_values: str | None = None,
    gap_buffer_values: str | None = None,
    gap_l_values: str | None = None,
    gap_h_values: str | None = None,
    remove_self: bool = True,
    write_submission: bool = True,
    submission_run_tag: str = "lgbm_cutoff",
) -> CutoffSearchConfig:
    payload = _load_config_payload(config_path)

    fixed_payload = payload.get("fixed_topk", {}) if isinstance(payload.get("fixed_topk", {}), dict) else {}
    ratio_payload = payload.get("ratio_cutoff", {}) if isinstance(payload.get("ratio_cutoff", {}), dict) else {}
    gap_payload = payload.get("largest_gap_cutoff", {}) if isinstance(payload.get("largest_gap_cutoff", {}), dict) else {}

    fixed = FixedTopKConfig(
        k_values=_parse_int_list(
            fixed_k_values,
            fixed_payload.get("k_values", DEFAULT_FIXED_TOPK_VALUES),
        )
    )
    ratio = RatioCutoffConfig(
        p_values=_parse_float_list(
            ratio_p_values,
            ratio_payload.get("p_values", DEFAULT_RATIO_P_VALUES),
        ),
        l_values=_parse_int_list(
            ratio_l_values,
            ratio_payload.get("l_values", DEFAULT_RATIO_L_VALUES),
        ),
        h_values=_parse_int_list(
            ratio_h_values,
            ratio_payload.get("h_values", DEFAULT_RATIO_H_VALUES),
        ),
    )
    gap = LargestGapCutoffConfig(
        N_values=_parse_int_list(
            gap_n_values,
            gap_payload.get("N_values", DEFAULT_GAP_N_VALUES),
        ),
        buffer_values=_parse_int_list(
            gap_buffer_values,
            gap_payload.get("buffer_values", DEFAULT_GAP_BUFFER_VALUES),
        ),
        l_values=_parse_int_list(
            gap_l_values,
            gap_payload.get("l_values", DEFAULT_GAP_L_VALUES),
        ),
        h_values=_parse_int_list(
            gap_h_values,
            gap_payload.get("h_values", DEFAULT_GAP_H_VALUES),
        ),
    )

    return CutoffSearchConfig(
        fixed_topk=fixed,
        ratio_cutoff=ratio,
        largest_gap_cutoff=gap,
        remove_self=bool(payload.get("remove_self", remove_self)),
        write_submission=bool(payload.get("write_submission", write_submission)),
        submission_run_tag=str(payload.get("submission_run_tag", submission_run_tag)),
    )


def load_rerank_predictions(path: Path, *, has_label: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    header = pd.read_csv(path, nrows=0)
    dtype_map: dict[str, object] = {}
    for column in header.columns:
        if column in {"query_id", "candidate_id"}:
            dtype_map[column] = str
        elif column == "label":
            dtype_map[column] = np.int8
        elif column in {"pred_rank", "rank_position"}:
            dtype_map[column] = np.int32
        elif column in {"pred_score", "rank_score"}:
            dtype_map[column] = np.float32
    df = pd.read_csv(path, dtype=dtype_map or None)
    rename_map = {}
    if "pred_score" in df.columns and "rank_score" not in df.columns:
        rename_map["pred_score"] = "rank_score"
    if "pred_rank" in df.columns and "rank_position" not in df.columns:
        rename_map["pred_rank"] = "rank_position"
    if rename_map:
        df = df.rename(columns=rename_map)

    required = {"query_id", "candidate_id", "rank_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    if has_label and "label" not in df.columns:
        raise ValueError(f"Validation prediction file must include label column: {path}")

    df["query_id"] = df["query_id"].map(normalize_case_id)
    df["candidate_id"] = df["candidate_id"].map(normalize_case_id)
    if has_label:
        df["label"] = df["label"].fillna(0).astype(np.int8)

    df = df.sort_values(
        ["query_id", "rank_score", "candidate_id"],
        ascending=[True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    df["rank_position"] = df.groupby("query_id", sort=False).cumcount() + 1
    return df


def apply_common_legal_filters(
    df: pd.DataFrame,
    *,
    scope: Mapping[str, Sequence[str]] | None,
    remove_self: bool,
    split_name: str,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    qid_order = list(dict.fromkeys(df["query_id"].tolist()))
    frames: list[pd.DataFrame] = []
    removed_by_scope = 0
    removed_by_self = 0
    missing_scope_queries: list[str] = []
    empty_queries: list[str] = []

    for qid, qdf in df.groupby("query_id", sort=False):
        current = qdf
        if scope is not None:
            allowed = scope.get(qid, scope.get(qid.lstrip("0"), None))
            if allowed is None:
                missing_scope_queries.append(qid)
            else:
                keep_scope = current["candidate_id"].isin(allowed)
                removed_by_scope += int((~keep_scope).sum())
                current = current.loc[keep_scope]

        if remove_self:
            keep_self = current["candidate_id"] != qid
            removed_by_self += int((~keep_self).sum())
            current = current.loc[keep_self]

        if current.empty:
            empty_queries.append(qid)
            continue

        current = current.sort_values(
            ["rank_score", "candidate_id"],
            ascending=[False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        current["rank_position"] = np.arange(1, len(current) + 1, dtype=np.int32)
        frames.append(current)

    if frames:
        filtered = pd.concat(frames, ignore_index=True)
    else:
        filtered = df.iloc[0:0].copy()

    stats = {
        "queries": len(qid_order),
        "rows_before": int(len(df)),
        "rows_after": int(len(filtered)),
        "removed_by_scope": int(removed_by_scope),
        "removed_by_self": int(removed_by_self),
        "missing_scope_queries": len(missing_scope_queries),
        "empty_queries": len(empty_queries),
    }
    logger.info(
        "[%s] legal filter rows_before=%d rows_after=%d removed_scope=%d removed_self=%d missing_scope=%d empty_queries=%d",
        split_name,
        stats["rows_before"],
        stats["rows_after"],
        stats["removed_by_scope"],
        stats["removed_by_self"],
        stats["missing_scope_queries"],
        stats["empty_queries"],
    )
    return filtered, stats, qid_order


def build_query_rankings(
    df: pd.DataFrame,
    *,
    all_query_ids: Sequence[str],
    has_label: bool,
) -> list[QueryRanking]:
    grouped = {qid: qdf for qid, qdf in df.groupby("query_id", sort=False)}
    rankings: list[QueryRanking] = []

    for raw_qid in all_query_ids:
        qid = normalize_case_id(raw_qid)
        qdf = grouped.get(qid)
        if qdf is None:
            empty_frame = df.iloc[0:0].copy()
            rankings.append(
                QueryRanking(
                    query_id=qid,
                    frame=empty_frame,
                    candidate_ids=np.asarray([], dtype=object),
                    rank_scores=np.asarray([], dtype=np.float32),
                    labels=np.asarray([], dtype=np.int8) if has_label else None,
                    total_relevant=0,
                    cumulative_relevant=np.asarray([], dtype=np.int32) if has_label else None,
                )
            )
            continue

        candidate_ids = qdf["candidate_id"].astype(str).to_numpy(copy=False)
        rank_scores = qdf["rank_score"].to_numpy(dtype=np.float32, copy=False)
        if has_label:
            labels = qdf["label"].to_numpy(dtype=np.int8, copy=False)
            cumulative_relevant = np.cumsum(labels.astype(np.int32, copy=False))
            total_relevant = int(labels.sum())
        else:
            labels = None
            cumulative_relevant = None
            total_relevant = 0

        rankings.append(
            QueryRanking(
                query_id=qid,
                frame=qdf.reset_index(drop=True),
                candidate_ids=np.asarray(candidate_ids, dtype=object),
                rank_scores=rank_scores,
                labels=labels,
                total_relevant=total_relevant,
                cumulative_relevant=cumulative_relevant,
            )
        )
    return rankings


def apply_fixed_topk(df_query: pd.DataFrame, k: int) -> pd.DataFrame:
    if k <= 0 or df_query.empty:
        return df_query.iloc[0:0].copy()
    return df_query.sort_values(
        ["rank_score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(int(k)).copy()


def apply_ratio_cutoff(df_query: pd.DataFrame, p: float, l: int, h: int) -> pd.DataFrame:
    ordered = df_query.sort_values(
        ["rank_score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    k = _compute_ratio_cutoff_k(ordered["rank_score"].to_numpy(dtype=np.float32, copy=False), p=p, l=l, h=h)
    return ordered.head(k).copy()


def apply_largest_gap_cutoff(df_query: pd.DataFrame, N: int, buffer: int, l: int, h: int) -> pd.DataFrame:
    ordered = df_query.sort_values(
        ["rank_score", "candidate_id"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    k = _compute_largest_gap_cutoff_k(
        ordered["rank_score"].to_numpy(dtype=np.float32, copy=False),
        N=N,
        buffer=buffer,
        l=l,
        h=h,
    )
    return ordered.head(k).copy()


def _compute_ratio_cutoff_k(scores: np.ndarray, *, p: float, l: int, h: int) -> int:
    size = int(scores.shape[0])
    if size <= 0:
        return 0

    l = max(int(l), 1)
    h = max(int(h), l)
    capped = min(size, h)
    if capped <= 0:
        return 0

    top_scores = scores[:capped]
    top1 = float(top_scores[0])
    if top1 > 0.0:
        threshold = p * top1
    elif top1 < 0.0:
        threshold = top1 / p
    else:
        threshold = 0.0

    selected = int(np.count_nonzero(top_scores >= threshold))
    if selected < l:
        selected = min(l, size)
    if selected > h:
        selected = h
    return min(selected, size)


def _compute_largest_gap_cutoff_k(scores: np.ndarray, *, N: int, buffer: int, l: int, h: int) -> int:
    size = int(scores.shape[0])
    if size <= 0:
        return 0

    l = max(int(l), 1)
    h = max(int(h), l)
    if size < 2:
        return min(size, max(l, 1))

    analysis_n = max(int(N), 1)
    analysis_n = min(analysis_n, size)
    window = scores[:analysis_n]
    if window.shape[0] < 2:
        base_k = 1
    else:
        gaps = window[:-1] - window[1:]
        max_index = int(np.argmax(gaps))
        base_k = max_index + 1

    selected = base_k + max(int(buffer), 0)
    selected = max(selected, l)
    selected = min(selected, h)
    return min(selected, size)


def _ndcg_at_k(labels: np.ndarray | None, k: int, total_relevant: int) -> float:
    if labels is None or k <= 0 or total_relevant <= 0:
        return 0.0
    cutoff = min(int(k), 10, int(labels.shape[0]))
    if cutoff <= 0:
        return 0.0
    gains = labels[:cutoff].astype(np.float64, copy=False)
    dcg = float(np.sum(gains * NDCG_DISCOUNTS_10[:cutoff]))
    ideal_cutoff = min(int(total_relevant), 10)
    if ideal_cutoff <= 0:
        return 0.0
    ideal = float(np.sum(np.ones(ideal_cutoff, dtype=np.float64) * NDCG_DISCOUNTS_10[:ideal_cutoff]))
    return dcg / ideal if ideal > 0.0 else 0.0


def _evaluate_k_predictions(
    rankings: Sequence[QueryRanking],
    selected_k: Mapping[str, int],
) -> tuple[dict[str, Any], pd.DataFrame]:
    query_stats_rows: list[dict[str, Any]] = []
    total_tp = 0
    total_predicted = 0
    total_relevant = 0
    sum_p_at_5 = 0.0
    sum_r_at_5 = 0.0
    sum_ndcg_at_10 = 0.0

    for ranking in rankings:
        k = max(int(selected_k.get(ranking.query_id, 0)), 0)
        k = min(k, ranking.size)
        total_predicted += k
        tp = 0
        rel_at_5 = 0
        if ranking.labels is not None and ranking.cumulative_relevant is not None and ranking.size > 0:
            if k > 0:
                tp = int(ranking.cumulative_relevant[k - 1])
                rel_cutoff = min(k, 5)
                rel_at_5 = int(ranking.cumulative_relevant[rel_cutoff - 1]) if rel_cutoff > 0 else 0
            total_relevant += ranking.total_relevant
            total_tp += tp
            sum_p_at_5 += rel_at_5 / 5.0
            sum_r_at_5 += (rel_at_5 / ranking.total_relevant) if ranking.total_relevant > 0 else 0.0
            sum_ndcg_at_10 += _ndcg_at_k(ranking.labels, k=10 if k >= 10 else k, total_relevant=ranking.total_relevant)

        query_stats_rows.append(
            {
                "query_id": ranking.query_id,
                "num_candidates": ranking.size,
                "retained_count": k,
                "retained_ratio": (k / ranking.size) if ranking.size > 0 else 0.0,
                "num_relevant": ranking.total_relevant,
                "true_positive": tp,
            }
        )

    total_fp = total_predicted - total_tp
    total_fn = total_relevant - total_tp
    precision = total_tp / total_predicted if total_predicted else 0.0
    recall = total_tp / total_relevant if total_relevant else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    query_count = max(len(rankings), 1)
    query_stats = pd.DataFrame(query_stats_rows)
    retained_values = query_stats["retained_count"].to_numpy(dtype=np.int32, copy=False) if not query_stats.empty else np.asarray([], dtype=np.int32)

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "p_at_5": float(sum_p_at_5 / query_count),
        "r_at_5": float(sum_r_at_5 / query_count),
        "ndcg_at_10": float(sum_ndcg_at_10 / query_count),
        "true_positive": int(total_tp),
        "false_positive": int(total_fp),
        "false_negative": int(total_fn),
        "predicted_total": int(total_predicted),
        "relevant_total": int(total_relevant),
        "query_count": int(len(rankings)),
        "avg_retained": float(retained_values.mean()) if retained_values.size else 0.0,
        "median_retained": float(np.median(retained_values)) if retained_values.size else 0.0,
        "min_retained": int(retained_values.min()) if retained_values.size else 0,
        "max_retained": int(retained_values.max()) if retained_values.size else 0,
    }
    return metrics, query_stats


def _build_selected_prediction_frame(
    rankings: Sequence[QueryRanking],
    selected_k: Mapping[str, int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for ranking in rankings:
        k = max(int(selected_k.get(ranking.query_id, 0)), 0)
        k = min(k, ranking.size)
        if k <= 0:
            continue
        selected = ranking.frame.head(k).copy()
        selected["cutoff_k"] = k
        selected["selected_rank"] = np.arange(1, k + 1, dtype=np.int32)
        frames.append(selected)

    if not frames:
        return pd.DataFrame(columns=["query_id", "candidate_id", "rank_score", "rank_position", "cutoff_k", "selected_rank"])

    selected_df = pd.concat(frames, ignore_index=True)
    selected_df = selected_df.sort_values(
        ["query_id", "selected_rank", "candidate_id"],
        ascending=[True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return selected_df


def _format_params(params: Mapping[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=True)


def _select_best_record(df: pd.DataFrame, mode_name: str) -> pd.Series:
    if df.empty:
        raise ValueError(f"No validation results available for mode: {mode_name}")

    def row_key(row: pd.Series) -> tuple[Any, ...]:
        base = (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
        )
        if mode_name == "fixed_topk":
            tiebreak = (-int(row["k"]),)
        elif mode_name == "ratio_cutoff":
            tiebreak = (-int(row["h"]), -int(row["l"]), -float(row["p"]))
        elif mode_name == "largest_gap_cutoff":
            tiebreak = (-int(row["h"]), -int(row["buffer"]), -int(row["N"]), -int(row["l"]))
        else:
            tiebreak = ()
        return base + tiebreak + (-float(row["avg_retained"]),) + (str(row["params_json"]),)

    best_idx = max(df.index, key=lambda idx: row_key(df.loc[idx]))
    return df.loc[best_idx]


def evaluate_cutoff_mode(
    validation_rankings: Sequence[QueryRanking],
    *,
    mode_name: str,
    param_grid: Sequence[dict[str, Any]],
    cutoff_k_fn: Callable[..., int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    per_param_query_stats: dict[str, pd.DataFrame] = {}

    total = len(param_grid)
    logger.info("[%s] grid search start: %d parameter settings", mode_name, total)

    for idx, params in enumerate(param_grid, start=1):
        selected_k = {
            ranking.query_id: int(cutoff_k_fn(ranking.rank_scores, **params))
            for ranking in validation_rankings
        }
        metrics, query_stats = _evaluate_k_predictions(validation_rankings, selected_k)
        record = {
            "mode_name": mode_name,
            **params,
            **metrics,
            "params_json": _format_params(params),
        }
        records.append(record)
        per_param_query_stats[record["params_json"]] = query_stats
        logger.info(
            "[%s] %d/%d params=%s -> F1=%.6f Precision=%.6f Recall=%.6f nDCG@10=%.6f P@5=%.6f R@5=%.6f",
            mode_name,
            idx,
            total,
            record["params_json"],
            record["f1"],
            record["precision"],
            record["recall"],
            record["ndcg_at_10"],
            record["p_at_5"],
            record["r_at_5"],
        )

    results_df = pd.DataFrame(records)
    best_row = _select_best_record(results_df, mode_name=mode_name)
    best_params = json.loads(str(best_row["params_json"]))
    best_query_stats = per_param_query_stats[str(best_row["params_json"])]
    best_selected_k = {
        ranking.query_id: int(cutoff_k_fn(ranking.rank_scores, **best_params))
        for ranking in validation_rankings
    }
    best_predictions = _build_selected_prediction_frame(validation_rankings, best_selected_k)
    best_metrics = {column: best_row[column] for column in results_df.columns if column not in {"mode_name", "params_json"} and column not in best_params}
    best_metrics["params_json"] = str(best_row["params_json"])
    return results_df, best_params, best_predictions, best_query_stats


def search_best_fixed_topk(
    validation_rankings: Sequence[QueryRanking],
    *,
    k_values: Sequence[int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    param_grid = [{"k": int(k)} for k in sorted({int(k) for k in k_values if int(k) > 0})]
    return evaluate_cutoff_mode(
        validation_rankings,
        mode_name="fixed_topk",
        param_grid=param_grid,
        cutoff_k_fn=lambda scores, k: min(max(int(k), 0), int(scores.shape[0])),
        logger=logger,
    )


def search_best_ratio_cutoff(
    validation_rankings: Sequence[QueryRanking],
    *,
    p_values: Sequence[float],
    l_values: Sequence[int],
    h_values: Sequence[int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    param_grid = [
        {"p": float(p), "l": int(l), "h": int(h)}
        for p, l, h in itertools.product(sorted(set(float(v) for v in p_values)), sorted(set(int(v) for v in l_values)), sorted(set(int(v) for v in h_values)))
        if int(l) <= int(h) and int(l) > 0
    ]
    return evaluate_cutoff_mode(
        validation_rankings,
        mode_name="ratio_cutoff",
        param_grid=param_grid,
        cutoff_k_fn=lambda scores, p, l, h: _compute_ratio_cutoff_k(scores, p=p, l=l, h=h),
        logger=logger,
    )


def search_best_largest_gap_cutoff(
    validation_rankings: Sequence[QueryRanking],
    *,
    N_values: Sequence[int],
    buffer_values: Sequence[int],
    l_values: Sequence[int],
    h_values: Sequence[int],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    param_grid = [
        {"N": int(N), "buffer": int(buffer), "l": int(l), "h": int(h)}
        for N, buffer, l, h in itertools.product(
            sorted(set(int(v) for v in N_values)),
            sorted(set(int(v) for v in buffer_values)),
            sorted(set(int(v) for v in l_values)),
            sorted(set(int(v) for v in h_values)),
        )
        if int(l) <= int(h) and int(l) > 0 and int(N) > 0
    ]
    return evaluate_cutoff_mode(
        validation_rankings,
        mode_name="largest_gap_cutoff",
        param_grid=param_grid,
        cutoff_k_fn=lambda scores, N, buffer, l, h: _compute_largest_gap_cutoff_k(scores, N=N, buffer=buffer, l=l, h=h),
        logger=logger,
    )


def _best_mode_sort_key(row: pd.Series) -> tuple[Any, ...]:
    return (
        float(row["f1"]),
        float(row["recall"]),
        float(row["precision"]),
        -float(row["avg_retained"]),
        str(row["mode_name"]),
        str(row["params_json"]),
    )


def apply_best_cutoff_to_test(
    test_rankings: Sequence[QueryRanking],
    *,
    best_mode: str,
    best_params: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if best_mode == "fixed_topk":
        cutoff_k_fn = lambda scores: min(max(int(best_params["k"]), 0), int(scores.shape[0]))
    elif best_mode == "ratio_cutoff":
        cutoff_k_fn = lambda scores: _compute_ratio_cutoff_k(
            scores,
            p=float(best_params["p"]),
            l=int(best_params["l"]),
            h=int(best_params["h"]),
        )
    elif best_mode == "largest_gap_cutoff":
        cutoff_k_fn = lambda scores: _compute_largest_gap_cutoff_k(
            scores,
            N=int(best_params["N"]),
            buffer=int(best_params["buffer"]),
            l=int(best_params["l"]),
            h=int(best_params["h"]),
        )
    else:
        raise ValueError(f"Unsupported cutoff mode: {best_mode}")

    selected_k = {ranking.query_id: int(cutoff_k_fn(ranking.rank_scores)) for ranking in test_rankings}
    _, query_stats = _evaluate_k_predictions(test_rankings, selected_k)
    predictions = _build_selected_prediction_frame(test_rankings, selected_k)
    return predictions, query_stats


def build_submission_from_cutoff_results(
    test_predictions: pd.DataFrame,
    *,
    output_path: Path,
    run_tag: str,
    query_order: Sequence[str] | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if query_order is None:
        query_order = list(dict.fromkeys(test_predictions["query_id"].tolist()))

    groups = {qid: qdf for qid, qdf in test_predictions.groupby("query_id", sort=False)}
    with output_path.open("w", encoding="utf-8") as fout:
        for raw_qid in query_order:
            qid = normalize_case_id(raw_qid)
            qdf = groups.get(qid)
            if qdf is None:
                continue
            for _, row in qdf.sort_values("selected_rank", ascending=True, kind="mergesort").iterrows():
                fout.write(f"{qid} {row['candidate_id']} {run_tag}\n")
    return output_path


def _write_candidate_json(df: pd.DataFrame, output_path: Path) -> Path:
    payload: dict[str, list[str]] = {}
    for qid, qdf in df.groupby("query_id", sort=False):
        payload[qid] = qdf.sort_values("selected_rank", ascending=True, kind="mergesort")["candidate_id"].astype(str).tolist()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def compare_all_cutoff_modes(
    validation_rankings: Sequence[QueryRanking],
    *,
    config: CutoffSearchConfig,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, ModeSearchResult], pd.Series]:
    fixed_results, fixed_params, fixed_predictions, fixed_query_stats = search_best_fixed_topk(
        validation_rankings,
        k_values=config.fixed_topk.k_values,
        logger=logger,
    )
    ratio_results, ratio_params, ratio_predictions, ratio_query_stats = search_best_ratio_cutoff(
        validation_rankings,
        p_values=config.ratio_cutoff.p_values,
        l_values=config.ratio_cutoff.l_values,
        h_values=config.ratio_cutoff.h_values,
        logger=logger,
    )
    gap_results, gap_params, gap_predictions, gap_query_stats = search_best_largest_gap_cutoff(
        validation_rankings,
        N_values=config.largest_gap_cutoff.N_values,
        buffer_values=config.largest_gap_cutoff.buffer_values,
        l_values=config.largest_gap_cutoff.l_values,
        h_values=config.largest_gap_cutoff.h_values,
        logger=logger,
    )

    mode_results: dict[str, ModeSearchResult] = {}
    for mode_name, results_df, best_params, best_predictions, best_query_stats in [
        ("fixed_topk", fixed_results, fixed_params, fixed_predictions, fixed_query_stats),
        ("ratio_cutoff", ratio_results, ratio_params, ratio_predictions, ratio_query_stats),
        ("largest_gap_cutoff", gap_results, gap_params, gap_predictions, gap_query_stats),
    ]:
        best_row = _select_best_record(results_df, mode_name=mode_name)
        best_metrics = best_row.to_dict()
        mode_results[mode_name] = ModeSearchResult(
            mode_name=mode_name,
            best_params=best_params,
            best_metrics=best_metrics,
            all_results=results_df,
            best_validation_predictions=best_predictions,
            best_validation_query_stats=best_query_stats,
        )

    comparison_df = pd.DataFrame([result.best_metrics for result in mode_results.values()])
    comparison_df = comparison_df.sort_values(
        ["f1", "recall", "precision", "avg_retained", "mode_name"],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    best_row = comparison_df.iloc[max(range(len(comparison_df)), key=lambda idx: _best_mode_sort_key(comparison_df.iloc[idx]))]
    return comparison_df, mode_results, best_row


def run_cutoff_postprocess(
    *,
    valid_predictions_path: Path,
    test_predictions_path: Path,
    valid_scope: Mapping[str, Sequence[str]] | None,
    test_scope: Mapping[str, Sequence[str]] | None,
    output_dir: Path,
    logger: logging.Logger,
    config: CutoffSearchConfig,
    valid_query_ids: Sequence[str] | None = None,
    test_query_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_df_raw = load_rerank_predictions(valid_predictions_path, has_label=True)
    test_df_raw = load_rerank_predictions(test_predictions_path, has_label=False)

    valid_filtered_df, valid_filter_stats, inferred_valid_qids = apply_common_legal_filters(
        valid_df_raw,
        scope=valid_scope,
        remove_self=config.remove_self,
        split_name="valid",
        logger=logger,
    )
    test_filtered_df, test_filter_stats, inferred_test_qids = apply_common_legal_filters(
        test_df_raw,
        scope=test_scope,
        remove_self=config.remove_self,
        split_name="test",
        logger=logger,
    )

    valid_qids = [normalize_case_id(qid) for qid in (valid_query_ids or inferred_valid_qids)]
    test_qids = [normalize_case_id(qid) for qid in (test_query_ids or inferred_test_qids)]

    valid_filtered_path = output_dir / "valid_predictions_legal_filtered.csv"
    test_filtered_path = output_dir / "test_predictions_legal_filtered.csv"
    valid_filtered_df.to_csv(valid_filtered_path, index=False)
    test_filtered_df.to_csv(test_filtered_path, index=False)

    validation_rankings = build_query_rankings(
        valid_filtered_df,
        all_query_ids=valid_qids,
        has_label=True,
    )
    test_rankings = build_query_rankings(
        test_filtered_df,
        all_query_ids=test_qids,
        has_label=False,
    )

    comparison_df, mode_results, best_row = compare_all_cutoff_modes(
        validation_rankings,
        config=config,
        logger=logger,
    )

    comparison_path = output_dir / "validation_mode_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    mode_summaries: list[dict[str, Any]] = []
    for mode_name, result in mode_results.items():
        mode_dir = output_dir / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)
        result.all_results.to_csv(mode_dir / "validation_grid_search.csv", index=False)
        result.best_validation_predictions.to_csv(mode_dir / "validation_best_predictions.csv", index=False)
        result.best_validation_query_stats.to_csv(mode_dir / "validation_best_query_stats.csv", index=False)

        test_predictions, test_query_stats = apply_best_cutoff_to_test(
            test_rankings,
            best_mode=mode_name,
            best_params=result.best_params,
        )
        test_predictions.to_csv(mode_dir / "test_best_predictions.csv", index=False)
        test_query_stats.to_csv(mode_dir / "test_best_query_stats.csv", index=False)
        _write_candidate_json(test_predictions, mode_dir / "test_best_candidates.json")
        if config.write_submission:
            build_submission_from_cutoff_results(
                test_predictions,
                output_path=mode_dir / "test_best_submission.txt",
                run_tag=f"{config.submission_run_tag}_{mode_name}",
                query_order=test_qids,
            )

        updated_result = ModeSearchResult(
            mode_name=result.mode_name,
            best_params=result.best_params,
            best_metrics=result.best_metrics,
            all_results=result.all_results,
            best_validation_predictions=result.best_validation_predictions,
            best_validation_query_stats=result.best_validation_query_stats,
            best_test_predictions=test_predictions,
            best_test_query_stats=test_query_stats,
        )
        mode_results[mode_name] = updated_result
        summary = dict(updated_result.best_metrics)
        summary["mode_name"] = mode_name
        summary["best_params"] = updated_result.best_params
        summary["validation_best_predictions_path"] = str(mode_dir / "validation_best_predictions.csv")
        summary["test_best_predictions_path"] = str(mode_dir / "test_best_predictions.csv")
        mode_summaries.append(summary)

    best_mode = str(best_row["mode_name"])
    best_params = json.loads(str(best_row["params_json"]))
    best_test_predictions = mode_results[best_mode].best_test_predictions
    best_test_query_stats = mode_results[best_mode].best_test_query_stats
    if best_test_predictions is None or best_test_query_stats is None:
        raise RuntimeError(f"Best mode test predictions missing for {best_mode}")

    best_dir = output_dir / "best_overall"
    best_dir.mkdir(parents=True, exist_ok=True)
    best_test_predictions.to_csv(best_dir / "test_predictions_best_mode.csv", index=False)
    best_test_query_stats.to_csv(best_dir / "test_query_stats_best_mode.csv", index=False)
    _write_candidate_json(best_test_predictions, best_dir / "test_candidates_best_mode.json")
    if config.write_submission:
        build_submission_from_cutoff_results(
            best_test_predictions,
            output_path=best_dir / "test_submission_best_mode.txt",
            run_tag=config.submission_run_tag,
            query_order=test_qids,
        )

    summary_payload = {
        "config": config.to_dict(),
        "valid_filter_stats": valid_filter_stats,
        "test_filter_stats": test_filter_stats,
        "validation_mode_comparison_path": str(comparison_path),
        "mode_summaries": mode_summaries,
        "best_mode": best_mode,
        "best_params": best_params,
        "best_validation_metrics": best_row.to_dict(),
        "best_overall_test_predictions_path": str(best_dir / "test_predictions_best_mode.csv"),
        "best_overall_test_candidates_path": str(best_dir / "test_candidates_best_mode.json"),
    }
    if config.write_submission:
        summary_payload["best_overall_submission_path"] = str(best_dir / "test_submission_best_mode.txt")

    summary_path = output_dir / "cutoff_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Cutoff postprocess best mode=%s params=%s F1=%.6f Precision=%.6f Recall=%.6f nDCG@10=%.6f",
        best_mode,
        _format_params(best_params),
        float(best_row["f1"]),
        float(best_row["precision"]),
        float(best_row["recall"]),
        float(best_row["ndcg_at_10"]),
    )
    return summary_payload


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search cutoff / top-k post-processing strategies for LightGBM rerank outputs.")
    parser.add_argument("--valid-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--valid-scope", type=Path, default=None)
    parser.add_argument("--test-scope", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--valid-qid", type=Path, default=None)
    parser.add_argument("--test-qid", type=Path, default=None)
    parser.add_argument("--cutoff-config-json", type=Path, default=None)
    parser.add_argument("--fixed-k-values", type=str, default=None)
    parser.add_argument("--ratio-p-values", type=str, default=None)
    parser.add_argument("--ratio-l-values", type=str, default=None)
    parser.add_argument("--ratio-h-values", type=str, default=None)
    parser.add_argument("--gap-n-values", type=str, default=None)
    parser.add_argument("--gap-buffer-values", type=str, default=None)
    parser.add_argument("--gap-l-values", type=str, default=None)
    parser.add_argument("--gap-h-values", type=str, default=None)
    parser.add_argument("--keep-self", action="store_true", help="Disable query self-removal.")
    parser.add_argument("--no-submission", action="store_true", help="Skip submission file output.")
    parser.add_argument("--submission-run-tag", type=str, default="lgbm_cutoff")
    return parser


def _load_qids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"QID file not found: {path}")
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


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("cutoff_postprocess")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logger = _build_logger()

    valid_scope = load_scope(args.valid_scope) if args.valid_scope else None
    test_scope = load_scope(args.test_scope) if args.test_scope else None
    config = build_cutoff_config(
        config_path=args.cutoff_config_json,
        fixed_k_values=args.fixed_k_values,
        ratio_p_values=args.ratio_p_values,
        ratio_l_values=args.ratio_l_values,
        ratio_h_values=args.ratio_h_values,
        gap_n_values=args.gap_n_values,
        gap_buffer_values=args.gap_buffer_values,
        gap_l_values=args.gap_l_values,
        gap_h_values=args.gap_h_values,
        remove_self=not args.keep_self,
        write_submission=not args.no_submission,
        submission_run_tag=args.submission_run_tag,
    )

    run_cutoff_postprocess(
        valid_predictions_path=args.valid_predictions,
        test_predictions_path=args.test_predictions,
        valid_scope=valid_scope,
        test_scope=test_scope,
        output_dir=args.output_dir,
        logger=logger,
        config=config,
        valid_query_ids=_load_qids(args.valid_qid),
        test_query_ids=_load_qids(args.test_qid),
    )


if __name__ == "__main__":
    main()
