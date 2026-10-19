import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.getenv(
    'PYTORCH_CUDA_ALLOC_CONF',
    'expandable_segments:True',
)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import json
import heapq
import math
from torch import nn
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Sequence
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    EarlyStoppingCallback,
    EvalPrediction
)
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
import random
random.seed(289)

# 添加路徑來import自定義模組
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))

from env_utils import load_chunkagg_dotenv
from content_utils import (
    has_minimum_document_content,
    scan_eligible_document_ids,
    summarize_rejected_documents,
)

_LOADED_DOTENV_PATH = load_chunkagg_dotenv(__file__)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_env_path, get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

from lcr.data import load_query_candidate_scope, normalize_case_id
from collections import OrderedDict, defaultdict
from lcr.data import load_query_ids as load_query_ids_from_utils
from lcr.device import get_device
from lcr.metrics import (
    my_classification_report,
    rel_file_to_dict as rel_file_convert,
)
from lcr.retrieval import _build_document_batch, _chunk_single_text, generate_similarity_artifacts

from modernbert_contrastive_model import (
    CHUNK_MICROBATCH_SIZE,
    DOCUMENT_CHUNK_LENGTH,
    MAX_DOCUMENT_CHUNKS,
    ModernBERTContrastive,
)

def _get_env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


# -------------
# QUICK TEST MODE
# -------------
# 中文註解：TASK1_CHUNKAGG_QUICK_TEST=1 時，只抽極少量資料驗證流程。
QUICK_TEST = _get_env_bool("TASK1_CHUNKAGG_QUICK_TEST", False)
# 中文註解：TASK1_RETRIEVAL_BATCH_SIZE 控制 retrieval / adaptive negative sampling 的文件編碼 batch size。
RETRIEVAL_BATCH_SIZE = max(1, _get_env_int("TASK1_RETRIEVAL_BATCH_SIZE", 8))
# 中文註解：TASK1_INIT_TEMPERATURE 是對比式學習初始溫度。
INIT_TEMPERATURE = _get_env_float("TASK1_INIT_TEMPERATURE", 0.2222)
if INIT_TEMPERATURE <= 0:
    raise ValueError(f"TASK1_INIT_TEMPERATURE must be > 0, got: {INIT_TEMPERATURE}")
TEMPERATURE_MIN = _get_env_float("TASK1_CHUNKAGG_TEMPERATURE_MIN", 0.05)
TEMPERATURE_MAX = _get_env_float("TASK1_CHUNKAGG_TEMPERATURE_MAX", 2.0)
EVIDENCE_LOSS_WEIGHT = _get_env_float("TASK1_CHUNKAGG_EVIDENCE_LOSS_WEIGHT", 0.2)
EVIDENCE_POOL_TEMPERATURE = _get_env_float(
    "TASK1_CHUNKAGG_EVIDENCE_POOL_TEMPERATURE",
    0.1,
)
FUSION_RESIDUAL_MAX_NORM = _get_env_float(
    "TASK1_CHUNKAGG_FUSION_RESIDUAL_MAX_NORM",
    0.5,
)
FUSION_RESIDUAL_PENALTY_WEIGHT = _get_env_float(
    "TASK1_CHUNKAGG_FUSION_RESIDUAL_PENALTY_WEIGHT",
    0.01,
)
MIN_DOCUMENT_CONTENT_CHARS = max(
    1,
    _get_env_int("TASK1_CHUNKAGG_MIN_DOCUMENT_CHARS", 100),
)
if FUSION_RESIDUAL_MAX_NORM <= 0:
    raise ValueError("TASK1_CHUNKAGG_FUSION_RESIDUAL_MAX_NORM must be > 0")
if FUSION_RESIDUAL_PENALTY_WEIGHT < 0:
    raise ValueError("TASK1_CHUNKAGG_FUSION_RESIDUAL_PENALTY_WEIGHT must be >= 0")
# 中文註解：TASK1_CHUNKAGG_ENABLE_TF32=1 時，允許 Ampere/Ada GPU 用 TF32 加速 matmul。
ENABLE_TF32 = _get_env_bool("TASK1_CHUNKAGG_ENABLE_TF32", False)
ENABLE_FP16 = _get_env_bool("TASK1_CHUNKAGG_FP16", True)
ENABLE_BF16 = _get_env_bool("TASK1_CHUNKAGG_BF16", False)
if ENABLE_FP16 and ENABLE_BF16:
    raise ValueError("TASK1_CHUNKAGG_FP16 and TASK1_CHUNKAGG_BF16 cannot both be enabled")
RETRIEVAL_AUTOCAST_DTYPE = (
    torch.bfloat16 if ENABLE_BF16 else torch.float16 if ENABLE_FP16 else None
)
# 中文註解：TASK1_CHUNKAGG_TEXT_CACHE_SIZE 控制資料集層的原始文本快取數量。
TEXT_CACHE_SIZE = max(0, _get_env_int("TASK1_CHUNKAGG_TEXT_CACHE_SIZE", 4096))
# 中文註解：TASK1_CHUNKAGG_CHUNK_CACHE_SIZE 控制 collator 層的 chunk tokenization 快取數量。
CHUNK_CACHE_SIZE = max(0, _get_env_int("TASK1_CHUNKAGG_CHUNK_CACHE_SIZE", 1024))
# 中文註解：TASK1_CHUNKAGG_PIN_MEMORY / TASK1_CHUNKAGG_PERSISTENT_WORKERS 控制 dataloader 傳輸效率。
ENABLE_PIN_MEMORY = _get_env_bool("TASK1_CHUNKAGG_PIN_MEMORY", True)
REQUESTED_PERSISTENT_WORKERS = _get_env_bool("TASK1_CHUNKAGG_PERSISTENT_WORKERS", False)
# 中文註解：每筆 query-positive 訓練樣本最多抽多少 scope-filtered negatives。
MAX_NEGATIVES = max(1, _get_env_int("TASK1_CHUNKAGG_MAX_NEGATIVES", _get_env_int("TASK1_HARD_NEG_MAX_NEGATIVES", 15)))
# 中文註解：超過 max chunks 時，完整切句後選首／中／尾，而不是只保留文件開頭。
CHUNK_SELECTION_STRATEGY = os.getenv(
    "TASK1_CHUNKAGG_LONG_DOCUMENT_SELECTION",
    "head_middle_tail",
).strip().lower()
if CHUNK_SELECTION_STRATEGY not in {"head", "head_middle_tail"}:
    raise ValueError(
        "TASK1_CHUNKAGG_LONG_DOCUMENT_SELECTION must be head or head_middle_tail"
    )

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = ENABLE_TF32
    torch.backends.cudnn.allow_tf32 = ENABLE_TF32
if ENABLE_TF32:
    torch.set_float32_matmul_precision('high')

# 由 main() 設定，用於 generate_similarity_artifacts 的覆寫資料
_QT_CANDIDATE_FILES = None   # List[str] 檔名（包含 .txt）
_QT_TRAIN_QIDS = None        # List[str]
_QT_VALID_QIDS = None        # List[str]

# 中文註解：TASK1_CHUNKAGG_QT_CAND_K / TASK1_CHUNKAGG_QT_QUERY_K 控制 QUICK_TEST 模式的樣本量。
QT_CAND_K = max(1, _get_env_int("TASK1_CHUNKAGG_QT_CAND_K", 20))
QT_QUERY_K = max(1, _get_env_int("TASK1_CHUNKAGG_QT_QUERY_K", 5))


def evaluate_model_retrieval(model, tokenizer, device, candidate_dataset_path, query_dataset_path,
                           train_qid_path, valid_qid_path, labels_path, topk=5,
                           retrieval_batch_size: int = RETRIEVAL_BATCH_SIZE,
                           query_candidate_scope_path: str | Path | None = None,
                           eligible_document_ids: Optional[Set[str]] = None,
                           retrieval_autocast_dtype: torch.dtype | None = RETRIEVAL_AUTOCAST_DTYPE,
                           chunk_selection_strategy: str = CHUNK_SELECTION_STRATEGY):
    """
    評估模型在整體train和valid data上的檢索性能
    """
    # 載入正確答案
    train_rel_dict = rel_file_convert(labels_path, train_qid_path)
    valid_rel_dict = rel_file_convert(labels_path, valid_qid_path)
    
    # 載入query IDs
    train_qids = load_query_ids_from_utils(train_qid_path)
    valid_qids = load_query_ids_from_utils(valid_qid_path)
    if QUICK_TEST:
        global _QT_TRAIN_QIDS, _QT_VALID_QIDS
        if _QT_TRAIN_QIDS:
            train_qids = list(_QT_TRAIN_QIDS)
        else:
            kt = min(QT_QUERY_K, len(train_qids))
            if len(train_qids) > kt:
                train_qids = random.sample(train_qids, kt)
        if _QT_VALID_QIDS:
            valid_qids = list(_QT_VALID_QIDS)
        else:
            kv = min(QT_QUERY_K, len(valid_qids))
            if len(valid_qids) > kv:
                valid_qids = random.sample(valid_qids, kv)
    
    # Train and valid share the same candidate corpus. Encode it once, then keep
    # only per-query top-k answers in memory; no large intermediate TREC file.
    combined_qids = list(dict.fromkeys([*train_qids, *valid_qids]))
    aggregation_variants = ("fused", "mean", "first")
    answer_dicts: Dict[str, Dict[int, List[str]]] = {
        name: {} for name in aggregation_variants
    }

    def consume_retrieval_scores(qid: str, scores: Dict[str, float]) -> None:
        consume_variant_scores("fused", qid, scores)

    def consume_variant_scores(
        variant: str,
        qid: str,
        scores: Dict[str, float],
    ) -> None:
        normalized_qid = normalize_case_id(qid)
        top_items = heapq.nlargest(
            topk,
            (
                (doc_id, score)
                for doc_id, score in scores.items()
                if normalize_case_id(doc_id) != normalized_qid
            ),
            key=lambda item: item[1],
        )
        answer_dicts[variant][int(normalize_case_id(qid))] = [
            normalize_case_id(doc_id) for doc_id, _ in top_items
        ]

    generate_similarity_artifacts(
        model,
        tokenizer,
        device,
        candidate_dir=candidate_dataset_path,
        query_dir=query_dataset_path,
        query_ids=combined_qids,
        trec_output_path=Path(os.devnull),
        run_tag="modernBert_eval_train_valid",
        batch_size=retrieval_batch_size,
        max_length=DOCUMENT_CHUNK_LENGTH,
        quick_test=False,
        candidate_files_override=_QT_CANDIDATE_FILES,
        candidate_limit=QT_CAND_K,
        query_limit=max(len(combined_qids), 1),
        query_candidate_scope_path=query_candidate_scope_path,
        persist_trec=False,
        retain_scores=False,
        score_consumer=consume_retrieval_scores,
        exclude_self=True,
        autocast_dtype=retrieval_autocast_dtype,
        chunk_selection_strategy=chunk_selection_strategy,
        reuse_query_embeddings=True,
        eligible_document_ids=eligible_document_ids,
        embedding_variants=aggregation_variants,
        variant_score_consumer=(
            lambda variant, qid, scores: (
                None
                if variant == "fused"
                else consume_variant_scores(variant, qid, scores)
            )
        ),
    )

    results = {}
    for split, rel_dict in (("train", train_rel_dict), ("valid", valid_rel_dict)):
        print(f"🔍 計算 {split} set 全域指標...")
        # 準備評估資料。所有 labeled query 都要進分母；若 scope 為空或
        # ranking 沒有該 query，視為空答案，而不是把該 query 略過。
        list_label_ohe = []   # 真實答案
        for qid in rel_dict.keys():
            list_label_ohe.append([int(pid) for pid in rel_dict[qid]])

        aggregation_results = {}
        for variant in aggregation_variants:
            list_answer_ohe = []
            missing_pred_qids = 0
            answer_dict = answer_dicts[variant]
            for qid in rel_dict.keys():
                one_answer = answer_dict.get(int(qid), [])
                if not one_answer:
                    missing_pred_qids += 1
                list_answer_ohe.append([int(pid) for pid in one_answer])

            f1, precision, recall = my_classification_report(
                list_label_ohe,
                list_answer_ohe,
            )
            aggregation_results[variant] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'num_queries': len(list_answer_ohe),
                'missing_pred_qids': missing_pred_qids,
            }

        results[split] = dict(aggregation_results["fused"])
        results[split]["aggregation"] = aggregation_results

        fused_results = aggregation_results["fused"]
        print(
            f"✅ {split} set fused: F1={fused_results['f1']:.4f}, "
            f"Precision={fused_results['precision']:.4f}, "
            f"Recall={fused_results['recall']:.4f}"
        )
        print(
            f"   aggregation ablation: mean_F1={aggregation_results['mean']['f1']:.4f}, "
            f"first_F1={aggregation_results['first']['f1']:.4f}"
        )
        if fused_results["missing_pred_qids"]:
            print(
                f"⚠️ {split} set 有 {fused_results['missing_pred_qids']} 個 query 無預測，"
                "已以空答案計分"
            )

    return results




def read_positive_pairs_from_json(json_path: str) -> Dict[str, Set[str]]:
    """從 JSON 讀取正樣本對映表，並去除 .txt 副檔名"""
    positives = defaultdict(set)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for q_txt, pos_list in data.items():
        qid = normalize_case_id(q_txt)
        for doc_txt in pos_list:
            doc_id = normalize_case_id(doc_txt)
            positives[qid].add(doc_id)
    return positives


def _sample_negative_ids(
    negative_candidates: Sequence[str],
    negative_scores: Sequence[float],
    *,
    max_negatives: int,
    temperature: float,
    target_ess: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> List[str]:
    if temperature <= 0:
        raise ValueError(f"sampling temperature must be > 0, got {temperature}")
    sample_size = min(max_negatives, len(negative_candidates))
    if sample_size <= 0:
        return []
    scores = np.asarray(negative_scores, dtype=np.float64)
    if scores.shape != (len(negative_candidates),):
        raise ValueError("negative candidate and score counts do not match")
    if not np.isfinite(scores).all():
        raise ValueError("dynamic-negative scores must all be finite")
    probs, _ = _build_sampling_distribution(
        scores,
        temperature=temperature,
        target_ess=target_ess,
    )
    choice = rng.choice if rng is not None else np.random.choice
    selected_negatives = choice(
        list(negative_candidates),
        size=sample_size,
        replace=False,
        p=probs,
    )
    return [str(item) for item in selected_negatives.tolist()]


def _softmax_sampling_probabilities(scores: np.ndarray, temperature: float) -> np.ndarray:
    logits = scores / float(temperature)
    logits -= logits.max()
    # Keep every scoped candidate in the numerical support even for very low
    # temperatures, so weighted sampling without replacement remains defined.
    weights = np.exp(np.clip(logits, -700.0, 0.0))
    probs = weights / weights.sum()
    if not np.isfinite(probs).all() or float(probs.sum()) <= 0:
        raise ValueError("non-finite dynamic-negative sampling probabilities")
    return probs


def _effective_sample_size(probabilities: np.ndarray) -> float:
    return float(1.0 / np.square(probabilities).sum())


def _calibrate_temperature_for_ess(
    scores: np.ndarray,
    *,
    target_ess: float,
    fallback_temperature: float,
) -> float:
    """Find a score-scale-aware temperature whose distribution reaches target ESS."""
    sample_count = len(scores)
    target = min(max(float(target_ess), 1.0), float(sample_count))
    if sample_count <= 1 or float(np.ptp(scores)) <= 1e-12:
        return float(fallback_temperature)

    score_scale = max(float(np.std(scores)), float(np.ptp(scores)) / 20.0, 1e-8)
    low = max(score_scale * 1e-6, 1e-8)
    high = max(score_scale, float(fallback_temperature), low * 10.0)

    low_ess = _effective_sample_size(_softmax_sampling_probabilities(scores, low))
    if low_ess >= target:
        return low
    high_ess = _effective_sample_size(_softmax_sampling_probabilities(scores, high))
    while high_ess < target and high < 1e8:
        high *= 2.0
        high_ess = _effective_sample_size(_softmax_sampling_probabilities(scores, high))

    for _ in range(48):
        midpoint = math.sqrt(low * high)
        midpoint_ess = _effective_sample_size(
            _softmax_sampling_probabilities(scores, midpoint)
        )
        if midpoint_ess < target:
            low = midpoint
        else:
            high = midpoint
    return high


def _build_sampling_distribution(
    scores: np.ndarray,
    *,
    temperature: float,
    target_ess: float = 0.0,
) -> tuple[np.ndarray, Dict[str, float]]:
    actual_temperature = float(temperature)
    if target_ess > 0:
        actual_temperature = _calibrate_temperature_for_ess(
            scores,
            target_ess=target_ess,
            fallback_temperature=temperature,
        )
    probabilities = _softmax_sampling_probabilities(scores, actual_temperature)
    top_count = min(100, len(scores))
    top_indices = np.argpartition(scores, -top_count)[-top_count:]
    entropy = float(-(probabilities * np.log(probabilities.clip(min=1e-300))).sum())
    diagnostics = {
        "temperature": actual_temperature,
        "effective_sample_size": _effective_sample_size(probabilities),
        "entropy": entropy,
        "top100_probability_mass": float(probabilities[top_indices].sum()),
        "max_probability": float(probabilities.max()),
        "pool_size": float(len(scores)),
    }
    return probabilities, diagnostics


def _get_scoped_candidate_ids(
    qid: str,
    query_candidate_scope: Optional[Dict[str, List[str]]],
) -> Optional[Set[str]]:
    if query_candidate_scope is None:
        return None
    normalized_qid = normalize_case_id(qid)
    scoped_ids = query_candidate_scope.get(normalized_qid)
    if scoped_ids is None and normalized_qid.isdigit():
        scoped_ids = query_candidate_scope.get(normalized_qid.zfill(6))
    return {normalize_case_id(doc_id) for doc_id in scoped_ids or []}


def _collect_scoped_negative_pool(
    qid: str,
    similarities: Dict[str, float],
    pos_set: Set[str],
    query_candidate_scope: Optional[Dict[str, List[str]]] = None,
) -> tuple[List[str], List[float]]:
    normalized_qid = normalize_case_id(qid)
    normalized_pos_set = {normalize_case_id(pos_id) for pos_id in pos_set}
    allowed_ids = _get_scoped_candidate_ids(normalized_qid, query_candidate_scope)

    negative_candidates: List[str] = []
    negative_scores: List[float] = []
    seen: Set[str] = set()
    for raw_doc_id, score in similarities.items():
        doc_id = normalize_case_id(raw_doc_id)
        score = float(score)
        if not doc_id or doc_id in seen:
            continue
        if not np.isfinite(score):
            continue
        if allowed_ids is not None and doc_id not in allowed_ids:
            continue
        if doc_id in normalized_pos_set or doc_id == normalized_qid:
            continue
        seen.add(doc_id)
        negative_candidates.append(doc_id)
        negative_scores.append(score)
    return negative_candidates, negative_scores


def _generate_query_adaptive_samples(
    qid: str,
    similarities: Dict[str, float],
    pos_set: Set[str],
    *,
    max_negatives: int,
    temperature: float,
    target_ess: float = 0.0,
    query_candidate_scope: Optional[Dict[str, List[str]]],
    rng: Optional[np.random.Generator] = None,
    diagnostics_sink: Optional[List[Dict[str, float]]] = None,
) -> tuple[List[Dict], int, int]:
    normalized_qid = normalize_case_id(qid)
    normalized_positives = {normalize_case_id(pos_id) for pos_id in pos_set}
    allowed_ids = _get_scoped_candidate_ids(normalized_qid, query_candidate_scope)
    scored_ids = {normalize_case_id(doc_id) for doc_id in similarities}
    train_positives = []
    skipped_out_of_scope = 0
    skipped_unavailable = 0
    for pos_id in sorted(normalized_positives):
        if allowed_ids is not None and pos_id not in allowed_ids:
            skipped_out_of_scope += 1
            continue
        if pos_id not in scored_ids:
            skipped_unavailable += 1
            continue
        train_positives.append(pos_id)

    negative_candidates, negative_scores = _collect_scoped_negative_pool(
        normalized_qid,
        similarities,
        normalized_positives,
        query_candidate_scope=query_candidate_scope,
    )
    if not negative_candidates:
        return [], skipped_out_of_scope, skipped_unavailable
    probabilities, sampling_diagnostics = _build_sampling_distribution(
        np.asarray(negative_scores, dtype=np.float64),
        temperature=temperature,
        target_ess=target_ess,
    )
    if diagnostics_sink is not None:
        diagnostics_sink.append(sampling_diagnostics)

    samples = []
    choice = rng.choice if rng is not None else np.random.choice
    sample_size = min(max_negatives, len(negative_candidates))
    for pos_id in train_positives:
        selected_negatives = choice(
            negative_candidates,
            size=sample_size,
            replace=False,
            p=probabilities,
        ).tolist()
        if selected_negatives:
            samples.append({
                "query_id": normalized_qid,
                "positive_id": pos_id,
                "negative_ids": selected_negatives,
            })
    return samples, skipped_out_of_scope, skipped_unavailable


def generate_adaptive_negative_samples(
    query_id_to_similarities,
    positives,
    max_negatives=MAX_NEGATIVES,
    temperature=1.0,
    target_ess: float = 0.0,
    query_candidate_scope: Optional[Dict[str, List[str]]] = None,
    rng: Optional[np.random.Generator] = None,
):
    """
    根據完整 scope 內相似度分數作為機率來選擇負樣本。
    同一筆樣本內不重複抽樣；若可用 pool 小於 max_negatives，直接使用可用數量。
    同一 query 的每個 positive 會獨立重抽 negatives。
    """
    dataset = []
    skipped_out_of_scope_positives = 0
    skipped_unavailable_positives = 0

    for qid, pos_set in positives.items():
        normalized_qid = normalize_case_id(qid)
        similarities = (
            query_id_to_similarities.get(normalized_qid)
            or query_id_to_similarities.get(f"{normalized_qid}.txt")
            or (query_id_to_similarities.get(normalized_qid.zfill(6)) if normalized_qid.isdigit() else None)
        )
        if similarities is None:
            continue
        query_samples, skipped_scope, skipped_unavailable = _generate_query_adaptive_samples(
            normalized_qid,
            similarities,
            pos_set,
            max_negatives=max_negatives,
            temperature=temperature,
            target_ess=target_ess,
            query_candidate_scope=query_candidate_scope,
            rng=rng,
        )
        dataset.extend(query_samples)
        skipped_out_of_scope_positives += skipped_scope
        skipped_unavailable_positives += skipped_unavailable

    if skipped_out_of_scope_positives:
        print(
            "🔹 Scope filter 排除 "
            f"{skipped_out_of_scope_positives} 個不在 query scope 內的 positive pairs"
        )
    if skipped_unavailable_positives:
        print(
            "⚠️ 略過 "
            f"{skipped_unavailable_positives} 個未出現在 scoped retrieval scores 的 positive pairs"
        )
    return dataset

# ------------------------
# Dataset (隨機子集用)
# ------------------------
class RandomSubsetDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.data = samples
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# ---------- compute_metrics ----------
def make_compute_metrics_for_retrieval(model, tokenizer,
                                       candidate_dataset_path,
                                       query_dataset_path,
                                       train_qid_path,
                                       valid_qid_path,
                                       labels_path,
                                       retrieval_batch_size: int = RETRIEVAL_BATCH_SIZE,
                                       query_candidate_scope_path: str | Path | None = None,
                                       eligible_document_ids: Optional[Set[str]] = None,
                                       retrieval_autocast_dtype: torch.dtype | None = RETRIEVAL_AUTOCAST_DTYPE,
                                       chunk_selection_strategy: str = CHUNK_SELECTION_STRATEGY):
    """Factory to create compute_metrics that also computes full-corpus retrieval metrics.

    Returns a function taking EvalPrediction and returning a metrics dict including:
    - global_f1 (will be surfaced as eval_global_f1)
    - acc1, acc5 (for reference)
    Full-corpus precision/recall/F1 are returned through the normal Trainer log
    stream, so TensorBoard receives one canonical event stream.
    """
    def _compute(eval_pred: EvalPrediction):
        # 1) Keep quick classification-style metrics for reference
        metrics = {}
        try:
            logits = torch.tensor(eval_pred.predictions)
            labels = torch.tensor(eval_pred.label_ids)
            preds_top1 = logits.argmax(dim=1).cpu().numpy()
            metrics["acc1"] = accuracy_score(labels.cpu().numpy(), preds_top1)
            topk_k = min(5, logits.size(1))
            top5_preds = torch.topk(logits, k=topk_k, dim=1).indices
            labels_expanded = labels.view(-1, 1).expand_as(top5_preds)
            metrics["acc5"] = (top5_preds == labels_expanded).any(dim=1).float().mean().item()
        except Exception:
            pass

        # 2) Full-corpus retrieval over train/valid.
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        results = evaluate_model_retrieval(
            model=model,
            tokenizer=tokenizer,
            device=device,
            candidate_dataset_path=candidate_dataset_path,
            query_dataset_path=query_dataset_path,
            train_qid_path=train_qid_path,
            valid_qid_path=valid_qid_path,
            labels_path=labels_path,
            topk=5,
            retrieval_batch_size=retrieval_batch_size,
            query_candidate_scope_path=query_candidate_scope_path,
            eligible_document_ids=eligible_document_ids,
            retrieval_autocast_dtype=retrieval_autocast_dtype,
            chunk_selection_strategy=chunk_selection_strategy,
        )

        # 3) Global metric for best model selection/early stopping
        global_f1 = float(results.get("valid", {}).get("f1", 0.0))
        metrics["global_f1"] = global_f1
        for split in ("train", "valid"):
            split_results = results.get(split, {})
            for metric_name in ("f1", "precision", "recall"):
                metrics[f"retrieval_{split}_top5_{metric_name}"] = float(
                    split_results.get(metric_name, 0.0)
                )
            metrics[f"retrieval_{split}_missing_queries"] = int(
                split_results.get("missing_pred_qids", 0)
            )
            aggregation_results = split_results.get("aggregation", {})
            for variant in ("fused", "mean", "first"):
                variant_results = aggregation_results.get(variant, {})
                for metric_name in ("f1", "precision", "recall"):
                    metrics[
                        f"retrieval_{split}_{variant}_top5_{metric_name}"
                    ] = float(variant_results.get(metric_name, 0.0))

        print(f"eval_global_f1: {global_f1:.6f}")

        return metrics

    return _compute

# ----------- Dataset 讀檔 -----------
class ContrastiveDataset(Dataset):
    def __init__(
        self,
        json_path: str = None,
        doc_folder: str = None,
        data: List[Dict] = None,
        query_candidate_scope: Optional[Dict[str, List[str]]] = None,
        eligible_document_ids: Optional[Set[str]] = None,
        minimum_content_characters: int = MIN_DOCUMENT_CONTENT_CHARS,
    ):
        self.query_candidate_scope = query_candidate_scope
        self.eligible_document_ids = (
            {normalize_case_id(doc_id) for doc_id in eligible_document_ids}
            if eligible_document_ids is not None
            else None
        )
        self.minimum_content_characters = max(1, int(minimum_content_characters))
        if data is not None:
            loaded_data = data
        elif json_path is not None:
            with open(json_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
        else:
            loaded_data = []
        self.data = self._filter_samples_to_scope(loaded_data)
        
        self.doc_folder = doc_folder
        # 中文註解：把常重複讀取的判決書文字放進 LRU cache，減少磁碟 I/O。
        self.text_cache_size = TEXT_CACHE_SIZE
        self._text_cache = OrderedDict()
        if json_path:
            print(f"🔹 對比式資料（{os.path.basename(json_path)}）共載入 {len(self.data)} 筆樣本")
        else:
            print(f"🔹 對比式資料共載入 {len(self.data)} 筆樣本")
    
    def _filter_samples_to_scope(self, samples: List[Dict]) -> List[Dict]:
        all_positives_by_query: Dict[str, Set[str]] = defaultdict(set)
        for sample in samples:
            query_id = normalize_case_id(sample.get("query_id", ""))
            positive_id = normalize_case_id(sample.get("positive_id", ""))
            if query_id and positive_id:
                all_positives_by_query[query_id].add(positive_id)
        self._all_positives_by_query = all_positives_by_query

        filtered = []
        removed = 0
        for sample in samples:
            query_id = normalize_case_id(sample.get("query_id", ""))
            positive_id = normalize_case_id(sample.get("positive_id", ""))
            allowed_ids = _get_scoped_candidate_ids(query_id, self.query_candidate_scope)
            query_is_eligible = (
                self.eligible_document_ids is None
                or query_id in self.eligible_document_ids
            )
            positive_is_eligible = (
                self.eligible_document_ids is None
                or positive_id in self.eligible_document_ids
            )
            positive_is_scoped = allowed_ids is None or positive_id in allowed_ids
            if (
                not query_id
                or not positive_id
                or not query_is_eligible
                or not positive_is_eligible
                or not positive_is_scoped
            ):
                removed += 1
                continue
            forbidden_ids = all_positives_by_query[query_id] | {query_id}
            negative_ids = []
            for raw_negative_id in sample.get("negative_ids", []):
                negative_id = normalize_case_id(raw_negative_id)
                negative_is_scoped = allowed_ids is None or negative_id in allowed_ids
                negative_is_eligible = (
                    self.eligible_document_ids is None
                    or negative_id in self.eligible_document_ids
                )
                if (
                    negative_id
                    and negative_is_scoped
                    and negative_is_eligible
                    and negative_id not in forbidden_ids
                ):
                    negative_ids.append(negative_id)
            negative_ids = list(dict.fromkeys(negative_ids))
            if not negative_ids:
                removed += 1
                continue
            filtered.append({
                "query_id": query_id,
                "positive_id": positive_id,
                "negative_ids": negative_ids,
            })
        if removed:
            print(
                "🔹 Scope/content filter 從 contrastive dataset 排除 "
                f"{removed} 筆不對齊樣本"
            )
        return filtered

    def update_data(self, new_data: List[Dict]):
        """更新資料集的負樣本"""
        self.data = self._filter_samples_to_scope(new_data)
        print(f"🔹 資料集已更新，現有 {len(self.data)} 筆樣本")
    
    def __len__(self):
        return len(self.data)
    
    def load_text(self, doc_id: str) -> str:
        if self.text_cache_size > 0 and doc_id in self._text_cache:
            text = self._text_cache.pop(doc_id)
            self._text_cache[doc_id] = text
            return text

        path = os.path.join(self.doc_folder, f"{doc_id}.txt")
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        if not has_minimum_document_content(text, self.minimum_content_characters):
            raise ValueError(
                f"Document {doc_id} has fewer than "
                f"{self.minimum_content_characters} non-whitespace characters"
            )

        if self.text_cache_size > 0:
            self._text_cache[doc_id] = text
            while len(self._text_cache) > self.text_cache_size:
                self._text_cache.popitem(last=False)
        return text
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        if "positive_id" not in sample:
            raise KeyError("single-positive sample must contain `positive_id`")
        query_id = normalize_case_id(sample["query_id"])
        positive_id = normalize_case_id(sample["positive_id"])
        if not query_id or not positive_id:
            raise ValueError(f"Sample has an empty query or positive id: {sample}")

        negative_ids = []
        seen_negatives = set()
        for raw_negative_id in sample.get("negative_ids", []):
            negative_id = normalize_case_id(raw_negative_id)
            if not negative_id or negative_id in seen_negatives:
                continue
            if negative_id in {query_id, positive_id}:
                raise ValueError(
                    f"Negative {negative_id} overlaps query/positive in sample {idx}"
                )
            seen_negatives.add(negative_id)
            negative_ids.append(negative_id)
        if not negative_ids:
            raise ValueError(f"Sample has no negatives: {sample}")

        return {
            "query_id": query_id,
            "query_text": self.load_text(query_id),
            "positive_id": positive_id,
            "positive_text": self.load_text(positive_id),
            "negative_ids": negative_ids,
            "negative_texts": [self.load_text(nid) for nid in negative_ids],
            "all_positive_ids": sorted(self._all_positives_by_query.get(query_id, {positive_id})),
        }

# ----------- Collator -----------
@dataclass
class ContrastiveCollator:
    tokenizer: AutoTokenizer
    max_length: int = DOCUMENT_CHUNK_LENGTH
    max_chunks: int = MAX_DOCUMENT_CHUNKS
    chunk_cache_size: int = CHUNK_CACHE_SIZE
    chunk_selection_strategy: str = CHUNK_SELECTION_STRATEGY

    def __post_init__(self):
        # 中文註解：把常見文件的 chunk tokenization 快取在 CPU，減少 tokenizer 開銷。
        self._chunk_cache = OrderedDict()

    def _get_cached_chunk(self, cache_key: str, text: str) -> Dict[str, torch.Tensor]:
        if self.chunk_cache_size > 0 and cache_key in self._chunk_cache:
            cached = self._chunk_cache.pop(cache_key)
            self._chunk_cache[cache_key] = cached
            return cached

        encoded = _chunk_single_text(
            text,
            self.tokenizer,
            max_length=self.max_length,
            max_chunks=self.max_chunks,
            selection_strategy=self.chunk_selection_strategy,
        )
        if self.chunk_cache_size > 0:
            self._chunk_cache[cache_key] = encoded
            while len(self._chunk_cache) > self.chunk_cache_size:
                self._chunk_cache.popitem(last=False)
        return encoded

    def _build_cached_document_batch(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if ids is not None and len(ids) != len(texts):
            raise ValueError(
                f"Document ids/texts length mismatch: ids={len(ids)}, texts={len(texts)}"
            )
        if not texts:
            return _build_document_batch(
                texts,
                self.tokenizer,
                max_length=self.max_length,
                max_chunks=self.max_chunks,
                chunk_selection_strategy=self.chunk_selection_strategy,
                device=None,
            )

        if ids is None or self.chunk_cache_size <= 0:
            return _build_document_batch(
                texts,
                self.tokenizer,
                max_length=self.max_length,
                max_chunks=self.max_chunks,
                chunk_selection_strategy=self.chunk_selection_strategy,
                device=None,
            )

        encoded_documents = [
            self._get_cached_chunk(cache_key=str(doc_id), text=text)
            for doc_id, text in zip(ids, texts)
        ]
        return {
            key: torch.stack([encoded[key] for encoded in encoded_documents], dim=0)
            for key in ("input_ids", "attention_mask", "chunk_mask")
        }

    def __call__(self, batch: List[Dict]) -> Dict[str, object]:
        bsz = len(batch)
        q_ids = [item["query_id"] for item in batch]
        q_texts = [item["query_text"] for item in batch]
        negative_counts = [len(item["negative_ids"]) for item in batch]
        if any(count <= 0 for count in negative_counts):
            raise ValueError(f"Every sample must have at least one negative: {negative_counts}")

        p_ids = [item["positive_id"] for item in batch]
        p_texts = [item["positive_text"] for item in batch]
        n_ids = [neg_id for item in batch for neg_id in item["negative_ids"]]
        n_texts = [neg for item in batch for neg in item["negative_texts"]]

        # 中文註解：query / positive / negative 全部共用同一套 3-chunk 文件編碼邏輯。
        anchor_batch = self._build_cached_document_batch(
            q_texts,
            ids=q_ids,
        )
        positive_batch = self._build_cached_document_batch(
            p_texts,
            ids=p_ids,
        )
        negative_batch = self._build_cached_document_batch(
            n_texts,
            ids=n_ids,
        )

        labels = torch.zeros(bsz, dtype=torch.long)
        return {
            "anchor_input": anchor_batch,
            "positive_input": positive_batch,
            "negative_input": negative_batch,
            "labels": labels,
            "negative_counts": torch.tensor(negative_counts, dtype=torch.long),
            # Metadata is consumed by the custom Trainer before calling model.forward.
            # It is used only to mask false negatives in the detached late cache.
            "anchor_ids": q_ids,
            "positive_ids": p_ids,
            "negative_id_rows": [list(item["negative_ids"]) for item in batch],
            "all_positive_id_rows": [list(item["all_positive_ids"]) for item in batch],
        }

# ----------- 自訂 Trainer：讓 temperature 有獨立 LR 並實現 adaptive negative sampling -----------
def _parse_sampling_temperature_schedule(raw_schedule: str) -> List[tuple[int, float]]:
    schedule: List[tuple[int, float]] = []
    for raw_item in raw_schedule.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "sampling schedule entries must use EPOCH:TEMPERATURE, got "
                f"{item!r}"
            )
        raw_epoch, raw_temperature = item.split(":", 1)
        epoch = int(raw_epoch)
        temperature = float(raw_temperature)
        if epoch < 0 or temperature <= 0:
            raise ValueError(f"invalid sampling schedule entry: {item!r}")
        schedule.append((epoch, temperature))
    if not schedule:
        raise ValueError("sampling temperature schedule cannot be empty")
    schedule.sort(key=lambda item: item[0])
    if schedule[0][0] != 0:
        raise ValueError("sampling temperature schedule must start at epoch 0")
    return schedule


DEFAULT_SAMPLING_TARGET_ESS_SCHEDULE = "0:4000,2:2000,4:1000,6:500,8:300"


def _parse_sampling_target_ess_schedule(raw_schedule: str) -> List[tuple[int, float]]:
    schedule: List[tuple[int, float]] = []
    for raw_item in raw_schedule.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "target ESS schedule entries must use EPOCH:ESS, got "
                f"{item!r}"
            )
        raw_epoch, raw_target = item.split(":", 1)
        epoch = int(raw_epoch)
        target = float(raw_target)
        if epoch < 0 or not math.isfinite(target) or target < 0:
            raise ValueError(f"invalid target ESS schedule entry: {item!r}")
        schedule.append((epoch, target))
    if not schedule:
        raise ValueError("target ESS schedule cannot be empty")
    schedule.sort(key=lambda item: item[0])
    if schedule[0][0] != 0:
        raise ValueError("target ESS schedule must start at epoch 0")
    return schedule


class AdaptiveNegativeSamplingTrainer(Trainer):
    def __init__(self, 
                 *args, 
                 candidate_dataset_path: str = None,
                 query_dataset_path: str = None,
                 train_qid_path: str = None,
                 positive_train_json_path: str = None,
                 finetune_data_dir: str = None,
                 sampling_temperature: float = 1.0,
                 sampling_temperature_schedule: str = "0:0.20,3:0.15,5:0.10",
                 sampling_target_ess: float = 300.0,
                 sampling_target_ess_schedule: str = DEFAULT_SAMPLING_TARGET_ESS_SCHEDULE,
                 retrieval_batch_size: int = RETRIEVAL_BATCH_SIZE,
                 query_candidate_scope_path: str | Path | None = None,
                 eligible_document_ids: Optional[Set[str]] = None,
                 max_negatives: int = MAX_NEGATIVES,
                 negative_cache_size: int = 0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_dataset_path = candidate_dataset_path
        self.query_dataset_path = query_dataset_path
        self.train_qid_path = train_qid_path
        self.positive_train_json_path = positive_train_json_path
        self.finetune_data_dir = finetune_data_dir
        if sampling_temperature <= 0:
            raise ValueError("sampling_temperature must be > 0")
        self.sampling_temperature = float(sampling_temperature)
        self.sampling_temperature_schedule = _parse_sampling_temperature_schedule(
            sampling_temperature_schedule
        )
        self.sampling_target_ess = max(0.0, float(sampling_target_ess))
        self.sampling_target_ess_schedule = _parse_sampling_target_ess_schedule(
            sampling_target_ess_schedule
        )
        self.retrieval_batch_size = max(1, retrieval_batch_size)
        if query_candidate_scope_path is None:
            raise ValueError("query_candidate_scope_path is required for chunkAgg training")
        self.query_candidate_scope_path = Path(query_candidate_scope_path)
        self.query_candidate_scope = load_query_candidate_scope(self.query_candidate_scope_path)
        self.eligible_document_ids = (
            {normalize_case_id(doc_id) for doc_id in eligible_document_ids}
            if eligible_document_ids is not None
            else None
        )
        self.max_negatives = max(1, int(max_negatives))
        self.retrieval_tokenizer = self.processing_class
        self.current_epoch = 0
        self._prepared_epoch_index = None  # resume 時若已預先準備該 epoch 的負樣本，callback 需略過一次
        self.negative_cache_size = max(0, int(negative_cache_size))
        self._negative_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._diagnostic_sums: Dict[str, float] = defaultdict(float)
        self._diagnostic_count = 0
        
        # 載入正樣本資料和query IDs
        if train_qid_path:
            self.train_qids = load_query_ids_from_utils(train_qid_path)
            # QUICK_TEST: 若主程式已提供縮小後的清單，採用之；否則在此抽樣最多5個
            if QUICK_TEST:
                global _QT_TRAIN_QIDS
                if _QT_TRAIN_QIDS:
                    self.train_qids = list(_QT_TRAIN_QIDS)
                else:
                    kq = min(QT_QUERY_K, len(self.train_qids))
                    if len(self.train_qids) > kq:
                        self.train_qids = random.sample(self.train_qids, kq)
            if self.eligible_document_ids is not None:
                original_query_count = len(self.train_qids)
                self.train_qids = [
                    normalize_case_id(qid)
                    for qid in self.train_qids
                    if normalize_case_id(qid) in self.eligible_document_ids
                ]
                excluded_query_count = original_query_count - len(self.train_qids)
                if excluded_query_count:
                    print(
                        "🔹 Content filter 排除 "
                        f"{excluded_query_count} 個空白或過短的 training queries"
                    )
        if positive_train_json_path:
            self.positives = read_positive_pairs_from_json(positive_train_json_path)
            
        print(f"🔹 適應性負樣本採樣設定：")
        print("   - 負樣本更新頻率：固定每個 epoch")
        print(f"   - 採樣溫度 fallback：{self.sampling_temperature}")
        print(f"   - 採樣溫度 schedule：{self.sampling_temperature_schedule}")
        print(f"   - Sampling target ESS：{self.sampling_target_ess}")
        print(f"   - Sampling target ESS schedule：{self.sampling_target_ess_schedule}")
        print("   - Negative source：current model for every epoch, including epoch 0")
        print("   - Loss：single-positive document InfoNCE + chunk-evidence InfoNCE")
        print(f"   - Detached late-cache size：{self.negative_cache_size}")
        print(f"   - Max negatives per sample：{self.max_negatives}")
        print(f"   - Candidate embedding batch size：{self.retrieval_batch_size}")
        print(f"   - Chunk microbatch size：{getattr(self.model, 'chunk_microbatch_size', CHUNK_MICROBATCH_SIZE)}")
        if self.query_candidate_scope_path is not None:
            print(f"   - Query candidate scope：{self.query_candidate_scope_path}")
        if hasattr(self, 'train_qids'):
            print(f"   - 訓練查詢數量：{len(self.train_qids)}")
        if hasattr(self, 'positives'):
            positive_pair_count = sum(len(items) for items in self.positives.values())
            print(f"   - 有標註 positives 的 query 數量：{len(self.positives)}")
            print(f"   - 標註 positive pairs 數量：{positive_pair_count}")
        if self.eligible_document_ids is not None:
            print(f"   - Eligible corpus documents：{len(self.eligible_document_ids)}")

    def _sampling_temperature_for_epoch(self, epoch_index: int) -> float:
        selected = self.sampling_temperature
        for start_epoch, temperature in self.sampling_temperature_schedule:
            if epoch_index < start_epoch:
                break
            selected = temperature
        return selected

    def _sampling_target_ess_for_epoch(self, epoch_index: int) -> float:
        selected = self.sampling_target_ess
        schedule = getattr(self, "sampling_target_ess_schedule", None)
        if not schedule:
            return selected
        for start_epoch, target_ess in schedule:
            if epoch_index < start_epoch:
                break
            selected = target_ess
        return selected

    def _adaptive_data_path(self, epoch_index: int) -> Path:
        return Path(self.finetune_data_dir) / f"adaptive_negative_epoch{epoch_index}_train.json"

    def _build_cached_negatives(
        self,
        *,
        anchor_ids: Sequence[str],
        negative_id_rows: Sequence[Sequence[str]],
        all_positive_id_rows: Sequence[Sequence[str]],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.negative_cache_size <= 0 or not self._negative_cache:
            return None, None

        selected_rows: List[List[torch.Tensor]] = []
        for query_id, explicit_negatives, positive_ids in zip(
            anchor_ids,
            negative_id_rows,
            all_positive_id_rows,
        ):
            normalized_query_id = normalize_case_id(query_id)
            complete_positive_ids = {
                normalize_case_id(item)
                for item in getattr(self, "positives", {}).get(normalized_query_id, set())
            }
            forbidden = {
                normalized_query_id,
                *[normalize_case_id(item) for item in explicit_negatives],
                *[normalize_case_id(item) for item in positive_ids],
                *complete_positive_ids,
            }
            allowed_ids = _get_scoped_candidate_ids(
                normalized_query_id,
                self.query_candidate_scope,
            ) or set()
            selected_rows.append([
                vector
                for doc_id, vector in self._negative_cache.items()
                if doc_id in allowed_ids and doc_id not in forbidden
            ])

        max_count = max((len(row) for row in selected_rows), default=0)
        if max_count == 0:
            return None, None
        hidden_dim = int(getattr(self.model, "hidden_dim"))
        vectors = torch.zeros(
            (len(selected_rows), max_count, hidden_dim),
            dtype=dtype,
            device=device,
        )
        mask = torch.zeros(
            (len(selected_rows), max_count),
            dtype=torch.bool,
            device=device,
        )
        for row_index, row in enumerate(selected_rows):
            if not row:
                continue
            stacked = torch.stack(row).to(device=device, dtype=dtype)
            vectors[row_index, : len(row)] = stacked
            mask[row_index, : len(row)] = True
        return vectors, mask

    def _update_negative_cache(
        self,
        positive_ids: Sequence[str],
        positive_vectors: torch.Tensor,
    ) -> None:
        if self.negative_cache_size <= 0:
            return
        for doc_id, vector in zip(positive_ids, positive_vectors.detach().float().cpu()):
            normalized_id = normalize_case_id(doc_id)
            self._negative_cache.pop(normalized_id, None)
            self._negative_cache[normalized_id] = vector
        while len(self._negative_cache) > self.negative_cache_size:
            self._negative_cache.popitem(last=False)

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        if self.negative_cache_size <= 0:
            return
        checkpoint_dir = Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        cache_path = checkpoint_dir / "chunkagg_negative_cache.pt"
        temporary_path = cache_path.with_suffix(".pt.tmp")
        cache_payload = {
            "ids": list(self._negative_cache.keys()),
            "vectors": (
                torch.stack(list(self._negative_cache.values()))
                if self._negative_cache
                else torch.empty((0, int(getattr(self.model, "hidden_dim"))))
            ),
        }
        torch.save(cache_payload, temporary_path)
        os.replace(temporary_path, cache_path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        result = super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        self._negative_cache.clear()
        cache_path = Path(resume_from_checkpoint) / "chunkagg_negative_cache.pt"
        if self.negative_cache_size > 0 and cache_path.exists():
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(cache_path, map_location="cpu")
            ids = payload.get("ids", [])
            vectors = payload.get("vectors", torch.empty((0, 0)))
            for doc_id, vector in zip(ids, vectors):
                self._negative_cache[normalize_case_id(doc_id)] = vector.float().cpu()
            while len(self._negative_cache) > self.negative_cache_size:
                self._negative_cache.popitem(last=False)
            print(
                f"🔹 已從 checkpoint 恢復 {len(self._negative_cache)} 個 late-cache negatives"
            )
        return result

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        """Inject masked late-cache negatives and retain scalar diagnostics."""
        model_inputs = dict(inputs)
        anchor_ids = model_inputs.pop("anchor_ids", [])
        positive_ids = model_inputs.pop("positive_ids", [])
        negative_id_rows = model_inputs.pop("negative_id_rows", [])
        all_positive_id_rows = model_inputs.pop("all_positive_id_rows", [])

        if model.training and self.negative_cache_size > 0:
            anchor_tensor = model_inputs["anchor_input"]["input_ids"]
            cached_vectors, cached_mask = self._build_cached_negatives(
                anchor_ids=anchor_ids,
                negative_id_rows=negative_id_rows,
                all_positive_id_rows=all_positive_id_rows,
                device=anchor_tensor.device,
                dtype=next(model.parameters()).dtype,
            )
            model_inputs["cached_negative_vectors"] = cached_vectors
            model_inputs["cached_negative_mask"] = cached_mask

        outputs = model(**model_inputs)
        diagnostics = outputs.pop("_diagnostics", {})
        positive_vectors = outputs.pop("_cache_positive_vectors", None)
        if model.training:
            for name, value in diagnostics.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    self._diagnostic_sums[name] += float(value.detach().cpu().item())
            self._diagnostic_count += 1
            if positive_vectors is not None:
                self._update_negative_cache(positive_ids, positive_vectors)

        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss
    
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        args = self.args
        model = self.model
        encoder_lr = getattr(args, "encoder_lr", args.learning_rate)
        fusion_lr = getattr(args, "fusion_lr", args.learning_rate)
        projector_lr = getattr(args, "projector_lr", encoder_lr)
        temperature_lr = getattr(args, "temperature_lr", args.learning_rate)

        no_decay_tokens = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
        optimizer_grouped_parameters = []

        def _append_named_group(
            group_name: str,
            named_params: Sequence[tuple[str, nn.Parameter]],
            *,
            lr: float,
            weight_decay: float,
        ):
            decay_params = []
            no_decay_params = []
            for name, param in named_params:
                if not param.requires_grad:
                    continue
                if any(token in name for token in no_decay_tokens):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
            if decay_params:
                optimizer_grouped_parameters.append({
                    "params": decay_params,
                    "weight_decay": weight_decay,
                    "lr": lr,
                    "group_name": f"{group_name}_decay",
                })
            if no_decay_params:
                optimizer_grouped_parameters.append({
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "lr": lr,
                    "group_name": f"{group_name}_no_decay",
                })

        _append_named_group(
            "encoder",
            [(f"encoder.{name}", param) for name, param in model.encoder.named_parameters()],
            lr=encoder_lr,
            weight_decay=args.weight_decay,
        )
        _append_named_group(
            "projector",
            [(f"projector.{name}", param) for name, param in model.projector.named_parameters()],
            lr=projector_lr,
            weight_decay=args.weight_decay,
        )
        _append_named_group(
            "fusion",
            list(model.chunk_aggregation_named_parameters()),
            lr=fusion_lr,
            weight_decay=args.weight_decay,
        )
        if model.log_temperature.requires_grad:
            optimizer_grouped_parameters.append({
                "params": [model.log_temperature],
                "weight_decay": 0.0,
                "lr": temperature_lr,
                "group_name": "log_temperature",
            })

        # 根據 TrainingArguments 選擇 AdamW；若要求 fused，嘗試啟用
        adamw_kwargs = dict(
            lr=encoder_lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        from torch.optim import AdamW
        try:
            if getattr(args, "optim", "") == "adamw_torch_fused":
                # torch>=2.0 支援 fused，若無支援會拋例外
                self.optimizer = AdamW(optimizer_grouped_parameters, **adamw_kwargs, fused=True)
            else:
                self.optimizer = AdamW(optimizer_grouped_parameters, **adamw_kwargs)
        except TypeError:
            # 沒有 fused 參數就退回一般 AdamW
            self.optimizer = AdamW(optimizer_grouped_parameters, **adamw_kwargs)

        return self.optimizer

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Attach chunkAgg extras to the standard Trainer TensorBoard stream.

        Keeping these values in the normal Trainer log avoids separate TensorBoard
        runs/event streams for temperature and LR groups.
        """
        if any(key in logs for key in ("loss", "grad_norm", "learning_rate")):
            logs = dict(logs)
            model = self.model
            if model is not None and hasattr(model, "log_temperature"):
                try:
                    logs["temperature"] = float(model.temperature.detach().cpu().item())
                except Exception:
                    pass
            if self.optimizer is not None:
                for i, group in enumerate(self.optimizer.param_groups):
                    try:
                        logs[f"lr_group_{i}"] = float(group.get("lr", 0.0))
                    except Exception:
                        pass
            if self._diagnostic_count > 0:
                for name, total in self._diagnostic_sums.items():
                    logs[f"diagnostics/{name}"] = total / self._diagnostic_count
                logs["cache/current_size"] = float(len(self._negative_cache))
                self._diagnostic_sums.clear()
                self._diagnostic_count = 0
        return super().log(logs, start_time=start_time)
    
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """Populate the mutable dataset before Trainer builds its dataloader."""
        train_size = len(self.train_dataset) if self.train_dataset is not None else 0
        if train_size == 0:
            epoch_index = 0
            if resume_from_checkpoint:
                print(f"🔹 偵測到 resume_from_checkpoint={resume_from_checkpoint}")
                completed_epoch = self.get_completed_epoch_from_checkpoint(resume_from_checkpoint)
                if completed_epoch is not None:
                    epoch_index = completed_epoch
                else:
                    print("⚠️ 無法從 checkpoint 解析 epoch，將以 epoch 0 命名重算負樣本")
            self.current_epoch = epoch_index
            print(
                f"🔹 使用目前模型重新推論並產生 epoch {epoch_index} 動態負樣本；"
                "不載入舊的 adaptive-negative JSON"
            )
            self.update_negative_samples()
            self._prepared_epoch_index = epoch_index
        
        return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)

    def get_completed_epoch_from_checkpoint(self, resume_from_checkpoint) -> Optional[int]:
        """解析 checkpoint trainer_state.json 的 epoch（已完成的 epoch 數，取整數）。"""
        if not resume_from_checkpoint:
            return None
        state_path = Path(resume_from_checkpoint) / "trainer_state.json"
        if not state_path.exists():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            epoch_value = state.get("epoch", None)
            if epoch_value is None:
                return None
            return int(float(epoch_value))
        except Exception as e:
            print(f"⚠️ 解析 checkpoint epoch 失敗: {state_path} ({e})")
            return None
    
    def update_negative_samples(self):
        """更新訓練資料集的負樣本"""
        
        print(f"🔹 正在為第{self.current_epoch}個epoch計算適應性負樣本...")
        
        try:
            # 確保輸出目錄存在
            os.makedirs(self.finetune_data_dir, exist_ok=True)

            new_data = []
            sampling_stats = {
                "out_of_scope": 0,
                "unavailable": 0,
            }
            sampling_diagnostics: List[Dict[str, float]] = []
            epoch_sampling_temperature = self._sampling_temperature_for_epoch(
                self.current_epoch
            )
            sampling_rng = np.random.default_rng(
                int(self.args.seed) + 1_000_003 * int(self.current_epoch)
            )
            current_target_ess = self._sampling_target_ess_for_epoch(
                self.current_epoch
            )

            quick_candidate_ids = None
            if QUICK_TEST and _QT_CANDIDATE_FILES:
                quick_candidate_ids = {
                    normalize_case_id(Path(filename).stem)
                    for filename in _QT_CANDIDATE_FILES
                }

            def consume_query_scores(qid: str, similarities: Dict[str, float]) -> None:
                if quick_candidate_ids is not None:
                    similarities = {
                        doc_id: score
                        for doc_id, score in similarities.items()
                        if normalize_case_id(doc_id) in quick_candidate_ids
                    }
                pos_set = self.positives.get(normalize_case_id(qid), set())
                query_samples, skipped_scope, skipped_unavailable = _generate_query_adaptive_samples(
                    qid,
                    similarities,
                    pos_set,
                    max_negatives=self.max_negatives,
                    temperature=epoch_sampling_temperature,
                    target_ess=current_target_ess,
                    query_candidate_scope=self.query_candidate_scope,
                    rng=sampling_rng,
                    diagnostics_sink=sampling_diagnostics,
                )
                new_data.extend(query_samples)
                sampling_stats["out_of_scope"] += skipped_scope
                sampling_stats["unavailable"] += skipped_unavailable

            if self.current_epoch == 0:
                print(
                    "🔹 Epoch 0 使用初始 chunkAgg 模型做 full-scope retrieval；"
                    "不使用 BM25 或預先產生的 negatives"
                )
            else:
                print(
                    f"🔹 Epoch {self.current_epoch} 使用上一個 epoch 結束後的模型"
                    "重新計算 full-scope similarities"
                )
            helper_device = (
                self.args.device
                if hasattr(self.args, 'device')
                else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
            generate_similarity_artifacts(
                self.model,
                self.retrieval_tokenizer,
                helper_device,
                candidate_dir=self.candidate_dataset_path,
                query_dir=self.query_dataset_path,
                query_ids=self.train_qids,
                trec_output_path=Path(self.finetune_data_dir) / f"similarity_scores_epoch{self.current_epoch}.tsv",
                run_tag=f"modernBert_epoch{self.current_epoch}",
                batch_size=self.retrieval_batch_size,
                max_length=DOCUMENT_CHUNK_LENGTH,
                quick_test=QUICK_TEST,
                candidate_files_override=_QT_CANDIDATE_FILES,
                candidate_limit=QT_CAND_K,
                query_limit=QT_QUERY_K,
                query_candidate_scope_path=self.query_candidate_scope_path,
                persist_trec=False,
                retain_scores=False,
                score_consumer=consume_query_scores,
                exclude_self=True,
                autocast_dtype=RETRIEVAL_AUTOCAST_DTYPE,
                chunk_selection_strategy=CHUNK_SELECTION_STRATEGY,
                reuse_query_embeddings=True,
                eligible_document_ids=getattr(self, "eligible_document_ids", None),
            )
            if sampling_stats["out_of_scope"]:
                print(
                    "🔹 Scope filter 排除 "
                    f"{sampling_stats['out_of_scope']} 個不在 query scope 內的 positive pairs"
                )
            if sampling_stats["unavailable"]:
                print(
                    "⚠️ 略過 "
                    f"{sampling_stats['unavailable']} 個未出現在 scoped retrieval scores 的 positive pairs"
                )
            if not new_data:
                raise RuntimeError(
                    "Adaptive negative sampling produced 0 samples. "
                    "請檢查 scope、train qids、positive labels 與 candidate files 是否對齊。"
                )

            if sampling_diagnostics:
                diagnostic_metrics = {}
                for metric_name in (
                    "temperature",
                    "effective_sample_size",
                    "entropy",
                    "top100_probability_mass",
                    "max_probability",
                    "pool_size",
                ):
                    values = [row[metric_name] for row in sampling_diagnostics]
                    diagnostic_metrics[f"sampling/{metric_name}_median"] = float(
                        np.median(values)
                    )
                    diagnostic_metrics[f"sampling/{metric_name}_mean"] = float(
                        np.mean(values)
                    )
                diagnostic_metrics["sampling/source_is_model"] = 1.0
                diagnostic_metrics["sampling/base_temperature"] = float(
                    epoch_sampling_temperature
                )
                diagnostic_metrics["sampling/target_ess"] = float(current_target_ess)
                self.log(diagnostic_metrics)
                print(
                    "🔹 Sampling diagnostics: "
                    "source=model, "
                    f"median_T={diagnostic_metrics['sampling/temperature_median']:.6g}, "
                    f"median_ESS={diagnostic_metrics['sampling/effective_sample_size_median']:.1f}, "
                    "top100_mass="
                    f"{diagnostic_metrics['sampling/top100_probability_mass_median']:.3f}"
                )
            
            # 更新訓練資料集
            if hasattr(self.train_dataset, 'update_data'):
                self.train_dataset.update_data(new_data)
                output_path = self._adaptive_data_path(self.current_epoch)
                temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
                with open(temporary_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, indent=2, ensure_ascii=False)
                os.replace(temporary_path, output_path)
                print(f"✅ 已儲存適應性負樣本資料到 {output_path}")
                print(f"✅ 成功更新 {len(new_data)} 筆負樣本")

            
        except Exception as e:
            print(f"❌ 更新負樣本時發生錯誤: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # on_epoch_begin 由 callback 處理，避免重複責任來源

    def evaluate(self, eval_dataset=None, *args, **kwargs):
        """QUICK_TEST 模式下，只抽樣一小部分 eval_dataset 來跑驗證迴圈。"""
        base_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if QUICK_TEST and base_dataset is not None:
            try:
                dataset_size = len(base_dataset)
                if dataset_size > 0:
                    k = min(QT_QUERY_K, dataset_size)
                    if k < dataset_size:
                        indices = random.sample(range(dataset_size), k)
                        subset = [base_dataset[i] for i in indices]
                        print(f"[QUICK_TEST] Eval subset: {k}/{dataset_size}")
                        return super().evaluate(eval_dataset=RandomSubsetDataset(subset), *args, **kwargs)
            except Exception:
                pass
        return super().evaluate(eval_dataset=eval_dataset, *args, **kwargs)


class AdaptiveNegativeSamplingCallback(TrainerCallback):
    """處理適應性負樣本採樣的Callback"""
    def __init__(self, trainer_instance):
        self.trainer_instance = trainer_instance
        self.last_epoch = -1
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """每個 epoch 開始時用上一個 epoch 結束的模型重挖負樣本。"""
        current_epoch = int(state.epoch) if state.epoch is not None else 0

        # 僅在 epoch 前進時處理一次
        if current_epoch > self.last_epoch:
            self.last_epoch = current_epoch
            if hasattr(self.trainer_instance, 'update_negative_samples'):
                self.trainer_instance.current_epoch = current_epoch
                prepared_epoch = getattr(self.trainer_instance, "_prepared_epoch_index", None)
                if prepared_epoch is not None and current_epoch == prepared_epoch:
                    print(
                        f"\n🔹 第{current_epoch}個epoch已預先準備動態負樣本，"
                        "本輪跳過重算一次"
                    )
                    self.trainer_instance._prepared_epoch_index = None
                    return
                if current_epoch >= 1:
                    print(
                        f"\n🔹 第{current_epoch}個epoch重新推論完整 scope，"
                        "並依上一個 epoch 結束模型的相似度重抽負樣本"
                    )
                    self.trainer_instance.update_negative_samples()


class TemperatureClampCallback(TrainerCallback):
    """Clamp log_temperature after optimizer updates without blocking recovery gradients."""

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None and hasattr(model, "clamp_temperature_parameter_"):
            model.clamp_temperature_parameter_()


def main():
    # 1. 檢查 CPU / GPU
    device = get_device()

    # 中文註解：TASK1_CHUNKAGG_BASE_ENCODER_DIR 指向 continued pretraining 後的 ModernBERT backbone checkpoint。
    ckpt_dir = get_env_path("TASK1_CHUNKAGG_BASE_ENCODER_DIR", required=True)
    assert ckpt_dir is not None
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"找不到繼續預訓練後的 ModernBERT checkpoint: {ckpt_dir}")

    model_name = str(ckpt_dir)
    print("🔹 載入 tokenizer 與模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # 中文註解：chunking 會自行控制長度，避免 tokenizer 對長文本發出截斷警告。
    tokenizer.model_max_length = 1_000_000_000

    # 2. 初始化自定義的 Contrastive Model
    model = ModernBERTContrastive(
        model_name,
        device,
        temperature=INIT_TEMPERATURE,
        max_chunks=MAX_DOCUMENT_CHUNKS,
        chunk_microbatch_size=CHUNK_MICROBATCH_SIZE,
        temperature_min=TEMPERATURE_MIN,
        temperature_max=TEMPERATURE_MAX,
        evidence_loss_weight=EVIDENCE_LOSS_WEIGHT,
        evidence_pool_temperature=EVIDENCE_POOL_TEMPERATURE,
        fusion_residual_max_norm=FUSION_RESIDUAL_MAX_NORM,
        fusion_residual_penalty_weight=FUSION_RESIDUAL_PENALTY_WEIGHT,
    ).to(device)
    print("✅ Tokenizer 與 Model 初始化完成")
    
    # Debug: 檢查模型組件
    print(f"🔍 Debug info:")
    print(f"   - Encoder: {type(model.encoder)}")
    print(f"   - Projector: {type(model.projector)}")
    print(f"   - Chunk fusion: {type(model.chunk_fusion)}")
    print(f"   - Architecture version: {model.ARCHITECTURE_VERSION}")
    print(f"   - Log temperature: {model.log_temperature}, temperature(actual): {model.temperature.item():.8f}")
    print("   - [DOC] seed: normalize(masked_mean(chunks)) + learned bias")
    print("   - Document vector: normalize(base + soft_cap(fusion_residual))")
    print(f"   - Fusion residual max norm: {FUSION_RESIDUAL_MAX_NORM}")
    print(f"   - Fusion residual penalty weight: {FUSION_RESIDUAL_PENALTY_WEIGHT}")
    print("   - Projector: residual MLP with zero-initialized output branch")
    print("   - Loss: single-positive document InfoNCE + chunk-evidence InfoNCE")
    print(f"   - Evidence loss weight: {EVIDENCE_LOSS_WEIGHT}")
    print(f"   - Evidence pooling temperature: {EVIDENCE_POOL_TEMPERATURE}")
    print(f"   - Max negatives per sample: {MAX_NEGATIVES}")
    print(f"   - Init temperature (from env): {INIT_TEMPERATURE}")
    print(f"   - Temperature bounds: [{TEMPERATURE_MIN}, {TEMPERATURE_MAX}]")
    print(f"   - Max document chunks: {MAX_DOCUMENT_CHUNKS}")
    print(f"   - Chunk length: {DOCUMENT_CHUNK_LENGTH}")
    print(f"   - Chunk microbatch size: {CHUNK_MICROBATCH_SIZE}")
    print(f"   - Long-document chunk selection: {CHUNK_SELECTION_STRATEGY}")
    print(f"   - Precision: fp16={ENABLE_FP16}, bf16={ENABLE_BF16}, tf32={ENABLE_TF32}")
    print(f"   - Text cache size: {TEXT_CACHE_SIZE}")
    print(f"   - Chunk cache size: {CHUNK_CACHE_SIZE}")
    print(f"   - Dataloader pin_memory: {ENABLE_PIN_MEMORY}")
    print("   - Dataloader persistent_workers: False (required by epoch-wise dataset mutation)")
    print(f"   - 新增聚合參數量: {model.added_parameter_count():,}")
    try:
        sample_param = next(model.parameters())
        print(f"   - Model dtype/device: {sample_param.dtype} @ {sample_param.device}")
    except StopIteration:
        pass
    print()

    # 3. 設定路徑
    doc_folder = f"{TASK1_DIR}/processed"
    query_dataset_path = f"{TASK1_DIR}/processed" #query可以用processed或processed_new資料夾下的文件
    eligible_document_ids, rejected_document_lengths = scan_eligible_document_ids(
        doc_folder,
        minimum_characters=MIN_DOCUMENT_CONTENT_CHARS,
    )
    if not eligible_document_ids:
        raise RuntimeError(
            f"No documents have at least {MIN_DOCUMENT_CONTENT_CHARS} non-whitespace characters"
        )
    print(
        "🔹 Document content filter: "
        f"kept {len(eligible_document_ids)}, rejected {len(rejected_document_lengths)}, "
        f"minimum={MIN_DOCUMENT_CONTENT_CHARS} non-whitespace characters"
    )
    if rejected_document_lengths:
        print(
            "   - Rejected documents: "
            f"{summarize_rejected_documents(rejected_document_lengths)}"
        )
    train_qid_path = f"{TASK1_DIR}/train_qid.tsv"
    positive_train_json_path = f"{TASK1_DIR}/task1_train_labels_{TASK1_YEAR}_train.json"
    default_valid_json_path = (
        f"{TASK1_DIR}/lht_process/modernBert_scopeFilteredRaw_plus0_emptyExclude/"
        "finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json"
    )
    valid_json_path = str(get_env_path(
        "TASK1_CHUNKAGG_VALID_JSON_PATH",
        default=default_valid_json_path,
        required=False,
    ))
    valid_qid_path = f"{TASK1_DIR}/valid_qid.tsv"  # Define valid_qid_path
    labels_path = f"{TASK1_DIR}/task1_train_labels_{TASK1_YEAR}.json"  # Define labels_path
    # 中文註解：TASK1_CHUNKAGG_FINETUNE_DATA_DIR 用來存 adaptive negatives 與 retrieval artifacts。
    finetune_data_dir = get_env_path("TASK1_CHUNKAGG_FINETUNE_DATA_DIR", required=True)
    assert finetune_data_dir is not None
    retrieval_batch_size = RETRIEVAL_BATCH_SIZE

    # 中文註解：TASK1_CHUNKAGG_OUTPUT_DIR 為 Trainer checkpoint 輸出根目錄。
    base_output_dir = get_env_path("TASK1_CHUNKAGG_OUTPUT_DIR", required=True)
    assert base_output_dir is not None
    base_output_dir = str(base_output_dir)
    finetune_data_dir = str(finetune_data_dir)
    os.makedirs(finetune_data_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)

    default_scope_path = (
        f"{TASK1_DIR}/lht_process/scope_compare/"
        "query_candidate_scope_raw_plus0_empty_exclude.json"
    )
    chunkagg_scope_path = get_env_path(
        "TASK1_CHUNKAGG_SCOPE_PATH",
        default=default_scope_path,
        required=True,
    )
    assert chunkagg_scope_path is not None
    if not chunkagg_scope_path.exists():
        raise FileNotFoundError(
            f"找不到必要的 TASK1_CHUNKAGG_SCOPE_PATH: {chunkagg_scope_path}"
        )
    os.environ["LCR_QUERY_CANDIDATE_SCOPE_JSON"] = str(chunkagg_scope_path)
    query_candidate_scope = load_query_candidate_scope(chunkagg_scope_path)
    print(f"🔹 使用必要的 query candidate scope: {chunkagg_scope_path}")
    

    # QUICK_TEST: 準備縮小的 candidate 與 query 清單（若啟用）
    if QUICK_TEST:
        try:
            global _QT_CANDIDATE_FILES, _QT_TRAIN_QIDS, _QT_VALID_QIDS
            available_candidates = set(eligible_document_ids)
            all_train_qids = load_query_ids_from_utils(train_qid_path)
            train_positives = read_positive_pairs_from_json(positive_train_json_path)
            eligible_qids = []
            for qid in all_train_qids:
                normalized_qid = normalize_case_id(qid)
                allowed = set(query_candidate_scope.get(normalized_qid, [])) & available_candidates
                positives = train_positives.get(normalized_qid, set())
                if allowed & positives and allowed - positives - {normalized_qid}:
                    eligible_qids.append(normalized_qid)
            if not eligible_qids:
                raise RuntimeError("QUICK_TEST 找不到同時具有 scoped positive/negative 的 query")
            _QT_TRAIN_QIDS = random.sample(
                eligible_qids,
                min(QT_QUERY_K, len(eligible_qids)),
            )
            _QT_VALID_QIDS = load_query_ids_from_utils(valid_qid_path)
            if len(_QT_VALID_QIDS) > QT_QUERY_K:
                _QT_VALID_QIDS = random.sample(_QT_VALID_QIDS, QT_QUERY_K)

            required_candidates = set()
            candidate_pool = set()
            for qid in _QT_TRAIN_QIDS:
                allowed = set(query_candidate_scope[qid]) & available_candidates
                positives = train_positives[qid] & allowed
                negatives = allowed - train_positives[qid] - {qid}
                required_candidates.add(sorted(positives)[0])
                required_candidates.add(random.choice(sorted(negatives)))
                candidate_pool.update(allowed)
            remaining_pool = sorted(candidate_pool - required_candidates)
            fill_count = min(max(0, QT_CAND_K - len(required_candidates)), len(remaining_pool))
            selected_candidates = required_candidates | set(random.sample(remaining_pool, fill_count))
            _QT_CANDIDATE_FILES = [f"{doc_id}.txt" for doc_id in sorted(selected_candidates)]

            print(f"[QUICK_TEST] Prepared {len(_QT_CANDIDATE_FILES)} candidates, train_q={len(_QT_TRAIN_QIDS)}, valid_q={len(_QT_VALID_QIDS)}")
        except Exception as e:
            print(f"[QUICK_TEST] init error: {e}")

    # 4. 建立初始訓練資料集（使用空的資料，稍後會被adaptive sampling更新）
    train_dataset = ContrastiveDataset(
        doc_folder=doc_folder,
        data=[],
        query_candidate_scope=query_candidate_scope,
        eligible_document_ids=eligible_document_ids,
        minimum_content_characters=MIN_DOCUMENT_CONTENT_CHARS,
    )

    # 建立驗證資料集
    valid_dataset = ContrastiveDataset(
        json_path=valid_json_path,
        doc_folder=doc_folder,
        query_candidate_scope=query_candidate_scope,
        eligible_document_ids=eligible_document_ids,
        minimum_content_characters=MIN_DOCUMENT_CONTENT_CHARS,
    )
    print(f"valid_dataset: {len(valid_dataset)}")
    print(
        "   - 上述 valid contrastive JSON 只計算輔助 eval_loss/acc；"
        "不會提供訓練 negatives，也不參與動態負樣本抽樣"
    )

    # 5. 設定 TrainingArguments
    logging_dir = os.path.join(base_output_dir, "tb")

    per_device_train_batch_size = max(1, _get_env_int("TASK1_CHUNKAGG_TRAIN_BATCH_SIZE", 1))
    if per_device_train_batch_size != 1:
        raise ValueError(
            "TASK1_CHUNKAGG_TRAIN_BATCH_SIZE must be 1 because dynamic scoped negatives "
            "can have variable counts per sample."
        )
    gradient_accumulation_steps = max(1, _get_env_int("TASK1_CHUNKAGG_GRAD_ACCUM_STEPS", 4))
    per_device_eval_batch_size = max(1, _get_env_int("TASK1_CHUNKAGG_EVAL_BATCH_SIZE", 1))
    num_train_epochs = _get_env_float("TASK1_CHUNKAGG_NUM_EPOCHS", 20.0)
    encoder_lr = _get_env_float("TASK1_CHUNKAGG_ENCODER_LR", 5e-6)
    projector_lr = _get_env_float("TASK1_CHUNKAGG_PROJECTOR_LR", 1e-5)
    fusion_lr = _get_env_float("TASK1_CHUNKAGG_FUSION_LR", 1e-5)
    temperature_lr = _get_env_float("TASK1_CHUNKAGG_TEMPERATURE_LR", 1e-4)
    warmup_ratio = _get_env_float("TASK1_CHUNKAGG_WARMUP_RATIO", 0.1)
    lr_scheduler_type = os.getenv("TASK1_CHUNKAGG_LR_SCHEDULER_TYPE", "linear").strip().lower()
    train_num_workers = max(0, _get_env_int("TASK1_CHUNKAGG_NUM_WORKERS", 8))
    save_total_limit = max(1, _get_env_int("TASK1_CHUNKAGG_SAVE_TOTAL_LIMIT", 20))
    logging_steps = max(1, _get_env_int("TASK1_CHUNKAGG_LOGGING_STEPS", 50))
    early_stopping_patience = max(
        1,
        _get_env_int(
            "TASK1_CHUNKAGG_EARLY_STOPPING_PATIENCE",
            _get_env_int("TASK1_EARLY_STOPPING_PATIENCE", 5),
        ),
    )
    sampling_temperature = _get_env_float("TASK1_CHUNKAGG_SAMPLING_TEMPERATURE", 0.15)
    sampling_temperature_schedule = os.getenv(
        "TASK1_CHUNKAGG_SAMPLING_TEMPERATURE_SCHEDULE",
        "0:0.20,3:0.15,5:0.10",
    ).strip()
    sampling_target_ess = max(
        0.0,
        _get_env_float("TASK1_CHUNKAGG_SAMPLING_TARGET_ESS", 300.0),
    )
    sampling_target_ess_schedule = os.getenv(
        "TASK1_CHUNKAGG_SAMPLING_TARGET_ESS_SCHEDULE",
        DEFAULT_SAMPLING_TARGET_ESS_SCHEDULE,
    ).strip()
    parsed_target_ess_schedule = _parse_sampling_target_ess_schedule(
        sampling_target_ess_schedule
    )
    # Formal model-only mining uses only explicit negatives sampled from the
    # previous epoch model. Cross-step cached positives would be another source.
    negative_cache_size = 0
    if REQUESTED_PERSISTENT_WORKERS and train_num_workers > 0:
        print(
            "⚠️ TASK1_CHUNKAGG_PERSISTENT_WORKERS=1 已被忽略："
            "dynamic dataset 每個 epoch 會更換，persistent workers 可能持續使用舊資料。"
        )

    args = TrainingArguments(
        output_dir=base_output_dir,
        dataloader_num_workers=train_num_workers,
        dataloader_pin_memory=ENABLE_PIN_MEMORY,
        dataloader_persistent_workers=False,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=ENABLE_FP16,
        bf16=ENABLE_BF16,
        tf32=ENABLE_TF32,
        learning_rate=encoder_lr,           # 給 encoder
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        optim="adamw_torch_fused",  # 使用穩定的 AdamW 以配合 bf16 + TF32
        logging_strategy="steps",
        logging_steps=logging_steps,
        logging_first_step=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        # Use full-corpus valid retrieval F1 as selection metric
        metric_for_best_model="eval_global_f1",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],  # 啟用TensorBoard
        prediction_loss_only=False,
        logging_dir=logging_dir,
    )
    if hasattr(args, "restore_callback_states_from_checkpoint"):
        # Match the single-chunk trainer: resume optimizer/scheduler/scaler,
        # but do not restore an already-exhausted EarlyStoppingCallback state.
        args.restore_callback_states_from_checkpoint = False
    args.encoder_lr = encoder_lr
    args.projector_lr = projector_lr
    args.fusion_lr = fusion_lr
    args.temperature_lr = temperature_lr
    print(f"   - Trainer bf16/fp16/tf32: bf16={args.bf16}, fp16={args.fp16}, tf32={args.tf32}")
    print(f"   - Retrieval batch size: {retrieval_batch_size}")
    print(f"   - Encoder LR: {args.encoder_lr}")
    print(f"   - Projector LR: {args.projector_lr}")
    print(f"   - Fusion LR: {args.fusion_lr}")
    print(f"   - Temperature LR: {args.temperature_lr}")
    print(f"   - Warmup ratio: {warmup_ratio}")
    print(f"   - LR scheduler: {lr_scheduler_type}")
    print(f"   - Train batch size: {per_device_train_batch_size}")
    print(f"   - Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   - Eval batch size: {per_device_eval_batch_size}")
    print(f"   - Logging steps: {logging_steps}")
    print(f"   - Num train epochs: {num_train_epochs}")
    print(f"   - Early stopping patience: {early_stopping_patience}")
    if hasattr(args, "restore_callback_states_from_checkpoint"):
        print(f"   - Restore callback states: {args.restore_callback_states_from_checkpoint}")
    print("   - Dataloader persistent workers: False")
    print(f"   - Sampling temperature: {sampling_temperature}")
    print(f"   - Sampling temperature schedule: {sampling_temperature_schedule}")
    print(f"   - Sampling target ESS: {sampling_target_ess}")
    print(f"   - Sampling target ESS schedule: {parsed_target_ess_schedule}")
    print(f"   - Minimum document content: {MIN_DOCUMENT_CONTENT_CHARS} non-whitespace chars")
    print("   - Dynamic-negative source: current model for every epoch")
    print(f"   - Detached negative cache size: {negative_cache_size}")
    print("   - Negative update frequency: every epoch (fixed)")
    print(f"   - Scope path: {chunkagg_scope_path}")

    # 6. 建立 Trainer（使用 AdaptiveNegativeSamplingTrainer）
    trainer = AdaptiveNegativeSamplingTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=ContrastiveCollator(
            tokenizer,
            max_length=DOCUMENT_CHUNK_LENGTH,
            max_chunks=MAX_DOCUMENT_CHUNKS,
            chunk_cache_size=CHUNK_CACHE_SIZE,
            chunk_selection_strategy=CHUNK_SELECTION_STRATEGY,
        ),
        processing_class=tokenizer,
        compute_metrics=make_compute_metrics_for_retrieval(
            model=model,
            tokenizer=tokenizer,
            candidate_dataset_path=doc_folder,
            query_dataset_path=query_dataset_path,
            train_qid_path=train_qid_path,
            valid_qid_path=valid_qid_path,
            labels_path=labels_path,
            retrieval_batch_size=retrieval_batch_size,
            query_candidate_scope_path=chunkagg_scope_path,
            eligible_document_ids=eligible_document_ids,
            retrieval_autocast_dtype=RETRIEVAL_AUTOCAST_DTYPE,
            chunk_selection_strategy=CHUNK_SELECTION_STRATEGY,
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            TemperatureClampCallback(),
        ],
        candidate_dataset_path=doc_folder,
        query_dataset_path=query_dataset_path,
        train_qid_path=train_qid_path,
        positive_train_json_path=positive_train_json_path,
        finetune_data_dir=finetune_data_dir,
        sampling_temperature=sampling_temperature,  # 可以調整這個參數來控制負樣本選擇的隨機性
        sampling_temperature_schedule=sampling_temperature_schedule,
        sampling_target_ess=sampling_target_ess,
        sampling_target_ess_schedule=sampling_target_ess_schedule,
        retrieval_batch_size=retrieval_batch_size,
        query_candidate_scope_path=chunkagg_scope_path,
        eligible_document_ids=eligible_document_ids,
        max_negatives=MAX_NEGATIVES,
        negative_cache_size=negative_cache_size,
    )

    # 在每個 epoch 開始時依據最新模型重算相似度並重抽負樣本
    trainer.add_callback(AdaptiveNegativeSamplingCallback(trainer))

    print("🔹 Trainer 設定完成，開始訓練並驗證...\n")
    # Summary line for QUICK_TEST
    try:
        cand_count = (
            len(_QT_CANDIDATE_FILES)
            if QUICK_TEST and _QT_CANDIDATE_FILES is not None
            else len(eligible_document_ids)
        )
        q_count = (
            len(_QT_TRAIN_QIDS)
            if QUICK_TEST and _QT_TRAIN_QIDS is not None
            else sum(
                normalize_case_id(qid) in eligible_document_ids
                for qid in load_query_ids_from_utils(train_qid_path)
            )
        )
        print(f"QUICK_TEST={QUICK_TEST} | candidates={cand_count} | queries={q_count}")
    except Exception:
        print("QUICK_TEST summary error")

    # 7. 可中斷後續訓：優先使用指定checkpoint，否則自動找 output_dir 最新checkpoint
    explicit_resume_ckpt = os.getenv("TASK1_RESUME_FROM_CHECKPOINT", "").strip()
    auto_resume_flag = os.getenv("TASK1_AUTO_RESUME", "1").strip().lower() not in {"0", "false", "no"}
    resume_from_checkpoint = None
    if explicit_resume_ckpt:
        if not os.path.isdir(explicit_resume_ckpt):
            raise FileNotFoundError(f"TASK1_RESUME_FROM_CHECKPOINT 不存在: {explicit_resume_ckpt}")
        resume_from_checkpoint = explicit_resume_ckpt
        print(f"🔹 使用指定 checkpoint 續訓: {resume_from_checkpoint}")
    elif auto_resume_flag:
        last_ckpt = get_last_checkpoint(args.output_dir)
        if last_ckpt:
            resume_from_checkpoint = last_ckpt
            print(f"🔹 偵測到最新 checkpoint，將自動續訓: {resume_from_checkpoint}")
        else:
            print("🔹 未找到 checkpoint，將從頭開始訓練")
    else:
        print("🔹 TASK1_AUTO_RESUME=0，將從頭開始訓練")
        output_path = Path(args.output_dir)
        existing_checkpoints = sorted(output_path.glob("checkpoint-*"))
        existing_events = sorted((output_path / "tb").glob("**/events.out.tfevents.*"))
        if existing_checkpoints or existing_events:
            raise RuntimeError(
                "TASK1_AUTO_RESUME=0 但 output directory 已有 checkpoints 或 TensorBoard events。"
                "為避免覆寫 checkpoint 或混合曲線，請先改用空的 TASK1_CHUNKAGG_OUTPUT_DIR。"
            )

    if resume_from_checkpoint:
        # Strictly validate architecture before Trainer restores optimizer/scheduler.
        # This prevents legacy pure-[DOC] checkpoints from being loaded partially.
        model.load_checkpoint(resume_from_checkpoint)
        print("🔹 續訓 checkpoint 架構與權重已嚴格驗證；Trainer 將接續 optimizer/scheduler/scaler")
    else:
        print(
            "🔹 從 continued-pretraining backbone 從頭訓練 chunkAgg："
            "encoder 來自 TASK1_CHUNKAGG_BASE_ENCODER_DIR，projector/fusion/temperature 為本次初始化，"
            "不載入 single-chunk retrieval warm-start"
        )

    # （可選）檢查各參數組 LR。需在可選 checkpoint 初始化後建立 optimizer，避免後續才改 requires_grad。
    for i, g in enumerate(trainer.create_optimizer().param_groups):
        sz = sum(p.numel() for p in g["params"])
        group_name = g.get("group_name", f"group_{i}")
        print(f"{group_name}: lr={g['lr']}  weight_decay={g['weight_decay']}  #params={sz}")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(
        "\n✅ single-positive document/evidence InfoNCE 訓練與驗證完成！"
        f"模型已儲存於：{args.output_dir}"
    )

if __name__ == "__main__":
    main()
