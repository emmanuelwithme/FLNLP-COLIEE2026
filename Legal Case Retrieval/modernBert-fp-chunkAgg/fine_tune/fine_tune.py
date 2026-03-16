import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import json
from torch import nn
import contextlib
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
from transformers import (
    AutoTokenizer,
    ModernBertModel,
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
import pynvml
pynvml.nvmlInit()
nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

# 添加路徑來import自定義模組
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'utils'))

from env_utils import load_chunkagg_dotenv

_LOADED_DOTENV_PATH = load_chunkagg_dotenv(__file__)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

from lcr.data import EmbeddingsData
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
import time
from lcr.data import load_query_ids as load_query_ids_from_utils
from lcr.device import get_device
from lcr.metrics import (
    my_classification_report,
    rel_file_to_dict as rel_file_convert,
    trec_file_to_dict as trec_file_convert,
)
from lcr.retrieval import _build_document_batch, _chunk_single_text, generate_similarity_artifacts

# Global holder for retrieval results to reuse in TB logging callback
_LATEST_RETRIEVAL_RESULTS = None
_EVAL_EPOCH_TAG = None  # 用於在評估時以 epoch 編號命名輸出檔


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
# 中文註解：TASK1_CHUNKAGG_SCOPE_FILTER=1 時，retrieval 只在同年份範圍內 candidates 計分。
SCOPE_FILTER = _get_env_bool("TASK1_CHUNKAGG_SCOPE_FILTER", True)
# 中文註解：TASK1_RETRIEVAL_BATCH_SIZE 控制 retrieval / adaptive negative sampling 的文件編碼 batch size。
RETRIEVAL_BATCH_SIZE = max(1, _get_env_int("TASK1_RETRIEVAL_BATCH_SIZE", 8))
# 中文註解：TASK1_INIT_TEMPERATURE 是對比式學習初始溫度。
INIT_TEMPERATURE = _get_env_float("TASK1_INIT_TEMPERATURE", 0.2222)
if INIT_TEMPERATURE <= 0:
    raise ValueError(f"TASK1_INIT_TEMPERATURE must be > 0, got: {INIT_TEMPERATURE}")
# 中文註解：TASK1_MAX_DOCUMENT_CHUNKS 控制每篇文件最多保留幾個 chunk。
MAX_DOCUMENT_CHUNKS = max(1, _get_env_int("TASK1_MAX_DOCUMENT_CHUNKS", 3))
# 中文註解：TASK1_DOCUMENT_CHUNK_LENGTH 控制每個 chunk 的最長 token 數，含 special tokens。
DOCUMENT_CHUNK_LENGTH = max(8, _get_env_int("TASK1_DOCUMENT_CHUNK_LENGTH", 4096))
# 中文註解：TASK1_CHUNK_MICROBATCH_SIZE 控制同一次 encoder forward 送幾個 chunk，避免 OOM。
CHUNK_MICROBATCH_SIZE = max(1, _get_env_int("TASK1_CHUNK_MICROBATCH_SIZE", 1))
# 中文註解：TASK1_CHUNKAGG_ENABLE_TF32=1 時，允許 Ampere/Ada GPU 用 TF32 加速 matmul。
ENABLE_TF32 = _get_env_bool("TASK1_CHUNKAGG_ENABLE_TF32", True)
# 中文註解：TASK1_CHUNKAGG_TEXT_CACHE_SIZE 控制資料集層的原始文本快取數量。
TEXT_CACHE_SIZE = max(0, _get_env_int("TASK1_CHUNKAGG_TEXT_CACHE_SIZE", 4096))
# 中文註解：TASK1_CHUNKAGG_CHUNK_CACHE_SIZE 控制 collator 層的 chunk tokenization 快取數量。
CHUNK_CACHE_SIZE = max(0, _get_env_int("TASK1_CHUNKAGG_CHUNK_CACHE_SIZE", 1024))
# 中文註解：TASK1_CHUNKAGG_PIN_MEMORY / TASK1_CHUNKAGG_PERSISTENT_WORKERS 控制 dataloader 傳輸效率。
ENABLE_PIN_MEMORY = _get_env_bool("TASK1_CHUNKAGG_PIN_MEMORY", True)
ENABLE_PERSISTENT_WORKERS = _get_env_bool("TASK1_CHUNKAGG_PERSISTENT_WORKERS", True)

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
                           train_qid_path, valid_qid_path, labels_path, output_dir, epoch_num, topk=5,
                           retrieval_batch_size: int = RETRIEVAL_BATCH_SIZE):
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
    
    results = {}
    
    for split, (qids, rel_dict) in [("train", (train_qids, train_rel_dict)), 
                                    ("valid", (valid_qids, valid_rel_dict))]:
        print(f"🔍 評估 {split} set...")
        
        # 生成相似度分數並保存TREC檔案
        epoch_tag = f"{epoch_num}_eval_{split}"
        artifacts = generate_similarity_artifacts(
            model,
            tokenizer,
            device,
            candidate_dir=candidate_dataset_path,
            query_dir=query_dataset_path,
            query_ids=qids,
            trec_output_path=Path(output_dir) / f"similarity_scores_{epoch_tag}.tsv",
            run_tag=f"modernBert_{epoch_tag}",
            batch_size=retrieval_batch_size,
            max_length=DOCUMENT_CHUNK_LENGTH,
            quick_test=QUICK_TEST,
            candidate_files_override=_QT_CANDIDATE_FILES,
            candidate_limit=QT_CAND_K,
            query_limit=QT_QUERY_K,
        )
        query_id_to_similarities = artifacts.scores
        
        # 讀取生成的TREC檔案
        trec_path = str(artifacts.trec_path)
        answer_dict = trec_file_convert(trec_path, topk)
        
        # 準備評估資料
        list_answer_ohe = []  # 預測答案
        list_label_ohe = []   # 真實答案
        
        for qid in rel_dict.keys():
            if qid in answer_dict:
                one_answer = answer_dict[qid]  # 預測
                one_rel = rel_dict[qid]        # 真實
                one_answer = [int(pid) for pid in one_answer]
                one_rel = [int(pid) for pid in one_rel]
                list_answer_ohe.append(one_answer)
                list_label_ohe.append(one_rel)
        
        # 計算評估指標
        f1, precision, recall = my_classification_report(list_label_ohe, list_answer_ohe)
        
        results[split] = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'num_queries': len(list_answer_ohe)
        }
        
        print(f"✅ {split} set 結果: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    
    return results




def read_positive_pairs_from_json(json_path: str) -> Dict[str, Set[str]]:
    """從 JSON 讀取正樣本對映表，並去除 .txt 副檔名"""
    positives = defaultdict(set)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for q_txt, pos_list in data.items():
        qid = q_txt.replace(".txt", "")
        for doc_txt in pos_list:
            doc_id = doc_txt.replace(".txt", "")
            positives[qid].add(doc_id)
    return positives


def generate_adaptive_negative_samples(query_id_to_similarities, positives, max_negatives=15, temperature=1.0):
    """
    根據相似度分數作為機率來選擇負樣本
    """
    dataset = []
    
    for qid, pos_set in positives.items():
        if qid not in query_id_to_similarities:
            continue
            
        similarities = query_id_to_similarities[qid]
        
        for pos_id in pos_set:
            # 過濾出不是正樣本的文件作為負樣本候選
            negative_candidates = []
            negative_scores = []
            
            for doc_id, score in similarities.items():
                # 排除正樣本與查詢自身
                if doc_id not in pos_set and str(doc_id) != str(qid):
                    negative_candidates.append(doc_id)
                    negative_scores.append(score)
            
            # 若可選負樣本數不足 max_negatives，允許重複抽樣以保持每筆樣本負樣本數一致
            if len(negative_candidates) > 0:
                # 將相似度分數轉換為機率（使用softmax with temperature）
                scores_tensor = torch.tensor(negative_scores) / temperature
                probs = F.softmax(scores_tensor, dim=0).numpy()

                replace_flag = len(negative_candidates) < max_negatives
                selected_negatives = np.random.choice(
                    negative_candidates,
                    size=max_negatives,
                    replace=replace_flag,
                    p=probs,
                )

                dataset.append({
                    "query_id": qid,
                    "positive_id": pos_id,
                    "negative_ids": selected_negatives.tolist(),
                })
    
    return dataset

# 自訂 Callback，把紀錄寫到 TensorBoard
class TensorBoardExtras(TrainerCallback):
    def __init__(self):
        self.writer = None

    def _ensure_writer(self, args):
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            logdir = os.path.join(args.output_dir, "tb", "extras")  # 與官方 writer 分開存
            os.makedirs(logdir, exist_ok=True)
            self.writer = SummaryWriter(logdir)

    def on_train_begin(self, args, state, control, **kwargs):
        self._ensure_writer(args)

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 中文註解：標準 train/eval 指標交給 Hugging Face 內建 TensorBoard writer，
        # 避免和 report_to=["tensorboard"] 重複寫入。
        self._ensure_writer(args)

    def on_step_end(self, args, state, control, **kwargs):
        # 這裡額外寫 temperature 與每組學習率
        self._ensure_writer(args)
        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)

        if model is not None and hasattr(model, "log_temperature"):
            temp = model.log_temperature.exp().item()
            self.writer.add_scalar("train/temperature", temp, state.global_step)

        if optimizer is not None:
            for i, g in enumerate(optimizer.param_groups):
                lr = float(g.get("lr", 0.0))
                self.writer.add_scalar(f"train/lr_group_{i}", lr, state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # 中文註解：eval_* 指標由官方 writer 產生，不在 extras writer 重複記錄。
        return

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer:
            self.writer.flush()
            self.writer.close()

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

# ------------------------
# 隨機抽樣評估的 Trainer（保留）
# ------------------------
class RandEvalTrainer(Trainer):
    def __init__(self, *args, num_eval_samples: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_eval_samples = num_eval_samples
    def evaluate(self, eval_dataset=None, *args, **kwargs):
        base_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        dataset_size = len(base_dataset)
        if dataset_size == 0:
            raise ValueError("基底驗證集長度為 0，無法抽樣。")
        k = min(self.num_eval_samples, dataset_size)
        picked_indices = random.sample(range(dataset_size), k)
        subset = [base_dataset[i] for i in picked_indices]
        rand_subset = RandomSubsetDataset(subset)
        return super().evaluate(eval_dataset=rand_subset, *args, **kwargs)

# 生成假的極限長度文本
def generate_fake_text(word_count=20000):
    vocab = ["city", "building", "traffic", "light", "road", "car", "signal", "street", "corner", "park",
             "tree", "bridge", "sky", "people", "crosswalk", "bus", "train", "station", "bike", "walk"]
    return " ".join(random.choices(vocab, k=word_count))

def generate_fake_sample():
    return {
        "query_text": generate_fake_text(),
        "positive_text": generate_fake_text(),
        "negative_texts": [generate_fake_text() for _ in range(15)]
    }

class FakeContrastiveDataset(Dataset):
    def __init__(self, n_samples: int = 4):
        self.data = [generate_fake_sample() for _ in range(n_samples)]
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
                                       output_dir,
                                       retrieval_batch_size: int = RETRIEVAL_BATCH_SIZE):
    """Factory to create compute_metrics that also computes full-corpus retrieval metrics.

    Returns a function taking EvalPrediction and returning a metrics dict including:
    - global_f1 (will be surfaced as eval_global_f1)
    - acc1, acc5 (for reference)
    - loss (recomputed if not provided)
    Also writes retrieval/* metrics to a global holder for TB logging.
    """
    def _compute(eval_pred: EvalPrediction):
        global _LATEST_RETRIEVAL_RESULTS, _EVAL_EPOCH_TAG

        # 1) Keep quick classification-style metrics for reference
        metrics = {}
        try:
            logits = torch.tensor(eval_pred.predictions)
            labels = torch.tensor(eval_pred.label_ids)
            preds_top1 = logits.argmax(dim=1).cpu().numpy()
            metrics["acc1"] = accuracy_score(labels.cpu().numpy(), preds_top1)
            top5_preds = torch.topk(logits, k=5, dim=1).indices
            labels_expanded = labels.view(-1, 1).expand_as(top5_preds)
            metrics["acc5"] = (top5_preds == labels_expanded).any(dim=1).float().mean().item()
            if hasattr(eval_pred, "losses") and eval_pred.losses is not None:
                metrics["loss"] = float(np.mean(eval_pred.losses))
            else:
                metrics["loss"] = nn.CrossEntropyLoss()(logits, labels).item()
        except Exception:
            pass

        # 2) Full-corpus retrieval over train/valid; 使用 epoch 編號命名（若可用），否則退回 timestamp
        unique_epoch_tag = str(_EVAL_EPOCH_TAG) if _EVAL_EPOCH_TAG is not None else f"cm_{int(time.time())}"
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
            output_dir=output_dir,
            epoch_num=unique_epoch_tag,
            topk=5,
            retrieval_batch_size=retrieval_batch_size,
        )

        # 3) Global metric for best model selection/early stopping
        global_f1 = float(results.get("valid", {}).get("f1", 0.0))
        metrics["global_f1"] = global_f1

        # Persist for TB logging callback and print to stdout
        _LATEST_RETRIEVAL_RESULTS = results
        print(f"eval_global_f1: {global_f1:.6f}")

        return metrics

    return _compute

# ----------- Dataset 讀檔 -----------
class ContrastiveDataset(Dataset):
    def __init__(self, json_path: str = None, doc_folder: str = None, data: List[Dict] = None):
        if data is not None:
            self.data = data
        elif json_path is not None:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = []
        
        self.doc_folder = doc_folder
        # 中文註解：把常重複讀取的判決書文字放進 LRU cache，減少磁碟 I/O。
        self.text_cache_size = TEXT_CACHE_SIZE
        self._text_cache = OrderedDict()
        if json_path:
            print(f"🔹 對比式資料（{os.path.basename(json_path)}）共載入 {len(self.data)} 筆樣本")
        else:
            print(f"🔹 對比式資料共載入 {len(self.data)} 筆樣本")
    
    def update_data(self, new_data: List[Dict]):
        """更新資料集的負樣本"""
        self.data = new_data
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

        if self.text_cache_size > 0:
            self._text_cache[doc_id] = text
            while len(self._text_cache) > self.text_cache_size:
                self._text_cache.popitem(last=False)
        return text
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "query_id": sample["query_id"],
            "query_text": self.load_text(sample["query_id"]),
            "positive_id": sample["positive_id"],
            "positive_text": self.load_text(sample["positive_id"]),
            "negative_ids": sample["negative_ids"],
            "negative_texts": [self.load_text(nid) for nid in sample["negative_ids"]]
        }

# ----------- Collator -----------
@dataclass
class ContrastiveCollator:
    tokenizer: AutoTokenizer
    max_length: int = DOCUMENT_CHUNK_LENGTH
    max_chunks: int = MAX_DOCUMENT_CHUNKS
    chunk_cache_size: int = CHUNK_CACHE_SIZE

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
        if not texts:
            return _build_document_batch(
                texts,
                self.tokenizer,
                max_length=self.max_length,
                max_chunks=self.max_chunks,
                device=None,
            )

        if ids is None or self.chunk_cache_size <= 0:
            return _build_document_batch(
                texts,
                self.tokenizer,
                max_length=self.max_length,
                max_chunks=self.max_chunks,
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

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        q_ids = [item["query_id"] for item in batch]
        q_texts = [item["query_text"] for item in batch]
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

        neg_count = len(n_texts) // bsz
        labels = torch.zeros(bsz, dtype=torch.long)
        return {
            "anchor_input": anchor_batch,
            "positive_input": positive_batch,
            "negative_input": negative_batch,
            "labels": labels,
            "negatives_per_query": neg_count,
        }

# ----------- Model with InfoNCE loss -----------
class ChunkFusionBlock(nn.Module):
    """中文註解：用 1 層 pre-norm transformer block 融合 1~3 個 chunk 向量。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        normed_states = self.attn_norm(hidden_states)
        attn_output, _ = self.self_attn(
            normed_states,
            normed_states,
            normed_states,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden_states = residual + self.attn_dropout(attn_output)

        residual = hidden_states
        normed_states = self.ffn_norm(hidden_states)
        hidden_states = residual + self.ffn(normed_states)
        return hidden_states


class ModernBERTContrastive(nn.Module):
    def __init__(
        self,
        model_name: str,
        device,
        temperature: float = 0.55555,
        max_chunks: int = MAX_DOCUMENT_CHUNKS,
        chunk_microbatch_size: int = CHUNK_MICROBATCH_SIZE,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        # 與 inference.py 保持一致的 encoder_kwargs 設定；fp16 由 Trainer 的 AMP 控制，權重維持 fp32
        device_str = str(device)
        dtype = torch.float32
        device_map = {"": device_str}
        self.encoder = ModernBertModel.from_pretrained(
            model_name,
            device_map=device_map,
            attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        # 確保權重直接放到指定裝置並維持 fp32
        self.encoder.to(device=device, dtype=dtype)
        hidden_dim = self.encoder.config.hidden_size
        self.hidden_dim = hidden_dim
        self.max_chunks = max_chunks
        self.chunk_microbatch_size = max(1, int(chunk_microbatch_size))
        self.supports_chunked_documents = True

        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 中文註解：learnable [DOC] token 與 chunk 位置向量，用來做 chunk-level 聚合。
        self.doc_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.chunk_pos_emb = nn.Parameter(torch.zeros(1, max_chunks + 1, hidden_dim))
        num_heads = getattr(self.encoder.config, "num_attention_heads", 12)
        self.chunk_fusion = ChunkFusionBlock(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=fusion_dropout,
        )
        # 可學習的 temperature（log-param 保證 >0）
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(float(temperature)), dtype=torch.float32)
        )
        self.temperature_min = 1e-2
        self.temperature_max = 2.0
        self.fusion_dropout = fusion_dropout

        self.encoder.config.use_cache = False
        self.encoder.enable_input_require_grads() # 打開訓練效果會好點，可以學習id->embedding
        self.encoder.gradient_checkpointing_enable()
        self._init_chunk_agg_parameters()

    # 供 Trainer 呼叫，維持介面與 huggingface 模型一致
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()

    def _init_chunk_agg_parameters(self):
        # 中文註解：新增聚合層參數沿用 BERT 類型初始化尺度。
        nn.init.normal_(self.doc_token, mean=0.0, std=0.02)
        nn.init.normal_(self.chunk_pos_emb, mean=0.0, std=0.02)
        for module in self.chunk_fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def added_parameter_count(self) -> int:
        # 中文註解：只統計此次 chunk aggregation 額外新增的參數量。
        return (
            self.doc_token.numel()
            + self.chunk_pos_emb.numel()
            + sum(p.numel() for p in self.chunk_fusion.parameters())
        )

    def chunk_aggregation_parameters(self):
        return (
            list(self.projector.parameters())
            + [self.doc_token, self.chunk_pos_emb]
            + list(self.chunk_fusion.parameters())
        )

    def _encode_projected_chunks(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, chunk_count, seq_len = input_ids.shape
        flat_input_ids = input_ids.view(batch_size * chunk_count, seq_len)
        flat_attention_mask = attention_mask.view(batch_size * chunk_count, seq_len)
        # 中文註解：以 attention_mask 再次校正有效 chunk，避免空文本產生 chunk_mask=1 但 token 數為 0。
        flat_chunk_mask = chunk_mask.view(batch_size * chunk_count).bool() & flat_attention_mask.any(dim=1)
        valid_indices = flat_chunk_mask.nonzero(as_tuple=False).flatten()

        flat_embeddings = self.projector[0].weight.new_zeros(
            batch_size * chunk_count,
            self.hidden_dim,
        )
        if valid_indices.numel() == 0:
            return flat_embeddings.view(batch_size, chunk_count, self.hidden_dim)

        for start in range(0, valid_indices.numel(), self.chunk_microbatch_size):
            batch_indices = valid_indices[start : start + self.chunk_microbatch_size]
            encoder_outputs = self.encoder(
                input_ids=flat_input_ids.index_select(0, batch_indices),
                attention_mask=flat_attention_mask.index_select(0, batch_indices),
            )
            cls_vectors = encoder_outputs.last_hidden_state[:, 0, :]
            projected_vectors = self.projector(cls_vectors).to(dtype=flat_embeddings.dtype)
            flat_embeddings.index_copy_(0, batch_indices, projected_vectors)

        return flat_embeddings.view(batch_size, chunk_count, self.hidden_dim)

    def encode_document(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids = input_batch["input_ids"]
        attention_mask = input_batch["attention_mask"]
        chunk_mask = input_batch.get("chunk_mask")

        if input_ids.dim() == 2:
            input_ids = input_ids.unsqueeze(1)
            attention_mask = attention_mask.unsqueeze(1)
            if chunk_mask is None:
                chunk_mask = attention_mask.any(dim=-1).long()
            else:
                chunk_mask = chunk_mask.view(-1, 1)
        elif input_ids.dim() == 3:
            if chunk_mask is None:
                chunk_mask = attention_mask.any(dim=-1).long()
        else:
            raise ValueError(f"Unsupported input_ids ndim: {input_ids.dim()}")

        # 中文註解：chunk_mask 與 attention_mask 必須一致，否則 ModernBERT flash attention 可能在空 chunk 上失敗。
        chunk_mask = (
            chunk_mask.to(device=input_ids.device, dtype=torch.long)
            * attention_mask.any(dim=-1).to(device=input_ids.device, dtype=torch.long)
        )
        chunk_vectors = self._encode_projected_chunks(input_ids, attention_mask, chunk_mask)

        batch_size, chunk_count, _ = chunk_vectors.shape
        doc_tokens = self.doc_token.expand(batch_size, -1, -1)
        fusion_inputs = torch.cat([doc_tokens, chunk_vectors], dim=1)
        fusion_inputs = fusion_inputs + self.chunk_pos_emb[:, : chunk_count + 1, :]

        # 中文註解：padding_mask=True 表示該位置不參與 self-attention。
        padding_mask = torch.cat(
            [
                torch.zeros(batch_size, 1, dtype=torch.bool, device=input_ids.device),
                ~chunk_mask.bool(),
            ],
            dim=1,
        )
        fused_states = self.chunk_fusion(fusion_inputs, padding_mask=padding_mask)
        document_vectors = fused_states[:, 0, :]
        return torch.nn.functional.normalize(document_vectors, p=2, dim=-1)

    def encode(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.encode_document(input_batch)

    def forward(self,
        anchor_input: Dict[str, torch.Tensor],
        positive_input: Dict[str, torch.Tensor],
        negative_input: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        negatives_per_query: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        temperature = self.log_temperature.exp().clamp(self.temperature_min, self.temperature_max)
        bsz = anchor_input["input_ids"].size(0)
        neg_count = negatives_per_query or (negative_input["input_ids"].size(0) // bsz)

        merged_batch = {
            "input_ids": torch.cat(
                [
                    anchor_input["input_ids"],
                    positive_input["input_ids"],
                    negative_input["input_ids"],
                ],
                dim=0,
            ),
            "attention_mask": torch.cat(
                [
                    anchor_input["attention_mask"],
                    positive_input["attention_mask"],
                    negative_input["attention_mask"],
                ],
                dim=0,
            ),
            "chunk_mask": torch.cat(
                [
                    anchor_input["chunk_mask"],
                    positive_input["chunk_mask"],
                    negative_input["chunk_mask"],
                ],
                dim=0,
            ),
        }
        vec_all = self.encode(merged_batch)
        anchor_vec   = vec_all[:bsz]
        positive_vec = vec_all[bsz:bsz*2]
        neg_flat     = vec_all[bsz*2:]
        negative_vec = neg_flat.view(bsz, neg_count, -1)

        # 中文註解：向量已 L2 normalize，內積即 cosine similarity。
        pos_sim = torch.sum(anchor_vec * positive_vec, dim=-1, keepdim=True)
        neg_sim = torch.sum(anchor_vec.unsqueeze(1) * negative_vec, dim=-1)

        logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
        if labels is None:
            labels = torch.zeros(bsz, dtype=torch.long, device=logits.device)
        loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}
    
    def print_gpu_status(self, tag=""):
        util = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
        print(f"🟩 [{tag}] GPU 使用率: {util.gpu}% │ 記憶體: {mem.used / 1024**2:.0f} MB / {mem.total / 1024**2:.0f} MB")

# ----------- 自訂 Trainer：讓 temperature 有獨立 LR 並實現 adaptive negative sampling -----------
class AdaptiveNegativeSamplingTrainer(Trainer):
    def __init__(self, 
                 *args, 
                 candidate_dataset_path: str = None,
                 query_dataset_path: str = None,
                 train_qid_path: str = None,
                 positive_train_json_path: str = None,
                 finetune_data_dir: str = None,
                 sampling_temperature: float = 1.0,
                 update_frequency: int = 1,  # 新增：多少epoch更新一次負樣本
                 retrieval_batch_size: int = RETRIEVAL_BATCH_SIZE,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_dataset_path = candidate_dataset_path
        self.query_dataset_path = query_dataset_path
        self.train_qid_path = train_qid_path
        self.positive_train_json_path = positive_train_json_path
        self.finetune_data_dir = finetune_data_dir
        self.sampling_temperature = sampling_temperature
        self.update_frequency = update_frequency
        self.retrieval_batch_size = max(1, retrieval_batch_size)
        self.current_epoch = 0
        self._prepared_epoch_index = None  # resume 時若已預先準備該 epoch 的負樣本，callback 需略過一次
        
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
        if positive_train_json_path:
            self.positives = read_positive_pairs_from_json(positive_train_json_path)
            
        print(f"🔹 適應性負樣本採樣設定：")
        print(f"   - 負樣本更新頻率：每 {self.update_frequency} 個epoch")
        print(f"   - 採樣溫度：{self.sampling_temperature}")
        print(f"   - Candidate embedding batch size：{self.retrieval_batch_size}")
        print(f"   - Chunk microbatch size：{getattr(self.model, 'chunk_microbatch_size', CHUNK_MICROBATCH_SIZE)}")
        if hasattr(self, 'train_qids'):
            print(f"   - 訓練查詢數量：{len(self.train_qids)}")
        if hasattr(self, 'positives'):
            print(f"   - 正樣本對數量：{len(self.positives)}")
    
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        args = self.args
        model = self.model

        encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
        temp_params = [model.log_temperature] if model.log_temperature.requires_grad else []
        encoder_param_ids = {id(p) for p in encoder_params}
        temp_param_ids = {id(p) for p in temp_params}
        head_params = [
            p for p in model.parameters()
            if p.requires_grad and id(p) not in encoder_param_ids and id(p) not in temp_param_ids
        ]
        encoder_lr = getattr(args, "encoder_lr", args.learning_rate)
        fusion_lr = getattr(args, "fusion_lr", args.learning_rate)
        temperature_lr = getattr(args, "temperature_lr", args.learning_rate)

        optimizer_grouped_parameters = [
            {
                "params": encoder_params,
                "weight_decay": args.weight_decay,
                "lr": encoder_lr,
                "group_name": "encoder",
            },
            {
                "params": head_params,
                "weight_decay": args.weight_decay,
                "lr": fusion_lr,
                "group_name": "fusion_head",
            },
            {
                "params": temp_params,
                "weight_decay": 0.0,
                "lr": temperature_lr,
                "group_name": "log_temperature",
            },
        ]

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
    
    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        """重寫training loop來在每個epoch開始前更新負樣本"""
        train_size = len(self.train_dataset) if self.train_dataset is not None else 0

        # train_dataset 一開始是空的，先確保可訓練資料存在
        if train_size == 0:
            restored = False
            if resume_from_checkpoint:
                print(f"🔹 偵測到 resume_from_checkpoint={resume_from_checkpoint}")
                completed_epoch = self.get_completed_epoch_from_checkpoint(resume_from_checkpoint)
                if completed_epoch is not None:
                    # HF trainer_state 的 epoch 是已完成的 epoch 數；下一輪訓練的 epoch index 即 completed_epoch
                    next_epoch_index = completed_epoch
                    self.current_epoch = next_epoch_index
                    print(
                        f"🔹 依 checkpoint 對齊動態負樣本："
                        f"已完成 epoch={completed_epoch}，將準備下一輪(第{next_epoch_index + 1}輪)資料"
                    )
                    restored = self.load_adaptive_data_for_epoch(next_epoch_index)
                    if not restored:
                        print(
                            f"🔹 找不到 adaptive_negative_epoch{next_epoch_index}_train.json，"
                            "將即時計算該輪動態負樣本"
                        )
                        self.update_negative_samples()
                        restored = len(self.train_dataset) > 0
                    if restored:
                        self._prepared_epoch_index = next_epoch_index
                else:
                    print("⚠️ 無法從 checkpoint 解析 epoch，改用最新 adaptive negatives 檔案")
                    restored = self.load_latest_adaptive_data()
            if not restored:
                print("🔹 找不到可用的 adaptive negatives，改為即時計算一次")
                self.update_negative_samples()
        elif self.current_epoch == 0 and not resume_from_checkpoint:
            print("🔹 第0個epoch開始使用適應性負樣本...")
            self.update_negative_samples()
        
        return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)

    def load_latest_adaptive_data(self) -> bool:
        """從 finetune_data_dir 載入最新 adaptive_negative_epoch*_train.json。"""
        if not self.finetune_data_dir:
            return False
        if not hasattr(self.train_dataset, "update_data"):
            return False

        data_dir = Path(self.finetune_data_dir)
        if not data_dir.exists():
            return False

        candidates = []
        for path in data_dir.glob("adaptive_negative_epoch*_train.json"):
            stem = path.stem  # adaptive_negative_epoch{N}_train
            epoch_str = stem.removeprefix("adaptive_negative_epoch").removesuffix("_train")
            if epoch_str.isdigit():
                candidates.append((int(epoch_str), path))

        if not candidates:
            return False

        candidates.sort(key=lambda x: x[0])
        latest_epoch, latest_path = candidates[-1]
        try:
            with open(latest_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"⚠️ 讀取 adaptive negatives 失敗: {latest_path} ({e})")
            return False

        if not payload:
            print(f"⚠️ adaptive negatives 檔案為空: {latest_path}")
            return False

        self.train_dataset.update_data(payload)
        print(
            f"✅ 已載入最新 adaptive negatives: {latest_path.name} "
            f"(epoch={latest_epoch}, samples={len(payload)})"
        )
        return True

    def load_adaptive_data_for_epoch(self, epoch_index: int) -> bool:
        """載入指定 epoch index 的 adaptive_negative_epoch{index}_train.json。"""
        if not self.finetune_data_dir:
            return False
        if not hasattr(self.train_dataset, "update_data"):
            return False

        target_path = Path(self.finetune_data_dir) / f"adaptive_negative_epoch{epoch_index}_train.json"
        if not target_path.exists():
            return False

        try:
            with open(target_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            print(f"⚠️ 讀取指定 adaptive negatives 失敗: {target_path} ({e})")
            return False

        if not payload:
            print(f"⚠️ 指定 adaptive negatives 檔案為空: {target_path}")
            return False

        self.train_dataset.update_data(payload)
        print(
            f"✅ 已載入對應 epoch 的 adaptive negatives: {target_path.name} "
            f"(epoch={epoch_index}, samples={len(payload)})"
        )
        return True

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
    
    def load_bm25_backup_data(self):
        """載入BM25備用資料作為初始訓練資料"""
        try:
            # 使用專案根目錄為基準的正確路徑
            backup_json_path = f"{TASK1_DIR}/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_train.json"
            if os.path.exists(backup_json_path):
                with open(backup_json_path, 'r', encoding='utf-8') as f:
                    backup_data = json.load(f)
                
                # 使用所有資料
                if hasattr(self.train_dataset, 'update_data'):
                    self.train_dataset.update_data(backup_data)
                    print(f"✅ 載入BM25備用資料，共 {len(backup_data)} 筆樣本")
                else:
                    print("❌ 無法更新訓練資料集")
            else:
                print(f"❌ BM25資料檔案不存在: {backup_json_path}")
        except Exception as e:
            print(f"❌ 載入BM25資料失敗: {e}")
    
    def update_negative_samples(self):
        """更新訓練資料集的負樣本"""
        
        print(f"🔹 正在為第{self.current_epoch}個epoch計算適應性負樣本...")
        
        try:
            # 確保輸出目錄存在
            os.makedirs(self.finetune_data_dir, exist_ok=True)
            
            # 使用模型計算相似度分數
            helper_device = self.args.device if hasattr(self.args, 'device') else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            artifacts = generate_similarity_artifacts(
                self.model,
                self.tokenizer,
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
            )
            query_id_to_similarities = artifacts.scores
            
            # 根據相似度分數生成新的負樣本
            new_data = generate_adaptive_negative_samples(
                query_id_to_similarities=query_id_to_similarities,
                positives=self.positives,
                max_negatives=15,
                temperature=self.sampling_temperature
            )
            
            # 更新訓練資料集
            if hasattr(self.train_dataset, 'update_data'):
                # 嚴格使用模型相似度抽樣的結果，不再退回 BM25
                self.train_dataset.update_data(new_data)
                # 儲存新的訓練資料（即使為空，也落檔以便檢查）
                output_path = os.path.join(self.finetune_data_dir, f"adaptive_negative_epoch{self.current_epoch}_train.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(new_data, f, indent=2, ensure_ascii=False)
                print(f"✅ 已儲存適應性負樣本資料到 {output_path}")
                print(f"✅ 成功更新 {len(new_data)} 筆負樣本")
            
        except Exception as e:
            print(f"❌ 更新負樣本時發生錯誤（僅使用模型相似度，不退回BM25）: {e}")
            import traceback
            traceback.print_exc()
    
    # on_epoch_begin 由 callback 處理，避免重複責任來源

    def evaluate(self, eval_dataset=None, *args, **kwargs):
        """QUICK_TEST 模式下，只抽樣一小部分 eval_dataset 來跑驗證迴圈。"""
        base_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        # 在 compute_metrics 之前，把本次 eval 的 epoch 編號寫入全域，以便輸出檔名使用 epoch 而非 timestamp
        try:
            global _EVAL_EPOCH_TAG
            _EVAL_EPOCH_TAG = int(self.state.epoch) if self.state.epoch is not None else 0
        except Exception:
            pass
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


# ----------- 評估回調類別 -----------
class EvaluationCallback(TrainerCallback):
    """評估回調：只負責將全語料檢索的Top-5指標寫入 TensorBoard（不重算）。"""

    def __init__(self, model, tokenizer, candidate_dataset_path, query_dataset_path,
                 train_qid_path, valid_qid_path, labels_path, output_dir):
        # 參數保留以便未來需要，但此處不再用於重新計算
        self.model = model
        self.tokenizer = tokenizer
        self.candidate_dataset_path = candidate_dataset_path
        self.query_dataset_path = query_dataset_path
        self.train_qid_path = train_qid_path
        self.valid_qid_path = valid_qid_path
        self.labels_path = labels_path
        self.output_dir = output_dir

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        print(f"\n🔍 完整檢索評估結果寫入 TensorBoard（epoch {current_epoch}）...")

        try:
            from torch.utils.tensorboard import SummaryWriter
            # 從全域變數中取得剛剛 compute_metrics 計算過的結果
            global _LATEST_RETRIEVAL_RESULTS
            results = _LATEST_RETRIEVAL_RESULTS
            if not results:
                print("⚠️ 找不到檢索評估結果（_LATEST_RETRIEVAL_RESULTS 为空）。略過 TensorBoard 追加寫入。")
                return

            logdir = os.path.join(args.output_dir, 'tb', 'retrieval')
            os.makedirs(logdir, exist_ok=True)
            writer = SummaryWriter(log_dir=logdir)

            # 寫入六個 retrieval/* 指標
            try:
                writer.add_scalar('retrieval/train_top5_f1',        results['train']['f1'],        current_epoch)
                writer.add_scalar('retrieval/train_top5_precision',  results['train']['precision'], current_epoch)
                writer.add_scalar('retrieval/train_top5_recall',     results['train']['recall'],    current_epoch)
                writer.add_scalar('retrieval/valid_top5_f1',        results['valid']['f1'],        current_epoch)
                writer.add_scalar('retrieval/valid_top5_precision',  results['valid']['precision'], current_epoch)
                writer.add_scalar('retrieval/valid_top5_recall',     results['valid']['recall'],    current_epoch)
                print("✅ 已寫入 TensorBoard: retrieval/* 六個指標")
            finally:
                writer.flush()
                writer.close()
        except Exception as e:
            print(f"❌ TensorBoard 記錄錯誤: {e}")

# （可選）在 optimizer.step() 之後印出溫度
from transformers import TrainerCallback
class TempWatch(TrainerCallback):
    def on_optimizer_step(self, args, state, control, **kwargs):
        m = kwargs.get("model", None)
        if hasattr(m, "log_temperature"):
            try:
                print(f"[after step {state.global_step}] T = {m.log_temperature.exp().item():.8f}")
            except Exception:
                pass


class AdaptiveNegativeSamplingCallback(TrainerCallback):
    """處理適應性負樣本採樣的Callback"""
    def __init__(self, trainer_instance):
        self.trainer_instance = trainer_instance
        self.last_epoch = -1
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """在每個 epoch 開始時，依 update_frequency 決定是否更新負樣本"""
        current_epoch = int(state.epoch) if state.epoch is not None else 0

        # 僅在 epoch 前進時處理一次
        if current_epoch > self.last_epoch:
            self.last_epoch = current_epoch
            if hasattr(self.trainer_instance, 'update_negative_samples'):
                self.trainer_instance.current_epoch = current_epoch
                prepared_epoch = getattr(self.trainer_instance, "_prepared_epoch_index", None)
                if prepared_epoch is not None and current_epoch == prepared_epoch:
                    print(
                        f"\n🔹 第{current_epoch}個epoch已依 checkpoint 預先準備動態負樣本，"
                        "本輪跳過重算一次"
                    )
                    self.trainer_instance._prepared_epoch_index = None
                    return
                upd_freq = getattr(self.trainer_instance, 'update_frequency', 1)
                if current_epoch >= 1 and (upd_freq <= 1 or current_epoch % upd_freq == 0):
                    print(f"\n🔹 第{current_epoch}個epoch需要更新負樣本 (更新頻率: 每{max(upd_freq,1)}個epoch)")
                    self.trainer_instance.update_negative_samples()
                else:
                    # 計算下次更新的 epoch
                    next_epoch = ((current_epoch // max(upd_freq,1)) + 1) * max(upd_freq,1)
                    print(f"\n🔹 第{current_epoch}個epoch跳過負樣本更新 (下次更新: 第{next_epoch}個epoch)")

def main():
    # 1. 檢查 CPU / GPU
    device = get_device()

    # 中文註解：TASK1_CHUNKAGG_BASE_ENCODER_DIR 指向 continued pretraining 後的 ModernBERT backbone checkpoint。
    ckpt_dir = Path(
        os.getenv(
            "TASK1_CHUNKAGG_BASE_ENCODER_DIR",
            str((PACKAGE_ROOT.parent / "modernbert-caselaw-accsteps-fp" / "checkpoint-29000").resolve()),
        )
    ).resolve()
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
    ).to(device)
    print("✅ Tokenizer 與 Model 初始化完成")
    
    # Debug: 檢查模型組件
    print(f"🔍 Debug info:")
    print(f"   - Encoder: {type(model.encoder)}")
    print(f"   - Projector: {type(model.projector)}")
    print(f"   - Chunk fusion: {type(model.chunk_fusion)}")
    print(f"   - Log temperature: {model.log_temperature}, temperature(actual): {model.log_temperature.exp().item():.8f}")
    print(f"   - Init temperature (from env): {INIT_TEMPERATURE}")
    print(f"   - Max document chunks: {MAX_DOCUMENT_CHUNKS}")
    print(f"   - Chunk length: {DOCUMENT_CHUNK_LENGTH}")
    print(f"   - Chunk microbatch size: {CHUNK_MICROBATCH_SIZE}")
    print(f"   - TF32 enabled: {ENABLE_TF32}")
    print(f"   - Text cache size: {TEXT_CACHE_SIZE}")
    print(f"   - Chunk cache size: {CHUNK_CACHE_SIZE}")
    print(f"   - Dataloader pin_memory: {ENABLE_PIN_MEMORY}")
    print(f"   - Dataloader persistent_workers: {ENABLE_PERSISTENT_WORKERS}")
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
    train_qid_path = f"{TASK1_DIR}/train_qid.tsv"
    positive_train_json_path = f"{TASK1_DIR}/task1_train_labels_{TASK1_YEAR}_train.json"
    valid_json_path = f"{TASK1_DIR}/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json"
    valid_qid_path = f"{TASK1_DIR}/valid_qid.tsv"  # Define valid_qid_path
    labels_path = f"{TASK1_DIR}/task1_train_labels_{TASK1_YEAR}.json"  # Define labels_path
    # 中文註解：TASK1_CHUNKAGG_FINETUNE_DATA_DIR 用來存 adaptive negatives 與 retrieval artifacts。
    finetune_data_dir_env = os.getenv("TASK1_CHUNKAGG_FINETUNE_DATA_DIR")
    finetune_data_dir = finetune_data_dir_env or f"{TASK1_DIR}/lht_process/modernBert-chunkAgg/finetune_data"
    retrieval_batch_size = RETRIEVAL_BATCH_SIZE

    # 中文註解：TASK1_CHUNKAGG_OUTPUT_DIR 為 Trainer checkpoint 輸出根目錄。
    base_output_dir_env = os.getenv("TASK1_CHUNKAGG_OUTPUT_DIR")
    base_output_dir = base_output_dir_env or "./modernBERT_contrastive_adaptive_fp_fp16_chunkAgg"
    if base_output_dir_env is None:
        if SCOPE_FILTER:
            base_output_dir += "_scopeFilteredRaw"
        if QUICK_TEST:
            base_output_dir += "_test"
        base_output_dir += f"_{TASK1_YEAR}"
    if QUICK_TEST and finetune_data_dir_env is None:
        finetune_data_dir += "_test"
    os.makedirs(finetune_data_dir, exist_ok=True)

    default_scope_path = f"{TASK1_DIR}/lht_process/modernBert/query_candidate_scope.json"
    env_scope_path = os.getenv("LCR_QUERY_CANDIDATE_SCOPE_JSON")
    if SCOPE_FILTER:
        if os.path.exists(default_scope_path):
            os.environ["LCR_QUERY_CANDIDATE_SCOPE_JSON"] = default_scope_path
            print(f"🔹 使用 query candidate scope: {os.environ['LCR_QUERY_CANDIDATE_SCOPE_JSON']}")
        elif env_scope_path:
            print(f"🔹 使用 query candidate scope: {env_scope_path}")
        else:
            raise FileNotFoundError(
                "SCOPE_FILTER=True 但找不到 query candidate scope。"
                f"請先生成 {TASK1_DIR}/lht_process/modernBert/query_candidate_scope.json"
            )
    else:
        if env_scope_path:
            print(f"🔹 使用 query candidate scope: {env_scope_path}")
        else:
            print("⚠️ 未設定 query candidate scope；將對全部 candidates 計算相似度。")
    

    # QUICK_TEST: 準備縮小的 candidate 與 query 清單（若啟用）
    if QUICK_TEST:
        try:
            global _QT_CANDIDATE_FILES, _QT_TRAIN_QIDS, _QT_VALID_QIDS
            all_cands = [fn for fn in os.listdir(doc_folder) if fn.endswith('.txt')]
            k_c = min(QT_CAND_K, len(all_cands))
            _QT_CANDIDATE_FILES = random.sample(all_cands, k_c) if k_c > 0 else []

            # 預先縮小 train/valid qids
            _QT_TRAIN_QIDS = load_query_ids_from_utils(train_qid_path)
            _QT_VALID_QIDS = load_query_ids_from_utils(valid_qid_path)
            if len(_QT_TRAIN_QIDS) > QT_QUERY_K:
                _QT_TRAIN_QIDS = random.sample(_QT_TRAIN_QIDS, QT_QUERY_K)
            if len(_QT_VALID_QIDS) > QT_QUERY_K:
                _QT_VALID_QIDS = random.sample(_QT_VALID_QIDS, QT_QUERY_K)

            print(f"[QUICK_TEST] Prepared {len(_QT_CANDIDATE_FILES)} candidates, train_q={len(_QT_TRAIN_QIDS)}, valid_q={len(_QT_VALID_QIDS)}")
        except Exception as e:
            print(f"[QUICK_TEST] init error: {e}")

    # 4. 建立初始訓練資料集（使用空的資料，稍後會被adaptive sampling更新）
    train_dataset = ContrastiveDataset(doc_folder=doc_folder, data=[])

    # 建立驗證資料集
    valid_dataset = ContrastiveDataset(json_path=valid_json_path, doc_folder=doc_folder)
    print(f"valid_dataset: {len(valid_dataset)}")

    # 5. 設定 TrainingArguments
    logging_dir = os.path.join(base_output_dir, "tb")

    per_device_train_batch_size = max(1, _get_env_int("TASK1_CHUNKAGG_TRAIN_BATCH_SIZE", 1))
    gradient_accumulation_steps = max(1, _get_env_int("TASK1_CHUNKAGG_GRAD_ACCUM_STEPS", 4))
    per_device_eval_batch_size = max(1, _get_env_int("TASK1_CHUNKAGG_EVAL_BATCH_SIZE", 1))
    num_train_epochs = _get_env_float("TASK1_CHUNKAGG_NUM_EPOCHS", 20)
    encoder_lr = _get_env_float("TASK1_CHUNKAGG_ENCODER_LR", 5e-6)
    fusion_lr = _get_env_float("TASK1_CHUNKAGG_FUSION_LR", 5e-5)
    temperature_lr = _get_env_float("TASK1_CHUNKAGG_TEMPERATURE_LR", 5e-4)
    warmup_ratio = _get_env_float("TASK1_CHUNKAGG_WARMUP_RATIO", 0.1)
    train_num_workers = max(0, _get_env_int("TASK1_CHUNKAGG_NUM_WORKERS", 8))
    save_total_limit = max(1, _get_env_int("TASK1_CHUNKAGG_SAVE_TOTAL_LIMIT", 20))
    sampling_temperature = _get_env_float("TASK1_CHUNKAGG_SAMPLING_TEMPERATURE", 1.0)
    update_frequency = max(1, _get_env_int("TASK1_CHUNKAGG_UPDATE_FREQUENCY", 1))

    args = TrainingArguments(
        output_dir=base_output_dir,
        dataloader_num_workers=train_num_workers,
        dataloader_pin_memory=ENABLE_PIN_MEMORY,
        dataloader_persistent_workers=ENABLE_PERSISTENT_WORKERS if train_num_workers > 0 else False,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=True,
        bf16=False,
        tf32=ENABLE_TF32,
        learning_rate=encoder_lr,           # 給 encoder
        num_train_epochs=num_train_epochs,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",  # 使用穩定的 AdamW 以配合 bf16 + TF32
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        # Use full-corpus valid retrieval F1 as selection metric
        metric_for_best_model="eval_global_f1",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],  # 啟用TensorBoard
        include_for_metrics=["loss"],
        prediction_loss_only=False,
        logging_dir=logging_dir,
    )
    args.encoder_lr = encoder_lr
    args.fusion_lr = fusion_lr
    args.temperature_lr = temperature_lr
    print(f"   - Trainer bf16/fp16/tf32: bf16={args.bf16}, fp16={args.fp16}, tf32={args.tf32}")
    print(f"   - Retrieval batch size: {retrieval_batch_size}")
    print(f"   - Encoder LR: {args.encoder_lr}")
    print(f"   - Aggregator LR: {args.fusion_lr}")
    print(f"   - Temperature LR: {args.temperature_lr}")
    print(f"   - Train batch size: {per_device_train_batch_size}")
    print(f"   - Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"   - Eval batch size: {per_device_eval_batch_size}")
    print(f"   - Num train epochs: {num_train_epochs}")
    print(f"   - Sampling temperature: {sampling_temperature}")
    print(f"   - Negative update frequency: {update_frequency}")

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
        ),
        tokenizer=tokenizer,  # 保留 tokenizer 以供自訂 Trainer 使用
        compute_metrics=make_compute_metrics_for_retrieval(
            model=model,
            tokenizer=tokenizer,
            candidate_dataset_path=doc_folder,
            query_dataset_path=query_dataset_path,
            train_qid_path=train_qid_path,
            valid_qid_path=valid_qid_path,
            labels_path=labels_path,
            output_dir=finetune_data_dir,
            retrieval_batch_size=retrieval_batch_size,
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5), 
            TempWatch(), 
            TensorBoardExtras(),
            EvaluationCallback(
                model=model,
                tokenizer=tokenizer,
                candidate_dataset_path=doc_folder,
                query_dataset_path=query_dataset_path,
                train_qid_path=train_qid_path,
                valid_qid_path=valid_qid_path,
                labels_path=labels_path,
                output_dir=finetune_data_dir
            )
        ],
        candidate_dataset_path=doc_folder,
        query_dataset_path=query_dataset_path,
        train_qid_path=train_qid_path,
        positive_train_json_path=positive_train_json_path,
        finetune_data_dir=finetune_data_dir,
        sampling_temperature=sampling_temperature,  # 可以調整這個參數來控制負樣本選擇的隨機性
        update_frequency=update_frequency,  # (整數)可以調整：1=每個epoch更新，2=每2個epoch更新一次，等等
        retrieval_batch_size=retrieval_batch_size,
    )

    # 在每個 epoch 開始時依據最新模型重算相似度並重抽負樣本
    trainer.add_callback(AdaptiveNegativeSamplingCallback(trainer))

    print("🔹 Trainer 設定完成，開始訓練並驗證...\n")
    # Summary line for QUICK_TEST
    try:
        cand_count = len(_QT_CANDIDATE_FILES) if QUICK_TEST and _QT_CANDIDATE_FILES is not None else len([fn for fn in os.listdir(doc_folder) if fn.endswith('.txt')])
        q_count = len(_QT_TRAIN_QIDS) if QUICK_TEST and _QT_TRAIN_QIDS is not None else len(load_query_ids_from_utils(train_qid_path))
        print(f"QUICK_TEST={QUICK_TEST} | candidates={cand_count} | queries={q_count}")
    except Exception:
        print("QUICK_TEST summary error")

    # （可選）檢查各參數組 LR
    for i, g in enumerate(trainer.create_optimizer().param_groups):
        sz = sum(p.numel() for p in g["params"])
        group_name = g.get("group_name", f"group_{i}")
        print(f"{group_name}: lr={g['lr']}  weight_decay={g['weight_decay']}  #params={sz}")

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

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"\n✅ 訓練與驗證完成！模型已儲存於：{args.output_dir}")

if __name__ == "__main__":
    main()
