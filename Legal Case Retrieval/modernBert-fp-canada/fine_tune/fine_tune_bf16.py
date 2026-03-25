import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
# torch.set_float32_matmul_precision('high')
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

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_env_flag, get_env_int, get_env_path, get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

from lcr.data import EmbeddingsData
import torch.nn.functional as F
from collections import defaultdict
import time
from lcr.data import load_query_ids as load_query_ids_from_utils
from lcr.device import get_device
from lcr.metrics import (
    my_classification_report,
    rel_file_to_dict as rel_file_convert,
    trec_file_to_dict as trec_file_convert,
)
from lcr.retrieval import generate_similarity_artifacts

# Global holder for retrieval results to reuse in TB logging callback
_LATEST_RETRIEVAL_RESULTS = None
_EVAL_EPOCH_TAG = None  # 用於在評估時以 epoch 編號命名輸出檔

# -------------
# QUICK TEST MODE
# -------------
# 切換快速測試模式：True 只取極少量資料以便快速驗證流程
QUICK_TEST = get_env_flag("TASK1_QUICK_TEST", required=True)

# 由 main() 設定，用於 generate_similarity_artifacts 的覆寫資料
_QT_CANDIDATE_FILES = None   # List[str] 檔名（包含 .txt）
_QT_TRAIN_QIDS = None        # List[str]
_QT_VALID_QIDS = None        # List[str]

# QUICK_TEST 數量上限（可由環境變數覆寫）
QT_CAND_K = max(1, get_env_int("TASK1_QUICK_TEST_CAND_K", required=True))
QT_QUERY_K = max(1, get_env_int("TASK1_QUICK_TEST_QUERY_K", required=True))


def evaluate_model_retrieval(model, tokenizer, device, candidate_dataset_path, query_dataset_path, 
                           train_qid_path, valid_qid_path, labels_path, output_dir, epoch_num, topk=5):
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
            batch_size=1,
            max_length=4096,
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
        # Trainer 每個 logging step 會呼叫這裡，logs 內含 loss、learning_rate 等
        self._ensure_writer(args)
        if logs:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"train/{k}", v, state.global_step)

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
        # 每次 evaluate() 後把 eval_* 寫到 TensorBoard
        self._ensure_writer(args)
        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f"eval/{k}", v, state.global_step)

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
                                       output_dir):
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
        path = os.path.join(self.doc_folder, f"{doc_id}.txt")
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "query_text": self.load_text(sample["query_id"]),
            "positive_text": self.load_text(sample["positive_id"]),
            "negative_texts": [self.load_text(nid) for nid in sample["negative_ids"]]
        }

# ----------- Collator -----------
@dataclass
class ContrastiveCollator:
    tokenizer: AutoTokenizer
    max_length: int = 4096
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        bsz = len(batch)
        q_texts = [item["query_text"] for item in batch]
        p_texts = [item["positive_text"] for item in batch]
        n_texts = [neg for item in batch for neg in item["negative_texts"]]
        all_texts = q_texts + p_texts + n_texts
        all_enc = self.tokenizer(
            all_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        neg_count = len(n_texts) // bsz
        sizes = [bsz, bsz, bsz * neg_count]
        anchor_ids,  positive_ids,  negative_ids  = all_enc["input_ids"].split(sizes, dim=0)
        anchor_mask, positive_mask, negative_mask = all_enc["attention_mask"].split(sizes, dim=0)
        labels = torch.zeros(bsz, dtype=torch.long)
        return {
            "anchor_input":   {"input_ids": anchor_ids,   "attention_mask": anchor_mask},
            "positive_input": {"input_ids": positive_ids, "attention_mask": positive_mask},
            "negative_input": {"input_ids": negative_ids, "attention_mask": negative_mask},
            "labels": labels,
        }

# ----------- Model with InfoNCE loss -----------
class ModernBERTContrastive(nn.Module):
    def __init__(self, model_name: str, device, temperature: float = 0.55555):
        super().__init__()
        # 與 inference.py 保持一致的 encoder_kwargs 設定；bf16 由 Trainer 的 AMP 控制，權重維持 fp32
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
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 可學習的 temperature（log-param 保證 >0）
        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(float(temperature)), dtype=torch.float32)
        )
        self.temperature_min = 1e-3
        self.temperature_max = 2.0

        self.encoder.config.use_cache = False
        self.encoder.enable_input_require_grads() # 打開訓練效果會好點，可以學習id->embedding
        self.encoder.gradient_checkpointing_enable()

    # 供 Trainer 呼叫，維持介面與 huggingface 模型一致
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()

    def encode(self, input_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.encoder(**input_batch)
        vec = output.last_hidden_state[:, 0, :]
        vec_project = self.projector(vec)
        vec_l2_norm = torch.nn.functional.normalize(vec_project, p=2, dim=-1)
        return vec_l2_norm
    
    def encode_in_chunks(self, negatives_input, chunk_size):
        ids = negatives_input["input_ids"]
        attn = negatives_input["attention_mask"]
        all_vecs = []
        for start in range(0, ids.size(0), chunk_size):
            end = start + chunk_size
            out = self.encoder(input_ids=ids[start:end], attention_mask=attn[start:end])
            cls = out.last_hidden_state[:,0,:]
            proj = self.projector(cls)
            all_vecs.append(torch.nn.functional.normalize(proj, p=2, dim=-1))
        return torch.cat(all_vecs, dim=0)

    def forward(self,
        anchor_input: Dict[str, torch.Tensor],
        positive_input: Dict[str, torch.Tensor],
        negative_input: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        print(f"forward...")
        temperature = self.log_temperature.exp().clamp(self.temperature_min, self.temperature_max)
        bsz = anchor_input["input_ids"].size(0)
        neg_count = negative_input["input_ids"].size(0) // bsz

        merged_batch = {
            "input_ids": torch.cat([
                anchor_input["input_ids"], positive_input["input_ids"], negative_input["input_ids"]
            ], dim=0),
            "attention_mask": torch.cat([
                anchor_input["attention_mask"], positive_input["attention_mask"], negative_input["attention_mask"]
            ], dim=0),
        }
        vec_all = self.encode(merged_batch)
        anchor_vec   = vec_all[:bsz]
        positive_vec = vec_all[bsz:bsz*2]
        neg_flat     = vec_all[bsz*2:]
        negative_vec = neg_flat.view(bsz, neg_count, -1)

        self.print_gpu_status("all encoded in one pass")
        pos_sim = torch.cosine_similarity(anchor_vec, positive_vec, dim=-1).unsqueeze(1)
        neg_sim = torch.cosine_similarity(anchor_vec.unsqueeze(1), negative_vec, dim=-1)
        self.print_gpu_status("cosine similarity computed")

        logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature
        loss = nn.CrossEntropyLoss()(logits, labels)
        print(f"labels: {labels} │ temperature(now)={temperature.item()}")
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
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_dataset_path = candidate_dataset_path
        self.query_dataset_path = query_dataset_path
        self.train_qid_path = train_qid_path
        self.positive_train_json_path = positive_train_json_path
        self.finetune_data_dir = finetune_data_dir
        self.sampling_temperature = sampling_temperature
        self.update_frequency = update_frequency
        self.current_epoch = 0
        
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
        if hasattr(self, 'train_qids'):
            print(f"   - 訓練查詢數量：{len(self.train_qids)}")
        if hasattr(self, 'positives'):
            print(f"   - 正樣本對數量：{len(self.positives)}")
    
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        args = self.args
        model = self.model

        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        decay_params, nodecay_params, temp_params = [], [], []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "log_temperature" in n:
                temp_params.append(p)
            elif any(nd in n for nd in no_decay):
                nodecay_params.append(p)
            else:
                decay_params.append(p)

        optimizer_grouped_parameters = [
            {"params": decay_params,   "weight_decay": args.weight_decay, "lr": args.learning_rate},
            {"params": nodecay_params, "weight_decay": 0.0,               "lr": args.learning_rate},
            {"params": temp_params,    "weight_decay": 0.0,               "lr": getattr(args, "temperature_lr", args.learning_rate)},
        ]

        # 根據 TrainingArguments 選擇 AdamW；若要求 fused，嘗試啟用
        adamw_kwargs = dict(
            lr=args.learning_rate,
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
        
        # 在第一個epoch開始前就使用適應性負樣本
        if self.current_epoch == 0:
            print("🔹 第0個epoch開始使用適應性負樣本...")
            self.update_negative_samples()
        
        return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
    
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
                batch_size=1,
                max_length=4096,
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

    ckpt_dir = get_env_path("TASK1_CANADA_BASE_ENCODER_DIR", required=True)
    assert ckpt_dir is not None
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"找不到繼續預訓練後的 ModernBERT checkpoint: {ckpt_dir}")

    model_name = str(ckpt_dir)
    print("🔹 載入 tokenizer 與模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # 2. 初始化自定義的 Contrastive Model
    model = ModernBERTContrastive(model_name, device)
    print("✅ Tokenizer 與 Model 初始化完成")
    
    # Debug: 檢查模型組件
    print(f"🔍 Debug info:")
    print(f"   - Encoder: {type(model.encoder)}")
    print(f"   - Projector: {type(model.projector)}")
    print(f"   - Log temperature: {model.log_temperature}, temperature(actual): {model.log_temperature.exp().item():.8f}")
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
    finetune_data_dir = get_env_path("TASK1_CANADA_BF16_FINETUNE_DATA_DIR", required=True)
    base_output_dir = get_env_path("TASK1_CANADA_BF16_MODEL_ROOT_DIR", required=True)
    assert finetune_data_dir is not None
    assert base_output_dir is not None
    finetune_data_dir = str(finetune_data_dir)
    base_output_dir = str(base_output_dir)
    os.makedirs(finetune_data_dir, exist_ok=True)

    default_scope_path = f"{TASK1_DIR}/lht_process/modernBert/query_candidate_scope.json"
    if os.path.exists(default_scope_path):
        os.environ.setdefault("LCR_QUERY_CANDIDATE_SCOPE_JSON", default_scope_path)
        print(f"🔹 使用 query candidate scope: {os.environ['LCR_QUERY_CANDIDATE_SCOPE_JSON']}")
    elif os.getenv("LCR_QUERY_CANDIDATE_SCOPE_JSON"):
        print(f"🔹 使用 query candidate scope: {os.environ['LCR_QUERY_CANDIDATE_SCOPE_JSON']}")
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

    args = TrainingArguments(
        output_dir=base_output_dir,
        dataloader_num_workers=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        fp16=False,
        bf16=True,
        tf32=False,
        learning_rate=7.5e-6,           # 給 encoder / projector
        num_train_epochs=20,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",  # 使用穩定的 AdamW 以配合 bf16 訓練
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=20,
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
    # ✅ 給 temperature 的專屬 LR（自行調整）
    args.temperature_lr = 5e-4
    print(f"   - Trainer bf16/fp16/tf32: bf16={args.bf16}, fp16={args.fp16}, tf32={args.tf32}")

    # 6. 建立 Trainer（使用 AdaptiveNegativeSamplingTrainer）
    trainer = AdaptiveNegativeSamplingTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=ContrastiveCollator(tokenizer),
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
        sampling_temperature=1.0,  # 可以調整這個參數來控制負樣本選擇的隨機性
        update_frequency=1,  # (整數)可以調整：1=每個epoch更新，2=每2個epoch更新一次，等等
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
        print(f"group {i}: lr={g['lr']}  weight_decay={g['weight_decay']}  #params={sz}")

    trainer.train()
    print(f"\n✅ 訓練與驗證完成！模型已儲存於：{args.output_dir}")

if __name__ == "__main__":
    main()
