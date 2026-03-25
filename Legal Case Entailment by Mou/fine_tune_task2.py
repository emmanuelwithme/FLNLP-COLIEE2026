#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_config import (
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_path,
    load_dotenv_if_present,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from modernbert_contrastive_model import ModernBERTContrastive

random.seed(289)
np.random.seed(289)

LCR_ROOT = REPO_ROOT / "Legal Case Retrieval"
if str(LCR_ROOT) not in sys.path:
    sys.path.insert(0, str(LCR_ROOT))

from lcr.data import load_query_ids as load_query_ids_from_utils
from lcr.device import get_device
from lcr.metrics import my_classification_report
from lcr.retrieval import generate_similarity_artifacts

_LATEST_RETRIEVAL_RESULTS = None
_EVAL_EPOCH_TAG = None


def normalize_id(raw_id: object) -> str:
    return Path(str(raw_id).strip()).stem


def normalize_mode(value: str) -> str:
    mode = value.strip().lower()
    if mode in {"test", "quick", "quick_test"}:
        return "test"
    return "full"


def sample_ids(ids: Sequence[str], limit: int, seed: int) -> List[str]:
    cleaned = [normalize_id(x) for x in ids if str(x).strip()]
    if limit <= 0 or len(cleaned) <= limit:
        return cleaned
    rng = random.Random(seed)
    return sorted(rng.sample(cleaned, limit))


def write_qid_tsv(path: Path, qids: Sequence[str]) -> None:
    lines = [normalize_id(qid) for qid in qids if str(qid).strip()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def filter_contrastive_json(
    src_path: Path,
    dst_path: Path,
    *,
    allow_qids: Set[str],
) -> int:
    rows = json.loads(src_path.read_text(encoding="utf-8"))
    filtered = [
        row
        for row in rows
        if normalize_id(row.get("query_id", "")) in allow_qids
    ]
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(json.dumps(filtered, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(filtered)


def build_candidate_files_override(
    *,
    selected_qids: Sequence[str],
    query_candidates_map: Dict[str, List[str]],
    candidate_dir: Path,
) -> List[str]:
    candidate_ids: Set[str] = set()
    for qid in selected_qids:
        candidate_ids.update(query_candidates_map.get(normalize_id(qid), []))
    return sorted(
        f"{candidate_id}.txt"
        for candidate_id in candidate_ids
        if (candidate_dir / f"{candidate_id}.txt").is_file()
    )


def accuracy_score_np(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return float((labels == preds).mean())


def retrieval_metrics_for_qids(
    *,
    qids: Sequence[str],
    rel_dict: Dict[str, List[str]],
    answer_dict: Dict[str, List[str]],
) -> Dict[str, float]:
    list_answer_ohe: List[Sequence[str]] = []
    list_label_ohe: List[Sequence[str]] = []
    for qid in qids:
        one_rel = rel_dict.get(qid, [])
        list_answer_ohe.append(answer_dict.get(qid, []))
        list_label_ohe.append(one_rel)
    f1, precision, recall = my_classification_report(list_label_ohe, list_answer_ohe)
    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "num_queries": float(len(list_label_ohe)),
    }


def load_query_candidates_map(path: Path) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping: Dict[str, List[str]] = {}
    for raw_qid, raw_candidates in payload.items():
        qid = normalize_id(raw_qid)
        seen = set()
        cleaned: List[str] = []
        for raw_doc in raw_candidates:
            doc_id = normalize_id(raw_doc)
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            cleaned.append(doc_id)
        mapping[qid] = cleaned
    return mapping


def rel_file_to_dict_str(rel_path: str, query_id_path: str) -> Dict[str, List[str]]:
    query_ids = {
        normalize_id(line)
        for line in Path(query_id_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    label_dict = json.loads(Path(rel_path).read_text(encoding="utf-8"))
    rel_dict: Dict[str, List[str]] = {}
    for raw_qid, label_list in label_dict.items():
        qid = normalize_id(raw_qid)
        if qid not in query_ids:
            continue
        rel_dict[qid] = sorted({normalize_id(label) for label in label_list})
    return rel_dict


def trec_file_to_dict_str(trec_path: str, topk: int, skip_self: bool = True) -> Dict[str, List[str]]:
    trec_dict: Dict[str, List[str]] = {}
    with Path(trec_path).open("r", encoding="utf-8") as trec_file:
        for line in trec_file:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            qid = normalize_id(parts[0])
            pid = normalize_id(parts[2])
            if skip_self and qid == pid:
                continue
            entries = trec_dict.setdefault(qid, [])
            if len(entries) < topk:
                entries.append(pid)
    return trec_dict


def read_positive_pairs_from_json(json_path: str) -> Dict[str, Set[str]]:
    positives = defaultdict(set)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for q_txt, pos_list in data.items():
        qid = normalize_id(q_txt)
        for doc_txt in pos_list:
            positives[qid].add(normalize_id(doc_txt))
    return positives


def generate_adaptive_negative_samples(
    query_id_to_similarities: Dict[str, Dict[str, float]],
    positives: Dict[str, Set[str]],
    *,
    max_negatives: int = 15,
    temperature: float = 1.0,
) -> List[Dict[str, object]]:
    dataset: List[Dict[str, object]] = []
    for qid, pos_set in positives.items():
        similarities = query_id_to_similarities.get(qid)
        if not similarities:
            continue

        for pos_id in sorted(pos_set):
            if pos_id not in similarities:
                continue
            negative_candidates = []
            negative_scores = []
            for doc_id, score in similarities.items():
                if doc_id not in pos_set and doc_id != qid:
                    negative_candidates.append(doc_id)
                    negative_scores.append(score)
            if not negative_candidates:
                continue

            probs = F.softmax(torch.tensor(negative_scores) / temperature, dim=0).numpy()
            replace_flag = len(negative_candidates) < max_negatives
            selected_negatives = np.random.choice(
                negative_candidates,
                size=max_negatives,
                replace=replace_flag,
                p=probs,
            ).tolist()
            dataset.append(
                {
                    "query_id": qid,
                    "positive_id": pos_id,
                    "negative_ids": selected_negatives,
                }
            )
    return dataset


def evaluate_model_retrieval(
    model,
    tokenizer,
    device,
    *,
    candidate_dataset_path: str,
    query_dataset_path: str,
    train_qid_path: str,
    valid_qid_path: str,
    labels_path: str,
    output_dir: str,
    epoch_num: int,
    primary_topk: int,
    secondary_topk: int,
    query_candidates_map: Dict[str, List[str]],
    similarity_batch_size: int,
    similarity_max_length: int,
    quick_test: bool = False,
    candidate_files_override: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    train_rel_dict = rel_file_to_dict_str(labels_path, train_qid_path)
    valid_rel_dict = rel_file_to_dict_str(labels_path, valid_qid_path)
    train_qids = load_query_ids_from_utils(train_qid_path)
    valid_qids = load_query_ids_from_utils(valid_qid_path)
    all_qids = sorted(set(train_qids + valid_qids))

    epoch_tag = f"{epoch_num}_eval_all"
    artifacts = generate_similarity_artifacts(
        model,
        tokenizer,
        device,
        candidate_dir=candidate_dataset_path,
        query_dir=query_dataset_path,
        query_ids=all_qids,
        trec_output_path=Path(output_dir) / f"similarity_scores_{epoch_tag}.tsv",
        run_tag=f"modernBert_task2_{epoch_tag}",
        batch_size=similarity_batch_size,
        max_length=similarity_max_length,
        quick_test=quick_test,
        candidate_files_override=candidate_files_override,
        query_to_candidate_ids=query_candidates_map,
        fallback_to_all_candidates_if_scope_missing=False,
    )
    answer_top1 = trec_file_to_dict_str(str(artifacts.trec_path), primary_topk)
    answer_top2 = trec_file_to_dict_str(str(artifacts.trec_path), secondary_topk)

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for split, (qids, rel_dict) in [
        ("train", (train_qids, train_rel_dict)),
        ("valid", (valid_qids, valid_rel_dict)),
    ]:
        print(f"[retrieval] evaluating {split}...")
        top1 = retrieval_metrics_for_qids(qids=qids, rel_dict=rel_dict, answer_dict=answer_top1)
        top2 = retrieval_metrics_for_qids(qids=qids, rel_dict=rel_dict, answer_dict=answer_top2)
        results[split] = {"top1": top1, "top2": top2}
        print(
            f"[retrieval] {split} top1 F1={top1['f1']:.6f}, P={top1['precision']:.6f}, R={top1['recall']:.6f}"
        )
        print(
            f"[retrieval] {split} top2 F1={top2['f1']:.6f}, P={top2['precision']:.6f}, R={top2['recall']:.6f}"
        )
    return results


def make_compute_metrics_for_retrieval(
    model,
    tokenizer,
    *,
    candidate_dataset_path: str,
    query_dataset_path: str,
    train_qid_path: str,
    valid_qid_path: str,
    labels_path: str,
    output_dir: str,
    primary_eval_topk: int,
    secondary_eval_topk: int,
    query_candidates_map: Dict[str, List[str]],
    retrieval_batch_size: int,
    retrieval_max_length: int,
    quick_test: bool = False,
    candidate_files_override: Optional[Sequence[str]] = None,
):
    def _compute(eval_pred: EvalPrediction):
        global _LATEST_RETRIEVAL_RESULTS

        logits = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)

        metrics = {}
        with contextlib.suppress(Exception):
            preds_top1 = torch.argmax(logits, dim=1).cpu().numpy()
            metrics["acc1"] = accuracy_score_np(labels.cpu().numpy(), preds_top1)
            k2 = min(2, logits.size(1))
            top2_preds = torch.topk(logits, k=k2, dim=1).indices
            labels_expanded = labels.view(-1, 1).expand_as(top2_preds)
            metrics["acc2"] = (top2_preds == labels_expanded).any(dim=1).float().mean().item()

        current_epoch = _EVAL_EPOCH_TAG if _EVAL_EPOCH_TAG is not None else int(time.time())
        helper_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results = evaluate_model_retrieval(
            model=model,
            tokenizer=tokenizer,
            device=helper_device,
            candidate_dataset_path=candidate_dataset_path,
            query_dataset_path=query_dataset_path,
            train_qid_path=train_qid_path,
            valid_qid_path=valid_qid_path,
            labels_path=labels_path,
            output_dir=output_dir,
            epoch_num=current_epoch,
            primary_topk=primary_eval_topk,
            secondary_topk=secondary_eval_topk,
            query_candidates_map=query_candidates_map,
            similarity_batch_size=retrieval_batch_size,
            similarity_max_length=retrieval_max_length,
            quick_test=quick_test,
            candidate_files_override=candidate_files_override,
        )

        valid_top1 = results.get("valid", {}).get("top1", {})
        valid_top2 = results.get("valid", {}).get("top2", {})
        metrics["global_f1"] = float(valid_top1.get("f1", 0.0))
        metrics["global_precision"] = float(valid_top1.get("precision", 0.0))
        metrics["global_recall"] = float(valid_top1.get("recall", 0.0))
        metrics["global_f1_top2"] = float(valid_top2.get("f1", 0.0))
        metrics["global_precision_top2"] = float(valid_top2.get("precision", 0.0))
        metrics["global_recall_top2"] = float(valid_top2.get("recall", 0.0))
        _LATEST_RETRIEVAL_RESULTS = results
        print(f"eval_global_f1={metrics['global_f1']:.6f}")
        return metrics

    return _compute


class ContrastiveDataset(Dataset):
    def __init__(
        self,
        *,
        query_folder: str,
        candidate_folder: str,
        json_path: Optional[str] = None,
        data: Optional[List[Dict]] = None,
        cache_texts: bool = True,
    ):
        if data is not None:
            self.data = data
        elif json_path is not None:
            with open(json_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = []
        self.query_folder = query_folder
        self.candidate_folder = candidate_folder
        self.cache_texts = cache_texts
        self._query_cache: Dict[str, str] = {}
        self._candidate_cache: Dict[str, str] = {}
        if json_path:
            print(f"loaded {len(self.data)} samples from {json_path}")
        else:
            print(f"loaded {len(self.data)} samples")

    def update_data(self, new_data: List[Dict]):
        self.data = new_data
        if self.cache_texts:
            self._query_cache.clear()
            self._candidate_cache.clear()
        print(f"dataset updated, size={len(self.data)}")

    def __len__(self):
        return len(self.data)

    def _load_query_text(self, query_id: str) -> str:
        if self.cache_texts and query_id in self._query_cache:
            return self._query_cache[query_id]
        path = Path(self.query_folder) / f"{query_id}.txt"
        with path.open("r", encoding="utf-8") as f:
            text = f.read().strip()
        if self.cache_texts:
            self._query_cache[query_id] = text
        return text

    def _load_candidate_text(self, doc_id: str) -> str:
        if self.cache_texts and doc_id in self._candidate_cache:
            return self._candidate_cache[doc_id]
        path = Path(self.candidate_folder) / f"{doc_id}.txt"
        with path.open("r", encoding="utf-8") as f:
            text = f.read().strip()
        if self.cache_texts:
            self._candidate_cache[doc_id] = text
        return text

    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "query_text": self._load_query_text(sample["query_id"]),
            "positive_text": self._load_candidate_text(sample["positive_id"]),
            "negative_texts": [self._load_candidate_text(nid) for nid in sample["negative_ids"]],
        }


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
            all_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        neg_count = len(n_texts) // bsz
        sizes = [bsz, bsz, bsz * neg_count]
        anchor_ids, positive_ids, negative_ids = all_enc["input_ids"].split(sizes, dim=0)
        anchor_mask, positive_mask, negative_mask = all_enc["attention_mask"].split(sizes, dim=0)
        labels = torch.zeros(bsz, dtype=torch.long)
        return {
            "anchor_input": {"input_ids": anchor_ids, "attention_mask": anchor_mask},
            "positive_input": {"input_ids": positive_ids, "attention_mask": positive_mask},
            "negative_input": {"input_ids": negative_ids, "attention_mask": negative_mask},
            "labels": labels,
        }


class AdaptiveNegativeSamplingTrainer(Trainer):
    def __init__(
        self,
        *args,
        candidate_dataset_path: str,
        query_dataset_path: str,
        train_qid_path: str,
        positive_train_json_path: str,
        finetune_data_dir: str,
        query_candidates_map: Dict[str, List[str]],
        sampling_temperature: float = 1.0,
        update_frequency: int = 1,
        max_negatives: int = 15,
        retrieval_batch_size: int = 8,
        retrieval_max_length: int = 4096,
        quick_test: bool = False,
        candidate_files_override: Optional[Sequence[str]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.candidate_dataset_path = candidate_dataset_path
        self.query_dataset_path = query_dataset_path
        self.train_qid_path = train_qid_path
        self.positive_train_json_path = positive_train_json_path
        self.finetune_data_dir = finetune_data_dir
        self.query_candidates_map = query_candidates_map
        self.sampling_temperature = sampling_temperature
        self.update_frequency = update_frequency
        self.max_negatives = max_negatives
        self.retrieval_batch_size = retrieval_batch_size
        self.retrieval_max_length = retrieval_max_length
        self.quick_test = quick_test
        self.candidate_files_override = (
            list(candidate_files_override) if candidate_files_override else None
        )
        self.current_epoch = 0
        self.train_qids = load_query_ids_from_utils(train_qid_path)
        self.positives = read_positive_pairs_from_json(positive_train_json_path)

        print("adaptive negative sampling config:")
        print(f"update_frequency={self.update_frequency}")
        print(f"sampling_temperature={self.sampling_temperature}")
        print(f"max_negatives={self.max_negatives}")
        print(f"retrieval_batch_size={self.retrieval_batch_size}")
        print(f"retrieval_max_length={self.retrieval_max_length}")
        print(f"quick_test={self.quick_test}")
        print(f"train queries={len(self.train_qids)}")
        print(f"positive query keys={len(self.positives)}")

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        args = self.args
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        decay_params, nodecay_params, temp_params = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "log_temperature" in name:
                temp_params.append(param)
            elif any(nd in name for nd in no_decay):
                nodecay_params.append(param)
            else:
                decay_params.append(param)

        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": args.weight_decay, "lr": args.learning_rate},
            {"params": nodecay_params, "weight_decay": 0.0, "lr": args.learning_rate},
            {
                "params": temp_params,
                "weight_decay": 0.0,
                "lr": getattr(args, "temperature_lr", args.learning_rate),
            },
        ]

        from torch.optim import AdamW

        adamw_kwargs = dict(
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
        try:
            if getattr(args, "optim", "") == "adamw_torch_fused":
                self.optimizer = AdamW(optimizer_grouped_parameters, **adamw_kwargs, fused=True)
            else:
                self.optimizer = AdamW(optimizer_grouped_parameters, **adamw_kwargs)
        except TypeError:
            self.optimizer = AdamW(optimizer_grouped_parameters, **adamw_kwargs)
        return self.optimizer

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        if self.current_epoch == 0:
            print("epoch0 initial adaptive negative update")
            self.update_negative_samples()
        return super()._inner_training_loop(
            batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval
        )

    def update_negative_samples(self):
        print(f"building adaptive negatives for epoch {self.current_epoch}")
        os.makedirs(self.finetune_data_dir, exist_ok=True)
        helper_device = (
            self.args.device
            if hasattr(self.args, "device")
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        artifacts = generate_similarity_artifacts(
            self.model,
            self.tokenizer,
            helper_device,
            candidate_dir=self.candidate_dataset_path,
            query_dir=self.query_dataset_path,
            query_ids=self.train_qids,
            trec_output_path=Path(self.finetune_data_dir) / f"similarity_scores_epoch{self.current_epoch}.tsv",
            run_tag=f"modernBert_task2_epoch{self.current_epoch}",
            batch_size=self.retrieval_batch_size,
            max_length=self.retrieval_max_length,
            quick_test=self.quick_test,
            candidate_files_override=self.candidate_files_override,
            query_to_candidate_ids=self.query_candidates_map,
            fallback_to_all_candidates_if_scope_missing=False,
        )
        new_data = generate_adaptive_negative_samples(
            query_id_to_similarities=artifacts.scores,
            positives=self.positives,
            max_negatives=self.max_negatives,
            temperature=self.sampling_temperature,
        )
        self.train_dataset.update_data(new_data)
        output_path = Path(self.finetune_data_dir) / f"adaptive_negative_epoch{self.current_epoch}_train.json"
        output_path.write_text(json.dumps(new_data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved adaptive negatives: {output_path} ({len(new_data)} samples)")

    def evaluate(self, eval_dataset=None, *args, **kwargs):
        global _EVAL_EPOCH_TAG
        with contextlib.suppress(Exception):
            _EVAL_EPOCH_TAG = int(self.state.epoch) if self.state.epoch is not None else 0
        return super().evaluate(eval_dataset=eval_dataset, *args, **kwargs)


class TensorBoardExtras(TrainerCallback):
    def __init__(self):
        self.writer = None

    def _ensure_writer(self, args):
        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter

            logdir = os.path.join(args.output_dir, "tb", "extras")
            os.makedirs(logdir, exist_ok=True)
            self.writer = SummaryWriter(logdir)

    def on_train_begin(self, args, state, control, **kwargs):
        self._ensure_writer(args)

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._ensure_writer(args)
        if logs:
            for key, value in logs.items():
                if key.startswith("eval_"):
                    continue
                if key in {"epoch", "total_flos"}:
                    continue
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"train/{key}", value, state.global_step)

    def on_step_end(self, args, state, control, **kwargs):
        self._ensure_writer(args)
        model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)
        if model is not None and hasattr(model, "log_temperature"):
            self.writer.add_scalar("train/temperature", model.log_temperature.exp().item(), state.global_step)
        if optimizer is not None:
            for i, g in enumerate(optimizer.param_groups):
                self.writer.add_scalar(f"train/lr_group_{i}", float(g.get("lr", 0.0)), state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if self.writer:
            self.writer.flush()
            self.writer.close()


class EvaluationCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        print(f"write retrieval metrics to TB (epoch {current_epoch})")
        global _LATEST_RETRIEVAL_RESULTS
        results = _LATEST_RETRIEVAL_RESULTS
        if not results:
            print("warning: no retrieval results to log")
            return

        from torch.utils.tensorboard import SummaryWriter

        logdir = os.path.join(args.output_dir, "tb", "retrieval")
        os.makedirs(logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=logdir)
        try:
            writer.add_scalar("retrieval_top1/train_micro_f1", results["train"]["top1"]["f1"], current_epoch)
            writer.add_scalar(
                "retrieval_top1/train_micro_precision",
                results["train"]["top1"]["precision"],
                current_epoch,
            )
            writer.add_scalar(
                "retrieval_top1/train_micro_recall",
                results["train"]["top1"]["recall"],
                current_epoch,
            )
            writer.add_scalar("retrieval_top1/valid_micro_f1", results["valid"]["top1"]["f1"], current_epoch)
            writer.add_scalar(
                "retrieval_top1/valid_micro_precision",
                results["valid"]["top1"]["precision"],
                current_epoch,
            )
            writer.add_scalar(
                "retrieval_top1/valid_micro_recall",
                results["valid"]["top1"]["recall"],
                current_epoch,
            )
            writer.add_scalar("retrieval_top2/train_micro_f1", results["train"]["top2"]["f1"], current_epoch)
            writer.add_scalar(
                "retrieval_top2/train_micro_precision",
                results["train"]["top2"]["precision"],
                current_epoch,
            )
            writer.add_scalar(
                "retrieval_top2/train_micro_recall",
                results["train"]["top2"]["recall"],
                current_epoch,
            )
            writer.add_scalar("retrieval_top2/valid_micro_f1", results["valid"]["top2"]["f1"], current_epoch)
            writer.add_scalar(
                "retrieval_top2/valid_micro_precision",
                results["valid"]["top2"]["precision"],
                current_epoch,
            )
            writer.add_scalar(
                "retrieval_top2/valid_micro_recall",
                results["valid"]["top2"]["recall"],
                current_epoch,
            )
            print("retrieval metrics logged")
        finally:
            writer.flush()
            writer.close()


class AdaptiveNegativeSamplingCallback(TrainerCallback):
    def __init__(self, trainer_instance):
        self.trainer_instance = trainer_instance
        self.last_epoch = -1

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) if state.epoch is not None else 0
        if current_epoch <= self.last_epoch:
            return
        self.last_epoch = current_epoch
        self.trainer_instance.current_epoch = current_epoch
        upd_freq = max(getattr(self.trainer_instance, "update_frequency", 1), 1)
        if current_epoch >= 1 and current_epoch % upd_freq == 0:
            print(f"epoch {current_epoch}: refresh adaptive negatives")
            self.trainer_instance.update_negative_samples()
        elif current_epoch >= 1:
            next_epoch = ((current_epoch // upd_freq) + 1) * upd_freq
            print(f"epoch {current_epoch}: skip refresh (next={next_epoch})")


def resolve_latest_checkpoint(root_or_checkpoint: Path) -> Path:
    if not root_or_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {root_or_checkpoint}")
    if root_or_checkpoint.is_file():
        raise ValueError(f"Expected directory, got file: {root_or_checkpoint}")
    if root_or_checkpoint.name.startswith("checkpoint-"):
        return root_or_checkpoint

    checkpoints = []
    for child in root_or_checkpoint.iterdir():
        if child.is_dir() and child.name.startswith("checkpoint-"):
            try:
                step = int(child.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            checkpoints.append((step, child))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint-* dirs found under: {root_or_checkpoint}")
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def resolve_best_checkpoint_by_metric(
    root_or_checkpoint: Path,
    *,
    metric: str = "eval_global_f1",
    mode: str = "max",
) -> Path:
    if not root_or_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {root_or_checkpoint}")
    if root_or_checkpoint.is_file():
        raise ValueError(f"Expected directory, got file: {root_or_checkpoint}")
    if root_or_checkpoint.name.startswith("checkpoint-"):
        return root_or_checkpoint

    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported mode: {mode}")

    best_path: Optional[Path] = None
    best_value: Optional[float] = None

    for child in root_or_checkpoint.iterdir():
        if not (child.is_dir() and child.name.startswith("checkpoint-")):
            continue
        state_path = child / "trainer_state.json"
        if not state_path.exists():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        global_step = state.get("global_step")
        if global_step is None:
            continue

        metric_value = None
        for record in state.get("log_history", []):
            if record.get("step") == global_step and metric in record:
                metric_value = record[metric]
                break
        if metric_value is None:
            continue

        metric_value = float(metric_value)
        if best_value is None:
            best_value = metric_value
            best_path = child
            continue
        if mode == "max" and metric_value > best_value:
            best_value = metric_value
            best_path = child
        elif mode == "min" and metric_value < best_value:
            best_value = metric_value
            best_path = child

    if best_path is None:
        raise FileNotFoundError(
            f"No checkpoint with metric `{metric}` found under: {root_or_checkpoint}"
        )
    print(f"best checkpoint by {metric} ({mode}) = {best_path} (value={best_value})")
    return best_path


def main():
    load_dotenv_if_present()
    enable_tf32 = get_env_bool("TASK2_ENABLE_TF32", required=True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
    with contextlib.suppress(Exception):
        torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")

    device = get_device()
    year = get_env("COLIEE_TASK2_YEAR", required=True)
    configured_eval_topk = get_env_int("TASK2_EVAL_TOPK", required=True)
    primary_eval_topk = 1
    secondary_eval_topk = 2
    if configured_eval_topk != 1:
        print(
            f"warning: TASK2_EVAL_TOPK={configured_eval_topk} ignored. "
            "Early stopping is fixed to validation top-1 F1."
        )

    prepared_dir = get_env_path("COLIEE_TASK2_PREPARED_DIR", required=True)
    init_model_root = get_env_path("TASK2_INIT_MODEL_ROOT", required=True)
    explicit_init_checkpoint = get_env("TASK2_INIT_CHECKPOINT", default="", allow_empty=True) or ""
    init_metric = get_env("TASK2_INIT_METRIC", required=True) or "eval_global_f1"
    init_metric_mode = get_env("TASK2_INIT_METRIC_MODE", required=True) or "max"
    output_dir = get_env_path("TASK2_OUTPUT_DIR", required=True)

    candidate_dir = prepared_dir / "processed_candidates"
    query_dir = prepared_dir / "processed_queries"
    train_qid_path = prepared_dir / "train_qid.tsv"
    valid_qid_path = prepared_dir / "valid_qid.tsv"
    labels_path = prepared_dir / f"task2_train_labels_{year}_flat.json"
    positive_train_json_path = prepared_dir / f"task2_train_labels_{year}_flat_train.json"
    valid_json_path = prepared_dir / "finetune_data" / "contrastive_task2_random15_valid.json"
    finetune_data_dir = prepared_dir / "finetune_data"
    query_candidates_map_path = prepared_dir / "query_candidates_map.json"

    for path in [
        candidate_dir,
        query_dir,
        train_qid_path,
        valid_qid_path,
        labels_path,
        positive_train_json_path,
        valid_json_path,
        query_candidates_map_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")

    task2_mode = normalize_mode(get_env("TASK2_MODE", required=True) or "full")
    quick_test = task2_mode == "test"
    test_seed = get_env_int("TASK2_TEST_SEED", required=True)
    test_train_query_limit = get_env_int("TASK2_TEST_TRAIN_QUERY_LIMIT", required=True)
    test_valid_query_limit = get_env_int("TASK2_TEST_VALID_QUERY_LIMIT", required=True)
    test_num_train_epochs = get_env_float("TASK2_TEST_NUM_TRAIN_EPOCHS", required=True)
    test_max_steps = get_env_int("TASK2_TEST_MAX_STEPS", required=True)
    test_logging_steps = get_env_int("TASK2_TEST_LOGGING_STEPS", required=True)
    test_save_total_limit = get_env_int("TASK2_TEST_SAVE_TOTAL_LIMIT", required=True)
    test_early_stopping_patience = get_env_int("TASK2_TEST_EARLY_STOPPING_PATIENCE", required=True)

    train_batch_size = get_env_int("TASK2_TRAIN_BATCH_SIZE", required=True)
    eval_batch_size = get_env_int("TASK2_EVAL_BATCH_SIZE", required=True)
    grad_accum_steps = get_env_int("TASK2_GRAD_ACCUM_STEPS", required=True)
    dataloader_num_workers = get_env_int("TASK2_DATALOADER_NUM_WORKERS", required=True)
    dataloader_pin_memory = get_env_bool("TASK2_DATALOADER_PIN_MEMORY", required=True)
    dataloader_persistent_workers = get_env_bool("TASK2_DATALOADER_PERSISTENT_WORKERS", required=True)
    use_gradient_checkpointing = get_env_bool("TASK2_GRADIENT_CHECKPOINTING", required=True)
    retrieval_batch_size = get_env_int("TASK2_RETRIEVAL_BATCH_SIZE", required=True)
    retrieval_max_length = get_env_int("TASK2_RETRIEVAL_MAX_LENGTH", required=True)
    cache_texts = get_env_bool("TASK2_CACHE_TEXTS", required=True)

    query_candidates_map = load_query_candidates_map(query_candidates_map_path)
    all_train_qids = load_query_ids_from_utils(train_qid_path)
    all_valid_qids = load_query_ids_from_utils(valid_qid_path)

    effective_train_qid_path = train_qid_path
    effective_valid_qid_path = valid_qid_path
    effective_valid_json_path = valid_json_path
    effective_query_candidates_map = query_candidates_map
    effective_finetune_data_dir = finetune_data_dir
    adaptive_candidate_files_override: Optional[List[str]] = build_candidate_files_override(
        selected_qids=all_train_qids,
        query_candidates_map=query_candidates_map,
        candidate_dir=candidate_dir,
    )
    eval_candidate_files_override: Optional[List[str]] = build_candidate_files_override(
        selected_qids=sorted(set(all_train_qids + all_valid_qids)),
        query_candidates_map=query_candidates_map,
        candidate_dir=candidate_dir,
    )

    if quick_test:
        selected_train_qids = sample_ids(all_train_qids, test_train_query_limit, test_seed)
        selected_valid_qids = sample_ids(all_valid_qids, test_valid_query_limit, test_seed + 1)
        if not selected_train_qids:
            raise ValueError("test mode selected zero training queries")
        if not selected_valid_qids:
            raise ValueError("test mode selected zero validation queries")

        quick_dir = finetune_data_dir / "quick_test"
        quick_dir.mkdir(parents=True, exist_ok=True)

        effective_train_qid_path = quick_dir / "train_qid.tsv"
        effective_valid_qid_path = quick_dir / "valid_qid.tsv"
        write_qid_tsv(effective_train_qid_path, selected_train_qids)
        write_qid_tsv(effective_valid_qid_path, selected_valid_qids)

        effective_valid_json_path = quick_dir / "contrastive_task2_random15_valid.json"
        valid_count = filter_contrastive_json(
            valid_json_path,
            effective_valid_json_path,
            allow_qids=set(selected_valid_qids),
        )
        if valid_count == 0:
            raise ValueError("test mode produced zero valid contrastive samples")

        selected_all_qids = sorted(set(selected_train_qids + selected_valid_qids))
        effective_query_candidates_map = {
            qid: query_candidates_map[qid]
            for qid in selected_all_qids
            if qid in query_candidates_map
        }
        quick_candidate_files_override = build_candidate_files_override(
            selected_qids=selected_all_qids,
            query_candidates_map=effective_query_candidates_map,
            candidate_dir=candidate_dir,
        )
        if not quick_candidate_files_override:
            raise ValueError("test mode found zero candidate files")
        adaptive_candidate_files_override = quick_candidate_files_override
        eval_candidate_files_override = quick_candidate_files_override

        if not output_dir.name.endswith("_test"):
            output_dir = output_dir.parent / f"{output_dir.name}_test"
        effective_finetune_data_dir = quick_dir
        print(
            "test mode enabled: "
            f"train_q={len(selected_train_qids)}, "
            f"valid_q={len(selected_valid_qids)}, "
            f"valid_samples={valid_count}, "
            f"candidate_files={len(quick_candidate_files_override)}"
        )

    if explicit_init_checkpoint:
        init_checkpoint = Path(explicit_init_checkpoint).resolve()
    else:
        try:
            init_checkpoint = resolve_best_checkpoint_by_metric(
                init_model_root,
                metric=init_metric,
                mode=init_metric_mode,
            )
        except Exception as err:
            print(
                f"warning: resolve_best_checkpoint_by_metric failed ({err}), "
                "fallback to latest checkpoint"
            )
            init_checkpoint = resolve_latest_checkpoint(init_model_root)
    print(f"init checkpoint: {init_checkpoint}")
    print(f"mode={task2_mode}")
    print(f"query->candidates mapping loaded: {len(effective_query_candidates_map)} queries")
    print("evaluation topk: top-1 (early stop) + top-2 (extra report)")
    print(
        f"train_batch_size={train_batch_size}, grad_accum_steps={grad_accum_steps}, "
        f"eval_batch_size={eval_batch_size}, retrieval_batch_size={retrieval_batch_size}"
    )
    print(
        f"tf32={enable_tf32}, gradient_checkpointing={use_gradient_checkpointing}, "
        f"retrieval_max_length={retrieval_max_length}, cache_texts={cache_texts}"
    )

    tokenizer = AutoTokenizer.from_pretrained(str(init_checkpoint), trust_remote_code=True)
    model = ModernBERTContrastive.from_pretrained(
        str(init_checkpoint),
        encoder_model_name_or_path=str(init_model_root),
        encoder_kwargs={
            "torch_dtype": torch.float32,
            "attn_implementation": "flash_attention_2",
            "trust_remote_code": True,
        },
    )
    model = model.to(device)
    if hasattr(model, "encoder") and model.encoder is not None:
        with contextlib.suppress(Exception):
            if use_gradient_checkpointing:
                model.encoder.gradient_checkpointing_enable()
            else:
                model.encoder.gradient_checkpointing_disable()
    print("tokenizer/model loaded")

    train_dataset = ContrastiveDataset(
        query_folder=str(query_dir),
        candidate_folder=str(candidate_dir),
        data=[],
        cache_texts=cache_texts,
    )
    valid_dataset = ContrastiveDataset(
        query_folder=str(query_dir),
        candidate_folder=str(candidate_dir),
        json_path=str(effective_valid_json_path),
        cache_texts=cache_texts,
    )
    print(f"valid_dataset size: {len(valid_dataset)}")

    num_train_epochs = get_env_float("TASK2_NUM_TRAIN_EPOCHS", required=True)
    logging_steps = get_env_int("TASK2_LOGGING_STEPS", required=True)
    save_total_limit = get_env_int("TASK2_SAVE_TOTAL_LIMIT", required=True)
    early_stopping_patience = get_env_int("TASK2_EARLY_STOPPING_PATIENCE", required=True)
    max_steps = get_env_int("TASK2_MAX_STEPS", required=True)
    if quick_test:
        num_train_epochs = test_num_train_epochs
        logging_steps = test_logging_steps
        save_total_limit = test_save_total_limit
        early_stopping_patience = test_early_stopping_patience
        max_steps = test_max_steps if test_max_steps > 0 else -1
    persistent_workers = dataloader_persistent_workers and dataloader_num_workers > 0

    args = TrainingArguments(
        output_dir=str(output_dir),
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=persistent_workers,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        per_device_eval_batch_size=eval_batch_size,
        fp16=True,
        bf16=False,
        tf32=enable_tf32,
        learning_rate=5e-6,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        optim="adamw_torch_fused",
        logging_strategy="steps",
        logging_steps=logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_global_f1",
        greater_is_better=True,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        include_for_metrics=["loss"],
        prediction_loss_only=False,
        logging_dir=str(output_dir / "tb"),
    )
    args.temperature_lr = 5e-4

    trainer = AdaptiveNegativeSamplingTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=ContrastiveCollator(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics_for_retrieval(
            model=model,
            tokenizer=tokenizer,
            candidate_dataset_path=str(candidate_dir),
            query_dataset_path=str(query_dir),
            train_qid_path=str(effective_train_qid_path),
            valid_qid_path=str(effective_valid_qid_path),
            labels_path=str(labels_path),
            output_dir=str(effective_finetune_data_dir),
            primary_eval_topk=primary_eval_topk,
            secondary_eval_topk=secondary_eval_topk,
            query_candidates_map=effective_query_candidates_map,
            retrieval_batch_size=retrieval_batch_size,
            retrieval_max_length=retrieval_max_length,
            quick_test=quick_test,
            candidate_files_override=eval_candidate_files_override,
        ),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            TensorBoardExtras(),
            EvaluationCallback(),
        ],
        candidate_dataset_path=str(candidate_dir),
        query_dataset_path=str(query_dir),
        train_qid_path=str(effective_train_qid_path),
        positive_train_json_path=str(positive_train_json_path),
        finetune_data_dir=str(effective_finetune_data_dir),
        query_candidates_map=effective_query_candidates_map,
        sampling_temperature=1.0,
        update_frequency=1,
        max_negatives=15,
        retrieval_batch_size=retrieval_batch_size,
        retrieval_max_length=retrieval_max_length,
        quick_test=quick_test,
        candidate_files_override=adaptive_candidate_files_override,
    )
    trainer.add_callback(AdaptiveNegativeSamplingCallback(trainer))

    print(
        "train start: "
        f"queries={len(load_query_ids_from_utils(effective_train_qid_path))}, "
        f"candidates="
        f"{len(adaptive_candidate_files_override) if adaptive_candidate_files_override else len(list(candidate_dir.glob('*.txt')))}"
    )
    resume_ckpt = get_env("TASK2_RESUME_CHECKPOINT", default="", allow_empty=True) or ""
    if resume_ckpt:
        print(f"resume_from_checkpoint={resume_ckpt}")
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()

    print(f"training finished. output_dir={output_dir}")


if __name__ == "__main__":
    main()
