from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Ensure project root (contains the lcr package) is importable when running from repo root.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

from lcr.data import EmbeddingsData
from find_best_model import find_best_checkpoint

QUICK_TEST = False
SCOPE_FILTER = True # 使用有依照判決書年份過濾的資料來訓練的模型推論
TEST_MODE = os.getenv("LCR_TEST_MODE", "0") == "1"

# Shared utilities package (contains reusable helpers for retrieval pipelines)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.device import get_device
from lcr.embeddings import process_directory_to_embeddings

# 將 modernBERT contrastive 模型加入路徑
sys.path.append(os.path.join(os.path.dirname(__file__), "fine_tune"))
from modernbert_contrastive_model import ModernBERTContrastive, ContrastiveConfig

# 檢查 GPU 是否可用
device = get_device()

# 繼續預訓練的 ModernBERT checkpoint（提供 backbone config）
BASE_ENCODER_DIR = Path(__file__).resolve().parents[2] / "modernbert-caselaw-accsteps-fp" / "checkpoint-29000"
if not BASE_ENCODER_DIR.exists():
    raise FileNotFoundError(f"找不到繼續預訓練後的 ModernBERT checkpoint: {BASE_ENCODER_DIR}")

# 找出最佳模型checkpoint
dir_suffix = "_scopeFiltered" if SCOPE_FILTER else ""
dir_suffix += "_test" if QUICK_TEST else ""
model_root_dir = f"./modernBERT_contrastive_adaptive_fp_fp16{dir_suffix}_{TASK1_YEAR}"
best_loss_ckpt = find_best_checkpoint(model_root_dir, "eval_loss", mode="min")
print("最佳 eval_loss checkpoint:", best_loss_ckpt)
best_acc1_ckpt = find_best_checkpoint(model_root_dir, "eval_acc1", mode="max")
print("最佳 eval_acc1 checkpoint:", best_acc1_ckpt)
best_acc5_ckpt = find_best_checkpoint(model_root_dir, "eval_acc5", mode="max")
print("最佳 eval_acc5 checkpoint:", best_acc5_ckpt)
best_f1_ckpt = find_best_checkpoint(model_root_dir, "eval_global_f1", mode="max")
print("最佳 eval_global_f1 checkpoint:", best_f1_ckpt)
best_loss_path, _ = best_loss_ckpt
best_acc1_path, _ = best_acc1_ckpt
best_acc5_path, _ = best_acc5_ckpt
best_f1_path, _ = best_f1_ckpt

# 載入 tokenizer + 模型權重
tokenizer = AutoTokenizer.from_pretrained(best_f1_path, trust_remote_code=True)
encoder_kwargs = {
    "device_map": {"": str(device)},
    "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
    "trust_remote_code": True,
}
if device.type == "cuda":
    encoder_kwargs["attn_implementation"] = "flash_attention_2"
# 印出 encoder_kwargs 以供除錯
print("Encoder kwargs:", encoder_kwargs)
model = ModernBERTContrastive.from_pretrained(
    best_f1_path,
    encoder_model_name_or_path=str(BASE_ENCODER_DIR),
    encoder_kwargs=encoder_kwargs,
)
model = model.to(device)
if device.type == "cuda":
    model = model.half()  # projector 的精度也轉成 fp16
model = model.eval()


def encode_batch(batch_inputs):
    with torch.no_grad():
        return model.encode(batch_inputs)


# Path to the processed documents
# 這裡固定同時編碼 processed 與 processed_new。
# processed_new 保留給 THUIR-style query 實驗使用；目前本 repo 的 similarity 預設 query / candidate 都使用 processed。
model_name = "modernBert_fp_fp16" # 有SFT，且SFT是fp16
print(f"------Using {model_name} to encode documents------\n")
suffix = "_test" if QUICK_TEST else ""
if TEST_MODE:
    candidate_dataset_path = f"{TASK1_DIR}/processed_test"
    query_dataset_path = candidate_dataset_path
    candidate_output_path = f"{TASK1_DIR}/processed_test/processed_test_document_{model_name}_embeddings{suffix}.pkl"
    query_output_path = f"{TASK1_DIR}/processed_test/processed_test_query_{model_name}_embeddings{suffix}.pkl"
    print("⚙️  TEST_MODE 啟用：使用 processed_test 作為 candidate/query 語料")
else:
    candidate_dataset_path = f"{TASK1_DIR}/processed"
    query_dataset_path = f"{TASK1_DIR}/processed_new"
    candidate_output_path = f"{TASK1_DIR}/processed/processed_document_{model_name}_embeddings{suffix}.pkl"
    query_output_path = f"{TASK1_DIR}/processed_new/processed_new_document_{model_name}_embeddings{suffix}.pkl"
if QUICK_TEST:
    print("⚙️  QUICK_TEST 模式啟用：使用測試模型與輸出路徑")
if not TEST_MODE:
    print("🔹 推論會同時輸出 processed 與 processed_new 兩份 embeddings。")

# -------------------------------
# Candidate 資料集處理
# -------------------------------
print("--------------------------")
print(f"\n🔹 Encoding candidate documents located at {candidate_dataset_path} ...")
candidate_data = process_directory_to_embeddings(
    candidate_dataset_path,
    candidate_output_path,
    tokenizer,
    encode_batch=encode_batch,
    batch_size=1,
    max_length=4096,
    device=device,
    show_progress=True,
)
print(f"💾 Candidate embeddings saved to {candidate_output_path} ({len(candidate_data)} documents)")

# -------------------------------
# Query 資料集處理
# -------------------------------
print("--------------------------")
if TEST_MODE:
    # test query 與 candidate 同一語料，直接重用 embeddings 以避免重複編碼。
    query_data = candidate_data
    query_data.save(query_output_path)
    print(f"\n🔹 TEST_MODE：query embeddings 重用 candidate embeddings from {candidate_dataset_path}")
else:
    print(f"\n🔹 Encoding query documents located at {query_dataset_path} ...")
    query_data = process_directory_to_embeddings(
        query_dataset_path,
        query_output_path,
        tokenizer,
        encode_batch=encode_batch,
        batch_size=1,
        max_length=4096,
        device=device,
        show_progress=True,
    )
print(f"💾 Query embeddings saved to {query_output_path} ({len(query_data)} documents)")

print("\n✅ All embeddings saved successfully.")
