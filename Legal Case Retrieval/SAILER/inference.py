from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

import torch
from transformers import AutoModel, AutoTokenizer

from lcr.data import EmbeddingsData
from lcr.device import get_device
from lcr.embeddings import process_directory_to_embeddings

# 檢查 GPU 是否可用
device = get_device()

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("CSHaitao/SAILER_en_finetune")
model = AutoModel.from_pretrained("CSHaitao/SAILER_en_finetune")

# 將模型移動到指定的設備 (GPU 或 CPU)
model.to(device)
model.eval()

def encode_batch(batch_inputs):
    with torch.no_grad():
        outputs = model(**batch_inputs)
    return outputs.last_hidden_state[:, 0, :]

# Path to the processed documents
# 這裡固定同時編碼 processed 與 processed_new。
# processed_new 保留給 THUIR-style query 實驗使用；目前本 repo 的 similarity 預設 query / candidate 都使用 processed。
model_name = "SAILER"
print(f"------Using {model_name} to encode documents------\n")
candidate_dataset_path = f"{TASK1_DIR}/processed"
query_dataset_path = f"{TASK1_DIR}/processed_new"
candidate_output_path = f"{TASK1_DIR}/processed/processed_document_{model_name}_embeddings.pkl"
query_output_path = f"{TASK1_DIR}/processed_new/processed_new_document_{model_name}_embeddings.pkl"
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
    batch_size=8,
    max_length=512,
    device=device,
    show_progress=True,
)
print(f"💾 Candidate embeddings saved to {candidate_output_path} ({len(candidate_data)} documents)")


# -------------------------------
# Query 資料集處理
# -------------------------------
print("--------------------------")
print(f"\n🔹 Encoding query documents located at {query_dataset_path} ...")
query_data = process_directory_to_embeddings(
    query_dataset_path,
    query_output_path,
    tokenizer,
    encode_batch=encode_batch,
    batch_size=8,
    max_length=512,
    device=device,
    show_progress=True,
)
print(f"💾 Query embeddings saved to {query_output_path} ({len(query_data)} documents)")

print("\n✅ All embeddings saved successfully.")
