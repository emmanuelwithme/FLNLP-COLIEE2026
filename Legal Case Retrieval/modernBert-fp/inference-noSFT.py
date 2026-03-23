from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

# Ensure project root (contains the lcr package) is importable when running from repo root.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

from lcr.device import get_device
from lcr.embeddings import process_directory_to_embeddings

MAX_LENGTH = 4096
MODEL_NAME = "modernBert_fp"  # 與 downstream 檔名保持一致（eval.py 會讀取）
QUICK_TEST = False

REPO_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = REPO_ROOT / "modernbert-caselaw-accsteps-fp" / "checkpoint-29000"

def load_tokenizer_and_model(device: torch.device):
    if not CKPT_DIR.exists():
        raise FileNotFoundError(f"找不到 continued pretraining checkpoint: {CKPT_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        CKPT_DIR,
        model_max_length=MAX_LENGTH,
        use_fast=True,
        trust_remote_code=True,
    )
    # 避免長文本警告，實際長度在 encode 時控制
    tokenizer.model_max_length = 1_000_000_000

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "device_map": {"": str(device)},
    }
    if device.type == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModel.from_pretrained(CKPT_DIR, **model_kwargs)
    model = model.eval()

    def encode_batch(batch_inputs):
        with torch.no_grad():
            outputs = model(**batch_inputs)
            # 取 CLS 表示，轉成 float32 方便後續 CPU 相似度運算
            return outputs.last_hidden_state[:, 0, :].float()

    return tokenizer, encode_batch

def main() -> None:
    device = get_device()
    tokenizer, encode_batch = load_tokenizer_and_model(device)

    suffix = "_test" if QUICK_TEST else ""
    candidate_dataset_path = Path(f"{TASK1_DIR}/processed")
    query_dataset_path = Path(f"{TASK1_DIR}/processed_new")
    candidate_output_path = Path(f"{TASK1_DIR}/processed/processed_document_{MODEL_NAME}_embeddings{suffix}.pkl")
    query_output_path = Path(f"{TASK1_DIR}/processed_new/processed_new_document_{MODEL_NAME}_embeddings{suffix}.pkl")

    print("------Using continued-pretrained ModernBERT to encode documents------\n")
    print("🔹 推論會同時輸出 processed 與 processed_new 兩份 embeddings。")

    # Candidate
    print("--------------------------")
    print(f"\n🔹 Encoding candidate documents located at {candidate_dataset_path} ...")
    candidate_data = process_directory_to_embeddings(
        candidate_dataset_path,
        candidate_output_path,
        tokenizer,
        encode_batch=encode_batch,
        batch_size=1,
        max_length=MAX_LENGTH,
        device=device,
        show_progress=True,
    )
    print(f"💾 Candidate embeddings saved to {candidate_output_path} ({len(candidate_data)} documents)")

    # Query
    print("--------------------------")
    print(f"\n🔹 Encoding query documents located at {query_dataset_path} ...")
    query_data = process_directory_to_embeddings(
        query_dataset_path,
        query_output_path,
        tokenizer,
        encode_batch=encode_batch,
        batch_size=1,
        max_length=MAX_LENGTH,
        device=device,
        show_progress=True,
    )
    print(f"💾 Query embeddings saved to {query_output_path} ({len(query_data)} documents)")

    print("\n✅ All embeddings saved successfully.")


if __name__ == "__main__":
    main()
