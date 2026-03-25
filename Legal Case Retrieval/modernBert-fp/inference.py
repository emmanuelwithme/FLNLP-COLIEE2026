from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.device import get_device
from lcr.embeddings import process_directory_to_embeddings
from lcr.task1_paths import (
    get_env_flag,
    get_env_int,
    get_task1_base_encoder_dir,
    get_task1_dir,
    get_task1_model_name,
    get_task1_model_root_dir,
    get_task1_year,
    resolve_repo_path,
)

FINE_TUNE_DIR = Path(__file__).resolve().parent / "fine_tune"
if str(FINE_TUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINE_TUNE_DIR))

from find_best_model import find_best_checkpoint
from modernbert_contrastive_model import ModernBERTContrastive

TASK1_DIR = Path(get_task1_dir())
TASK1_YEAR = get_task1_year()
MODEL_NAME = get_task1_model_name()
QUICK_TEST = get_env_flag("TASK1_QUICK_TEST", required=True)
SCOPE_FILTER = get_env_flag("TASK1_SCOPE_FILTER", required=True)
TEST_MODE = os.getenv("LCR_TEST_MODE", "0") == "1"
RETRIEVAL_BATCH_SIZE = max(1, get_env_int("TASK1_RETRIEVAL_BATCH_SIZE", required=True))
RETRIEVAL_MAX_LENGTH = max(1, get_env_int("TASK1_RETRIEVAL_MAX_LENGTH", required=True))
CHECKPOINT_METRIC = get_env("TASK1_CHECKPOINT_METRIC", required=True)
CHECKPOINT_MODE = get_env("TASK1_CHECKPOINT_MODE", required=True)
assert CHECKPOINT_METRIC is not None
assert CHECKPOINT_MODE is not None


def _resolved_env_path(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name, "").strip()
    resolved = resolve_repo_path(raw) if raw else default.resolve()
    assert resolved is not None
    return resolved


device = get_device()

BASE_ENCODER_DIR = Path(get_task1_base_encoder_dir())
if not BASE_ENCODER_DIR.exists():
    raise FileNotFoundError(f"找不到繼續預訓練後的 ModernBERT checkpoint: {BASE_ENCODER_DIR}")

MODEL_ROOT_DIR = Path(get_task1_model_root_dir(scope_filter=SCOPE_FILTER, quick_test=QUICK_TEST))
if not MODEL_ROOT_DIR.exists():
    raise FileNotFoundError(f"找不到 Task 1 dense model root: {MODEL_ROOT_DIR}")

best_checkpoint_path, best_checkpoint_value = find_best_checkpoint(
    str(MODEL_ROOT_DIR),
    CHECKPOINT_METRIC,
    CHECKPOINT_MODE,
)
best_checkpoint_path = Path(best_checkpoint_path)
print(
    f"最佳 checkpoint by {CHECKPOINT_METRIC} ({CHECKPOINT_MODE}): "
    f"{best_checkpoint_path} (value={best_checkpoint_value})"
)

tokenizer = AutoTokenizer.from_pretrained(str(best_checkpoint_path), trust_remote_code=True)
encoder_kwargs = {
    "device_map": {"": str(device)},
    "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32,
    "trust_remote_code": True,
}
if device.type == "cuda":
    encoder_kwargs["attn_implementation"] = "flash_attention_2"

print("Encoder kwargs:", encoder_kwargs)
model = ModernBERTContrastive.from_pretrained(
    str(best_checkpoint_path),
    encoder_model_name_or_path=str(BASE_ENCODER_DIR),
    encoder_kwargs=encoder_kwargs,
)
model = model.to(device)
if device.type == "cuda":
    model = model.half()
model = model.eval()


def encode_batch(batch_inputs):
    with torch.no_grad():
        return model.encode(batch_inputs)


suffix = "_test" if QUICK_TEST else ""
if TEST_MODE:
    candidate_dataset_path = _resolved_env_path("TASK1_TEST_CANDIDATE_DIR", TASK1_DIR / "processed_test")
    query_dataset_path = _resolved_env_path("TASK1_TEST_QUERY_DIR", candidate_dataset_path)
    candidate_output_path = _resolved_env_path(
        "TASK1_TEST_CANDIDATE_EMBEDDINGS_PATH",
        TASK1_DIR / "processed_test" / f"processed_test_document_{MODEL_NAME}_embeddings{suffix}.pkl",
    )
    query_output_path = _resolved_env_path(
        "TASK1_TEST_QUERY_EMBEDDINGS_PATH",
        TASK1_DIR / "processed_test" / f"processed_test_query_{MODEL_NAME}_embeddings{suffix}.pkl",
    )
else:
    candidate_dataset_path = _resolved_env_path("TASK1_CANDIDATE_DIR", TASK1_DIR / "processed")
    query_dataset_path = _resolved_env_path("TASK1_QUERY_DIR", TASK1_DIR / "processed_new")
    candidate_output_path = _resolved_env_path(
        "TASK1_CANDIDATE_EMBEDDINGS_OUTPUT",
        TASK1_DIR / "processed" / f"processed_document_{MODEL_NAME}_embeddings{suffix}.pkl",
    )
    query_output_path = _resolved_env_path(
        "TASK1_QUERY_EMBEDDINGS_OUTPUT",
        TASK1_DIR / "processed_new" / f"processed_new_document_{MODEL_NAME}_embeddings{suffix}.pkl",
    )

print(f"------Using {MODEL_NAME} to encode documents------\n")
print(f"MODEL_ROOT_DIR   : {MODEL_ROOT_DIR}")
print(f"BASE_ENCODER_DIR : {BASE_ENCODER_DIR}")
print(f"CANDIDATE_DIR    : {candidate_dataset_path}")
print(f"QUERY_DIR        : {query_dataset_path}")
print(f"CANDIDATE_OUTPUT : {candidate_output_path}")
print(f"QUERY_OUTPUT     : {query_output_path}")
print(f"BATCH_SIZE       : {RETRIEVAL_BATCH_SIZE}")
print(f"MAX_LENGTH       : {RETRIEVAL_MAX_LENGTH}")
if QUICK_TEST:
    print("⚙️  QUICK_TEST 模式啟用：使用 quick-test 後綴輸出。")
if TEST_MODE:
    print("⚙️  TEST_MODE 啟用：使用 processed_test 作為 candidate/query 語料。")
else:
    print("🔹 推論會同時輸出 processed 與 processed_new 兩份 embeddings。")

print("--------------------------")
print(f"\n🔹 Encoding candidate documents located at {candidate_dataset_path} ...")
candidate_data = process_directory_to_embeddings(
    str(candidate_dataset_path),
    str(candidate_output_path),
    tokenizer,
    encode_batch=encode_batch,
    batch_size=RETRIEVAL_BATCH_SIZE,
    max_length=RETRIEVAL_MAX_LENGTH,
    device=device,
    show_progress=True,
)
print(f"💾 Candidate embeddings saved to {candidate_output_path} ({len(candidate_data)} documents)")

print("--------------------------")
if TEST_MODE:
    query_data = candidate_data
    query_data.save(str(query_output_path))
    print(f"\n🔹 TEST_MODE：query embeddings 重用 candidate embeddings from {candidate_dataset_path}")
else:
    print(f"\n🔹 Encoding query documents located at {query_dataset_path} ...")
    query_data = process_directory_to_embeddings(
        str(query_dataset_path),
        str(query_output_path),
        tokenizer,
        encode_batch=encode_batch,
        batch_size=RETRIEVAL_BATCH_SIZE,
        max_length=RETRIEVAL_MAX_LENGTH,
        device=device,
        show_progress=True,
    )
print(f"💾 Query embeddings saved to {query_output_path} ({len(query_data)} documents)")

print("\n✅ All embeddings saved successfully.")
