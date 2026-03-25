from __future__ import annotations

import logging
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
from lcr.task1_paths import get_env_flag, get_task1_dir, resolve_repo_path
from repo_config import get_env, get_env_path

TASK1_DIR = Path(get_task1_dir())
MODEL_NAME = get_env("TASK1_LEGACY_RETRIEVAL_MODEL_NAME", required=True)
CHECKPOINT_METRIC = get_env("TASK1_CHECKPOINT_METRIC", required=True)
CHECKPOINT_MODE = get_env("TASK1_CHECKPOINT_MODE", required=True)
QUICK_TEST = get_env_flag("TASK1_QUICK_TEST", required=True)
assert MODEL_NAME is not None
assert CHECKPOINT_METRIC is not None
assert CHECKPOINT_MODE is not None

from find_best_model import find_best_checkpoint

FINE_TUNE_DIR = Path(__file__).resolve().parent / "fine_tune"
if str(FINE_TUNE_DIR) not in sys.path:
    sys.path.insert(0, str(FINE_TUNE_DIR))

from modernbert_contrastive_model import ModernBERTContrastive


def enable_my_patch(enabled: bool = True):
    level = logging.DEBUG if enabled else logging.WARNING
    logging.getLogger("my_debug").setLevel(level)


def _resolved_env_path(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name, "").strip()
    resolved = resolve_repo_path(raw) if raw else default.resolve()
    assert resolved is not None
    return resolved


enable_my_patch(True)
device = get_device()

model_root_dir = get_env_path("TASK1_LEGACY_MODEL_ROOT_DIR", required=True)
assert model_root_dir is not None

best_loss_ckpt = find_best_checkpoint(str(model_root_dir), "eval_loss", mode="min")
print("最佳 eval_loss checkpoint:", best_loss_ckpt)
best_acc1_ckpt = find_best_checkpoint(str(model_root_dir), "eval_acc1", mode="max")
print("最佳 eval_acc1 checkpoint:", best_acc1_ckpt)
best_acc5_ckpt = find_best_checkpoint(str(model_root_dir), "eval_acc5", mode="max")
print("最佳 eval_acc5 checkpoint:", best_acc5_ckpt)
best_f1_ckpt = find_best_checkpoint(str(model_root_dir), CHECKPOINT_METRIC, mode=CHECKPOINT_MODE)
print(f"最佳 {CHECKPOINT_METRIC} checkpoint:", best_f1_ckpt)
best_f1_path, _ = best_f1_ckpt

tokenizer = AutoTokenizer.from_pretrained(best_f1_path)
encoder_kwargs = {"device_map": {"": str(device)}, "torch_dtype": torch.float16 if device.type == "cuda" else torch.float32}
if device.type == "cuda":
    encoder_kwargs["attn_implementation"] = "flash_attention_2"
model = ModernBERTContrastive.from_pretrained(best_f1_path, encoder_kwargs=encoder_kwargs)
model = model.to(device)
if device.type == "cuda":
    model = model.half()
model = model.eval()


def encode_batch(batch_inputs):
    return model.encode(batch_inputs)


print(f"------Using {MODEL_NAME} to encode documents------\n")
candidate_dataset_path = _resolved_env_path("TASK1_CANDIDATE_DIR", TASK1_DIR / "processed")
query_dataset_path = _resolved_env_path("TASK1_QUERY_DIR", TASK1_DIR / "processed_new")
suffix = "_test" if QUICK_TEST else ""
candidate_output_path = _resolved_env_path(
    "TASK1_CANDIDATE_EMBEDDINGS_OUTPUT",
    TASK1_DIR / "processed" / f"processed_document_{MODEL_NAME}_embeddings{suffix}.pkl",
)
query_output_path = _resolved_env_path(
    "TASK1_QUERY_EMBEDDINGS_OUTPUT",
    TASK1_DIR / "processed_new" / f"processed_new_document_{MODEL_NAME}_embeddings{suffix}.pkl",
)
if QUICK_TEST:
    print("⚙️  QUICK_TEST 模式啟用：使用測試模型與輸出路徑")
print("🔹 推論會同時輸出 processed 與 processed_new 兩份 embeddings。")

print("--------------------------")
print(f"\n🔹 Encoding candidate documents located at {candidate_dataset_path} ...")
candidate_data = process_directory_to_embeddings(
    str(candidate_dataset_path),
    str(candidate_output_path),
    tokenizer,
    encode_batch=encode_batch,
    batch_size=1,
    max_length=4096,
    device=device,
    show_progress=True,
)
print(f"💾 Candidate embeddings saved to {candidate_output_path} ({len(candidate_data)} documents)")

print("--------------------------")
print(f"\n🔹 Encoding query documents located at {query_dataset_path} ...")
query_data = process_directory_to_embeddings(
    str(query_dataset_path),
    str(query_output_path),
    tokenizer,
    encode_batch=encode_batch,
    batch_size=1,
    max_length=4096,
    device=device,
    show_progress=True,
)
print(f"💾 Query embeddings saved to {query_output_path} ({len(query_data)} documents)")

print("\n✅ All embeddings saved successfully.")
