from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Ensure project root (contains the lcr package) is importable when running from repo root.
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from env_utils import load_chunkagg_dotenv

_LOADED_DOTENV_PATH = load_chunkagg_dotenv(__file__)

from lcr.task1_paths import get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

from lcr.data import EmbeddingsData, read_text_directory
from lcr.device import get_device
from lcr.retrieval import _generate_document_embeddings

# 將 chunkAgg 版模型載入器加入路徑
sys.path.append(os.path.join(os.path.dirname(__file__), "fine_tune"))
from modernbert_contrastive_model import (  # noqa: E402
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


# 中文註解：TASK1_CHUNKAGG_INFER_QUICK_TEST=1 時，只編碼少量文件做 smoke test。
QUICK_TEST = _get_env_bool("TASK1_CHUNKAGG_INFER_QUICK_TEST", False)
# 中文註解：TASK1_CHUNKAGG_INFER_SCOPE_FILTER=1 時，推論預設對應 scopeFilteredRaw 的訓練輸出目錄。
SCOPE_FILTER = _get_env_bool("TASK1_CHUNKAGG_INFER_SCOPE_FILTER", True)
# 中文註解：TASK1_CHUNKAGG_MODEL_NAME 控制 embeddings 檔名與 ranking 輸出資料夾名稱。
MODEL_NAME = os.getenv("TASK1_CHUNKAGG_MODEL_NAME", "modernBert_fp_chunkAgg_fp16")
# 中文註解：TASK1_RETRIEVAL_BATCH_SIZE 控制一次編碼多少篇文件。
EMBED_BATCH_SIZE = max(1, int(os.getenv("TASK1_RETRIEVAL_BATCH_SIZE", "1")))
# 中文註解：TASK1_CHUNKAGG_INFER_QT_LIMIT 控制 QUICK_TEST 模式最多取幾篇文件。
QUICK_TEST_LIMIT = max(1, int(os.getenv("TASK1_CHUNKAGG_INFER_QT_LIMIT", "20")))

REPO_ROOT = Path(__file__).resolve().parents[2]
# 中文註解：TASK1_CHUNKAGG_BASE_ENCODER_DIR 指向 continued pretraining 後的 backbone checkpoint。
BASE_ENCODER_DIR = Path(
    os.getenv(
        "TASK1_CHUNKAGG_BASE_ENCODER_DIR",
        str(REPO_ROOT / "modernbert-caselaw-accsteps-fp" / "checkpoint-29000"),
    )
)


def find_best_checkpoint(checkpoint_root: str | Path, metric: str, mode: str) -> tuple[str, float]:
    checkpoint_root = str(checkpoint_root)
    if mode not in {"min", "max"}:
        raise ValueError("mode 只能是 'min' 或 'max'")

    best_checkpoint = None
    best_value = None
    greater_is_better = mode == "max"

    for folder in os.listdir(checkpoint_root):
        if not folder.startswith("checkpoint-"):
            continue

        ckpt_dir = os.path.join(checkpoint_root, folder)
        state_path = os.path.join(ckpt_dir, "trainer_state.json")
        if not os.path.isfile(state_path):
            continue

        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            continue

        cur_step = state.get("global_step")
        if cur_step is None:
            continue

        value = None
        for record in state.get("log_history", []):
            if record.get("step") == cur_step and metric in record:
                value = record[metric]
                break

        if value is None:
            continue
        if best_value is None:
            best_checkpoint = ckpt_dir
            best_value = value
        elif greater_is_better and value > best_value:
            best_checkpoint = ckpt_dir
            best_value = value
        elif (not greater_is_better) and value < best_value:
            best_checkpoint = ckpt_dir
            best_value = value

    if best_checkpoint is None or best_value is None:
        raise ValueError(f"在 {checkpoint_root} 底下找不到包含 `{metric}` 的 checkpoint")
    return best_checkpoint, float(best_value)


def encode_directory(
    directory: Path,
    output_path: Path,
    *,
    model,
    tokenizer,
    device: torch.device,
) -> EmbeddingsData:
    ids, texts = read_text_directory(directory)
    if QUICK_TEST:
        ids = ids[:QUICK_TEST_LIMIT]
        texts = texts[:QUICK_TEST_LIMIT]
        print(f"[QUICK_TEST] {directory.name}: 只編碼前 {len(ids)} 篇文件")

    embeddings = _generate_document_embeddings(
        texts,
        tokenizer,
        model,
        batch_size=EMBED_BATCH_SIZE,
        max_length=DOCUMENT_CHUNK_LENGTH,
        max_chunks=MAX_DOCUMENT_CHUNKS,
        device=device,
        show_progress=True,
        progress_desc=f"Encoding {directory.name}",
    )
    data = EmbeddingsData(ids, embeddings)
    data.save(output_path)
    return data


def main() -> None:
    device = get_device()
    if not BASE_ENCODER_DIR.exists():
        raise FileNotFoundError(f"找不到 continued pretraining checkpoint: {BASE_ENCODER_DIR}")

    dir_suffix = "_scopeFilteredRaw" if SCOPE_FILTER else ""
    dir_suffix += "_test" if QUICK_TEST else ""
    # 中文註解：TASK1_CHUNKAGG_OUTPUT_DIR 指向 chunkAgg 訓練輸出的根目錄。
    model_root_dir = Path(
        os.getenv(
            "TASK1_CHUNKAGG_OUTPUT_DIR",
            f"./modernBERT_contrastive_adaptive_fp_fp16_chunkAgg{dir_suffix}_{TASK1_YEAR}",
        )
    )
    if not model_root_dir.exists():
        raise FileNotFoundError(f"找不到 chunkAgg 訓練輸出目錄: {model_root_dir}")

    best_loss_ckpt = find_best_checkpoint(model_root_dir, "eval_loss", mode="min")
    print("最佳 eval_loss checkpoint:", best_loss_ckpt)
    best_acc1_ckpt = find_best_checkpoint(model_root_dir, "eval_acc1", mode="max")
    print("最佳 eval_acc1 checkpoint:", best_acc1_ckpt)
    best_acc5_ckpt = find_best_checkpoint(model_root_dir, "eval_acc5", mode="max")
    print("最佳 eval_acc5 checkpoint:", best_acc5_ckpt)
    best_f1_ckpt = find_best_checkpoint(model_root_dir, "eval_global_f1", mode="max")
    print("最佳 eval_global_f1 checkpoint:", best_f1_ckpt)
    best_f1_path, _ = best_f1_ckpt

    tokenizer = AutoTokenizer.from_pretrained(best_f1_path, trust_remote_code=True)
    # 中文註解：chunking 會自行控制長度，避免 tokenizer 對長文本發出截斷警告。
    tokenizer.model_max_length = 1_000_000_000

    model = ModernBERTContrastive(
        str(BASE_ENCODER_DIR),
        device=device,
        max_chunks=MAX_DOCUMENT_CHUNKS,
        chunk_microbatch_size=CHUNK_MICROBATCH_SIZE,
    )
    model.load_checkpoint(best_f1_path)
    model = model.to(device)
    model = model.eval()

    print("------Using chunkAgg ModernBERT to encode documents------\n")
    print(f"   - Embedding batch size: {EMBED_BATCH_SIZE}")
    print(f"   - Max chunks: {MAX_DOCUMENT_CHUNKS}")
    print(f"   - Chunk length: {DOCUMENT_CHUNK_LENGTH}")
    print(f"   - Chunk microbatch size: {CHUNK_MICROBATCH_SIZE}")

    suffix = "_test" if QUICK_TEST else ""
    # 中文註解：可用 env 覆寫輸入資料夾，方便切換不同預處理版本。
    candidate_dataset_path = Path(os.getenv("TASK1_CHUNKAGG_CANDIDATE_DIR", f"{TASK1_DIR}/processed"))
    query_dataset_path = Path(os.getenv("TASK1_CHUNKAGG_QUERY_DIR", f"{TASK1_DIR}/processed_new"))
    candidate_output_path = Path(
        os.getenv(
            "TASK1_CHUNKAGG_CAND_EMB_PATH",
            f"{TASK1_DIR}/processed/processed_document_{MODEL_NAME}_embeddings{suffix}.pkl",
        )
    )
    query_output_path = Path(
        os.getenv(
            "TASK1_CHUNKAGG_QUERY_EMB_PATH",
            f"{TASK1_DIR}/processed_new/processed_new_document_{MODEL_NAME}_embeddings{suffix}.pkl",
        )
    )

    print("--------------------------")
    print(f"\n🔹 Encoding candidate documents located at {candidate_dataset_path} ...")
    candidate_data = encode_directory(
        candidate_dataset_path,
        candidate_output_path,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    print(f"💾 Candidate embeddings saved to {candidate_output_path} ({len(candidate_data)} documents)")

    print("--------------------------")
    print(f"\n🔹 Encoding query documents located at {query_dataset_path} ...")
    query_data = encode_directory(
        query_dataset_path,
        query_output_path,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    print(f"💾 Query embeddings saved to {query_output_path} ({len(query_data)} documents)")

    print("\n✅ All chunkAgg embeddings saved successfully.")


if __name__ == "__main__":
    main()
