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

from env_utils import load_chunkagg_v6_dotenv
from content_utils import filter_id_text_pairs, summarize_rejected_documents

_LOADED_DOTENV_PATH = load_chunkagg_v6_dotenv(__file__)

from lcr.task1_paths import get_env_path, get_task1_dir, get_task1_year

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


# 中文註解：TASK1_CHUNKAGG_V6_INFER_QUICK_TEST=1 時，只編碼少量文件做 smoke test。
QUICK_TEST = _get_env_bool("TASK1_CHUNKAGG_V6_INFER_QUICK_TEST", False)
INFERENCE_FP16 = _get_env_bool("TASK1_CHUNKAGG_V6_FP16", True)
INFERENCE_BF16 = _get_env_bool("TASK1_CHUNKAGG_V6_BF16", False)
if INFERENCE_FP16 and INFERENCE_BF16:
    raise ValueError("TASK1_CHUNKAGG_V6_FP16 and TASK1_CHUNKAGG_V6_BF16 cannot both be enabled")
INFERENCE_AUTOCAST_DTYPE = (
    torch.bfloat16 if INFERENCE_BF16 else torch.float16 if INFERENCE_FP16 else None
)
# 中文註解：TASK1_CHUNKAGG_V6_MODEL_NAME 控制 embeddings 檔名與 ranking 輸出資料夾名稱。
MODEL_NAME = os.environ["TASK1_CHUNKAGG_V6_MODEL_NAME"].strip()
# 中文註解：TASK1_CHUNKAGG_V6_RETRIEVAL_BATCH_SIZE 控制一次編碼多少篇文件。
EMBED_BATCH_SIZE = max(1, int(os.getenv("TASK1_CHUNKAGG_V6_RETRIEVAL_BATCH_SIZE", "2")))
# 中文註解：TASK1_CHUNKAGG_V6_INFER_QT_LIMIT 控制 QUICK_TEST 模式最多取幾篇文件。
QUICK_TEST_LIMIT = max(1, int(os.getenv("TASK1_CHUNKAGG_V6_INFER_QT_LIMIT", "20")))
MIN_DOCUMENT_CONTENT_CHARS = max(
    1,
    int(os.getenv("TASK1_CHUNKAGG_V6_MIN_DOCUMENT_CHARS", "100")),
)
CHUNK_SELECTION_STRATEGY = os.getenv(
    "TASK1_CHUNKAGG_V6_LONG_DOCUMENT_SELECTION",
    "head_middle_tail",
).strip().lower()

# 中文註解：TASK1_CHUNKAGG_V6_BASE_ENCODER_DIR 指向 continued pretraining 後的 backbone checkpoint。
BASE_ENCODER_DIR = get_env_path("TASK1_CHUNKAGG_V6_BASE_ENCODER_DIR", required=True)
assert BASE_ENCODER_DIR is not None


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
    ids, texts, rejected_lengths = filter_id_text_pairs(
        ids,
        texts,
        minimum_characters=MIN_DOCUMENT_CONTENT_CHARS,
    )
    if rejected_lengths:
        print(
            f"🔹 {directory.name} content filter rejected "
            f"{len(rejected_lengths)} documents: "
            f"{summarize_rejected_documents(rejected_lengths)}"
        )
    if not ids:
        raise RuntimeError(
            f"No documents in {directory} have at least "
            f"{MIN_DOCUMENT_CONTENT_CHARS} non-whitespace characters"
        )
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
        chunk_selection_strategy=CHUNK_SELECTION_STRATEGY,
        device=device,
        show_progress=True,
        progress_desc=f"Encoding {directory.name}",
        autocast_dtype=INFERENCE_AUTOCAST_DTYPE,
    )
    data = EmbeddingsData(ids, embeddings)
    data.save(output_path)
    return data


def _quick_output_path(path: Path) -> Path:
    if not QUICK_TEST:
        return path
    return path.with_name(f"{path.stem}_quick{path.suffix}")


def main() -> None:
    device = get_device()
    if not BASE_ENCODER_DIR.exists():
        raise FileNotFoundError(f"找不到 continued pretraining checkpoint: {BASE_ENCODER_DIR}")

    # 中文註解：TASK1_CHUNKAGG_V6_OUTPUT_DIR 指向 chunkAgg 訓練輸出的根目錄。
    model_root_dir = get_env_path("TASK1_CHUNKAGG_V6_OUTPUT_DIR", required=True)
    assert model_root_dir is not None
    if not model_root_dir.exists():
        raise FileNotFoundError(f"找不到 chunkAgg 訓練輸出目錄: {model_root_dir}")

    scope_path = get_env_path(
        "TASK1_CHUNKAGG_V6_SCOPE_PATH",
        default=f"{TASK1_DIR}/lht_process/scope_compare/query_candidate_scope_raw_plus0_empty_exclude.json",
        required=True,
    )
    assert scope_path is not None
    if not scope_path.exists():
        raise FileNotFoundError(f"找不到 TASK1_CHUNKAGG_V6_SCOPE_PATH: {scope_path}")
    os.environ["LCR_QUERY_CANDIDATE_SCOPE_JSON"] = str(scope_path)
    print(f"🔹 推論/後續 ranking 使用必要的 query candidate scope: {scope_path}")

    for metric, mode in (("eval_loss", "min"), ("eval_acc1", "max"), ("eval_acc5", "max")):
        try:
            print(f"最佳 {metric} checkpoint:", find_best_checkpoint(model_root_dir, metric, mode))
        except ValueError as exc:
            print(f"⚠️ {exc}")
    best_f1_ckpt = find_best_checkpoint(model_root_dir, "eval_global_f1", mode="max")
    print("最佳 eval_global_f1 checkpoint:", best_f1_ckpt)
    best_f1_path, _ = best_f1_ckpt

    tokenizer = AutoTokenizer.from_pretrained(best_f1_path, trust_remote_code=True)
    # 中文註解：chunking 會自行控制長度，避免 tokenizer 對長文本發出截斷警告。
    tokenizer.model_max_length = 1_000_000_000

    model = ModernBERTContrastive(
        str(BASE_ENCODER_DIR),
        device=device,
        temperature=float(os.getenv("TASK1_CHUNKAGG_V6_INIT_TEMPERATURE", "0.55555")),
        max_chunks=MAX_DOCUMENT_CHUNKS,
        chunk_microbatch_size=CHUNK_MICROBATCH_SIZE,
        temperature_min=float(os.getenv("TASK1_CHUNKAGG_V6_TEMPERATURE_MIN", "0.05")),
        temperature_max=float(os.getenv("TASK1_CHUNKAGG_V6_TEMPERATURE_MAX", "2.0")),
        evidence_loss_weight=float(
            os.getenv("TASK1_CHUNKAGG_V6_EVIDENCE_LOSS_WEIGHT", "0.2")
        ),
        evidence_pool_temperature=float(
            os.getenv("TASK1_CHUNKAGG_V6_EVIDENCE_POOL_TEMPERATURE", "0.1")
        ),
        fusion_residual_max_norm=float(
            os.getenv("TASK1_CHUNKAGG_V6_FUSION_RESIDUAL_MAX_NORM", "0.5")
        ),
        fusion_residual_penalty_weight=float(
            os.getenv("TASK1_CHUNKAGG_V6_FUSION_RESIDUAL_PENALTY_WEIGHT", "0.01")
        ),
    )
    model.load_checkpoint(best_f1_path)
    model = model.to(device)
    model = model.eval()

    print("------Using chunkAgg ModernBERT to encode documents------\n")
    print(f"   - Embedding batch size: {EMBED_BATCH_SIZE}")
    print(f"   - Max chunks: {MAX_DOCUMENT_CHUNKS}")
    print(f"   - Chunk length: {DOCUMENT_CHUNK_LENGTH}")
    print(f"   - Chunk microbatch size: {CHUNK_MICROBATCH_SIZE}")
    print(f"   - Long-document chunk selection: {CHUNK_SELECTION_STRATEGY}")
    print(
        "   - Minimum document content: "
        f"{MIN_DOCUMENT_CONTENT_CHARS} non-whitespace characters"
    )
    print(
        "   - Retrieval precision: "
        + (str(INFERENCE_AUTOCAST_DTYPE) if INFERENCE_AUTOCAST_DTYPE is not None else "float32")
    )

    # Query and candidate roles intentionally share the same eligible `processed`
    # corpus and one embedding file in the canonical v6 comparison.
    processed_dir = Path(f"{TASK1_DIR}/processed")
    configured_embeddings_path = get_env_path(
        "TASK1_CHUNKAGG_V6_EMBEDDINGS_PATH", required=True
    )
    assert configured_embeddings_path is not None
    embeddings_path = _quick_output_path(configured_embeddings_path)
    print("🔹 Query/candidate 共用 processed 與同一份 embeddings。")
    print(f"\n🔹 Encoding eligible processed documents at {processed_dir} ...")
    embedding_data = encode_directory(
        processed_dir,
        embeddings_path,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    if not QUICK_TEST and len(embedding_data) != 7703:
        raise RuntimeError(
            f"v6 inference data contract drift: encoded={len(embedding_data)}, expected=7703"
        )
    print(f"💾 Embeddings saved to {embeddings_path} ({len(embedding_data)} documents)")

    print("\n✅ All chunkAgg embeddings saved successfully.")


if __name__ == "__main__":
    main()
