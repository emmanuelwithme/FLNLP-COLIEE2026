# 有 SFT + chunk aggregation 的模型推論相似度計算
from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from env_utils import load_chunkagg_dotenv

_LOADED_DOTENV_PATH = load_chunkagg_dotenv(__file__)

from lcr.task1_paths import get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

from lcr.data import EmbeddingsData, load_query_ids
from lcr.device import get_device
from lcr.embedding_selection import (
    log_task1_embedding_choices,
    select_task1_embedding_path,
)
from lcr.similarity import compute_similarity_and_save

# 中文註解：TASK1_CHUNKAGG_MODEL_NAME 需與 inference.py 產出的 embeddings 檔名一致。
MODEL_NAME = os.getenv("TASK1_CHUNKAGG_MODEL_NAME", "modernBert_fp_chunkAgg_fp16")

if __name__ == "__main__":
    _device = get_device()
    print(f"🔹 similarity_and_rank device: {_device}")

    # 中文註解：以下 env 方便切換不同 embeddings 與 query id 清單，不必改程式。
    processed_doc_embedding_path = os.getenv(
        "TASK1_CHUNKAGG_CAND_EMB_PATH",
        f"{TASK1_DIR}/processed/processed_document_{MODEL_NAME}_embeddings.pkl",
    )
    processed_new_doc_embedding_path = os.getenv(
        "TASK1_CHUNKAGG_QUERY_EMB_PATH",
        f"{TASK1_DIR}/processed_new/processed_new_document_{MODEL_NAME}_embeddings.pkl",
    )
    valid_qid_path = os.getenv("TASK1_CHUNKAGG_VALID_QID_PATH", f"{TASK1_DIR}/valid_qid.tsv")
    train_qid_path = os.getenv("TASK1_CHUNKAGG_TRAIN_QID_PATH", f"{TASK1_DIR}/train_qid.tsv")
    output_dot_train_path = os.getenv(
        "TASK1_CHUNKAGG_OUTPUT_DOT_TRAIN",
        f"{TASK1_DIR}/lht_process/{MODEL_NAME}/output_{MODEL_NAME}_dot_train.tsv",
    )
    output_dot_valid_path = os.getenv(
        "TASK1_CHUNKAGG_OUTPUT_DOT_VALID",
        f"{TASK1_DIR}/lht_process/{MODEL_NAME}/output_{MODEL_NAME}_dot_valid.tsv",
    )
    output_cos_valid_path = os.getenv(
        "TASK1_CHUNKAGG_OUTPUT_COS_VALID",
        f"{TASK1_DIR}/lht_process/{MODEL_NAME}/output_{MODEL_NAME}_cos_valid.tsv",
    )
    output_cos_train_path = os.getenv(
        "TASK1_CHUNKAGG_OUTPUT_COS_TRAIN",
        f"{TASK1_DIR}/lht_process/{MODEL_NAME}/output_{MODEL_NAME}_cos_train.tsv",
    )

    # 中文註解：TASK1_CHUNKAGG_SCOPE_PATH 可直接指定 query candidate scope JSON。
    model_scope_path = Path(
        os.getenv(
            "TASK1_CHUNKAGG_SCOPE_PATH",
            f"{TASK1_DIR}/lht_process/{MODEL_NAME}/query_candidate_scope.json",
        )
    )
    shared_scope_path = Path(f"{TASK1_DIR}/lht_process/modernBert/query_candidate_scope.json")
    if model_scope_path.exists():
        query_candidate_scope_path = model_scope_path
        print(f"🔹 使用 query candidate scope: {query_candidate_scope_path}")
    elif shared_scope_path.exists():
        query_candidate_scope_path = shared_scope_path
        print(f"🔹 未找到 {model_scope_path}，改用共用 scope: {query_candidate_scope_path}")
    else:
        print(f"⚠️ 未找到 {model_scope_path}，也未找到 {shared_scope_path}。")
        print("⚠️ 將對全部 candidates 計分。")
        query_candidate_scope_path = None

    query_selection = select_task1_embedding_path(
        role="query",
        processed_path=processed_doc_embedding_path,
        processed_new_path=processed_new_doc_embedding_path,
        source_env_names=("TASK1_CHUNKAGG_QUERY_EMB_SOURCE", "LCR_QUERY_EMBED_SOURCE"),
        path_env_names=("TASK1_CHUNKAGG_QUERY_EMB_PATH", "LCR_QUERY_EMBEDDINGS_PATH"),
    )
    candidate_selection = select_task1_embedding_path(
        role="candidate",
        processed_path=processed_doc_embedding_path,
        processed_new_path=processed_new_doc_embedding_path,
        source_env_names=("TASK1_CHUNKAGG_CAND_EMB_SOURCE", "LCR_CANDIDATE_EMBED_SOURCE"),
        path_env_names=("TASK1_CHUNKAGG_CAND_EMB_PATH", "LCR_CANDIDATE_EMBEDDINGS_PATH"),
    )
    log_task1_embedding_choices(
        processed_path=processed_doc_embedding_path,
        processed_new_path=processed_new_doc_embedding_path,
        query_selection=query_selection,
        candidate_selection=candidate_selection,
    )

    query_doc_data = EmbeddingsData.load(query_selection.path)
    candidate_doc_data = EmbeddingsData.load(candidate_selection.path)

    valid_qids = load_query_ids(valid_qid_path)
    train_qids = load_query_ids(train_qid_path)

    for split_name, qids in [("valid", valid_qids), ("train", train_qids)]:
        for metric in ["dot", "cos"]:
            output_path = {
                ("valid", "dot"): output_dot_valid_path,
                ("train", "dot"): output_dot_train_path,
                ("valid", "cos"): output_cos_valid_path,
                ("train", "cos"): output_cos_train_path,
            }[(split_name, metric)]
            missing = compute_similarity_and_save(
                qids,
                query_doc_data,
                candidate_doc_data,
                output_path,
                metric=metric,
                run_tag=f"{MODEL_NAME}_{metric}",
                query_candidate_scope_path=query_candidate_scope_path,
            )
            if missing:
                print(f"⚠️ {split_name} split 缺少 {len(missing)} 個查詢向量：{missing}")
            print(f"✅ 已輸出 {split_name} split / {metric} 相似度至 {output_path}")
