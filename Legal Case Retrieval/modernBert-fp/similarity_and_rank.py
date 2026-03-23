# 有SFT的模型推論相似度計算
from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()
TEST_MODE = os.getenv("LCR_TEST_MODE", "0") == "1"

from lcr.data import EmbeddingsData, load_query_ids
from lcr.device import get_device
from lcr.embedding_selection import (
    log_task1_embedding_choices,
    select_task1_embedding_path,
)
from lcr.similarity import compute_similarity_and_save

if __name__ == "__main__":
    _device = get_device()
    model_name = "modernBert_fp_fp16"
    if TEST_MODE:
        processed_doc_embedding_path = f"{TASK1_DIR}/processed_test/processed_test_document_{model_name}_embeddings.pkl"
        processed_new_doc_embedding_path = f"{TASK1_DIR}/processed_test/processed_test_query_{model_name}_embeddings.pkl"
        test_qid_path = f"{TASK1_DIR}/test_qid.tsv"
        output_dot_test_path = f"{TASK1_DIR}/lht_process/{model_name}/output_{model_name}_dot_test.tsv"
        output_cos_test_path = f"{TASK1_DIR}/lht_process/{model_name}/output_{model_name}_cos_test.tsv"
        split_to_qid_path = [("test", test_qid_path)]
        split_metric_to_output = {
            ("test", "dot"): output_dot_test_path,
            ("test", "cos"): output_cos_test_path,
        }
        model_scope_path = Path(f"{TASK1_DIR}/lht_process/{model_name}/query_candidate_scope_test.json")
        shared_scope_path = Path(f"{TASK1_DIR}/lht_process/modernBert/query_candidate_scope_test.json")
        print("⚙️  TEST_MODE 啟用：輸出 test split 排名")
    else:
        processed_doc_embedding_path = f"{TASK1_DIR}/processed/processed_document_{model_name}_embeddings.pkl"
        processed_new_doc_embedding_path = f"{TASK1_DIR}/processed_new/processed_new_document_{model_name}_embeddings.pkl"
        valid_qid_path = f"{TASK1_DIR}/valid_qid.tsv"
        train_qid_path = f"{TASK1_DIR}/train_qid.tsv"
        output_dot_train_path = f"{TASK1_DIR}/lht_process/{model_name}/output_{model_name}_dot_train.tsv"
        output_dot_valid_path = f"{TASK1_DIR}/lht_process/{model_name}/output_{model_name}_dot_valid.tsv"
        output_cos_valid_path = f"{TASK1_DIR}/lht_process/{model_name}/output_{model_name}_cos_valid.tsv"
        output_cos_train_path = f"{TASK1_DIR}/lht_process/{model_name}/output_{model_name}_cos_train.tsv"
        split_to_qid_path = [("valid", valid_qid_path), ("train", train_qid_path)]
        split_metric_to_output = {
            ("valid", "dot"): output_dot_valid_path,
            ("train", "dot"): output_dot_train_path,
            ("valid", "cos"): output_cos_valid_path,
            ("train", "cos"): output_cos_train_path,
        }
        model_scope_path = Path(f"{TASK1_DIR}/lht_process/{model_name}/query_candidate_scope.json")
        shared_scope_path = Path(f"{TASK1_DIR}/lht_process/modernBert/query_candidate_scope.json")

    env_scope_path = os.getenv("LCR_QUERY_CANDIDATE_SCOPE_JSON")
    if env_scope_path:
        query_candidate_scope_path = Path(env_scope_path)
        print(f"🔹 使用環境變數 scope: {query_candidate_scope_path}")
    elif model_scope_path.exists():
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
        source_env_names=("LCR_QUERY_EMBED_SOURCE",),
        path_env_names=("LCR_QUERY_EMBEDDINGS_PATH",),
    )
    candidate_selection = select_task1_embedding_path(
        role="candidate",
        processed_path=processed_doc_embedding_path,
        processed_new_path=processed_new_doc_embedding_path,
        source_env_names=("LCR_CANDIDATE_EMBED_SOURCE",),
        path_env_names=("LCR_CANDIDATE_EMBEDDINGS_PATH",),
    )
    log_task1_embedding_choices(
        processed_path=processed_doc_embedding_path,
        processed_new_path=processed_new_doc_embedding_path,
        query_selection=query_selection,
        candidate_selection=candidate_selection,
    )

    query_doc_data = EmbeddingsData.load(query_selection.path)
    candidate_doc_data = EmbeddingsData.load(candidate_selection.path)

    split_to_qids = [(split_name, load_query_ids(qid_path)) for split_name, qid_path in split_to_qid_path]

    for split_name, qids in split_to_qids:
        for metric in ["dot", "cos"]:
            output_path = split_metric_to_output[(split_name, metric)]
            missing = compute_similarity_and_save(
                qids,
                query_doc_data,
                candidate_doc_data,
                output_path,
                metric=metric,
                run_tag=f"{model_name}_{metric}",
                query_candidate_scope_path=query_candidate_scope_path,
            )
            if missing:
                print(f"⚠️ {split_name} split 缺少 {len(missing)} 個查詢向量：{missing}")
            print(f"✅ 已輸出 {split_name} split / {metric} 相似度至 {output_path}")
