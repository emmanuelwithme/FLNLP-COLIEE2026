from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.data import EmbeddingsData, load_query_ids
from lcr.device import get_device
from lcr.embedding_selection import (
    log_task1_embedding_choices,
    select_task1_embedding_path,
)
from lcr.similarity import compute_similarity_and_save
from lcr.task1_paths import (
    get_env_flag,
    get_task1_dir,
    get_task1_model_name,
    resolve_repo_path,
)

TASK1_DIR = Path(get_task1_dir())
MODEL_NAME = get_task1_model_name()
TEST_MODE = os.getenv("LCR_TEST_MODE", "0") == "1"
QUICK_TEST = get_env_flag("TASK1_QUICK_TEST", required=True)


def _resolved_env_path(env_name: str, default: Path) -> Path:
    raw = os.getenv(env_name, "").strip()
    resolved = resolve_repo_path(raw) if raw else default.resolve()
    assert resolved is not None
    return resolved


if __name__ == "__main__":
    _device = get_device()
    print(f"🔹 similarity_and_rank model name: {MODEL_NAME} @ {_device}")
    embed_suffix = "_test" if QUICK_TEST else ""

    if TEST_MODE:
        processed_doc_embedding_path = _resolved_env_path(
            "TASK1_TEST_CANDIDATE_EMBEDDINGS_PATH",
            TASK1_DIR / "processed_test" / f"processed_test_document_{MODEL_NAME}_embeddings{embed_suffix}.pkl",
        )
        processed_new_doc_embedding_path = _resolved_env_path(
            "TASK1_TEST_QUERY_EMBEDDINGS_PATH",
            TASK1_DIR / "processed_test" / f"processed_test_query_{MODEL_NAME}_embeddings{embed_suffix}.pkl",
        )
        test_qid_path = _resolved_env_path("TASK1_TEST_QID_PATH", TASK1_DIR / "test_qid.tsv")
        output_dot_test_path = _resolved_env_path(
            "TASK1_OUTPUT_DOT_TEST_PATH",
            TASK1_DIR / "lht_process" / MODEL_NAME / f"output_{MODEL_NAME}_dot_test.tsv",
        )
        output_cos_test_path = _resolved_env_path(
            "TASK1_OUTPUT_COS_TEST_PATH",
            TASK1_DIR / "lht_process" / MODEL_NAME / f"output_{MODEL_NAME}_cos_test.tsv",
        )
        split_to_qid_path = [("test", test_qid_path)]
        split_metric_to_output = {
            ("test", "dot"): output_dot_test_path,
            ("test", "cos"): output_cos_test_path,
        }
        model_scope_path = _resolved_env_path(
            "TASK1_MODEL_TEST_SCOPE_PATH",
            TASK1_DIR / "lht_process" / MODEL_NAME / "query_candidate_scope_test.json",
        )
        shared_scope_path = _resolved_env_path(
            "TASK1_TEST_SCOPE_PATH",
            TASK1_DIR / "lht_process" / "modernBert" / "query_candidate_scope_test_raw.json",
        )
        print("⚙️  TEST_MODE 啟用：輸出 test split 排名")
    else:
        processed_doc_embedding_path = _resolved_env_path(
            "TASK1_CANDIDATE_EMBEDDINGS_OUTPUT",
            TASK1_DIR / "processed" / f"processed_document_{MODEL_NAME}_embeddings{embed_suffix}.pkl",
        )
        processed_new_doc_embedding_path = _resolved_env_path(
            "TASK1_QUERY_EMBEDDINGS_OUTPUT",
            TASK1_DIR / "processed_new" / f"processed_new_document_{MODEL_NAME}_embeddings{embed_suffix}.pkl",
        )
        valid_qid_path = _resolved_env_path("TASK1_VALID_QID_PATH", TASK1_DIR / "valid_qid.tsv")
        train_qid_path = _resolved_env_path("TASK1_TRAIN_QID_PATH", TASK1_DIR / "train_qid.tsv")
        output_dot_train_path = _resolved_env_path(
            "TASK1_OUTPUT_DOT_TRAIN_PATH",
            TASK1_DIR / "lht_process" / MODEL_NAME / f"output_{MODEL_NAME}_dot_train.tsv",
        )
        output_dot_valid_path = _resolved_env_path(
            "TASK1_OUTPUT_DOT_VALID_PATH",
            TASK1_DIR / "lht_process" / MODEL_NAME / f"output_{MODEL_NAME}_dot_valid.tsv",
        )
        output_cos_valid_path = _resolved_env_path(
            "TASK1_OUTPUT_COS_VALID_PATH",
            TASK1_DIR / "lht_process" / MODEL_NAME / f"output_{MODEL_NAME}_cos_valid.tsv",
        )
        output_cos_train_path = _resolved_env_path(
            "TASK1_OUTPUT_COS_TRAIN_PATH",
            TASK1_DIR / "lht_process" / MODEL_NAME / f"output_{MODEL_NAME}_cos_train.tsv",
        )
        split_to_qid_path = [("valid", valid_qid_path), ("train", train_qid_path)]
        split_metric_to_output = {
            ("valid", "dot"): output_dot_valid_path,
            ("train", "dot"): output_dot_train_path,
            ("valid", "cos"): output_cos_valid_path,
            ("train", "cos"): output_cos_train_path,
        }
        model_scope_path = _resolved_env_path(
            "TASK1_MODEL_SCOPE_PATH",
            TASK1_DIR / "lht_process" / MODEL_NAME / "query_candidate_scope.json",
        )
        shared_scope_path = _resolved_env_path(
            "TASK1_SCOPE_PATH",
            TASK1_DIR / "lht_process" / "modernBert" / "query_candidate_scope.json",
        )

    env_scope_path = os.getenv("LCR_QUERY_CANDIDATE_SCOPE_JSON", "").strip()
    if env_scope_path:
        query_candidate_scope_path = resolve_repo_path(env_scope_path) or Path(env_scope_path)
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
        processed_path=str(processed_doc_embedding_path),
        processed_new_path=str(processed_new_doc_embedding_path),
        source_env_names=("LCR_QUERY_EMBED_SOURCE",),
        path_env_names=("LCR_QUERY_EMBEDDINGS_PATH",),
    )
    candidate_selection = select_task1_embedding_path(
        role="candidate",
        processed_path=str(processed_doc_embedding_path),
        processed_new_path=str(processed_new_doc_embedding_path),
        source_env_names=("LCR_CANDIDATE_EMBED_SOURCE",),
        path_env_names=("LCR_CANDIDATE_EMBEDDINGS_PATH",),
    )
    log_task1_embedding_choices(
        processed_path=str(processed_doc_embedding_path),
        processed_new_path=str(processed_new_doc_embedding_path),
        query_selection=query_selection,
        candidate_selection=candidate_selection,
    )

    query_doc_data = EmbeddingsData.load(query_selection.path)
    candidate_doc_data = EmbeddingsData.load(candidate_selection.path)

    split_to_qids = [(split_name, load_query_ids(str(qid_path))) for split_name, qid_path in split_to_qid_path]

    for split_name, qids in split_to_qids:
        for metric in ["dot", "cos"]:
            output_path = split_metric_to_output[(split_name, metric)]
            missing = compute_similarity_and_save(
                qids,
                query_doc_data,
                candidate_doc_data,
                str(output_path),
                metric=metric,
                run_tag=f"{MODEL_NAME}_{metric}",
                query_candidate_scope_path=query_candidate_scope_path,
            )
            if missing:
                print(f"⚠️ {split_name} split 缺少 {len(missing)} 個查詢向量：{missing}")
            print(f"✅ 已輸出 {split_name} split / {metric} 相似度至 {output_path}")
