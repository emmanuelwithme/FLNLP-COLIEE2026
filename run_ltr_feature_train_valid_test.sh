#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

# shellcheck source=scripts/common_env.sh
source "${REPO_ROOT}/scripts/common_env.sh"
load_env_file_if_present "${REPO_ROOT}/.env"

require_envs \
  COLIEE_TASK1_YEAR \
  COLIEE_TASK1_DIR \
  TASK1_TRAIN_SPLIT_LABELS_PATH \
  TASK1_VALID_SPLIT_LABELS_PATH \
  TASK1_TRAIN_QID_PATH \
  TASK1_VALID_QID_PATH \
  TASK1_TEST_QID_PATH \
  TASK1_TRAIN_RAW_DIR \
  TASK1_PROCESSED_DIR \
  TASK1_TEST_RAW_DIR \
  TASK1_TEST_PROCESSED_DIR \
  TASK1_BM25_DIR \
  TASK1_BM25_TEST_DIR \
  TASK1_RETRIEVAL_MODEL_NAME \
  TASK1_MODEL_ROOT_DIR \
  TASK1_BASE_ENCODER_DIR \
  TASK1_QUICK_TEST \
  COLIEE_LTR_OUTPUT_DIR \
  COLIEE_LTR_VALID_SCOPE_PATH \
  COLIEE_LTR_TEST_SCOPE_PATH \
  COLIEE_LTR_NUM_WORKERS \
  COLIEE_LTR_LEXICAL_PREFETCH_BATCH_SIZE \
  COLIEE_LTR_LEXICAL_BATCH_MAX_THREADS \
  COLIEE_LTR_LEXICAL_BATCH_MAX_QUERIES \
  COLIEE_LTR_LEXICAL_BATCH_MAX_TOTAL_HITS \
  COLIEE_LTR_LEXICAL_BATCH_MAX_K \
  COLIEE_LTR_DENSE_BATCH_SIZE \
  COLIEE_LTR_CHUNK_WARMUP_CASE_BATCH_SIZE \
  COLIEE_LTR_FEATURE_SCORE_BATCH_SIZE \
  COLIEE_LTR_LGBM_DEVICE \
  COLIEE_LTR_FIXED_TOPK \
  COLIEE_LTR_FIXED_TOPK_RUN_TAG \
  COLIEE_LTR_FIXED_TOPK_FINAL_SUBMISSION_PATH

resolve_env_path_var "${REPO_ROOT}" COLIEE_TASK1_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_TRAIN_SPLIT_LABELS_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_VALID_SPLIT_LABELS_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_TRAIN_QID_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_VALID_QID_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_TEST_QID_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_TRAIN_RAW_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_PROCESSED_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_TEST_RAW_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_TEST_PROCESSED_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_BM25_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_BM25_TEST_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_MODEL_ROOT_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_BASE_ENCODER_DIR
resolve_env_path_var "${REPO_ROOT}" COLIEE_LTR_OUTPUT_DIR
resolve_env_path_var "${REPO_ROOT}" COLIEE_LTR_VALID_SCOPE_PATH
resolve_env_path_var "${REPO_ROOT}" COLIEE_LTR_TEST_SCOPE_PATH
resolve_env_path_var "${REPO_ROOT}" COLIEE_LTR_FIXED_TOPK_FINAL_SUBMISSION_PATH
resolve_env_path_if_set_var "${REPO_ROOT}" COLIEE_LTR_FIXED_TOPK_OUTPUT_DIR

EMBED_SUFFIX=""
if is_truthy "${TASK1_QUICK_TEST}"; then
  EMBED_SUFFIX="_test"
fi

TRAIN_VALID_EMBEDDINGS="${TASK1_PROCESSED_DIR}/processed_document_${TASK1_RETRIEVAL_MODEL_NAME}_embeddings${EMBED_SUFFIX}.pkl"
TEST_EMBEDDINGS="${TASK1_TEST_PROCESSED_DIR}/processed_test_document_${TASK1_RETRIEVAL_MODEL_NAME}_embeddings${EMBED_SUFFIX}.pkl"

export JAVA_TOOL_OPTIONS="${COLIEE_JAVA_TOOL_OPTIONS:-${JAVA_TOOL_OPTIONS:-}}"

PIPELINE_ARGS=(
  --task1-dir "${COLIEE_TASK1_DIR}"
  --year "${COLIEE_TASK1_YEAR}"
  --train-labels "${TASK1_TRAIN_SPLIT_LABELS_PATH}"
  --valid-labels "${TASK1_VALID_SPLIT_LABELS_PATH}"
  --train-qid "${TASK1_TRAIN_QID_PATH}"
  --valid-qid "${TASK1_VALID_QID_PATH}"
  --test-qid "${TASK1_TEST_QID_PATH}"
  --train-valid-scope "${COLIEE_LTR_VALID_SCOPE_PATH}"
  --test-scope "${COLIEE_LTR_TEST_SCOPE_PATH}"
  --train-valid-raw-dir "${TASK1_TRAIN_RAW_DIR}"
  --train-valid-clean-dir "${TASK1_PROCESSED_DIR}"
  --test-raw-dir "${TASK1_TEST_RAW_DIR}"
  --test-clean-dir "${TASK1_TEST_PROCESSED_DIR}"
  --train-valid-embeddings "${TRAIN_VALID_EMBEDDINGS}"
  --test-embeddings "${TEST_EMBEDDINGS}"
  --bm25-index "${TASK1_BM25_DIR}/index"
  --qld-index "${TASK1_BM25_DIR}/index"
  --bm25-ngram-index "${TASK1_BM25_DIR%/}/../BM25_ngram/index"
  --bm25-test-index "${TASK1_BM25_TEST_DIR}/index"
  --qld-test-index "${TASK1_BM25_TEST_DIR}/index"
  --bm25-ngram-test-index "${TASK1_BM25_TEST_DIR%/}/../BM25_ngram_test/index"
  --model-root-dir "${TASK1_MODEL_ROOT_DIR}"
  --base-encoder-dir "${TASK1_BASE_ENCODER_DIR}"
  --output-dir "${COLIEE_LTR_OUTPUT_DIR}"
  --num-workers "${COLIEE_LTR_NUM_WORKERS}"
  --lexical-prefetch-batch-size "${COLIEE_LTR_LEXICAL_PREFETCH_BATCH_SIZE}"
  --lexical-batch-max-threads "${COLIEE_LTR_LEXICAL_BATCH_MAX_THREADS}"
  --lexical-batch-max-queries "${COLIEE_LTR_LEXICAL_BATCH_MAX_QUERIES}"
  --lexical-batch-max-total-hits "${COLIEE_LTR_LEXICAL_BATCH_MAX_TOTAL_HITS}"
  --lexical-batch-max-k "${COLIEE_LTR_LEXICAL_BATCH_MAX_K}"
  --dense-batch-size "${COLIEE_LTR_DENSE_BATCH_SIZE}"
  --chunk-warmup-case-batch-size "${COLIEE_LTR_CHUNK_WARMUP_CASE_BATCH_SIZE}"
  --feature-score-batch-size "${COLIEE_LTR_FEATURE_SCORE_BATCH_SIZE}"
  --lgbm-device "${COLIEE_LTR_LGBM_DEVICE}"
  --fixed-topk-k "${COLIEE_LTR_FIXED_TOPK}"
  --fixed-topk-submission-run-tag "${COLIEE_LTR_FIXED_TOPK_RUN_TAG}"
  --fixed-topk-final-submission-path "${COLIEE_LTR_FIXED_TOPK_FINAL_SUBMISSION_PATH}"
  --skip-cutoff-search
)

if [[ -n "${COLIEE_LTR_FIXED_TOPK_OUTPUT_DIR:-}" ]]; then
  PIPELINE_ARGS+=(--fixed-topk-output-dir "${COLIEE_LTR_FIXED_TOPK_OUTPUT_DIR}")
fi

python "Legal Case Retrieval/lightgbm/ltr_feature_pipeline.py" \
  "${PIPELINE_ARGS[@]}" \
  "$@"
