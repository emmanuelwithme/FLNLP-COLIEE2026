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
  TASK1_TEST_RAW_DIR \
  TASK1_TEST_LABELS_PATH \
  TASK1_TEST_PROCESSED_DIR \
  TASK1_TEST_QID_PATH \
  TASK1_TEST_SCOPE_PATH \
  TASK1_BM25_TEST_DIR \
  TASK1_SUBMISSION_DIR \
  TASK1_MODEL_RESULTS_DIR \
  TASK1_RETRIEVAL_MODEL_NAME \
  TASK1_MODEL_ROOT_DIR \
  TASK1_BASE_ENCODER_DIR \
  TASK1_QUICK_TEST \
  TASK1_SUBMISSION_TOPK \
  TASK1_BM25_RUN_TAG \
  TASK1_EMBED_RUN_TAG \
  TASK1_TEST_SCOPE_YEAR_SLACK \
  TASK1_TEST_SCOPE_UNKNOWN_QUERY_YEAR_POLICY \
  TASK1_TEST_SCOPE_EXCLUDE_SELF

resolve_env_path_var "${REPO_ROOT}" COLIEE_TASK1_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_TEST_RAW_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_TEST_LABELS_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_TEST_PROCESSED_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_TEST_QID_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_TEST_SCOPE_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_BM25_TEST_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_SUBMISSION_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_MODEL_RESULTS_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_MODEL_ROOT_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_BASE_ENCODER_DIR

TASK1_DIR="${COLIEE_TASK1_DIR}"
MODEL_NAME="${TASK1_RETRIEVAL_MODEL_NAME}"
EMBED_SUFFIX=""
if is_truthy "${TASK1_QUICK_TEST}"; then
  EMBED_SUFFIX="_test"
fi

TEST_RAW_DIR="${TASK1_TEST_RAW_DIR}"
TEST_LABELS_PATH="${TASK1_TEST_LABELS_PATH}"
TEST_PROCESSED_DIR="${TASK1_TEST_PROCESSED_DIR}"
TEST_QID_PATH="${TASK1_TEST_QID_PATH}"
TEST_SCOPE_PATH="${TASK1_TEST_SCOPE_PATH}"
BM25_TEST_RAW="${TASK1_BM25_TEST_DIR}/output_bm25_test_raw.tsv"
BM25_TEST_SCOPED="${TASK1_BM25_TEST_DIR}/output_bm25_test_rawscope.tsv"
BM25_SUBMISSION_PATH="${TASK1_SUBMISSION_DIR}/task1_${TASK1_BM25_RUN_TAG}.txt"
EMBED_SUBMISSION_PATH="${TASK1_SUBMISSION_DIR}/task1_${TASK1_EMBED_RUN_TAG}.txt"
TEST_QUERY_PATH="${TASK1_TEST_PROCESSED_DIR}"
EMBED_OUTPUT_PATH="${TASK1_TEST_PROCESSED_DIR}/processed_test_document_${MODEL_NAME}_embeddings${EMBED_SUFFIX}.pkl"
QUERY_EMBED_OUTPUT_PATH="${TASK1_TEST_PROCESSED_DIR}/processed_test_query_${MODEL_NAME}_embeddings${EMBED_SUFFIX}.pkl"
RANK_DOT_TEST_PATH="${TASK1_MODEL_RESULTS_DIR}/output_${MODEL_NAME}_dot_test.tsv"
SUBMISSION_TOPK="${TASK1_SUBMISSION_TOPK}"

run_step() {
  local title="$1"
  shift
  echo
  echo "============================================================"
  echo "[STEP] ${title}"
  echo "============================================================"
  "$@"
}

validate_submission() {
  local path="$1"
  python - "$path" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
raw = path.read_bytes()
try:
    text = raw.decode("ascii")
except UnicodeDecodeError as exc:
    print(f"[CHECK] {path}: not ASCII ({exc})")
    raise SystemExit(1)

bad_cols = 0
bad_spaces = 0
lines = 0
for line in text.splitlines():
    if not line.strip():
        continue
    lines += 1
    if "\t" in line or line.strip() != line or "  " in line:
        bad_spaces += 1
    parts = line.split(" ")
    if len(parts) != 3 or any(not part for part in parts):
        bad_cols += 1
print(f"[CHECK] {path}: lines={lines}, bad_columns={bad_cols}, bad_spaces={bad_spaces}, ascii_ok=True")
if bad_cols or bad_spaces:
    raise SystemExit(1)
PY
}

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] COLIEE_TASK1_YEAR=${COLIEE_TASK1_YEAR}"
echo "[INFO] TASK1_DIR=${TASK1_DIR}"
echo "[INFO] TASK1_RETRIEVAL_MODEL_NAME=${MODEL_NAME}"
echo "[INFO] TASK1_MODEL_ROOT_DIR=${TASK1_MODEL_ROOT_DIR}"
echo "[INFO] TASK1_BASE_ENCODER_DIR=${TASK1_BASE_ENCODER_DIR}"

require_file "${TEST_LABELS_PATH}"
require_dir "${TEST_RAW_DIR}"
require_dir "${TASK1_MODEL_ROOT_DIR}"
require_dir "${TASK1_BASE_ENCODER_DIR}"

mkdir -p "${TASK1_BM25_TEST_DIR}" "${TASK1_MODEL_RESULTS_DIR}" "${TASK1_SUBMISSION_DIR}"

run_step "Build processed_test corpus from raw test files" \
  python "Legal Case Retrieval/pre-process/process_test_files.py" \
    --input-dir "${TEST_RAW_DIR}" \
    --output-dir "${TEST_PROCESSED_DIR}"
require_dir "${TEST_PROCESSED_DIR}"

run_step "Prepare test_qid / BM25 query_test.tsv / BM25 corpus.json" \
  python "Legal Case Retrieval/pre-process/prepare_test_pipeline_inputs.py" \
    --test-label-json "${TEST_LABELS_PATH}" \
    --processed-test-dir "${TEST_PROCESSED_DIR}" \
    --test-qid-path "${TEST_QID_PATH}" \
    --bm25-query-path "${TASK1_BM25_TEST_DIR}/query_test.tsv" \
    --bm25-corpus-path "${TASK1_BM25_TEST_DIR}/corpus/corpus.json"
require_nonempty_file "${TEST_QID_PATH}"
require_nonempty_file "${TASK1_BM25_TEST_DIR}/query_test.tsv"
require_nonempty_file "${TASK1_BM25_TEST_DIR}/corpus/corpus.json"

TEST_SCOPE_ARGS=(
  --candidate-dir "${TEST_RAW_DIR}"
  --query-dir "${TEST_RAW_DIR}"
  --query-ids-path "${TEST_QID_PATH}"
  --output-path "${TEST_SCOPE_PATH}"
  --year-slack "${TASK1_TEST_SCOPE_YEAR_SLACK}"
  --unknown-query-year-policy "${TASK1_TEST_SCOPE_UNKNOWN_QUERY_YEAR_POLICY}"
)
if is_truthy "${TASK1_TEST_SCOPE_EXCLUDE_SELF}"; then
  TEST_SCOPE_ARGS+=(--exclude-self)
fi

run_step "Build test scope JSON" \
  python "Legal Case Retrieval/pre-process/build_query_candidate_scope.py" \
    "${TEST_SCOPE_ARGS[@]}"
require_nonempty_file "${TEST_SCOPE_PATH}"

run_step "Build BM25 test index" \
  bash "Legal Case Retrieval/lexical models/linux/index_test.sh"
require_dir "${TASK1_BM25_TEST_DIR}/index"

run_step "Run BM25 search on test queries" \
  env BM25_TEST_OUTPUT_PATH="${BM25_TEST_RAW}" \
    bash "Legal Case Retrieval/lexical models/linux/query_search_test.sh"
require_nonempty_file "${BM25_TEST_RAW}"

run_step "Apply raw scope filter to BM25 test" \
  python "Legal Case Retrieval/utils/filter_trec_by_scope.py" \
    --input-path "${BM25_TEST_RAW}" \
    --output-path "${BM25_TEST_SCOPED}" \
    --scope-path "${TEST_SCOPE_PATH}" \
    --qid-path "${TEST_QID_PATH}" \
    --skip-self \
    --strict-scope
require_nonempty_file "${BM25_TEST_SCOPED}"

run_step "Convert BM25 TREC to submission format with raw scope filter" \
  python "Legal Case Retrieval/utils/trec_to_submission.py" \
    --trec-path "${BM25_TEST_SCOPED}" \
    --output-path "${BM25_SUBMISSION_PATH}" \
    --run-tag "${TASK1_BM25_RUN_TAG}" \
    --topk "${SUBMISSION_TOPK}" \
    --scope-path "${TEST_SCOPE_PATH}" \
    --qid-path "${TEST_QID_PATH}" \
    --skip-self \
    --strict-scope
require_nonempty_file "${BM25_SUBMISSION_PATH}"
validate_submission "${BM25_SUBMISSION_PATH}"

run_step "Encode test corpus with ${MODEL_NAME}" \
  env \
    LCR_TEST_MODE=1 \
    TASK1_TEST_CANDIDATE_DIR="${TEST_PROCESSED_DIR}" \
    TASK1_TEST_QUERY_DIR="${TEST_QUERY_PATH}" \
    TASK1_TEST_CANDIDATE_EMBEDDINGS_PATH="${EMBED_OUTPUT_PATH}" \
    TASK1_TEST_QUERY_EMBEDDINGS_PATH="${QUERY_EMBED_OUTPUT_PATH}" \
    TASK1_MODEL_ROOT_DIR="${TASK1_MODEL_ROOT_DIR}" \
    TASK1_BASE_ENCODER_DIR="${TASK1_BASE_ENCODER_DIR}" \
    python "Legal Case Retrieval/modernBert-fp/inference.py"
require_nonempty_file "${EMBED_OUTPUT_PATH}"
require_nonempty_file "${QUERY_EMBED_OUTPUT_PATH}"

run_step "Rank test queries with encoder embeddings (raw scope-filtered)" \
  env \
    LCR_TEST_MODE=1 \
    TASK1_TEST_CANDIDATE_EMBEDDINGS_PATH="${EMBED_OUTPUT_PATH}" \
    TASK1_TEST_QUERY_EMBEDDINGS_PATH="${QUERY_EMBED_OUTPUT_PATH}" \
    TASK1_TEST_QID_PATH="${TEST_QID_PATH}" \
    TASK1_OUTPUT_DOT_TEST_PATH="${RANK_DOT_TEST_PATH}" \
    LCR_QUERY_CANDIDATE_SCOPE_JSON="${TEST_SCOPE_PATH}" \
    python "Legal Case Retrieval/modernBert-fp/similarity_and_rank.py"
require_nonempty_file "${RANK_DOT_TEST_PATH}"

run_step "Convert encoder TREC to submission format with raw scope filter" \
  python "Legal Case Retrieval/utils/trec_to_submission.py" \
    --trec-path "${RANK_DOT_TEST_PATH}" \
    --output-path "${EMBED_SUBMISSION_PATH}" \
    --run-tag "${TASK1_EMBED_RUN_TAG}" \
    --topk "${SUBMISSION_TOPK}" \
    --scope-path "${TEST_SCOPE_PATH}" \
    --qid-path "${TEST_QID_PATH}" \
    --skip-self \
    --strict-scope
require_nonempty_file "${EMBED_SUBMISSION_PATH}"
validate_submission "${EMBED_SUBMISSION_PATH}"

echo
echo "[DONE] Test retrieval pipeline complete."
echo "[OUT ] BM25  -> ${BM25_SUBMISSION_PATH}"
echo "[OUT ] EMBED -> ${EMBED_SUBMISSION_PATH}"
