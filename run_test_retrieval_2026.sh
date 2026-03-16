#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export COLIEE_TASK1_YEAR=2026
export COLIEE_TASK1_ROOT="${COLIEE_TASK1_ROOT:-./coliee_dataset/task1}"
export COLIEE_TASK1_DIR="${COLIEE_TASK1_DIR:-${COLIEE_TASK1_ROOT}/${COLIEE_TASK1_YEAR}}"

TASK1_DIR="${COLIEE_TASK1_DIR}"
MODEL_NAME="modernBert_fp_fp16"
BM25_RUN_TAG="FLNLPBM25"
EMBED_RUN_TAG="FLNLPEMBED"
BM25_TEST_RAW="${TASK1_DIR}/lht_process/BM25_test/output_bm25_test_raw.tsv"
BM25_TEST_SCOPED="${TASK1_DIR}/lht_process/BM25_test/output_bm25_test_rawscope.tsv"
TEST_SCOPE_PATH="${TASK1_DIR}/lht_process/modernBert/query_candidate_scope_test_raw.json"
BM25_SUBMISSION_PATH="${TASK1_DIR}/lht_process/submission/task1_${BM25_RUN_TAG}.txt"
EMBED_SUBMISSION_PATH="${TASK1_DIR}/lht_process/submission/task1_${EMBED_RUN_TAG}.txt"
SUBMISSION_TOPK="${SUBMISSION_TOPK:-5}"

run_step() {
  local title="$1"
  shift
  echo
  echo "============================================================"
  echo "[STEP] ${title}"
  echo "============================================================"
  "$@"
}

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Required file not found: ${path}" >&2
    exit 1
  fi
}

require_nonempty_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "[ERROR] Required non-empty file not found (or empty): ${path}" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Required directory not found: ${path}" >&2
    exit 1
  fi
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

require_file "${TASK1_DIR}/task1_test_no_labels_${COLIEE_TASK1_YEAR}.json"
require_dir "${TASK1_DIR}/task1_test_files_${COLIEE_TASK1_YEAR}"
require_dir "${REPO_ROOT}/modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_${COLIEE_TASK1_YEAR}"
require_dir "${REPO_ROOT}/modernbert-caselaw-accsteps-fp/checkpoint-29000"

mkdir -p "${TASK1_DIR}/lht_process/BM25_test"
mkdir -p "${TASK1_DIR}/lht_process/${MODEL_NAME}"
mkdir -p "${TASK1_DIR}/lht_process/submission"

run_step "Build processed_test corpus from raw test files" \
  python "Legal Case Retrieval/pre-process/process_test_files.py"
require_dir "${TASK1_DIR}/processed_test"

run_step "Prepare test_qid / BM25 query_test.tsv / BM25 corpus.json" \
  python "Legal Case Retrieval/pre-process/prepare_test_pipeline_inputs.py"
require_nonempty_file "${TASK1_DIR}/test_qid.tsv"
require_nonempty_file "${TASK1_DIR}/lht_process/BM25_test/query_test.tsv"
require_nonempty_file "${TASK1_DIR}/lht_process/BM25_test/corpus/corpus.json"

run_step "Build test scope JSON (raw+0)" \
  python "Legal Case Retrieval/pre-process/build_query_candidate_scope.py" \
    --candidate-dir "${TASK1_DIR}/task1_test_files_${COLIEE_TASK1_YEAR}" \
    --query-dir "${TASK1_DIR}/task1_test_files_${COLIEE_TASK1_YEAR}" \
    --query-ids-path "${TASK1_DIR}/test_qid.tsv" \
    --output-path "${TEST_SCOPE_PATH}"
require_nonempty_file "${TEST_SCOPE_PATH}"

run_step "Build BM25 test index" \
  bash "Legal Case Retrieval/lexical models/linux/index_test.sh"
require_dir "${TASK1_DIR}/lht_process/BM25_test/index"

run_step "Run BM25 search on test queries" \
  env BM25_TEST_OUTPUT_PATH="${BM25_TEST_RAW}" \
    bash "Legal Case Retrieval/lexical models/linux/query_search_test.sh"
require_nonempty_file "${BM25_TEST_RAW}"

run_step "Apply raw scope filter to BM25 test" \
  python "Legal Case Retrieval/utils/filter_trec_by_scope.py" \
    --input-path "${BM25_TEST_RAW}" \
    --output-path "${BM25_TEST_SCOPED}" \
    --scope-path "${TEST_SCOPE_PATH}" \
    --qid-path "${TASK1_DIR}/test_qid.tsv" \
    --skip-self \
    --strict-scope
require_nonempty_file "${BM25_TEST_SCOPED}"

run_step "Convert BM25 TREC to submission format with raw scope filter" \
  python "Legal Case Retrieval/utils/trec_to_submission.py" \
    --trec-path "${BM25_TEST_SCOPED}" \
    --output-path "${BM25_SUBMISSION_PATH}" \
    --run-tag "${BM25_RUN_TAG}" \
    --topk "${SUBMISSION_TOPK}" \
    --scope-path "${TEST_SCOPE_PATH}" \
    --qid-path "${TASK1_DIR}/test_qid.tsv" \
    --skip-self \
    --strict-scope
require_nonempty_file "${BM25_SUBMISSION_PATH}"
validate_submission "${BM25_SUBMISSION_PATH}"

run_step "Encode test corpus with modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_2026" \
  env LCR_TEST_MODE=1 python "Legal Case Retrieval/modernBert-fp/inference.py"
require_nonempty_file "${TASK1_DIR}/processed_test/processed_test_document_${MODEL_NAME}_embeddings.pkl"

run_step "Rank test queries with encoder embeddings (raw scope-filtered)" \
  env LCR_TEST_MODE=1 LCR_QUERY_CANDIDATE_SCOPE_JSON="${TEST_SCOPE_PATH}" \
    python "Legal Case Retrieval/modernBert-fp/similarity_and_rank.py"
require_nonempty_file "${TASK1_DIR}/lht_process/${MODEL_NAME}/output_${MODEL_NAME}_dot_test.tsv"

run_step "Convert encoder TREC to submission format with raw scope filter" \
  python "Legal Case Retrieval/utils/trec_to_submission.py" \
    --trec-path "${TASK1_DIR}/lht_process/${MODEL_NAME}/output_${MODEL_NAME}_dot_test.tsv" \
    --output-path "${EMBED_SUBMISSION_PATH}" \
    --run-tag "${EMBED_RUN_TAG}" \
    --topk "${SUBMISSION_TOPK}" \
    --scope-path "${TEST_SCOPE_PATH}" \
    --qid-path "${TASK1_DIR}/test_qid.tsv" \
    --skip-self \
    --strict-scope
require_nonempty_file "${EMBED_SUBMISSION_PATH}"
validate_submission "${EMBED_SUBMISSION_PATH}"

echo
echo "[DONE] Test retrieval pipeline complete."
echo "[OUT ] BM25  -> ${BM25_SUBMISSION_PATH}"
echo "[OUT ] EMBED -> ${EMBED_SUBMISSION_PATH}"
