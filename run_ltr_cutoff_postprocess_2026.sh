#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export COLIEE_TASK1_YEAR="${COLIEE_TASK1_YEAR:-2026}"
export COLIEE_TASK1_ROOT="${COLIEE_TASK1_ROOT:-./coliee_dataset/task1}"
export COLIEE_TASK1_DIR="${COLIEE_TASK1_DIR:-${COLIEE_TASK1_ROOT}/${COLIEE_TASK1_YEAR}}"

TASK1_DIR="${COLIEE_TASK1_DIR}"
LTR_OUTPUT_DIR="${COLIEE_LTR_OUTPUT_DIR:-${TASK1_DIR}/lht_process/lightgbm_ltr_scope_raw}"
VALID_PRED_PATH="${COLIEE_LTR_VALID_PRED_PATH:-${LTR_OUTPUT_DIR}/valid_predictions_raw.csv}"
TEST_PRED_PATH="${COLIEE_LTR_TEST_PRED_PATH:-${LTR_OUTPUT_DIR}/test_predictions_raw.csv}"
VALID_SCOPE_PATH="${COLIEE_LTR_VALID_SCOPE_PATH:-${TASK1_DIR}/lht_process/scope_compare/query_candidate_scope_raw_plus0.json}"
TEST_SCOPE_PATH="${COLIEE_LTR_TEST_SCOPE_PATH:-${TASK1_DIR}/lht_process/modernBert/query_candidate_scope_test_raw.json}"
VALID_QID_PATH="${COLIEE_LTR_VALID_QID_PATH:-${TASK1_DIR}/valid_qid.tsv}"
TEST_QID_PATH="${COLIEE_LTR_TEST_QID_PATH:-${TASK1_DIR}/test_qid.tsv}"
CUTOFF_OUTPUT_DIR="${COLIEE_LTR_CUTOFF_OUTPUT_DIR:-${LTR_OUTPUT_DIR}/cutoff_search}"
PREFERRED_CONFIG_JSON="${LTR_OUTPUT_DIR}/cutoff_search_expanded_config.json"
CUTOFF_CONFIG_JSON="${COLIEE_LTR_CUTOFF_CONFIG_JSON:-}"
SUBMISSION_RUN_TAG="${COLIEE_LTR_SUBMISSION_RUN_TAG:-FLNLPLTR}"
FINAL_SUBMISSION_PATH="${COLIEE_LTR_FINAL_SUBMISSION_PATH:-${REPO_ROOT}/task1_${SUBMISSION_RUN_TAG}.txt}"
SKIP_SUBMISSION_COPY="${COLIEE_LTR_SKIP_SUBMISSION_COPY:-0}"

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

resolve_cutoff_config() {
  if [[ -n "${CUTOFF_CONFIG_JSON}" ]]; then
    printf '%s\n' "${CUTOFF_CONFIG_JSON}"
    return 0
  fi
  if [[ -f "${PREFERRED_CONFIG_JSON}" ]]; then
    printf '%s\n' "${PREFERRED_CONFIG_JSON}"
    return 0
  fi
  printf '\n'
}

COPY_FROM_SUBMISSION_PATH="${CUTOFF_OUTPUT_DIR}/best_overall/test_submission_best_mode.txt"

CUTOFF_CONFIG_RESOLVED="$(resolve_cutoff_config)"

mkdir -p "${CUTOFF_OUTPUT_DIR}"

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] TASK1_DIR=${TASK1_DIR}"
echo "[INFO] LTR_OUTPUT_DIR=${LTR_OUTPUT_DIR}"
echo "[INFO] VALID_PRED_PATH=${VALID_PRED_PATH}"
echo "[INFO] TEST_PRED_PATH=${TEST_PRED_PATH}"
echo "[INFO] VALID_SCOPE_PATH=${VALID_SCOPE_PATH}"
echo "[INFO] TEST_SCOPE_PATH=${TEST_SCOPE_PATH}"
echo "[INFO] CUTOFF_OUTPUT_DIR=${CUTOFF_OUTPUT_DIR}"
if [[ -n "${CUTOFF_CONFIG_RESOLVED}" ]]; then
  echo "[INFO] CUTOFF_CONFIG_JSON=${CUTOFF_CONFIG_RESOLVED}"
else
  echo "[INFO] CUTOFF_CONFIG_JSON=<builtin defaults>"
fi

echo "[INFO] SUBMISSION_RUN_TAG=${SUBMISSION_RUN_TAG}"
echo "[INFO] FINAL_SUBMISSION_PATH=${FINAL_SUBMISSION_PATH}"

require_nonempty_file "${VALID_PRED_PATH}"
require_nonempty_file "${TEST_PRED_PATH}"
require_nonempty_file "${VALID_SCOPE_PATH}"
require_nonempty_file "${TEST_SCOPE_PATH}"
require_nonempty_file "${VALID_QID_PATH}"
require_nonempty_file "${TEST_QID_PATH}"

CUTOFF_ARGS=(
  --valid-predictions "${VALID_PRED_PATH}"
  --test-predictions "${TEST_PRED_PATH}"
  --valid-scope "${VALID_SCOPE_PATH}"
  --test-scope "${TEST_SCOPE_PATH}"
  --valid-qid "${VALID_QID_PATH}"
  --test-qid "${TEST_QID_PATH}"
  --output-dir "${CUTOFF_OUTPUT_DIR}"
  --submission-run-tag "${SUBMISSION_RUN_TAG}"
)

if [[ -n "${CUTOFF_CONFIG_RESOLVED}" ]]; then
  CUTOFF_ARGS+=(--cutoff-config-json "${CUTOFF_CONFIG_RESOLVED}")
fi

run_step "Run LTR cutoff postprocess only" \
  python "Legal Case Retrieval/lightgbm/cutoff_postprocess.py" \
  "${CUTOFF_ARGS[@]}"

require_nonempty_file "${CUTOFF_OUTPUT_DIR}/cutoff_summary.json"
require_nonempty_file "${COPY_FROM_SUBMISSION_PATH}"

if [[ "${SKIP_SUBMISSION_COPY}" != "1" ]]; then
  run_step "Copy final submission to repo root" \
    cp "${COPY_FROM_SUBMISSION_PATH}" "${FINAL_SUBMISSION_PATH}"
  require_nonempty_file "${FINAL_SUBMISSION_PATH}"
fi

echo
echo "[DONE] LTR cutoff postprocess complete."
echo "[OUT ] summary     -> ${CUTOFF_OUTPUT_DIR}/cutoff_summary.json"
echo "[OUT ] submission  -> ${COPY_FROM_SUBMISSION_PATH}"
if [[ "${SKIP_SUBMISSION_COPY}" != "1" ]]; then
  echo "[OUT ] copied file -> ${FINAL_SUBMISSION_PATH}"
fi
