#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

# shellcheck source=scripts/common_env.sh
source "${REPO_ROOT}/scripts/common_env.sh"
load_env_file_if_present "${REPO_ROOT}/.env"

require_envs \
  COLIEE_LTR_OUTPUT_DIR \
  COLIEE_LTR_VALID_SCOPE_PATH \
  COLIEE_LTR_TEST_SCOPE_PATH \
  TASK1_VALID_QID_PATH \
  TASK1_TEST_QID_PATH \
  COLIEE_LTR_SUBMISSION_RUN_TAG \
  COLIEE_LTR_FINAL_SUBMISSION_PATH \
  COLIEE_LTR_SKIP_SUBMISSION_COPY

resolve_env_path_var "${REPO_ROOT}" COLIEE_LTR_OUTPUT_DIR
resolve_env_path_var "${REPO_ROOT}" COLIEE_LTR_VALID_SCOPE_PATH
resolve_env_path_var "${REPO_ROOT}" COLIEE_LTR_TEST_SCOPE_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_VALID_QID_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_TEST_QID_PATH
resolve_env_path_var "${REPO_ROOT}" COLIEE_LTR_FINAL_SUBMISSION_PATH
resolve_env_path_if_set_var "${REPO_ROOT}" COLIEE_LTR_CUTOFF_CONFIG_JSON

VALID_PRED_PATH="${COLIEE_LTR_OUTPUT_DIR}/valid_predictions_raw.csv"
TEST_PRED_PATH="${COLIEE_LTR_OUTPUT_DIR}/test_predictions_raw.csv"
CUTOFF_OUTPUT_DIR="${COLIEE_LTR_OUTPUT_DIR}/cutoff_search"
COPY_FROM_SUBMISSION_PATH="${CUTOFF_OUTPUT_DIR}/best_overall/test_submission_best_mode.txt"

run_step() {
  local title="$1"
  shift
  echo
  echo "============================================================"
  echo "[STEP] ${title}"
  echo "============================================================"
  "$@"
}

mkdir -p "${CUTOFF_OUTPUT_DIR}"

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] LTR_OUTPUT_DIR=${COLIEE_LTR_OUTPUT_DIR}"
echo "[INFO] VALID_PRED_PATH=${VALID_PRED_PATH}"
echo "[INFO] TEST_PRED_PATH=${TEST_PRED_PATH}"
echo "[INFO] VALID_SCOPE_PATH=${COLIEE_LTR_VALID_SCOPE_PATH}"
echo "[INFO] TEST_SCOPE_PATH=${COLIEE_LTR_TEST_SCOPE_PATH}"
echo "[INFO] CUTOFF_OUTPUT_DIR=${CUTOFF_OUTPUT_DIR}"
echo "[INFO] SUBMISSION_RUN_TAG=${COLIEE_LTR_SUBMISSION_RUN_TAG}"
echo "[INFO] FINAL_SUBMISSION_PATH=${COLIEE_LTR_FINAL_SUBMISSION_PATH}"

require_nonempty_file "${VALID_PRED_PATH}"
require_nonempty_file "${TEST_PRED_PATH}"
require_nonempty_file "${COLIEE_LTR_VALID_SCOPE_PATH}"
require_nonempty_file "${COLIEE_LTR_TEST_SCOPE_PATH}"
require_nonempty_file "${TASK1_VALID_QID_PATH}"
require_nonempty_file "${TASK1_TEST_QID_PATH}"

CUTOFF_ARGS=(
  --valid-predictions "${VALID_PRED_PATH}"
  --test-predictions "${TEST_PRED_PATH}"
  --valid-scope "${COLIEE_LTR_VALID_SCOPE_PATH}"
  --test-scope "${COLIEE_LTR_TEST_SCOPE_PATH}"
  --valid-qid "${TASK1_VALID_QID_PATH}"
  --test-qid "${TASK1_TEST_QID_PATH}"
  --output-dir "${CUTOFF_OUTPUT_DIR}"
  --submission-run-tag "${COLIEE_LTR_SUBMISSION_RUN_TAG}"
)

if [[ -n "${COLIEE_LTR_CUTOFF_CONFIG_JSON:-}" ]]; then
  CUTOFF_ARGS+=(--cutoff-config-json "${COLIEE_LTR_CUTOFF_CONFIG_JSON}")
fi

run_step "Run LTR cutoff postprocess only" \
  python "Legal Case Retrieval/lightgbm/cutoff_postprocess.py" \
  "${CUTOFF_ARGS[@]}"

require_nonempty_file "${CUTOFF_OUTPUT_DIR}/cutoff_summary.json"
require_nonempty_file "${COPY_FROM_SUBMISSION_PATH}"

if ! is_truthy "${COLIEE_LTR_SKIP_SUBMISSION_COPY}"; then
  run_step "Copy final submission to repo root" \
    cp "${COPY_FROM_SUBMISSION_PATH}" "${COLIEE_LTR_FINAL_SUBMISSION_PATH}"
  require_nonempty_file "${COLIEE_LTR_FINAL_SUBMISSION_PATH}"
fi

echo
echo "[DONE] LTR cutoff postprocess complete."
echo "[OUT ] summary     -> ${CUTOFF_OUTPUT_DIR}/cutoff_summary.json"
echo "[OUT ] submission  -> ${COPY_FROM_SUBMISSION_PATH}"
if ! is_truthy "${COLIEE_LTR_SKIP_SUBMISSION_COPY}"; then
  echo "[OUT ] copied file -> ${COLIEE_LTR_FINAL_SUBMISSION_PATH}"
fi
