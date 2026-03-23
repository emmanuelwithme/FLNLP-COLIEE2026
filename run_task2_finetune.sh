#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

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

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Required directory not found: ${path}" >&2
    exit 1
  fi
}

_CALLER_CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"
_CALLER_COLIEE_TASK2_YEAR="${COLIEE_TASK2_YEAR:-}"
_CALLER_COLIEE_TASK2_ROOT="${COLIEE_TASK2_ROOT:-}"
_CALLER_COLIEE_TASK2_DIR="${COLIEE_TASK2_DIR:-}"
_CALLER_COLIEE_TASK2_PREPARED_DIR="${COLIEE_TASK2_PREPARED_DIR:-}"
_CALLER_TASK2_MODE="${TASK2_MODE:-}"
_CALLER_TASK2_SKIP_STATS="${TASK2_SKIP_STATS:-}"
_CALLER_TASK2_OUTPUT_DIR="${TASK2_OUTPUT_DIR:-}"
_CALLER_TASK2_TEST_TRAIN_QUERY_LIMIT="${TASK2_TEST_TRAIN_QUERY_LIMIT:-}"
_CALLER_TASK2_TEST_VALID_QUERY_LIMIT="${TASK2_TEST_VALID_QUERY_LIMIT:-}"
_CALLER_TASK2_TEST_NUM_TRAIN_EPOCHS="${TASK2_TEST_NUM_TRAIN_EPOCHS:-}"
_CALLER_TASK2_TEST_MAX_STEPS="${TASK2_TEST_MAX_STEPS:-}"
_CALLER_TASK2_TEST_LOGGING_STEPS="${TASK2_TEST_LOGGING_STEPS:-}"
_CALLER_TASK2_TEST_SAVE_TOTAL_LIMIT="${TASK2_TEST_SAVE_TOTAL_LIMIT:-}"
_CALLER_TASK2_TEST_EARLY_STOPPING_PATIENCE="${TASK2_TEST_EARLY_STOPPING_PATIENCE:-}"
_CALLER_TASK2_TRAIN_BATCH_SIZE="${TASK2_TRAIN_BATCH_SIZE:-}"
_CALLER_TASK2_EVAL_BATCH_SIZE="${TASK2_EVAL_BATCH_SIZE:-}"
_CALLER_TASK2_GRAD_ACCUM_STEPS="${TASK2_GRAD_ACCUM_STEPS:-}"
_CALLER_TASK2_DATALOADER_NUM_WORKERS="${TASK2_DATALOADER_NUM_WORKERS:-}"
_CALLER_TASK2_DATALOADER_PERSISTENT_WORKERS="${TASK2_DATALOADER_PERSISTENT_WORKERS:-}"
_CALLER_TASK2_RETRIEVAL_BATCH_SIZE="${TASK2_RETRIEVAL_BATCH_SIZE:-}"
_CALLER_TASK2_RETRIEVAL_MAX_LENGTH="${TASK2_RETRIEVAL_MAX_LENGTH:-}"

if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ -n "${_CALLER_CONDA_ENV_NAME}" ]]; then
  CONDA_ENV_NAME="${_CALLER_CONDA_ENV_NAME}"
fi
if [[ -n "${_CALLER_COLIEE_TASK2_YEAR}" ]]; then
  COLIEE_TASK2_YEAR="${_CALLER_COLIEE_TASK2_YEAR}"
fi
if [[ -n "${_CALLER_COLIEE_TASK2_ROOT}" ]]; then
  COLIEE_TASK2_ROOT="${_CALLER_COLIEE_TASK2_ROOT}"
fi
if [[ -n "${_CALLER_COLIEE_TASK2_DIR}" ]]; then
  COLIEE_TASK2_DIR="${_CALLER_COLIEE_TASK2_DIR}"
fi
if [[ -n "${_CALLER_COLIEE_TASK2_PREPARED_DIR}" ]]; then
  COLIEE_TASK2_PREPARED_DIR="${_CALLER_COLIEE_TASK2_PREPARED_DIR}"
fi
if [[ -n "${_CALLER_TASK2_MODE}" ]]; then
  TASK2_MODE="${_CALLER_TASK2_MODE}"
fi
if [[ -n "${_CALLER_TASK2_SKIP_STATS}" ]]; then
  TASK2_SKIP_STATS="${_CALLER_TASK2_SKIP_STATS}"
fi
if [[ -n "${_CALLER_TASK2_OUTPUT_DIR}" ]]; then
  TASK2_OUTPUT_DIR="${_CALLER_TASK2_OUTPUT_DIR}"
fi
if [[ -n "${_CALLER_TASK2_TEST_TRAIN_QUERY_LIMIT}" ]]; then
  TASK2_TEST_TRAIN_QUERY_LIMIT="${_CALLER_TASK2_TEST_TRAIN_QUERY_LIMIT}"
fi
if [[ -n "${_CALLER_TASK2_TEST_VALID_QUERY_LIMIT}" ]]; then
  TASK2_TEST_VALID_QUERY_LIMIT="${_CALLER_TASK2_TEST_VALID_QUERY_LIMIT}"
fi
if [[ -n "${_CALLER_TASK2_TEST_NUM_TRAIN_EPOCHS}" ]]; then
  TASK2_TEST_NUM_TRAIN_EPOCHS="${_CALLER_TASK2_TEST_NUM_TRAIN_EPOCHS}"
fi
if [[ -n "${_CALLER_TASK2_TEST_MAX_STEPS}" ]]; then
  TASK2_TEST_MAX_STEPS="${_CALLER_TASK2_TEST_MAX_STEPS}"
fi
if [[ -n "${_CALLER_TASK2_TEST_LOGGING_STEPS}" ]]; then
  TASK2_TEST_LOGGING_STEPS="${_CALLER_TASK2_TEST_LOGGING_STEPS}"
fi
if [[ -n "${_CALLER_TASK2_TEST_SAVE_TOTAL_LIMIT}" ]]; then
  TASK2_TEST_SAVE_TOTAL_LIMIT="${_CALLER_TASK2_TEST_SAVE_TOTAL_LIMIT}"
fi
if [[ -n "${_CALLER_TASK2_TEST_EARLY_STOPPING_PATIENCE}" ]]; then
  TASK2_TEST_EARLY_STOPPING_PATIENCE="${_CALLER_TASK2_TEST_EARLY_STOPPING_PATIENCE}"
fi
if [[ -n "${_CALLER_TASK2_TRAIN_BATCH_SIZE}" ]]; then
  TASK2_TRAIN_BATCH_SIZE="${_CALLER_TASK2_TRAIN_BATCH_SIZE}"
fi
if [[ -n "${_CALLER_TASK2_EVAL_BATCH_SIZE}" ]]; then
  TASK2_EVAL_BATCH_SIZE="${_CALLER_TASK2_EVAL_BATCH_SIZE}"
fi
if [[ -n "${_CALLER_TASK2_GRAD_ACCUM_STEPS}" ]]; then
  TASK2_GRAD_ACCUM_STEPS="${_CALLER_TASK2_GRAD_ACCUM_STEPS}"
fi
if [[ -n "${_CALLER_TASK2_DATALOADER_NUM_WORKERS}" ]]; then
  TASK2_DATALOADER_NUM_WORKERS="${_CALLER_TASK2_DATALOADER_NUM_WORKERS}"
fi
if [[ -n "${_CALLER_TASK2_DATALOADER_PERSISTENT_WORKERS}" ]]; then
  TASK2_DATALOADER_PERSISTENT_WORKERS="${_CALLER_TASK2_DATALOADER_PERSISTENT_WORKERS}"
fi
if [[ -n "${_CALLER_TASK2_RETRIEVAL_BATCH_SIZE}" ]]; then
  TASK2_RETRIEVAL_BATCH_SIZE="${_CALLER_TASK2_RETRIEVAL_BATCH_SIZE}"
fi
if [[ -n "${_CALLER_TASK2_RETRIEVAL_MAX_LENGTH}" ]]; then
  TASK2_RETRIEVAL_MAX_LENGTH="${_CALLER_TASK2_RETRIEVAL_MAX_LENGTH}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-FLNLP-COLIEE2026-WSL}"
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  echo "[ERROR] conda.sh not found under ~/miniconda3 or ~/anaconda3" >&2
  exit 1
fi
conda activate "${CONDA_ENV_NAME}"

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Conda env: ${CONDA_DEFAULT_ENV:-<unknown>}"
echo "[INFO] COLIEE_TASK2_YEAR=${COLIEE_TASK2_YEAR:-2026}"
echo "[INFO] TASK2_MODE=${TASK2_MODE:-full}"

TASK2_YEAR="${COLIEE_TASK2_YEAR:-2026}"
TASK2_ROOT="${COLIEE_TASK2_ROOT:-./coliee_dataset/task2}"
TASK2_DIR="${COLIEE_TASK2_DIR:-${TASK2_ROOT}/task2_train_files_${TASK2_YEAR}}"
TASK2_PREPARED_DIR="${COLIEE_TASK2_PREPARED_DIR:-./Legal Case Entailment by Mou/data/task2_${TASK2_YEAR}_prepared}"
TASK2_SKIP_STATS="${TASK2_SKIP_STATS:-0}"

require_dir "${TASK2_DIR}/cases"
require_file "${TASK2_DIR}/task2_train_labels_${TASK2_YEAR}.json"

run_step "Prepare task2 paragraph data" \
  python "Legal Case Entailment by Mou/prepare_task2_paragraph_data.py"

require_dir "${TASK2_PREPARED_DIR}/processed_queries"
require_dir "${TASK2_PREPARED_DIR}/processed_candidates"
require_file "${TASK2_PREPARED_DIR}/query_candidates_map.json"
require_file "${TASK2_PREPARED_DIR}/train_qid.tsv"
require_file "${TASK2_PREPARED_DIR}/valid_qid.tsv"
require_file "${TASK2_PREPARED_DIR}/finetune_data/contrastive_task2_random15_valid.json"

if [[ "${TASK2_SKIP_STATS}" == "1" ]]; then
  echo "[INFO] Skip statistics step (TASK2_SKIP_STATS=1)"
else
  run_step "Generate task2 statistics" \
    python "Legal Case Entailment by Mou/analyze_task2_stats.py"

  require_file "${TASK2_PREPARED_DIR}/stats/summary.json"
  require_file "${TASK2_PREPARED_DIR}/stats/relevant_count_distribution.csv"
  require_file "${TASK2_PREPARED_DIR}/stats/query_token_length_hist.png"
  require_file "${TASK2_PREPARED_DIR}/stats/candidate_token_length_hist.png"
fi

run_step "Train task2 paragraph encoder" \
  python "Legal Case Entailment by Mou/fine_tune_task2.py"

echo
echo "[DONE] Task2 preprocess + fine-tune finished."
