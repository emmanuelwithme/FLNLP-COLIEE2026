#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

# shellcheck source=scripts/common_env.sh
source "${REPO_ROOT}/scripts/common_env.sh"
load_env_file_if_present "${REPO_ROOT}/.env"

require_envs \
  CONDA_ENV_NAME \
  COLIEE_TASK2_YEAR \
  COLIEE_TASK2_DIR \
  COLIEE_TASK2_PREPARED_DIR \
  TASK2_LABELS_FILENAME \
  TASK2_PREPARE_TRAIN_RATIO \
  TASK2_PREPARE_SPLIT_SEED \
  TASK2_PREPARE_NEGATIVE_SEED \
  TASK2_PREPARE_MAX_NEGATIVES \
  TASK2_STATS_TOKENIZER_NAME \
  TASK2_STATS_BATCH_SIZE \
  TASK2_MODE \
  TASK2_SKIP_STATS

resolve_env_path_var "${REPO_ROOT}" COLIEE_TASK2_DIR
resolve_env_path_var "${REPO_ROOT}" COLIEE_TASK2_PREPARED_DIR

run_step() {
  local title="$1"
  shift
  echo
  echo "============================================================"
  echo "[STEP] ${title}"
  echo "============================================================"
  "$@"
}

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

TASK2_LABELS_PATH="${COLIEE_TASK2_DIR}/${TASK2_LABELS_FILENAME}"

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] Conda env: ${CONDA_DEFAULT_ENV:-<unknown>}"
echo "[INFO] COLIEE_TASK2_YEAR=${COLIEE_TASK2_YEAR}"
echo "[INFO] TASK2_MODE=${TASK2_MODE}"

require_dir "${COLIEE_TASK2_DIR}/cases"
require_file "${TASK2_LABELS_PATH}"

run_step "Prepare task2 paragraph data" \
  python "Legal Case Entailment by Mou/prepare_task2_paragraph_data.py" \
    --input-dir "${COLIEE_TASK2_DIR}" \
    --labels-file "${TASK2_LABELS_FILENAME}" \
    --output-dir "${COLIEE_TASK2_PREPARED_DIR}" \
    --train-ratio "${TASK2_PREPARE_TRAIN_RATIO}" \
    --split-seed "${TASK2_PREPARE_SPLIT_SEED}" \
    --negative-seed "${TASK2_PREPARE_NEGATIVE_SEED}" \
    --max-negatives "${TASK2_PREPARE_MAX_NEGATIVES}"

require_dir "${COLIEE_TASK2_PREPARED_DIR}/processed_queries"
require_dir "${COLIEE_TASK2_PREPARED_DIR}/processed_candidates"
require_file "${COLIEE_TASK2_PREPARED_DIR}/query_candidates_map.json"
require_file "${COLIEE_TASK2_PREPARED_DIR}/train_qid.tsv"
require_file "${COLIEE_TASK2_PREPARED_DIR}/valid_qid.tsv"
require_file "${COLIEE_TASK2_PREPARED_DIR}/finetune_data/contrastive_task2_random15_valid.json"

if is_truthy "${TASK2_SKIP_STATS}"; then
  echo "[INFO] Skip statistics step (TASK2_SKIP_STATS=1)"
else
  run_step "Generate task2 statistics" \
    python "Legal Case Entailment by Mou/analyze_task2_stats.py" \
      --prepared-dir "${COLIEE_TASK2_PREPARED_DIR}" \
      --tokenizer-name "${TASK2_STATS_TOKENIZER_NAME}" \
      --batch-size "${TASK2_STATS_BATCH_SIZE}"

  require_file "${COLIEE_TASK2_PREPARED_DIR}/stats/summary.json"
  require_file "${COLIEE_TASK2_PREPARED_DIR}/stats/relevant_count_distribution.csv"
  require_file "${COLIEE_TASK2_PREPARED_DIR}/stats/query_token_length_hist.png"
  require_file "${COLIEE_TASK2_PREPARED_DIR}/stats/candidate_token_length_hist.png"
fi

run_step "Train task2 paragraph encoder" \
  python "Legal Case Entailment by Mou/fine_tune_task2.py"

echo
echo "[DONE] Task2 preprocess + fine-tune finished."
