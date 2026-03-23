#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export COLIEE_TASK1_YEAR="${COLIEE_TASK1_YEAR:-2026}"
export COLIEE_TASK1_ROOT="${COLIEE_TASK1_ROOT:-./coliee_dataset/task1}"
export COLIEE_TASK1_DIR="${COLIEE_TASK1_DIR:-${COLIEE_TASK1_ROOT}/${COLIEE_TASK1_YEAR}}"
LTR_OUTPUT_DIR="${COLIEE_LTR_OUTPUT_DIR:-${COLIEE_TASK1_DIR}/lht_process/lightgbm_ltr_scope_raw}"
FIXED_TOPK_VALUE="${COLIEE_LTR_FIXED_TOPK:-5}"
FIXED_TOPK_RUN_TAG="${COLIEE_LTR_FIXED_TOPK_RUN_TAG:-FLNLPLTRTOP5}"
FIXED_TOPK_FINAL_SUBMISSION_PATH="${COLIEE_LTR_FIXED_TOPK_FINAL_SUBMISSION_PATH:-${REPO_ROOT}/task1_${FIXED_TOPK_RUN_TAG}.txt}"
# Give Lucene/Pyserini a safer default heap; user-defined JAVA_TOOL_OPTIONS still has priority.
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms2g -Xmx12g}"

CPU_CORES="${COLIEE_LTR_NUM_WORKERS:-$(nproc)}"
if (( CPU_CORES > 12 )); then
  LEXICAL_THREADS_DEFAULT=12
else
  LEXICAL_THREADS_DEFAULT="${CPU_CORES}"
fi

PIPELINE_ARGS=(
  --output-dir "${LTR_OUTPUT_DIR}"
  --num-workers "${CPU_CORES}"
  --lexical-prefetch-batch-size "${COLIEE_LTR_LEXICAL_PREFETCH_BATCH_SIZE:-64}"
  --lexical-batch-max-threads "${COLIEE_LTR_LEXICAL_BATCH_MAX_THREADS:-${LEXICAL_THREADS_DEFAULT}}"
  --lexical-batch-max-queries "${COLIEE_LTR_LEXICAL_BATCH_MAX_QUERIES:-16}"
  --lexical-batch-max-total-hits "${COLIEE_LTR_LEXICAL_BATCH_MAX_TOTAL_HITS:-120000}"
  --lexical-batch-max-k "${COLIEE_LTR_LEXICAL_BATCH_MAX_K:-8000}"
  --dense-batch-size "${COLIEE_LTR_DENSE_BATCH_SIZE:-24}"
  --chunk-warmup-case-batch-size "${COLIEE_LTR_CHUNK_WARMUP_CASE_BATCH_SIZE:-256}"
  --feature-score-batch-size "${COLIEE_LTR_FEATURE_SCORE_BATCH_SIZE:-8192}"
  --lgbm-device "${COLIEE_LTR_LGBM_DEVICE:-cuda}"
  --fixed-topk-k "${FIXED_TOPK_VALUE}"
  --fixed-topk-submission-run-tag "${FIXED_TOPK_RUN_TAG}"
  --fixed-topk-final-submission-path "${FIXED_TOPK_FINAL_SUBMISSION_PATH}"
  --skip-cutoff-search
)

if [[ -n "${COLIEE_LTR_FIXED_TOPK_OUTPUT_DIR:-}" ]]; then
  PIPELINE_ARGS+=(--fixed-topk-output-dir "${COLIEE_LTR_FIXED_TOPK_OUTPUT_DIR}")
fi

python "Legal Case Retrieval/lightgbm/ltr_feature_pipeline.py" \
  "${PIPELINE_ARGS[@]}" \
  "$@"
