#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export COLIEE_TASK1_YEAR="${COLIEE_TASK1_YEAR:-2026}"
export COLIEE_TASK1_ROOT="${COLIEE_TASK1_ROOT:-./coliee_dataset/task1}"
export COLIEE_TASK1_DIR="${COLIEE_TASK1_DIR:-${COLIEE_TASK1_ROOT}/${COLIEE_TASK1_YEAR}}"
# Give Lucene/Pyserini a safer default heap; user-defined JAVA_TOOL_OPTIONS still has priority.
export JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:--Xms2g -Xmx12g}"

CPU_CORES="${COLIEE_LTR_NUM_WORKERS:-$(nproc)}"
if (( CPU_CORES > 12 )); then
  LEXICAL_THREADS_DEFAULT=12
else
  LEXICAL_THREADS_DEFAULT="${CPU_CORES}"
fi

python "Legal Case Retrieval/lightgbm/src/trees/ltr_feature_pipeline.py" \
  --num-workers "${CPU_CORES}" \
  --lexical-prefetch-batch-size "${COLIEE_LTR_LEXICAL_PREFETCH_BATCH_SIZE:-64}" \
  --lexical-batch-max-threads "${COLIEE_LTR_LEXICAL_BATCH_MAX_THREADS:-${LEXICAL_THREADS_DEFAULT}}" \
  --lexical-batch-max-queries "${COLIEE_LTR_LEXICAL_BATCH_MAX_QUERIES:-16}" \
  --lexical-batch-max-total-hits "${COLIEE_LTR_LEXICAL_BATCH_MAX_TOTAL_HITS:-120000}" \
  --lexical-batch-max-k "${COLIEE_LTR_LEXICAL_BATCH_MAX_K:-8000}" \
  --dense-batch-size "${COLIEE_LTR_DENSE_BATCH_SIZE:-16}" \
  --chunk-warmup-case-batch-size "${COLIEE_LTR_CHUNK_WARMUP_CASE_BATCH_SIZE:-128}" \
  --feature-score-batch-size "${COLIEE_LTR_FEATURE_SCORE_BATCH_SIZE:-4096}" \
  --lgbm-device "${COLIEE_LTR_LGBM_DEVICE:-cpu}" \
  "$@"
