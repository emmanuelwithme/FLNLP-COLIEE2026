#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# shellcheck source=scripts/common_env.sh
source "${REPO_ROOT}/scripts/common_env.sh"
load_env_file_if_present "${REPO_ROOT}/.env"

require_envs \
  TASK1_BM25_DIR \
  TASK1_PROCESSED_DIR \
  TASK1_BM25_K1 \
  TASK1_BM25_B \
  TASK1_BM25_THREADS \
  TASK1_BM25_BATCH_SIZE

resolve_env_path_var "${REPO_ROOT}" TASK1_BM25_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_PROCESSED_DIR

BM25_HITS="${TASK1_BM25_HITS:-}"
if [[ -z "${BM25_HITS}" ]]; then
  BM25_HITS="$(find "${TASK1_PROCESSED_DIR}" -maxdepth 1 -type f -name '*.txt' | wc -l)"
fi

# 處理驗證集查詢
echo "運行驗證集BM25檢索..."
python -m pyserini.search.lucene \
  --index "${TASK1_BM25_DIR}/index" \
  --topics "${TASK1_BM25_DIR}/query_valid.tsv" \
  --output "${TASK1_BM25_DIR}/output_bm25_valid.tsv" \
  --bm25 \
  --k1 "${TASK1_BM25_K1}" \
  --b "${TASK1_BM25_B}" \
  --hits "${BM25_HITS}" \
  --threads "${TASK1_BM25_THREADS}" \
  --batch-size "${TASK1_BM25_BATCH_SIZE}"

# 處理訓練集查詢
echo "運行訓練集BM25檢索..."
python -m pyserini.search.lucene \
  --index "${TASK1_BM25_DIR}/index" \
  --topics "${TASK1_BM25_DIR}/query_train.tsv" \
  --output "${TASK1_BM25_DIR}/output_bm25_train.tsv" \
  --bm25 \
  --k1 "${TASK1_BM25_K1}" \
  --b "${TASK1_BM25_B}" \
  --hits "${BM25_HITS}" \
  --threads "${TASK1_BM25_THREADS}" \
  --batch-size "${TASK1_BM25_BATCH_SIZE}"

echo "BM25檢索完成！"
echo "驗證集結果保存在: ${TASK1_BM25_DIR}/output_bm25_valid.tsv"
echo "訓練集結果保存在: ${TASK1_BM25_DIR}/output_bm25_train.tsv"
