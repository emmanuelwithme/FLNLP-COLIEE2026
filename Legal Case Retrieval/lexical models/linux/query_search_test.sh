#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# shellcheck source=scripts/common_env.sh
source "${REPO_ROOT}/scripts/common_env.sh"
load_env_file_if_present "${REPO_ROOT}/.env"

require_envs \
  TASK1_BM25_TEST_DIR \
  TASK1_BM25_K1 \
  TASK1_BM25_B \
  TASK1_BM25_TEST_THREADS \
  TASK1_BM25_TEST_BATCH_SIZE

resolve_env_path_var "${REPO_ROOT}" TASK1_BM25_TEST_DIR

HITS="${BM25_TEST_HITS:-${TASK1_BM25_TEST_HITS}}"
OUTPUT_PATH="${BM25_TEST_OUTPUT_PATH:-${TASK1_BM25_TEST_DIR}/output_bm25_test.tsv}"

python -m pyserini.search.lucene \
  --index "${TASK1_BM25_TEST_DIR}/index" \
  --topics "${TASK1_BM25_TEST_DIR}/query_test.tsv" \
  --output "${OUTPUT_PATH}" \
  --bm25 \
  --k1 "${TASK1_BM25_K1}" \
  --b "${TASK1_BM25_B}" \
  --hits "${HITS}" \
  --threads "${TASK1_BM25_TEST_THREADS}" \
  --batch-size "${TASK1_BM25_TEST_BATCH_SIZE}"

echo "BM25 test retrieval done: ${OUTPUT_PATH}"
