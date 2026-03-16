#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

_CALLER_COLIEE_TASK1_YEAR="${COLIEE_TASK1_YEAR:-}"
_CALLER_COLIEE_TASK1_ROOT="${COLIEE_TASK1_ROOT:-}"
_CALLER_COLIEE_TASK1_DIR="${COLIEE_TASK1_DIR:-}"

if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  . "${REPO_ROOT}/.env"
  set +a
fi

if [ -n "${_CALLER_COLIEE_TASK1_YEAR}" ]; then
  COLIEE_TASK1_YEAR="${_CALLER_COLIEE_TASK1_YEAR}"
fi
if [ -n "${_CALLER_COLIEE_TASK1_ROOT}" ]; then
  COLIEE_TASK1_ROOT="${_CALLER_COLIEE_TASK1_ROOT}"
fi
if [ -n "${_CALLER_COLIEE_TASK1_DIR}" ]; then
  COLIEE_TASK1_DIR="${_CALLER_COLIEE_TASK1_DIR}"
fi

COLIEE_TASK1_YEAR="${COLIEE_TASK1_YEAR:-2025}"
COLIEE_TASK1_ROOT="${COLIEE_TASK1_ROOT:-./coliee_dataset/task1}"
TASK1_DIR="${COLIEE_TASK1_DIR:-${COLIEE_TASK1_ROOT}/${COLIEE_TASK1_YEAR}}"

HITS="${BM25_TEST_HITS:-2000}"
OUTPUT_PATH="${BM25_TEST_OUTPUT_PATH:-${TASK1_DIR}/lht_process/BM25_test/output_bm25_test.tsv}"

python -m pyserini.search.lucene \
  --index "${TASK1_DIR}/lht_process/BM25_test/index" \
  --topics "${TASK1_DIR}/lht_process/BM25_test/query_test.tsv" \
  --output "${OUTPUT_PATH}" \
  --bm25 \
  --k1 3 \
  --b 1 \
  --hits "${HITS}" \
  --threads 10 \
  --batch-size 16

echo "BM25 test retrieval done: ${OUTPUT_PATH}"
