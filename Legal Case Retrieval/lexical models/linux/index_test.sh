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

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "${TASK1_DIR}/lht_process/BM25_test/corpus" \
  --index "${TASK1_DIR}/lht_process/BM25_test/index" \
  --generator DefaultLuceneDocumentGenerator \
  --threads 16 \
  --storePositions --storeDocvectors --storeRaw
