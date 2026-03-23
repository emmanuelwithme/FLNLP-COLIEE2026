#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${REPO_ROOT}"

echo "[chunkAgg] repo root: ${REPO_ROOT}"
echo "[chunkAgg] script dir: ${SCRIPT_DIR}"
echo "[chunkAgg] python: ${PYTHON_BIN}"
echo "[chunkAgg] computing ranking outputs..."

exec "${PYTHON_BIN}" "Legal Case Retrieval/modernBert-fp-chunkAgg/similarity_and_rank.py"
