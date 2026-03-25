#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

# shellcheck source=scripts/common_env.sh
source "${REPO_ROOT}/scripts/common_env.sh"
load_env_file_if_present "${REPO_ROOT}/.env"

require_envs TASK1_BM25_DIR TASK1_BM25_INDEX_THREADS
resolve_env_path_var "${REPO_ROOT}" TASK1_BM25_DIR

export JAVA_TOOL_OPTIONS="${COLIEE_JAVA_TOOL_OPTIONS:-${JAVA_TOOL_OPTIONS:-}}"

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input "${TASK1_BM25_DIR}/corpus" \
  --index "${TASK1_BM25_DIR}/index" \
  --generator DefaultLuceneDocumentGenerator \
  --threads "${TASK1_BM25_INDEX_THREADS}" \
  --storePositions --storeDocvectors --storeRaw \
