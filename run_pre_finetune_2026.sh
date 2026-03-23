#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export COLIEE_TASK1_YEAR=2026
export COLIEE_TASK1_ROOT="${COLIEE_TASK1_ROOT:-./coliee_dataset/task1}"
export COLIEE_TASK1_DIR="${COLIEE_TASK1_DIR:-${COLIEE_TASK1_ROOT}/${COLIEE_TASK1_YEAR}}"

TASK1_DIR="${COLIEE_TASK1_DIR}"

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

require_nonempty_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "[ERROR] Required non-empty file not found (or empty): ${path}" >&2
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

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] COLIEE_TASK1_YEAR=${COLIEE_TASK1_YEAR}"
echo "[INFO] TASK1_DIR=${TASK1_DIR}"

require_dir "${TASK1_DIR}/task1_train_files_${COLIEE_TASK1_YEAR}"
require_file "${TASK1_DIR}/task1_train_labels_${COLIEE_TASK1_YEAR}.json"

mkdir -p "${TASK1_DIR}/lht_process/BM25"
mkdir -p "${TASK1_DIR}/lht_process/modernBert/finetune_data"

run_step "Extract summary from raw files" \
  python "Legal Case Retrieval/pre-process/summary.py"
require_dir "${TASK1_DIR}/summary"

run_step "Build processed corpus" \
  python "Legal Case Retrieval/pre-process/process.py"
require_dir "${TASK1_DIR}/processed"

run_step "Split labels into train/valid and generate valid_qid.tsv" \
  python "Legal Case Retrieval/pre-process/split_dataset.py"
require_file "${TASK1_DIR}/task1_train_labels_${COLIEE_TASK1_YEAR}_train.json"
require_file "${TASK1_DIR}/task1_train_labels_${COLIEE_TASK1_YEAR}_valid.json"
require_file "${TASK1_DIR}/valid_qid.tsv"

run_step "Generate BM25 query files and train_qid.tsv" \
  python "Legal Case Retrieval/lexical models/form_query.py"
require_file "${TASK1_DIR}/train_qid.tsv"
require_nonempty_file "${TASK1_DIR}/lht_process/BM25/query_train.tsv"
require_nonempty_file "${TASK1_DIR}/lht_process/BM25/query_valid.tsv"

run_step "Generate BM25 corpus jsonl" \
  python "Legal Case Retrieval/lexical models/form_corpus.py"
require_file "${TASK1_DIR}/lht_process/BM25/corpus/corpus.json"

run_step "Build BM25 index (Pyserini)" \
  bash "Legal Case Retrieval/lexical models/linux/index.sh"
require_dir "${TASK1_DIR}/lht_process/BM25/index"

run_step "Run BM25 search for train/valid queries" \
  bash "Legal Case Retrieval/lexical models/linux/query_search.sh"
require_nonempty_file "${TASK1_DIR}/lht_process/BM25/output_bm25_train.tsv"
require_nonempty_file "${TASK1_DIR}/lht_process/BM25/output_bm25_valid.tsv"

run_step "Create contrastive BM25 hard-negative JSON (top100 random15)" \
  python "Legal Case Retrieval/modernBert/fine_tune/create_bm25_hard_negative_data_top100_random15.py"
require_file "${TASK1_DIR}/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_train.json"
require_file "${TASK1_DIR}/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json"

run_step "Build query candidate scope JSON for year filter" \
  python "Legal Case Retrieval/pre-process/build_query_candidate_scope.py"
require_file "${TASK1_DIR}/lht_process/modernBert/query_candidate_scope.json"

echo
echo "[DONE] Fine-tune 前置流程完成。"
echo "[NEXT] python \"Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py\""
