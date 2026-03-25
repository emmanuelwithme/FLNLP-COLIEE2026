#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

# shellcheck source=scripts/common_env.sh
source "${REPO_ROOT}/scripts/common_env.sh"
load_env_file_if_present "${REPO_ROOT}/.env"

require_envs \
  COLIEE_TASK1_YEAR \
  COLIEE_TASK1_DIR \
  TASK1_TRAIN_RAW_DIR \
  TASK1_TRAIN_LABELS_PATH \
  TASK1_SUMMARY_DIR \
  TASK1_PROCESSED_DIR \
  TASK1_TRAIN_SPLIT_LABELS_PATH \
  TASK1_VALID_SPLIT_LABELS_PATH \
  TASK1_TRAIN_QID_PATH \
  TASK1_VALID_QID_PATH \
  TASK1_BM25_DIR \
  TASK1_FINETUNE_DATA_DIR \
  TASK1_SCOPE_PATH \
  TASK1_SPLIT_TRAIN_RATIO \
  TASK1_SPLIT_SEED \
  TASK1_SCOPE_YEAR_SLACK \
  TASK1_SCOPE_UNKNOWN_QUERY_YEAR_POLICY \
  TASK1_SCOPE_EXCLUDE_SELF \
  TASK1_SUMMARY_NUM_WORKERS \
  TASK1_PROCESS_NUM_WORKERS \
  TASK1_HARD_NEG_TOPK \
  TASK1_HARD_NEG_MAX_NEGATIVES \
  TASK1_HARD_NEG_SEED

resolve_env_path_var "${REPO_ROOT}" COLIEE_TASK1_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_TRAIN_RAW_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_TRAIN_LABELS_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_SUMMARY_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_PROCESSED_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_TRAIN_SPLIT_LABELS_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_VALID_SPLIT_LABELS_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_TRAIN_QID_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_VALID_QID_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_BM25_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_FINETUNE_DATA_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_SCOPE_PATH

TASK1_BM25_CORPUS_DIR="${TASK1_BM25_DIR}/corpus"
TASK1_BM25_CORPUS_PATH="${TASK1_BM25_CORPUS_DIR}/corpus.json"
TASK1_BM25_INDEX_DIR="${TASK1_BM25_DIR}/index"
TASK1_BM25_QUERY_TRAIN_PATH="${TASK1_BM25_DIR}/query_train.tsv"
TASK1_BM25_QUERY_VALID_PATH="${TASK1_BM25_DIR}/query_valid.tsv"
TASK1_BM25_OUTPUT_TRAIN_PATH="${TASK1_BM25_DIR}/output_bm25_train.tsv"
TASK1_BM25_OUTPUT_VALID_PATH="${TASK1_BM25_DIR}/output_bm25_valid.tsv"

run_step() {
  local title="$1"
  shift
  echo
  echo "============================================================"
  echo "[STEP] ${title}"
  echo "============================================================"
  "$@"
}

echo "[INFO] Repo root: ${REPO_ROOT}"
echo "[INFO] COLIEE_TASK1_YEAR=${COLIEE_TASK1_YEAR}"
echo "[INFO] TASK1_DIR=${COLIEE_TASK1_DIR}"

require_dir "${TASK1_TRAIN_RAW_DIR}"
require_file "${TASK1_TRAIN_LABELS_PATH}"

mkdir -p "${TASK1_BM25_DIR}" "${TASK1_FINETUNE_DATA_DIR}"

run_step "Extract summary from raw files" \
  python "Legal Case Retrieval/pre-process/summary.py" \
    --input-dir "${TASK1_TRAIN_RAW_DIR}" \
    --output-dir "${TASK1_SUMMARY_DIR}" \
    --num-workers "${TASK1_SUMMARY_NUM_WORKERS}"
require_dir "${TASK1_SUMMARY_DIR}"

run_step "Build processed corpus" \
  python "Legal Case Retrieval/pre-process/process.py" \
    --input-dir "${TASK1_TRAIN_RAW_DIR}" \
    --summary-dir "${TASK1_SUMMARY_DIR}" \
    --output-dir "${TASK1_PROCESSED_DIR}" \
    --num-workers "${TASK1_PROCESS_NUM_WORKERS}"
require_dir "${TASK1_PROCESSED_DIR}"

run_step "Split labels into train/valid and generate valid_qid.tsv" \
  python "Legal Case Retrieval/pre-process/split_dataset.py" \
    --input-file "${TASK1_TRAIN_LABELS_PATH}" \
    --train-ratio "${TASK1_SPLIT_TRAIN_RATIO}" \
    --seed "${TASK1_SPLIT_SEED}" \
    --output-dir "${COLIEE_TASK1_DIR}"
require_file "${TASK1_TRAIN_SPLIT_LABELS_PATH}"
require_file "${TASK1_VALID_SPLIT_LABELS_PATH}"
require_file "${TASK1_VALID_QID_PATH}"

run_step "Generate BM25 query files and train_qid.tsv" \
  python "Legal Case Retrieval/lexical models/form_query.py" \
    --raw-dir "${TASK1_PROCESSED_DIR}" \
    --output-dir "${TASK1_BM25_DIR}" \
    --labels-path "${TASK1_TRAIN_LABELS_PATH}" \
    --valid-qid-path "${TASK1_VALID_QID_PATH}" \
    --train-qid-path "${TASK1_TRAIN_QID_PATH}" \
    --truncate-threshold "${TASK1_LEXICAL_QUERY_TRUNCATE_THRESHOLD}" \
    --truncate-length "${TASK1_LEXICAL_QUERY_TRUNCATE_LENGTH}"
require_file "${TASK1_TRAIN_QID_PATH}"
require_nonempty_file "${TASK1_BM25_QUERY_TRAIN_PATH}"
require_nonempty_file "${TASK1_BM25_QUERY_VALID_PATH}"

run_step "Generate BM25 corpus jsonl" \
  python "Legal Case Retrieval/lexical models/form_corpus.py" \
    --raw-dir "${TASK1_PROCESSED_DIR}" \
    --output-dir "${TASK1_BM25_DIR}"
require_file "${TASK1_BM25_CORPUS_PATH}"

run_step "Build BM25 index (Pyserini)" \
  bash "Legal Case Retrieval/lexical models/linux/index.sh"
require_dir "${TASK1_BM25_INDEX_DIR}"

run_step "Run BM25 search for train/valid queries" \
  bash "Legal Case Retrieval/lexical models/linux/query_search.sh"
require_nonempty_file "${TASK1_BM25_OUTPUT_TRAIN_PATH}"
require_nonempty_file "${TASK1_BM25_OUTPUT_VALID_PATH}"

run_step "Create contrastive BM25 hard-negative JSON (top100 random15)" \
  python "Legal Case Retrieval/modernBert/fine_tune/create_bm25_hard_negative_data_top100_random15.py" \
    --bm25-train-path "${TASK1_BM25_OUTPUT_TRAIN_PATH}" \
    --bm25-valid-path "${TASK1_BM25_OUTPUT_VALID_PATH}" \
    --train-labels-path "${TASK1_TRAIN_SPLIT_LABELS_PATH}" \
    --valid-labels-path "${TASK1_VALID_SPLIT_LABELS_PATH}" \
    --train-output-path "${TASK1_FINETUNE_DATA_DIR}/contrastive_bm25_hard_negative_top100_random15_train.json" \
    --valid-output-path "${TASK1_FINETUNE_DATA_DIR}/contrastive_bm25_hard_negative_top100_random15_valid.json" \
    --top-k "${TASK1_HARD_NEG_TOPK}" \
    --max-negatives "${TASK1_HARD_NEG_MAX_NEGATIVES}" \
    --random-seed "${TASK1_HARD_NEG_SEED}"
require_file "${TASK1_FINETUNE_DATA_DIR}/contrastive_bm25_hard_negative_top100_random15_train.json"
require_file "${TASK1_FINETUNE_DATA_DIR}/contrastive_bm25_hard_negative_top100_random15_valid.json"

SCOPE_ARGS=(
  --candidate-dir "${TASK1_TRAIN_RAW_DIR}"
  --query-dir "${TASK1_TRAIN_RAW_DIR}"
  --output-path "${TASK1_SCOPE_PATH}"
  --year-slack "${TASK1_SCOPE_YEAR_SLACK}"
  --unknown-query-year-policy "${TASK1_SCOPE_UNKNOWN_QUERY_YEAR_POLICY}"
)
if is_truthy "${TASK1_SCOPE_EXCLUDE_SELF}"; then
  SCOPE_ARGS+=(--exclude-self)
fi

run_step "Build query candidate scope JSON for year filter" \
  python "Legal Case Retrieval/pre-process/build_query_candidate_scope.py" \
    "${SCOPE_ARGS[@]}"
require_file "${TASK1_SCOPE_PATH}"

echo
echo "[DONE] Fine-tune 前置流程完成。"
echo "[NEXT] python \"Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py\""
