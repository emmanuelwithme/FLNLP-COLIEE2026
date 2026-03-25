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
  TASK1_PROCESSED_DIR \
  TASK1_QUERY_DIR \
  TASK1_TRAIN_LABELS_PATH \
  TASK1_TRAIN_QID_PATH \
  TASK1_VALID_QID_PATH \
  TASK1_SCOPE_PATH \
  TASK1_BM25_DIR \
  TASK1_MODEL_ROOT_DIR \
  TASK1_BASE_ENCODER_DIR \
  TASK1_MODEL_RESULTS_DIR \
  TASK1_RETRIEVAL_MODEL_NAME \
  TASK1_FORCE_REENCODE \
  TASK1_RUN_FULL_EVAL \
  TASK1_SKIP_BM25 \
  TASK1_QUICK_TEST \
  TASK1_BM25_K1 \
  TASK1_BM25_B \
  TASK1_BM25_THREADS \
  TASK1_BM25_BATCH_SIZE

resolve_env_path_var "${REPO_ROOT}" COLIEE_TASK1_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_PROCESSED_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_QUERY_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_TRAIN_LABELS_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_TRAIN_QID_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_VALID_QID_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_SCOPE_PATH
resolve_env_path_var "${REPO_ROOT}" TASK1_BM25_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_MODEL_ROOT_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_BASE_ENCODER_DIR
resolve_env_path_var "${REPO_ROOT}" TASK1_MODEL_RESULTS_DIR

TASK1_DIR="${COLIEE_TASK1_DIR}"
MODEL_NAME="${TASK1_RETRIEVAL_MODEL_NAME}"
FORCE_REENCODE="${TASK1_FORCE_REENCODE}"
RUN_FULL_EVAL="${TASK1_RUN_FULL_EVAL}"
SKIP_BM25="${TASK1_SKIP_BM25}"
EMBED_SUFFIX=""
if is_truthy "${TASK1_QUICK_TEST}"; then
  EMBED_SUFFIX="_test"
fi

BM25_INDEX_DIR="${TASK1_BM25_DIR}/index"
BM25_QUERY_VALID_PATH="${TASK1_BM25_DIR}/query_valid.tsv"
BM25_QUERY_TRAIN_PATH="${TASK1_BM25_DIR}/query_train.tsv"
BM25_VALID_RAW="${TASK1_BM25_DIR}/output_bm25_valid_raw.tsv"
BM25_TRAIN_RAW="${TASK1_BM25_DIR}/output_bm25_train_raw.tsv"
BM25_VALID_SCOPED="${TASK1_BM25_DIR}/output_bm25_valid.tsv"
BM25_TRAIN_SCOPED="${TASK1_BM25_DIR}/output_bm25_train.tsv"

CANDIDATE_EMBED_PATH="${TASK1_PROCESSED_DIR}/processed_document_${MODEL_NAME}_embeddings${EMBED_SUFFIX}.pkl"
QUERY_EMBED_PATH="${TASK1_QUERY_DIR}/processed_new_document_${MODEL_NAME}_embeddings${EMBED_SUFFIX}.pkl"
OUTPUT_DOT_VALID_PATH="${TASK1_MODEL_RESULTS_DIR}/output_${MODEL_NAME}_dot_valid.tsv"
OUTPUT_COS_VALID_PATH="${TASK1_MODEL_RESULTS_DIR}/output_${MODEL_NAME}_cos_valid.tsv"
OUTPUT_DOT_TRAIN_PATH="${TASK1_MODEL_RESULTS_DIR}/output_${MODEL_NAME}_dot_train.tsv"
OUTPUT_COS_TRAIN_PATH="${TASK1_MODEL_RESULTS_DIR}/output_${MODEL_NAME}_cos_train.tsv"

BM25_HITS="${TASK1_BM25_HITS:-}"
if [[ -z "${BM25_HITS}" ]]; then
  BM25_HITS="$(find "${TASK1_PROCESSED_DIR}" -maxdepth 1 -type f -name '*.txt' | wc -l)"
fi

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
echo "[INFO] TASK1_DIR=${TASK1_DIR}"
echo "[INFO] TASK1_RETRIEVAL_MODEL_NAME=${MODEL_NAME}"
echo "[INFO] TASK1_MODEL_ROOT_DIR=${TASK1_MODEL_ROOT_DIR}"
echo "[INFO] TASK1_BASE_ENCODER_DIR=${TASK1_BASE_ENCODER_DIR}"

require_dir "${TASK1_PROCESSED_DIR}"
require_dir "${TASK1_QUERY_DIR}"
require_file "${TASK1_TRAIN_LABELS_PATH}"
require_nonempty_file "${TASK1_TRAIN_QID_PATH}"
require_nonempty_file "${TASK1_VALID_QID_PATH}"
require_nonempty_file "${TASK1_SCOPE_PATH}"
require_dir "${TASK1_MODEL_ROOT_DIR}"
require_dir "${TASK1_BASE_ENCODER_DIR}"

mkdir -p "${TASK1_MODEL_RESULTS_DIR}" "${TASK1_BM25_DIR}"

if is_truthy "${SKIP_BM25}"; then
  echo "[INFO] TASK1_SKIP_BM25=1, skip BM25 valid/train retrieval."
else
  require_dir "${BM25_INDEX_DIR}"
  require_nonempty_file "${BM25_QUERY_VALID_PATH}"
  require_nonempty_file "${BM25_QUERY_TRAIN_PATH}"

  run_step "Run BM25 on valid split" \
    python -m pyserini.search.lucene \
      --index "${BM25_INDEX_DIR}" \
      --topics "${BM25_QUERY_VALID_PATH}" \
      --output "${BM25_VALID_RAW}" \
      --bm25 \
      --k1 "${TASK1_BM25_K1}" \
      --b "${TASK1_BM25_B}" \
      --hits "${BM25_HITS}" \
      --threads "${TASK1_BM25_THREADS}" \
      --batch-size "${TASK1_BM25_BATCH_SIZE}"

  run_step "Run BM25 on train split" \
    python -m pyserini.search.lucene \
      --index "${BM25_INDEX_DIR}" \
      --topics "${BM25_QUERY_TRAIN_PATH}" \
      --output "${BM25_TRAIN_RAW}" \
      --bm25 \
      --k1 "${TASK1_BM25_K1}" \
      --b "${TASK1_BM25_B}" \
      --hits "${BM25_HITS}" \
      --threads "${TASK1_BM25_THREADS}" \
      --batch-size "${TASK1_BM25_BATCH_SIZE}"

  require_nonempty_file "${BM25_VALID_RAW}"
  require_nonempty_file "${BM25_TRAIN_RAW}"

  run_step "Apply processed scope filter to BM25 valid" \
    python "Legal Case Retrieval/utils/filter_trec_by_scope.py" \
      --input-path "${BM25_VALID_RAW}" \
      --output-path "${BM25_VALID_SCOPED}" \
      --scope-path "${TASK1_SCOPE_PATH}" \
      --qid-path "${TASK1_VALID_QID_PATH}" \
      --skip-self \
      --strict-scope

  run_step "Apply processed scope filter to BM25 train" \
    python "Legal Case Retrieval/utils/filter_trec_by_scope.py" \
      --input-path "${BM25_TRAIN_RAW}" \
      --output-path "${BM25_TRAIN_SCOPED}" \
      --scope-path "${TASK1_SCOPE_PATH}" \
      --qid-path "${TASK1_TRAIN_QID_PATH}" \
      --skip-self \
      --strict-scope

  require_nonempty_file "${BM25_VALID_SCOPED}"
  require_nonempty_file "${BM25_TRAIN_SCOPED}"
fi

if is_truthy "${FORCE_REENCODE}" || [[ ! -s "${CANDIDATE_EMBED_PATH}" ]]; then
  run_step "Encode processed/query corpora with ${MODEL_NAME}" \
    env \
      LCR_TEST_MODE=0 \
      TASK1_CANDIDATE_DIR="${TASK1_PROCESSED_DIR}" \
      TASK1_QUERY_DIR="${TASK1_QUERY_DIR}" \
      TASK1_CANDIDATE_EMBEDDINGS_OUTPUT="${CANDIDATE_EMBED_PATH}" \
      TASK1_QUERY_EMBEDDINGS_OUTPUT="${QUERY_EMBED_PATH}" \
      TASK1_MODEL_ROOT_DIR="${TASK1_MODEL_ROOT_DIR}" \
      TASK1_BASE_ENCODER_DIR="${TASK1_BASE_ENCODER_DIR}" \
      python "Legal Case Retrieval/modernBert-fp/inference.py"
else
  echo "[INFO] Reuse existing embeddings: ${CANDIDATE_EMBED_PATH}"
fi

require_nonempty_file "${CANDIDATE_EMBED_PATH}"
require_nonempty_file "${QUERY_EMBED_PATH}"

run_step "Rank train/valid with dense similarity" \
  env \
    LCR_TEST_MODE=0 \
    TASK1_CANDIDATE_EMBEDDINGS_OUTPUT="${CANDIDATE_EMBED_PATH}" \
    TASK1_QUERY_EMBEDDINGS_OUTPUT="${QUERY_EMBED_PATH}" \
    TASK1_TRAIN_QID_PATH="${TASK1_TRAIN_QID_PATH}" \
    TASK1_VALID_QID_PATH="${TASK1_VALID_QID_PATH}" \
    TASK1_SCOPE_PATH="${TASK1_SCOPE_PATH}" \
    TASK1_OUTPUT_DOT_VALID_PATH="${OUTPUT_DOT_VALID_PATH}" \
    TASK1_OUTPUT_COS_VALID_PATH="${OUTPUT_COS_VALID_PATH}" \
    TASK1_OUTPUT_DOT_TRAIN_PATH="${OUTPUT_DOT_TRAIN_PATH}" \
    TASK1_OUTPUT_COS_TRAIN_PATH="${OUTPUT_COS_TRAIN_PATH}" \
    python "Legal Case Retrieval/modernBert-fp/similarity_and_rank.py"

require_nonempty_file "${OUTPUT_DOT_VALID_PATH}"
require_nonempty_file "${OUTPUT_COS_VALID_PATH}"
require_nonempty_file "${OUTPUT_DOT_TRAIN_PATH}"
require_nonempty_file "${OUTPUT_COS_TRAIN_PATH}"

run_step "Quick sanity check: prediction coverage by qid" \
  env \
    TASK1_SANITY_VALID_QID_PATH="${TASK1_VALID_QID_PATH}" \
    TASK1_SANITY_TRAIN_QID_PATH="${TASK1_TRAIN_QID_PATH}" \
    TASK1_OUTPUT_DOT_VALID_PATH="${OUTPUT_DOT_VALID_PATH}" \
    TASK1_OUTPUT_COS_VALID_PATH="${OUTPUT_COS_VALID_PATH}" \
    TASK1_OUTPUT_DOT_TRAIN_PATH="${OUTPUT_DOT_TRAIN_PATH}" \
    TASK1_OUTPUT_COS_TRAIN_PATH="${OUTPUT_COS_TRAIN_PATH}" \
    python - <<'PY'
import os
from pathlib import Path


def load_qids(path: Path) -> list[str]:
    return [line.split()[0].strip().replace(".txt", "") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def count_predicted_qids(path: Path) -> int:
    qids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        qids.add(parts[0].replace(".txt", ""))
    return len(qids)


expected_valid = len(load_qids(Path(os.environ["TASK1_SANITY_VALID_QID_PATH"])))
expected_train = len(load_qids(Path(os.environ["TASK1_SANITY_TRAIN_QID_PATH"])))

files = {
    "dot_valid": Path(os.environ["TASK1_OUTPUT_DOT_VALID_PATH"]),
    "cos_valid": Path(os.environ["TASK1_OUTPUT_COS_VALID_PATH"]),
    "dot_train": Path(os.environ["TASK1_OUTPUT_DOT_TRAIN_PATH"]),
    "cos_train": Path(os.environ["TASK1_OUTPUT_COS_TRAIN_PATH"]),
}

print(f"Expected valid qids: {expected_valid}")
print(f"Expected train qids: {expected_train}")
for name, path in files.items():
    got = count_predicted_qids(path)
    expected = expected_valid if "valid" in name else expected_train
    print(f"{name:>9}: predicted_qids={got}, expected={expected}")
PY

if is_truthy "${RUN_FULL_EVAL}"; then
  run_step "Run full eval script (legacy all-model summary)" \
    python "Legal Case Retrieval/utils/eval.py"
fi

run_step "Compute focused metrics (BM25 + ${MODEL_NAME} train/valid)" \
  env \
    TASK1_METRICS_MODEL_NAME="${MODEL_NAME}" \
    TASK1_LABELS_PATH="${TASK1_TRAIN_LABELS_PATH}" \
    TASK1_TRAIN_QID_PATH="${TASK1_TRAIN_QID_PATH}" \
    TASK1_VALID_QID_PATH="${TASK1_VALID_QID_PATH}" \
    TASK1_BM25_VALID_PATH="${BM25_VALID_SCOPED}" \
    TASK1_BM25_TRAIN_PATH="${BM25_TRAIN_SCOPED}" \
    TASK1_OUTPUT_DOT_VALID_PATH="${OUTPUT_DOT_VALID_PATH}" \
    TASK1_OUTPUT_COS_VALID_PATH="${OUTPUT_COS_VALID_PATH}" \
    TASK1_OUTPUT_DOT_TRAIN_PATH="${OUTPUT_DOT_TRAIN_PATH}" \
    TASK1_OUTPUT_COS_TRAIN_PATH="${OUTPUT_COS_TRAIN_PATH}" \
    python - <<'PY'
import os
import sys
from pathlib import Path

repo = Path(".").resolve()
sys.path.insert(0, str(repo / "Legal Case Retrieval"))

from lcr.metrics import classification_report, rel_file_to_dict, trec_file_to_dict

model_name = os.environ["TASK1_METRICS_MODEL_NAME"]
labels_path = Path(os.environ["TASK1_LABELS_PATH"])
valid_qid_path = Path(os.environ["TASK1_VALID_QID_PATH"])
train_qid_path = Path(os.environ["TASK1_TRAIN_QID_PATH"])


def evaluate(name: str, split: str, trec_path: Path, rel_dict):
    if not trec_path.exists() or trec_path.stat().st_size == 0:
        print(f"{name:<24} {split:<6} missing")
        return
    preds = trec_file_to_dict(trec_path, topk=5)
    labels, answers = [], []
    missing = 0
    for qid in rel_dict.keys():
        ans = preds.get(qid, [])
        if not ans:
            missing += 1
        labels.append(rel_dict[qid])
        answers.append(ans)
    f1, p, r = classification_report(labels, answers)
    print(f"{name:<24} {split:<6} P={p:.6f} R={r:.6f} F1={f1:.6f} missing_qid={missing}")


valid_rel = rel_file_to_dict(labels_path, valid_qid_path)
train_rel = rel_file_to_dict(labels_path, train_qid_path)

evaluate("BM25", "valid", Path(os.environ["TASK1_BM25_VALID_PATH"]), valid_rel)
evaluate("BM25", "train", Path(os.environ["TASK1_BM25_TRAIN_PATH"]), train_rel)
evaluate(f"{model_name}_dot", "valid", Path(os.environ["TASK1_OUTPUT_DOT_VALID_PATH"]), valid_rel)
evaluate(f"{model_name}_cos", "valid", Path(os.environ["TASK1_OUTPUT_COS_VALID_PATH"]), valid_rel)
evaluate(f"{model_name}_dot", "train", Path(os.environ["TASK1_OUTPUT_DOT_TRAIN_PATH"]), train_rel)
evaluate(f"{model_name}_cos", "train", Path(os.environ["TASK1_OUTPUT_COS_TRAIN_PATH"]), train_rel)
PY

echo
echo "[DONE] Train/valid inference + evaluation pipeline complete."
