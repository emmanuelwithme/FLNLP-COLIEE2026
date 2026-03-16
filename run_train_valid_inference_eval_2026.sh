#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

export COLIEE_TASK1_YEAR=2026
export COLIEE_TASK1_ROOT="${COLIEE_TASK1_ROOT:-./coliee_dataset/task1}"
export COLIEE_TASK1_DIR="${COLIEE_TASK1_DIR:-${COLIEE_TASK1_ROOT}/${COLIEE_TASK1_YEAR}}"

TASK1_DIR="${COLIEE_TASK1_DIR}"
MODEL_NAME="modernBert_fp_fp16"

# Optional toggles
FORCE_REENCODE="${FORCE_REENCODE:-0}"   # 1: force re-run inference.py
RUN_FULL_EVAL="${RUN_FULL_EVAL:-0}"     # 1: additionally run utils/eval.py
SKIP_BM25="${SKIP_BM25:-0}"             # 1: skip BM25 valid retrieval

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

require_dir "${TASK1_DIR}/processed"
require_file "${TASK1_DIR}/task1_train_labels_${COLIEE_TASK1_YEAR}.json"
require_nonempty_file "${TASK1_DIR}/train_qid.tsv"
require_nonempty_file "${TASK1_DIR}/valid_qid.tsv"
require_nonempty_file "${TASK1_DIR}/lht_process/modernBert/query_candidate_scope.json"
require_dir "${REPO_ROOT}/modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_${COLIEE_TASK1_YEAR}"
require_dir "${REPO_ROOT}/modernbert-caselaw-accsteps-fp/checkpoint-29000"

mkdir -p "${TASK1_DIR}/lht_process/${MODEL_NAME}"
mkdir -p "${TASK1_DIR}/lht_process/BM25"

BM25_SCOPE_PATH="${TASK1_DIR}/lht_process/modernBert/query_candidate_scope.json"
BM25_VALID_RAW="${TASK1_DIR}/lht_process/BM25/output_bm25_valid_raw.tsv"
BM25_TRAIN_RAW="${TASK1_DIR}/lht_process/BM25/output_bm25_train_raw.tsv"
BM25_VALID_SCOPED="${TASK1_DIR}/lht_process/BM25/output_bm25_valid.tsv"
BM25_TRAIN_SCOPED="${TASK1_DIR}/lht_process/BM25/output_bm25_train.tsv"
BM25_HITS="${BM25_HITS:-$(find "${TASK1_DIR}/processed" -maxdepth 1 -type f -name '*.txt' | wc -l)}"

if [[ "${SKIP_BM25}" == "1" ]]; then
  echo "[INFO] SKIP_BM25=1, skip BM25 valid retrieval."
else
  require_dir "${TASK1_DIR}/lht_process/BM25/index"
  require_nonempty_file "${TASK1_DIR}/lht_process/BM25/query_valid.tsv"
  require_nonempty_file "${TASK1_DIR}/lht_process/BM25/query_train.tsv"
  run_step "Run BM25 on valid split" \
    python -m pyserini.search.lucene \
      --index "${TASK1_DIR}/lht_process/BM25/index" \
      --topics "${TASK1_DIR}/lht_process/BM25/query_valid.tsv" \
      --output "${BM25_VALID_RAW}" \
      --bm25 \
      --k1 3 \
      --b 1 \
      --hits "${BM25_HITS}" \
      --threads 10 \
      --batch-size 16
  run_step "Run BM25 on train split" \
    python -m pyserini.search.lucene \
      --index "${TASK1_DIR}/lht_process/BM25/index" \
      --topics "${TASK1_DIR}/lht_process/BM25/query_train.tsv" \
      --output "${BM25_TRAIN_RAW}" \
      --bm25 \
      --k1 3 \
      --b 1 \
      --hits "${BM25_HITS}" \
      --threads 10 \
      --batch-size 16
  require_nonempty_file "${BM25_VALID_RAW}"
  require_nonempty_file "${BM25_TRAIN_RAW}"

  run_step "Apply processed scope filter to BM25 valid/train" \
    python "Legal Case Retrieval/utils/filter_trec_by_scope.py" \
      --input-path "${BM25_VALID_RAW}" \
      --output-path "${BM25_VALID_SCOPED}" \
      --scope-path "${BM25_SCOPE_PATH}" \
      --qid-path "${TASK1_DIR}/valid_qid.tsv" \
      --skip-self \
      --strict-scope
  run_step "Apply processed scope filter to BM25 train" \
    python "Legal Case Retrieval/utils/filter_trec_by_scope.py" \
      --input-path "${BM25_TRAIN_RAW}" \
      --output-path "${BM25_TRAIN_SCOPED}" \
      --scope-path "${BM25_SCOPE_PATH}" \
      --qid-path "${TASK1_DIR}/train_qid.tsv" \
      --skip-self \
      --strict-scope
  require_nonempty_file "${BM25_VALID_SCOPED}"
  require_nonempty_file "${BM25_TRAIN_SCOPED}"
fi

EMB_PATH="${TASK1_DIR}/processed/processed_document_${MODEL_NAME}_embeddings.pkl"
if [[ "${FORCE_REENCODE}" == "1" || ! -s "${EMB_PATH}" ]]; then
  run_step "Encode processed corpus with modernBERT contrastive model" \
    env LCR_TEST_MODE=0 python "Legal Case Retrieval/modernBert-fp/inference.py"
else
  echo "[INFO] Reuse existing embeddings: ${EMB_PATH}"
fi
require_nonempty_file "${EMB_PATH}"

run_step "Rank train/valid with scope-filtered encoder similarity" \
  env LCR_TEST_MODE=0 python "Legal Case Retrieval/modernBert-fp/similarity_and_rank.py"

require_nonempty_file "${TASK1_DIR}/lht_process/${MODEL_NAME}/output_${MODEL_NAME}_dot_valid.tsv"
require_nonempty_file "${TASK1_DIR}/lht_process/${MODEL_NAME}/output_${MODEL_NAME}_cos_valid.tsv"
require_nonempty_file "${TASK1_DIR}/lht_process/${MODEL_NAME}/output_${MODEL_NAME}_dot_train.tsv"
require_nonempty_file "${TASK1_DIR}/lht_process/${MODEL_NAME}/output_${MODEL_NAME}_cos_train.tsv"

run_step "Quick sanity check: prediction coverage by qid" \
  python - <<'PY'
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

task1 = Path("./coliee_dataset/task1/2026")
expected_valid = len(load_qids(task1 / "valid_qid.tsv"))
expected_train = len(load_qids(task1 / "train_qid.tsv"))

files = {
    "dot_valid": task1 / "lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_valid.tsv",
    "cos_valid": task1 / "lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_valid.tsv",
    "dot_train": task1 / "lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_train.tsv",
    "cos_train": task1 / "lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_train.tsv",
}

print(f"Expected valid qids: {expected_valid}")
print(f"Expected train qids: {expected_train}")
for name, path in files.items():
    got = count_predicted_qids(path)
    expected = expected_valid if "valid" in name else expected_train
    print(f"{name:>9}: predicted_qids={got}, expected={expected}")
PY

if [[ "${RUN_FULL_EVAL}" == "1" ]]; then
  run_step "Run full eval script (all configured models)" \
    python "Legal Case Retrieval/utils/eval.py"
fi

run_step "Compute focused metrics (BM25 valid + modernBert_fp_fp16 train/valid)" \
  python - <<'PY'
import sys
from pathlib import Path

repo = Path(".").resolve()
sys.path.insert(0, str(repo / "Legal Case Retrieval"))

from lcr.metrics import classification_report, rel_file_to_dict, trec_file_to_dict
from lcr.task1_paths import get_task1_dir, get_task1_year

task1_dir = Path(get_task1_dir())
year = get_task1_year()
labels_path = task1_dir / f"task1_train_labels_{year}.json"
valid_qid_path = task1_dir / "valid_qid.tsv"
train_qid_path = task1_dir / "train_qid.tsv"

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

evaluate(
    "BM25",
    "valid",
    task1_dir / "lht_process/BM25/output_bm25_valid.tsv",
    valid_rel,
)
evaluate(
    "BM25",
    "train",
    task1_dir / "lht_process/BM25/output_bm25_train.tsv",
    train_rel,
)
evaluate(
    "modernBert_fp_fp16_dot",
    "valid",
    task1_dir / "lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_valid.tsv",
    valid_rel,
)
evaluate(
    "modernBert_fp_fp16_cos",
    "valid",
    task1_dir / "lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_valid.tsv",
    valid_rel,
)
evaluate(
    "modernBert_fp_fp16_dot",
    "train",
    task1_dir / "lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_train.tsv",
    train_rel,
)
evaluate(
    "modernBert_fp_fp16_cos",
    "train",
    task1_dir / "lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_train.tsv",
    train_rel,
)
PY

echo
echo "[DONE] Train/valid inference + evaluation pipeline complete."
