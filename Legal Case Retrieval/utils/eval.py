from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
for import_root in (PACKAGE_ROOT, REPO_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from repo_config import get_env_int, get_env_path
from lcr.metrics import (
    classification_report as my_classification_report,
    random_guess_baseline,
    random_guess_predictions,
    rel_file_to_dict as rel_file_convert,
    trec_file_to_dict as trec_file_convert,
)
from lcr.results import record_result
from lcr.task1_paths import get_task1_dir, get_task1_model_name, get_task1_year

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()
TASK1_SCOPE_PATH = get_env_path(
    "TASK1_SCOPE_PATH",
    default=f"{TASK1_DIR}/lht_process/modernBert/query_candidate_scope.json",
)
TASK1_BM25_DIR = get_env_path(
    "TASK1_BM25_DIR",
    default=f"{TASK1_DIR}/lht_process/BM25",
)
TASK1_RETRIEVAL_MODEL_NAME = get_task1_model_name()
TASK1_MODEL_RESULTS_DIR = get_env_path(
    "TASK1_MODEL_RESULTS_DIR",
    default=f"{TASK1_DIR}/lht_process/{TASK1_RETRIEVAL_MODEL_NAME}",
)
TASK1_RANDOM_BASELINE_SEED = get_env_int("TASK1_RANDOM_BASELINE_SEED", 42)


if __name__ == '__main__':
  # === 標準答案設定 ===
  rel_path = f"{TASK1_DIR}/task1_train_labels_{TASK1_YEAR}.json"
  valid_path = f"{TASK1_DIR}/valid_qid.tsv"
  train_path = f"{TASK1_DIR}/train_qid.tsv"
  topk = 5 # topk

  # === 讀取正確答案（只取 valid 查詢）===
  valid_rel_dict = rel_file_convert(rel_path, valid_path)
  train_rel_dict = rel_file_convert(rel_path, train_path)
  combine_rel_dict = {"valid": valid_rel_dict, "train": train_rel_dict}

  # === 預測答案設定(需在value相似度分數與排名路徑中有valid或train，如果沒有會跳過) ===
  models = {
    "BM25_valid": f"{TASK1_BM25_DIR}/output_bm25_valid.tsv",
    "SAILER_en_finetune_by_CSHaitao_Dot_valid": f"{TASK1_DIR}/lht_process/SAILER/output_SAILER_dot_valid.tsv",
    "SAILER_en_finetune_by_CSHaitao_Cos_valid": f"{TASK1_DIR}/lht_process/SAILER/output_SAILER_cos_valid.tsv",
    "SAILER_en_finetune_by_CSHaitao_Dot_train": f"{TASK1_DIR}/lht_process/SAILER/output_SAILER_dot_train.tsv",
    "SAILER_en_finetune_by_CSHaitao_Cos_train": f"{TASK1_DIR}/lht_process/SAILER/output_SAILER_cos_train.tsv",
    "moderBert_dot_valid": f"{TASK1_DIR}/lht_process/modernBert/output_modernBert_dot_valid.tsv",
    "moderBert_cos_valid": f"{TASK1_DIR}/lht_process/modernBert/output_modernBert_cos_valid.tsv",
    "moderBert_dot_train": f"{TASK1_DIR}/lht_process/modernBert/output_modernBert_dot_train.tsv",
    "moderBert_cos_train": f"{TASK1_DIR}/lht_process/modernBert/output_modernBert_cos_train.tsv",
    "modernBert_origin_dot_valid": f"{TASK1_DIR}/lht_process/modernBert_origin/output_modernBert_origin_dot_valid.tsv",
    "modernBert_origin_cos_valid": f"{TASK1_DIR}/lht_process/modernBert_origin/output_modernBert_origin_cos_valid.tsv",
    "modernBert_origin_dot_train": f"{TASK1_DIR}/lht_process/modernBert_origin/output_modernBert_origin_dot_train.tsv",
    "modernBert_origin_cos_train": f"{TASK1_DIR}/lht_process/modernBert_origin/output_modernBert_origin_cos_train.tsv",
    "modernBert_fp_dot_valid": f"{TASK1_DIR}/lht_process/modernBert_fp/output_modernBert_fp_dot_valid.tsv",
    "modernBert_fp_cos_valid": f"{TASK1_DIR}/lht_process/modernBert_fp/output_modernBert_fp_cos_valid.tsv",
    "modernBert_fp_dot_train": f"{TASK1_DIR}/lht_process/modernBert_fp/output_modernBert_fp_dot_train.tsv",
    "modernBert_fp_cos_train": f"{TASK1_DIR}/lht_process/modernBert_fp/output_modernBert_fp_cos_train.tsv",
    "modernBert_fp_fp16_dot_valid": f"{TASK1_DIR}/lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_valid.tsv",
    "modernBert_fp_fp16_cos_valid": f"{TASK1_DIR}/lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_valid.tsv",
    "modernBert_fp_fp16_dot_train": f"{TASK1_DIR}/lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_train.tsv",
    "modernBert_fp_fp16_cos_train": f"{TASK1_DIR}/lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_train.tsv",
    "raw_plus0_emptyExclude_dot_valid": f"{TASK1_DIR}/lht_process/modernBert_fp_fp16_scopeFilteredRaw_plus0_emptyExclude/output_modernBert_fp_fp16_dot_valid.tsv",
    "raw_plus0_emptyExclude_cos_valid": f"{TASK1_DIR}/lht_process/modernBert_fp_fp16_scopeFilteredRaw_plus0_emptyExclude/output_modernBert_fp_fp16_cos_valid.tsv",
    "raw_plus0_emptyExclude_dot_train": f"{TASK1_DIR}/lht_process/modernBert_fp_fp16_scopeFilteredRaw_plus0_emptyExclude/output_modernBert_fp_fp16_dot_train.tsv",
    "raw_plus0_emptyExclude_cos_train": f"{TASK1_DIR}/lht_process/modernBert_fp_fp16_scopeFilteredRaw_plus0_emptyExclude/output_modernBert_fp_fp16_cos_train.tsv",
    f"current_{TASK1_RETRIEVAL_MODEL_NAME}_dot_valid": f"{TASK1_MODEL_RESULTS_DIR}/output_{TASK1_RETRIEVAL_MODEL_NAME}_dot_valid.tsv",
    f"current_{TASK1_RETRIEVAL_MODEL_NAME}_cos_valid": f"{TASK1_MODEL_RESULTS_DIR}/output_{TASK1_RETRIEVAL_MODEL_NAME}_cos_valid.tsv",
    f"current_{TASK1_RETRIEVAL_MODEL_NAME}_dot_train": f"{TASK1_MODEL_RESULTS_DIR}/output_{TASK1_RETRIEVAL_MODEL_NAME}_dot_train.tsv",
    f"current_{TASK1_RETRIEVAL_MODEL_NAME}_cos_train": f"{TASK1_MODEL_RESULTS_DIR}/output_{TASK1_RETRIEVAL_MODEL_NAME}_cos_train.tsv",
    "BM25_train": f"{TASK1_BM25_DIR}/output_bm25_train.tsv",
  }

  scope_note = f"scope={TASK1_SCOPE_PATH}" if TASK1_SCOPE_PATH is not None else "scope=<none>"
  print(f"使用 BM25 路徑: {TASK1_BM25_DIR}")
  print(f"使用 current dense result 路徑: {TASK1_MODEL_RESULTS_DIR}")
  print(f"使用 Random Guess scope: {scope_note}")
  print(f"使用 Random Guess seed: {TASK1_RANDOM_BASELINE_SEED}")

  for split, rel_dict in combine_rel_dict.items():
    print(f"現在在比對{split}正確答案...")
    for model_name, trec_path in models.items():
      # 如果trec_path沒有valid或train就跳過
      if split not in trec_path:
         print(f"現在在比對{split}正確答案...")
         print(f"{model_name}: {trec_path} 這個路徑中沒有{split}，跳過!")
         continue
      trec_file = Path(trec_path)
      if not trec_file.exists():
        print(f"{model_name}: 找不到檔案 {trec_path}，跳過!")
        continue
      answer_dict = trec_file_convert(trec_path, topk)

      list_answer_ohe = [] #預測答案
      list_label_ohe = [] #真實答案
      missing_pred_qids = 0

      for qid in rel_dict.keys(): #遍歷在驗證答案裡的每個query id
        one_answer = answer_dict.get(qid, []) #預測 e.g., ['000123', '000456', '000789']
        if not one_answer:
          missing_pred_qids += 1
        one_rel = rel_dict[qid] #真實
        one_answer = [int(pid) for pid in one_answer] #轉型 e.g., [123, 456, 789]
        one_rel = [int(pid) for pid in one_rel] #轉型
        list_answer_ohe.append(one_answer) #list of list e.g., [[123, 456, 789](對應qid=100), [234, 567, 890](對應qid=101), ...]
        list_label_ohe.append(one_rel)

      f1, precision, recall = my_classification_report(list_label_ohe, list_answer_ohe)
      print(f"=== {model_name} ===")
      print('Precision',precision)
      print('Recall',recall)
      print('F1_Score',f1)
      if missing_pred_qids:
        print(f"⚠️ {model_name} 在 {split} 缺少 {missing_pred_qids} 個 query 的預測結果（以空集合計分）")

      notes = ""
      if model_name.startswith("BM25_") and TASK1_SCOPE_PATH is not None:
        notes = "scope_filtered"

      # record to csv
      record_result(
          model_name=model_name,
          topk=topk,
          trec_file=trec_path,
          f1=f1,
          precision=precision,
          recall=recall,
          notes=notes,
          csv_path="./Legal Case Retrieval/results/experiment_results.csv"
      )
  for split, rel_dict in combine_rel_dict.items():
    _, random_stats = random_guess_predictions(
        rel_dict,
        topk=topk,
        seed=TASK1_RANDOM_BASELINE_SEED,
        query_candidate_scope_path=TASK1_SCOPE_PATH,
    )
    random_f1, random_precision, random_recall = random_guess_baseline(
        rel_dict,
        topk=topk,
        seed=TASK1_RANDOM_BASELINE_SEED,
        query_candidate_scope_path=TASK1_SCOPE_PATH,
    )
    print(f"=== Random Guess ({split})===")
    print('Precision',random_precision)
    print('Recall',random_recall)
    print('F1_Score',random_f1)
    print(f"scope_missing={random_stats['missing_scope_queries']}")
    print(f"empty_candidate_pool={random_stats['empty_candidate_pool_queries']}")
    # record to csv
    record_result(
        model_name=f"Random Guess ({split})",
        topk=topk,
        trec_file="",
        f1=random_f1,
        precision=random_precision,
        recall=random_recall,
        notes=f"scope_filtered;seed={TASK1_RANDOM_BASELINE_SEED};scope_missing={random_stats['missing_scope_queries']};empty_pool={random_stats['empty_candidate_pool_queries']}",
        csv_path="./Legal Case Retrieval/results/experiment_results.csv"
    )
