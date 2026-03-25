# lightgbm

[English](README.en.md) | [Task 1 總覽](../README.md)

這個目錄提供 Task 1 的 rerank、fixed top-k 匯出與 cutoff postprocess。

## 主要檔案

- `ltr_feature_pipeline.py`: 建 feature、訓練 `LGBMRanker`、輸出 prediction CSV、匯出 fixed top-k。
- `cutoff_postprocess.py`: 讀取 prediction CSV，比較 cutoff 模式並輸出最終 submission。
- `fixed_topk_postprocess.py`: 根據 test prediction 直接匯出 fixed top-k submission。

## 前置條件

執行 LTR 前通常需要先準備：

- `processed/`
- `processed_test/`
- `train_qid.tsv`
- `valid_qid.tsv`
- `test_qid.tsv`
- train / valid dense embeddings
- test dense embeddings
- `lht_process/BM25/index`
- `lht_process/BM25_test/index`
- `TASK1_MODEL_ROOT_DIR`
- `TASK1_BASE_ENCODER_DIR`

最直接的準備順序：

```bash
bash run_pre_finetune.sh
python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
bash run_test_retrieval.sh
```

## 步驟 1: 建 feature、訓練與 fixed top-k

```bash
bash run_ltr_feature_train_valid_test.sh
```

這一步會產生：

- `train_features.csv`
- `valid_features.csv`
- `test_features.csv`
- `valid_predictions_raw.csv`
- `test_predictions_raw.csv`
- `fixed_top<k>/fixed_topk_summary.json`
- fixed top-k submission

常用變數：

- `COLIEE_LTR_OUTPUT_DIR`
- `COLIEE_LTR_NUM_WORKERS`
- `COLIEE_LTR_DENSE_BATCH_SIZE`
- `COLIEE_LTR_FEATURE_SCORE_BATCH_SIZE`
- `COLIEE_LTR_LGBM_DEVICE`
- `COLIEE_LTR_FIXED_TOPK`
- `COLIEE_LTR_FIXED_TOPK_RUN_TAG`
- `COLIEE_LTR_FIXED_TOPK_FINAL_SUBMISSION_PATH`

## 步驟 2: cutoff search 與最終 submission

```bash
bash run_ltr_cutoff_postprocess.sh
```

這一步只會讀取既有的 prediction CSV，不會重建 feature 或重訓模型。

輸出：

- `cutoff_search/cutoff_summary.json`
- `cutoff_search/best_overall/test_submission_best_mode.txt`
- `COLIEE_LTR_FINAL_SUBMISSION_PATH`

常用變數：

- `COLIEE_LTR_OUTPUT_DIR`
- `COLIEE_LTR_VALID_SCOPE_PATH`
- `COLIEE_LTR_TEST_SCOPE_PATH`
- `COLIEE_LTR_CUTOFF_CONFIG_JSON`
- `COLIEE_LTR_SUBMISSION_RUN_TAG`
- `COLIEE_LTR_FINAL_SUBMISSION_PATH`

## 直接執行 Python CLI

```bash
python "Legal Case Retrieval/lightgbm/ltr_feature_pipeline.py" --help
python "Legal Case Retrieval/lightgbm/cutoff_postprocess.py" --help
```

`ltr_feature_pipeline.py` 會從 repo root `.env` 讀取預設資料路徑、dense embeddings、模型目錄與輸出目錄；shell 變數可做單次覆寫。
