# Legal Case Retrieval

[English](README.en.md) | [回到根目錄](../README.md)

這個目錄提供 COLIEE Task 1 的完整流程，並且是這個 repository 的主要工作區，包含前處理、BM25、dense retrieval、LightGBM rerank 與 submission 匯出。

## 目錄結構

- `pre-process/`: raw 檔整理、label split、scope JSON 建立。
- `lexical models/`: BM25 / QLD 索引與搜尋。
- `modernBert-fp/`: dense encoder 訓練、embedding 與 ranking。
- `lightgbm/`: feature 建立、`LGBMRanker` 訓練、cutoff postprocess。
- `lcr/`: Task 1 共用模組。
- `modernBert-fp-chunkAgg/`: chunk aggregation 實驗路線。
- `modernBert/fine_tune/`: BM25 hard-negative 產生器與額外訓練工具。

## 資料配置

預設資料結構如下：

```text
coliee_dataset/task1/<YEAR>/
  task1_train_files_<YEAR>/
  task1_test_files_<YEAR>/
  task1_train_labels_<YEAR>.json
  task1_test_no_labels_<YEAR>.json
```

常用根變數：

- `COLIEE_TASK1_YEAR`
- `COLIEE_TASK1_ROOT`
- `COLIEE_TASK1_DIR`

## 執行流程

### 1. 前處理

```bash
bash run_pre_finetune.sh
```

這一步會建立：

- `summary/`
- `processed/`
- `task1_train_labels_<YEAR>_train.json`
- `task1_train_labels_<YEAR>_valid.json`
- `train_qid.tsv`
- `valid_qid.tsv`
- `lht_process/BM25/`
- `lht_process/modernBert/finetune_data/`
- `lht_process/modernBert/query_candidate_scope.json`

### 2. dense encoder 訓練

```bash
python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
```

若已經有 checkpoint，只要在 `.env` 或 shell 設定：

- `TASK1_MODEL_ROOT_DIR`
- `TASK1_BASE_ENCODER_DIR`
- `TASK1_RETRIEVAL_MODEL_NAME`

後續步驟會沿用同一組設定。

### 3. train / valid retrieval 檢查

```bash
bash run_train_valid_inference_eval.sh
```

這一步會產生：

- train / valid BM25 ranking
- dense embeddings
- dense dot / cosine ranking
- train / valid 指標摘要

### 4. test retrieval 與 submission

```bash
bash run_test_retrieval.sh
```

這一步會產生：

- `processed_test/`
- `test_qid.tsv`
- test BM25 ranking
- test dense embeddings
- BM25 submission
- dense submission

### 5. LTR feature、訓練與 fixed top-k

```bash
bash run_ltr_feature_train_valid_test.sh
```

這一步會產生：

- `train_features.csv`
- `valid_features.csv`
- `test_features.csv`
- `valid_predictions_raw.csv`
- `test_predictions_raw.csv`
- fixed top-k submission

### 6. cutoff search 與最終 submission

```bash
bash run_ltr_cutoff_postprocess.sh
```

這一步會讀取前一步的 prediction CSV，輸出：

- `cutoff_search/cutoff_summary.json`
- `cutoff_search/best_overall/test_submission_best_mode.txt`
- `COLIEE_LTR_FINAL_SUBMISSION_PATH`

## 設定方式

repo root wrapper 會先讀 `.env`，shell 先設定的值會優先於 `.env`。

最常調整的變數：

- 資料: `COLIEE_TASK1_YEAR`, `COLIEE_TASK1_DIR`, `TASK1_TRAIN_RAW_DIR`, `TASK1_TEST_RAW_DIR`
- dense: `TASK1_MODEL_ROOT_DIR`, `TASK1_BASE_ENCODER_DIR`, `TASK1_RETRIEVAL_MODEL_NAME`
- test 匯出: `TASK1_SUBMISSION_TOPK`, `TASK1_BM25_RUN_TAG`, `TASK1_EMBED_RUN_TAG`
- LTR: `COLIEE_LTR_OUTPUT_DIR`, `COLIEE_LTR_VALID_SCOPE_PATH`, `COLIEE_LTR_TEST_SCOPE_PATH`, `COLIEE_LTR_CUTOFF_CONFIG_JSON`

## 相關文件

- Dense retrieval: [modernBert-fp/README.md](./modernBert-fp/README.md)
- LightGBM rerank: [lightgbm/README.md](./lightgbm/README.md)
- Chunk aggregation: [modernBert-fp-chunkAgg/README.md](./modernBert-fp-chunkAgg/README.md)
- BM25 hard negatives 與額外訓練工具: [modernBert/fine_tune/README.md](./modernBert/fine_tune/README.md)
