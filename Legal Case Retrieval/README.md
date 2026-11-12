# Legal Case Retrieval

[English](README.en.md) | [回到根目錄](../README.md)

這個目錄提供 COLIEE Task 1 的完整流程，並且是這個 repository 的主要工作區，包含前處理、BM25、dense retrieval、LightGBM rerank 與 submission 匯出。

## 目錄結構

- `pre-process/`: raw 檔整理、label split、scope JSON 建立。
- `lexical models/`: BM25 / QLD 索引與搜尋。
- `modernBert-fp/`: dense encoder 訓練、embedding 與 ranking。
- [`modernBert-bm25/`](modernBert-bm25/README.md): 原始 ModernBERT-base 的 COLIEE 2025 實驗，每 epoch 從 BM25 top-100 重抽 15 個負樣本；另支援 [SAILER-style in-batch negatives／Label Mask／GradCache](modernBert-bm25/SAILER_GRADCACHE.md) 與 [四 query 分組驗證](modernBert-bm25/GROUPING.md)。正式與 debug 共用訓練流程，三份 config 隔離 pairwise、v3 與 legacy v2 的輸出及續訓。
- `lightgbm/`: feature 建立、`LGBMRanker` 訓練、cutoff postprocess。
- `lcr/`: Task 1 共用模組。
- `modernBert-fp-chunkAgg/`: chunk aggregation 實驗路線。
- `modernBert-fp-chunkAgg-v6/`: 與 single-chunk baseline 負樣本公式對齊的隔離 multi-chunk 實驗。
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
python "Legal Case Retrieval/modernBert-fp/inference.py"
python "Legal Case Retrieval/modernBert-fp/similarity_and_rank.py"
python "Legal Case Retrieval/utils/eval.py"
```

這一步會產生：

- dense embeddings
- dense dot / cosine ranking
- train / valid 指標，並追加到 `Legal Case Retrieval/results/experiment_results.csv`

### 4. test retrieval 與 submission

舊的 `run_test_retrieval.sh` wrapper 已移除。現在直接使用可組合的前處理、index、推論、ranking 與 submission CLI；各命令可先加 `--help` 確認參數：

```bash
python "Legal Case Retrieval/pre-process/process_test_files.py" --help
python "Legal Case Retrieval/pre-process/prepare_test_pipeline_inputs.py" --help
python "Legal Case Retrieval/pre-process/build_query_candidate_scope.py" --help
bash "Legal Case Retrieval/lexical models/linux/index_test.sh"
bash "Legal Case Retrieval/lexical models/linux/query_search_test.sh"
LCR_TEST_MODE=1 python "Legal Case Retrieval/modernBert-fp/inference.py"
LCR_TEST_MODE=1 python "Legal Case Retrieval/modernBert-fp/similarity_and_rank.py"
python "Legal Case Retrieval/utils/trec_to_submission.py" --help
```

依 `.env` 指定輸入／輸出路徑後，這些步驟會產生：

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
- Chunk aggregation v6: [modernBert-fp-chunkAgg-v6/README.md](./modernBert-fp-chunkAgg-v6/README.md)
- BM25 hard negatives 與額外訓練工具: [modernBert/fine_tune/README.md](./modernBert/fine_tune/README.md)
