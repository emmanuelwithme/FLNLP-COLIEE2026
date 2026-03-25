# lightgbm

[English](README.en.md) | [Task 1 總覽](../README.md)

這個目錄是目前維護中的 Task 1 rerank 與 submission 主流程。若你的目標是「把 test 資料做成最後可提交的 rerank 結果」，這裡就是主程式。

## 這個目錄負責什麼

- 建立 train / valid / test 的 LTR feature
- 訓練 `LightGBM LGBMRanker`
- 輸出 raw rerank prediction CSV
- 套用 scope / self-removal 等 legal filters
- 產生 fixed top-k baseline submission
- 在 validation 上做 cutoff grid search，選最佳 mode 後套用到 test

## 主要檔案

- `ltr_feature_pipeline.py`
  完整 feature building + LightGBM 訓練 + raw prediction 輸出 + fixed top-k / cutoff postprocess 整合入口。
- `cutoff_postprocess.py`
  只吃既有 rerank prediction CSV，不重建 feature、不重訓 model，專門做 cutoff mode 比較與 submission 匯出。
- `fixed_topk_postprocess.py`
  對 test rerank 結果直接做固定 top-k 匯出的小型 wrapper。

## 目前建議的完整 test submission 流程

如果你要從目前 repo 的主流程一路跑到 final submission，建議順序如下：

1. 先完成 `bash run_pre_finetune_2026.sh`
2. 準備或沿用可用的 dense checkpoint
3. 先跑 `bash run_test_retrieval_2026.sh`
   這一步會把 `processed_test/`、`test_qid.tsv`、`BM25_test` index、test scope、test embeddings 準備好
4. 執行 `bash run_ltr_feature_train_valid_test_2026.sh`
   這一步會產生 fixed top-k baseline submission
5. 執行 `bash run_ltr_cutoff_postprocess_2026.sh`
   這一步才做唯一一次 cutoff grid search，並產生最後 submission

## 這個流程不會幫你自動補齊的前置條件

`run_ltr_feature_train_valid_test_2026.sh` 與 `ltr_feature_pipeline.py` 本身不會替你建立下列內容，所以要先準備好：

- `processed/`
- `processed_test/`
- `train_qid.tsv`
- `valid_qid.tsv`
- `test_qid.tsv`
- `processed/processed_document_modernBert_fp_fp16_embeddings.pkl`
- `processed_test/processed_test_document_modernBert_fp_fp16_embeddings.pkl`
- `lht_process/BM25/index`
- `lht_process/BM25_test/index`
- `modernbert-caselaw-accsteps-fp/checkpoint-29000`
- 目標 dense checkpoint 根目錄

若缺少 test 側的 `processed_test`、`test_qid.tsv`、`BM25_test/index`、test embeddings，最簡單的做法就是先跑一次：

```bash
bash run_test_retrieval_2026.sh
```

## feature 內容

`ltr_feature_pipeline.py` 目前組合的特徵包含：

- lexical scores
  `bm25_score`、`qld_score`、`bm25_ngram_score`
- dense retrieval scores
  `dense_score`
- rank features
  `bm25_rank`、`dense_rank`
- length features
  `query_length`、`doc_length`、`len_ratio`、`len_diff`
- placeholder features
  citation / reference / fragment 的數量與比例
- year features
  `query_year`、`doc_year`、`year_diff`
- chunk similarity aggregation
  `chunk_sim_max`、`chunk_sim_mean`、`chunk_sim_top2_mean`

也就是說，這條 LTR pipeline 不只是吃單一 dense 分數，而是把 lexical、dense、長度、placeholder、年份與 chunk-level 相似度一起交給 LightGBM。

## `run_ltr_feature_train_valid_test_2026.sh`

這支 wrapper 會做完整 LTR 訓練與 fixed top-k 匯出。

```bash
bash run_ltr_feature_train_valid_test_2026.sh
```

預設行為：

- 讀取 `Legal Case Retrieval/lightgbm/ltr_feature_pipeline.py`
- 以 `coliee_dataset/task1/<YEAR>/lht_process/lightgbm_ltr_scope_raw/` 為輸出根目錄
- 產生 `train_features.csv`、`valid_features.csv`、`test_features.csv`
- 訓練 `lgbm_ranker_scope_raw.txt`
- 輸出 `valid_predictions_raw.csv`、`test_predictions_raw.csv`
- 額外輸出 scope-filtered 的 `valid_predictions.csv`、`test_predictions.csv`
- 再跑一次 fixed top-k 匯出
- 刻意跳過 cutoff search

主要輸出：

- `train_features.csv`
- `valid_features.csv`
- `test_features.csv`
- `lgbm_ranker_scope_raw.txt`
- `valid_predictions_raw.csv`
- `valid_predictions.csv`
- `test_predictions_raw.csv`
- `test_predictions.csv`
- `fixed_top5/test_submission_fixed_topk.txt`
- `fixed_top5/fixed_topk_summary.json`

repo root 另外會複製：

- `task1_FLNLPLTRTOP5.txt`

### 這支 wrapper 常用的環境變數

- `COLIEE_TASK1_YEAR`
- `COLIEE_TASK1_ROOT`
- `COLIEE_TASK1_DIR`
- `COLIEE_LTR_OUTPUT_DIR`
- `COLIEE_LTR_NUM_WORKERS`
- `COLIEE_LTR_DENSE_BATCH_SIZE`
- `COLIEE_LTR_FEATURE_SCORE_BATCH_SIZE`
- `COLIEE_LTR_LGBM_DEVICE`
- `COLIEE_LTR_FIXED_TOPK`
- `COLIEE_LTR_FIXED_TOPK_RUN_TAG`
- `COLIEE_LTR_FIXED_TOPK_FINAL_SUBMISSION_PATH`

這支 wrapper 不會自己 `source .env`。若要覆寫年份或輸出目錄，請在 shell 先帶環境變數，例如：

```bash
COLIEE_TASK1_YEAR=2026 \
COLIEE_LTR_OUTPUT_DIR=./coliee_dataset/task1/2026/lht_process/lightgbm_ltr_scope_raw \
bash run_ltr_feature_train_valid_test_2026.sh
```

## `run_ltr_cutoff_postprocess_2026.sh`

這支 wrapper 只做 cutoff postprocess，不重跑 feature、不重訓 model。

```bash
bash run_ltr_cutoff_postprocess_2026.sh
```

預設流程：

1. 讀取 `valid_predictions_raw.csv`
2. 讀取 `test_predictions_raw.csv`
3. 套用 valid / test scope 與 self-removal
4. 在 validation 上比較三種 mode
5. 選出最佳 mode 與參數
6. 對 test 套用一次最佳 cutoff
7. 寫出 `cutoff_summary.json` 與最終 submission

目前比較的 mode：

- fixed top-k
- ratio cutoff
- largest-gap adaptive cutoff

主要輸出：

- `cutoff_search/valid_predictions_legal_filtered.csv`
- `cutoff_search/test_predictions_legal_filtered.csv`
- `cutoff_search/validation_mode_comparison.csv`
- `cutoff_search/fixed_topk/`
- `cutoff_search/ratio_cutoff/`
- `cutoff_search/largest_gap_cutoff/`
- `cutoff_search/best_overall/test_submission_best_mode.txt`
- `cutoff_search/cutoff_summary.json`

repo root 另外會複製：

- `task1_FLNLPLTR.txt`

### 這支 wrapper 常用的環境變數

- `COLIEE_LTR_OUTPUT_DIR`
- `COLIEE_LTR_VALID_PRED_PATH`
- `COLIEE_LTR_TEST_PRED_PATH`
- `COLIEE_LTR_VALID_SCOPE_PATH`
- `COLIEE_LTR_TEST_SCOPE_PATH`
- `COLIEE_LTR_CUTOFF_OUTPUT_DIR`
- `COLIEE_LTR_CUTOFF_CONFIG_JSON`
- `COLIEE_LTR_SUBMISSION_RUN_TAG`
- `COLIEE_LTR_FINAL_SUBMISSION_PATH`

若你只想換 cutoff grid，不想重訓 LightGBM：

```bash
COLIEE_LTR_CUTOFF_CONFIG_JSON=/path/to/cutoff_config.json \
bash run_ltr_cutoff_postprocess_2026.sh
```

## 預設 scope 路徑要特別注意

這套流程現在有兩份不同用途的預設 scope：

- valid scope:
  `coliee_dataset/task1/<YEAR>/lht_process/scope_compare/query_candidate_scope_raw_plus0.json`
- test scope:
  `coliee_dataset/task1/<YEAR>/lht_process/modernBert/query_candidate_scope_test_raw.json`

其中 test scope 通常可由 `run_test_retrieval_2026.sh` 自動生成。

但 valid scope 的 `query_candidate_scope_raw_plus0.json` 只是目前 wrapper 的預設檔名，並不是 `run_pre_finetune_2026.sh` 自動建立的輸出。如果你的 valid scope 檔案不在這個位置，請顯式覆寫。

## cutoff search 的評分邏輯

`cutoff_postprocess.py` 會在 validation 上記錄多種指標，包含：

- F1
- Precision
- Recall
- nDCG@10
- P@5
- R@5

最後選 mode 時主要以 validation F1 為先，再依 recall、precision 等次序做 tie-break。

## 目前你最需要知道的限制

### 1. LTR wrapper 不會替你生 test embeddings

若 test embeddings 還沒產生，`ltr_feature_pipeline.py` 只會直接失敗，不會幫你先呼叫 `modernBert-fp/inference.py`。這就是為什麼建議先跑 `run_test_retrieval_2026.sh`。

### 2. dense checkpoint 根目錄預設仍偏向 `scopeFiltered`

`ltr_feature_pipeline.py` 的預設 `--model-root-dir` 是：

```text
modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_<YEAR>
```

如果你最近訓練出的模型其實放在 `scopeFilteredRaw_<YEAR>`，請記得覆寫 `--model-root-dir` 或修改 wrapper。

### 3. `run_ltr_feature_train_valid_test_2026.sh` 故意不做 cutoff search

這不是漏掉，而是目前設計上刻意把 cutoff grid search集中到 `run_ltr_cutoff_postprocess_2026.sh` 做唯一一次，避免重複搜尋與重複覆蓋 submission。
