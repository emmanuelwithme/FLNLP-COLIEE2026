# Legal Case Retrieval

[English](README.en.md) | [回到根目錄](../README.md)

這個目錄是 COLIEE Task 1 的維護中流程，負責從案例前處理、dense retrieval、LightGBM rerank，到最終 submission 匯出。

## 目前主要看哪些子目錄

- `modernBert-fp/`
  維護中的 dense retrieval 主流程。Task 1 的 ModernBERT contrastive fine-tune、embedding 推論、相似度排名都在這裡。
- `lightgbm/`
  維護中的 LightGBM learning-to-rank 與 submission 後處理主流程。
- `pre-process/`
  原始案例前處理、test 前處理、scope JSON 生成等工具。
- `lcr/`
  共用的 Task 1 工具模組，例如資料路徑、embedding 選擇、相似度計算、metrics 與 retrieval helpers。
- `lexical models/`
  BM25 / QLD 相關流程與 Pyserini index/search wrapper。

## 仍保留但不是主流程的子目錄

- `modernBert-fp-chunkAgg/`
  chunk aggregation 版本的 ModernBERT 實驗。
- `modernBert/`
  舊版或歷史實驗相關程式。`run_pre_finetune_2026.sh` 目前仍會用到其中的 BM25 hard-negative 生成腳本。
- `SAILER/`
  保留的上游實驗資料夾，不是目前主要提交流程。

## 資料與中間產物慣例

預設 Task 1 目錄：

```text
coliee_dataset/task1/2026/
```

常見輸入：

- `task1_train_files_2026/`
- `task1_test_files_2026/`
- `task1_train_labels_2026.json`
- `task1_test_no_labels_2026.json`

常見中間產物：

- `summary/`
  由 `pre-process/summary.py` 產生的摘要檔。
- `processed/`
  train/valid 使用的清理後案例。
- `processed_new/`
  保留給舊 THUIR-style query 實驗的另一份 query 文本來源。
- `processed_test/`
  test 使用的清理後案例。
- `lht_process/BM25/`
  train/valid 的 BM25 index、topics、檢索結果。
- `lht_process/BM25_test/`
  test 的 BM25 index、topics、檢索結果。
- `lht_process/modernBert/`
  dense retrieval 前置產物，例如 `finetune_data` 與 `query_candidate_scope.json`。
- `lht_process/modernBert_fp_fp16/`
  dense retrieval train/valid/test ranking 輸出。
- `lht_process/lightgbm_ltr_scope_raw/`
  LightGBM feature、模型、raw rerank 結果、fixed top-k 與 cutoff search 輸出。

## 建議執行順序

以下順序是目前最接近「從原始資料一路到 Task 1 submission」的做法。

### 1. 前處理與 fine-tune 前置

從 repo root 執行：

```bash
bash run_pre_finetune_2026.sh
```

這一步會依序建立：

- `summary/`
- `processed/`
- `task1_train_labels_2026_train.json`
- `task1_train_labels_2026_valid.json`
- `train_qid.tsv`
- `valid_qid.tsv`
- `lht_process/BM25/` index 與 train/valid 檢索結果
- `lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_{train,valid}.json`
- `lht_process/modernBert/query_candidate_scope.json`

其中 `query_candidate_scope.json` 是 `build_query_candidate_scope.py` 的內建預設模式產物：

- query / candidate 文字都取自 `processed/`
- 年份抽取來源取自原始 `task1_train_files_<YEAR>/`
- 會合併 train + valid qids
- `year_slack=1`
- 不排除 self

### 2. Dense retrieval 訓練或沿用既有 checkpoint

主流程文件：

- [modernBert-fp/README.md](./modernBert-fp/README.md)

直接訓練：

```bash
python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
```

如果你只是要沿用既有 checkpoint 做 test submission，也可以跳過重新訓練，直接使用後續推論與 rerank 流程。

### 3. Train/valid retrieval 檢查

```bash
bash run_train_valid_inference_eval_2026.sh
```

這一步會：

- 視需要重新產生 `processed` embeddings
- 產生 train/valid 的 dot / cosine TREC ranking
- 可選擇重跑 BM25 valid/train 檢索
- 輸出 focused metrics 與 coverage sanity check

### 4. Retrieval-only test prediction

```bash
bash run_test_retrieval_2026.sh
```

這一步會建立或更新：

- `processed_test/`
- `test_qid.tsv`
- `lht_process/BM25_test/`
- `lht_process/modernBert/query_candidate_scope_test_raw.json`
- `lht_process/submission/task1_FLNLPBM25.txt`
- `lht_process/submission/task1_FLNLPEMBED.txt`

如果你只想用 dense retrieval 直接提交，這一步已經是完整流程。

### 5. LightGBM rerank submission pipeline

主流程文件：

- [lightgbm/README.md](./lightgbm/README.md)

從 repo root 執行：

```bash
bash run_ltr_feature_train_valid_test_2026.sh
```

這一步會：

- 重建 train / valid / test feature CSV
- 訓練 LightGBM ranker
- 輸出 raw 與 scope-filtered prediction CSV
- 直接產出 fixed top-k test submission

目前 wrapper 刻意加上 `--skip-cutoff-search`，因此這一步只做固定 top-k baseline，不做 cutoff grid search。

預設固定 top-k 結果：

- `coliee_dataset/task1/2026/lht_process/lightgbm_ltr_scope_raw/fixed_top5/test_submission_fixed_topk.txt`
- repo root 複本：`task1_FLNLPLTRTOP5.txt`

### 6. Cutoff postprocess only

```bash
bash run_ltr_cutoff_postprocess_2026.sh
```

這一步只重用既有 rerank 輸出，不重建 features，也不重訓 LightGBM。它會：

- 套用 legal filters
- 在 validation 上比較 fixed top-k、ratio cutoff、largest-gap cutoff
- 選出最佳 mode
- 只對 test 套用一次最佳 cutoff

預設最終 submission：

- `coliee_dataset/task1/2026/lht_process/lightgbm_ltr_scope_raw/cutoff_search/best_overall/test_submission_best_mode.txt`
- repo root 複本：`task1_FLNLPLTR.txt`

## 重要慣例與注意事項

### 1. Task 1 wrapper 多半是 2026 專用

- `run_pre_finetune_2026.sh`
- `run_train_valid_inference_eval_2026.sh`
- `run_test_retrieval_2026.sh`

這三支腳本都不是靠 `.env` 自動切年份，而是腳本本身偏向 2026 workflow。

### 2. `lcr.task1_paths` 會讀 `.env`

Task 1 的許多 Python 腳本透過 `lcr/task1_paths.py` 讀取：

- `COLIEE_TASK1_YEAR`
- `COLIEE_TASK1_ROOT`
- `COLIEE_TASK1_DIR`

但 repo root wrapper 是否會主動讀 `.env`，要看該 shell script 自己的寫法，不要把兩者混為一談。

### 3. embedding 預設改成 `processed`

目前維護中的 Task 1 設定是：

- 推論仍會同時輸出 `processed` 與 `processed_new` embeddings
- ranking 預設 query / candidate 都使用 `processed`
- 若要切回 THUIR-style query，可以設定：

```bash
export LCR_QUERY_EMBED_SOURCE=processed_new
export LCR_CANDIDATE_EMBED_SOURCE=processed
```

### 4. scope 檔案有多種，腳本使用的不是同一份

常見 scope 檔：

- `lht_process/modernBert/query_candidate_scope.json`
  dense retrieval train/valid 常用。
- `lht_process/modernBert/query_candidate_scope_test_raw.json`
  `run_test_retrieval_2026.sh` 產生的 test scope。
- `lht_process/scope_compare/query_candidate_scope_raw_plus0.json`
  LightGBM wrapper 目前預設拿來做 valid scope 的檔案路徑。

最後這個 valid scope 檔案是 LightGBM wrapper 的預設輸入，不是 `run_pre_finetune_2026.sh` 自動生成的結果。若你的檔名或生成規則不同，請用環境變數或 CLI 參數改掉。

### 5. 目前存在 checkpoint 目錄命名差異

目前程式碼中有一個需要特別注意的地方：

- `modernBert-fp/fine_tune/fine_tune.py` 目前預設輸出到 `modernBERT_contrastive_adaptive_fp_fp16_scopeFilteredRaw_<YEAR>`
- 但 `modernBert-fp/inference.py`、`run_train_valid_inference_eval_2026.sh`、`run_test_retrieval_2026.sh`、`lightgbm/ltr_feature_pipeline.py` 的預設 checkpoint 根目錄仍偏向 `modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_<YEAR>`

也就是說，如果你剛訓練完一個新的 `scopeFilteredRaw` 實驗，後續 wrapper 不一定會自動接到那個目錄。需要手動對齊 checkpoint 路徑、調整腳本，或在支援 CLI 的地方明確覆寫。

## 相關文件

- Dense retrieval 主流程: [modernBert-fp/README.md](./modernBert-fp/README.md)
- LightGBM rerank 主流程: [lightgbm/README.md](./lightgbm/README.md)
- chunk aggregation 實驗: [modernBert-fp-chunkAgg/README.md](./modernBert-fp-chunkAgg/README.md)
- 舊版 modernBERT fine-tune 與 BM25 hard negatives: [modernBert/fine_tune/README.md](./modernBert/fine_tune/README.md)
