# modernBert-fp

[English](README.en.md) | [Task 1 總覽](../README.md)

這個目錄是目前維護中的 Task 1 dense retrieval 主流程。它負責：

- 對比式 fine-tune ModernBERT encoder
- 根據最佳 checkpoint 產生 embeddings
- 產生 train / valid / test 的相似度排名
- 供 retrieval-only submission 與 LightGBM rerank 使用

如果你現在要做的是「實驗提交測試資料預測」，這個目錄就是 Task 1 dense encoder 的主程式入口。

## 你會用到哪些檔案

- `fine_tune/fine_tune.py`
  維護中的對比式 fine-tune 入口。每個 epoch 都會重算相似度並動態抽 hard negatives。
- `inference.py`
  自動挑選最佳 `eval_global_f1` checkpoint，對 `processed` / `processed_new` 或 `processed_test` 產生 embeddings。
- `similarity_and_rank.py`
  讀取 embeddings 後輸出 dot / cosine TREC ranking。
- `find_best_model.py`
  從 checkpoint 目錄中挑出指定 metric 最佳的 checkpoint。
- `train_modernbert_caselaw_fp.py`
  繼續預訓練 backbone 的腳本，對應目前下游流程依賴的 `modernbert-caselaw-accsteps-fp/checkpoint-29000`。

以下檔案不是目前主流程，但仍保留：

- `inference-noSFT.py`
- `inference-test-noSFT.py`
- `similarity_and_rank_noSFT.py`

## 主流程分兩種情境

### 情境 A: 訓練新的 dense model

1. 先從 repo root 執行 `bash run_pre_finetune_2026.sh`
2. 執行 `python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"`
3. 若要做 train / valid 驗證，執行 `bash run_train_valid_inference_eval_2026.sh`
4. 若要做 test retrieval-only submission，執行 `bash run_test_retrieval_2026.sh`
5. 若要接 LightGBM rerank，先確認下游讀到的是你要用的 checkpoint，再執行 `bash run_ltr_feature_train_valid_test_2026.sh`

### 情境 B: 不重訓，只用既有 checkpoint 做 test prediction

1. 先確保已有可用 checkpoint 與 backbone checkpoint
2. 執行 `bash run_test_retrieval_2026.sh`
3. 若要接 LTR，繼續執行 `bash run_ltr_feature_train_valid_test_2026.sh`
4. 若要重搜 cutoff，執行 `bash run_ltr_cutoff_postprocess_2026.sh`

## 主要輸入條件

### 訓練需要

- `coliee_dataset/task1/2026/processed/`
- `coliee_dataset/task1/2026/train_qid.tsv`
- `coliee_dataset/task1/2026/valid_qid.tsv`
- `coliee_dataset/task1/2026/task1_train_labels_2026.json`
- `coliee_dataset/task1/2026/task1_train_labels_2026_train.json`
- `coliee_dataset/task1/2026/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json`
- `coliee_dataset/task1/2026/lht_process/modernBert/query_candidate_scope.json`

### 推論需要

- `modernbert-caselaw-accsteps-fp/checkpoint-29000`
- 目標 checkpoint 根目錄
- `processed/` 或 `processed_test/`

### test retrieval-only pipeline 額外需要

- `coliee_dataset/task1/2026/task1_test_files_2026/`
- `coliee_dataset/task1/2026/task1_test_no_labels_2026.json`

## 實際輸出內容

### 1. fine-tune 輸出

`fine_tune.py` 會產生：

- checkpoint 目錄
- TensorBoard logs
- 每個 epoch 的相似度檔與 adaptive negative JSON

目前腳本內建的命名是：

- 模型輸出：`./modernBERT_contrastive_adaptive_fp_fp16_scopeFilteredRaw_<YEAR>`
- fine-tune 中間資料：`coliee_dataset/task1/<YEAR>/lht_process/modernBert/finetune_data/`

常見輸出檔：

- `similarity_scores_epoch<E>.tsv`
- `adaptive_negative_epoch<E>_train.json`
- `similarity_scores_<epoch>_eval_train.tsv`
- `similarity_scores_<epoch>_eval_valid.tsv`

### 2. embedding 輸出

train / valid 模式：

- `processed/processed_document_modernBert_fp_fp16_embeddings.pkl`
- `processed_new/processed_new_document_modernBert_fp_fp16_embeddings.pkl`

test 模式：

- `processed_test/processed_test_document_modernBert_fp_fp16_embeddings.pkl`
- `processed_test/processed_test_query_modernBert_fp_fp16_embeddings.pkl`

### 3. ranking 輸出

train / valid：

- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_train.tsv`
- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_train.tsv`
- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_valid.tsv`
- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_valid.tsv`

test：

- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_test.tsv`
- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_test.tsv`

### 4. retrieval-only submission 輸出

`run_test_retrieval_2026.sh` 會另外輸出：

- `lht_process/submission/task1_FLNLPBM25.txt`
- `lht_process/submission/task1_FLNLPEMBED.txt`

## 建議操作順序

### 1. 建立前置資料

```bash
bash run_pre_finetune_2026.sh
```

這一步不是在本目錄裡，但它會幫你把本目錄訓練真正需要的資料先準備好。

### 2. 開始 fine-tune

```bash
python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
```

目前 `fine_tune.py` 的行為重點：

- query / positive / negative 共用同一個 encoder
- loss 是 document-level InfoNCE / CrossEntropy
- `TASK1_RETRIEVAL_BATCH_SIZE` 會影響整體 retrieval 評估與 adaptive sampling
- `TASK1_INIT_TEMPERATURE` 會設定可學習溫度的初值
- 每個 epoch 都會以 validation retrieval F1 作為最佳模型選擇依據
- `TASK1_AUTO_RESUME=1` 時會自動接最新 checkpoint
- 也可用 `TASK1_RESUME_FROM_CHECKPOINT` 指定續訓 checkpoint

### 3. 產生 train / valid embeddings 與 ranking

```bash
bash run_train_valid_inference_eval_2026.sh
```

常用可調環境變數：

- `FORCE_REENCODE=1`
  忽略既有 embeddings，重新跑 `inference.py`
- `SKIP_BM25=1`
  跳過 BM25 valid/train 檢索
- `RUN_FULL_EVAL=1`
  額外執行 `Legal Case Retrieval/utils/eval.py`

### 4. 產生 test retrieval-only submission

```bash
bash run_test_retrieval_2026.sh
```

常用可調環境變數：

- `SUBMISSION_TOPK`
  預設 5，用來控制輸出幾筆結果

這一步會做完整 test 前置：

- raw test files -> `processed_test/`
- 產生 `test_qid.tsv`
- 建 `BM25_test` index
- 建 test scope JSON
- 產生 BM25 與 dense retrieval submission

## embedding 使用慣例

目前維護中的 Task 1 dense ranking 預設：

- query 用 `processed`
- candidate 也用 `processed`

也就是說，雖然 `inference.py` 仍會同時輸出 `processed_new` embeddings，但 `similarity_and_rank.py` 預設不會拿它來算 query。

若要切換來源：

```bash
export LCR_QUERY_EMBED_SOURCE=processed_new
export LCR_CANDIDATE_EMBED_SOURCE=processed
python "Legal Case Retrieval/modernBert-fp/similarity_and_rank.py"
```

也可以直接指定自訂檔案：

```bash
export LCR_QUERY_EMBEDDINGS_PATH=/abs/path/query_embeddings.pkl
export LCR_CANDIDATE_EMBEDDINGS_PATH=/abs/path/candidate_embeddings.pkl
```

## 常用環境變數

### 路徑與年份

- `COLIEE_TASK1_YEAR`
- `COLIEE_TASK1_ROOT`
- `COLIEE_TASK1_DIR`

### 訓練與續訓

- `TASK1_RETRIEVAL_BATCH_SIZE`
- `TASK1_INIT_TEMPERATURE`
- `TASK1_AUTO_RESUME`
- `TASK1_RESUME_FROM_CHECKPOINT`

### 推論 / 排名

- `LCR_TEST_MODE`
- `LCR_QUERY_CANDIDATE_SCOPE_JSON`
- `LCR_QUERY_EMBED_SOURCE`
- `LCR_CANDIDATE_EMBED_SOURCE`
- `LCR_QUERY_EMBEDDINGS_PATH`
- `LCR_CANDIDATE_EMBEDDINGS_PATH`

## 目前程式碼要注意的地方

### 1. checkpoint 根目錄命名不一致

目前有一個實際存在的差異：

- `fine_tune.py` 預設輸出到 `modernBERT_contrastive_adaptive_fp_fp16_scopeFilteredRaw_<YEAR>`
- `inference.py` 預設讀的是 `modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_<YEAR>`

所以如果你剛訓練完一個新的 `scopeFilteredRaw` 目錄，`inference.py` 和 repo root 的部分 wrapper 不會自動改讀它。這時要自行對齊使用的 checkpoint 路徑。

### 2. `inference.py` 不接受 CLI 指定 model root

目前 `inference.py` 是固定在程式裡決定 checkpoint 根目錄，不像 `lightgbm/ltr_feature_pipeline.py` 那樣可直接用參數覆寫。這也是為什麼上面那個命名差異在文件裡必須特別提醒。

### 3. backbone checkpoint 是硬性依賴

`inference.py`、`run_train_valid_inference_eval_2026.sh`、`run_test_retrieval_2026.sh` 都要求：

```text
modernbert-caselaw-accsteps-fp/checkpoint-29000
```

若這個目錄不存在，下游流程不會開始。
