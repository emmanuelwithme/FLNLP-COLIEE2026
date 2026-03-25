# modernBert-fp

[English](README.en.md) | [Task 1 總覽](../README.md)

這個目錄提供 Task 1 的 dense encoder 訓練、embedding 產生與 ranking 輸出。

## 主要檔案

- `fine_tune/fine_tune.py`: 對比式 fine-tune 入口。
- `inference.py`: 根據 `TASK1_MODEL_ROOT_DIR` 選 checkpoint，產生 embeddings。
- `similarity_and_rank.py`: 讀取 embeddings，輸出 dot / cosine ranking。
- `find_best_model.py`: 根據 metric 找最佳 checkpoint。
- `train_modernbert_caselaw_fp.py`: backbone continued pretraining。

## 常見流程

### 訓練新的 dense model

```bash
bash run_pre_finetune.sh
python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
bash run_train_valid_inference_eval.sh
bash run_test_retrieval.sh
```

### 沿用既有 checkpoint

先設定：

- `TASK1_MODEL_ROOT_DIR`
- `TASK1_BASE_ENCODER_DIR`
- `TASK1_RETRIEVAL_MODEL_NAME`

然後執行：

```bash
bash run_train_valid_inference_eval.sh
bash run_test_retrieval.sh
```

若要接 LTR：

```bash
bash run_ltr_feature_train_valid_test.sh
bash run_ltr_cutoff_postprocess.sh
```

## 主要輸入

- `TASK1_FINETUNE_DATA_DIR/contrastive_bm25_hard_negative_top100_random15_train.json`
- `TASK1_FINETUNE_DATA_DIR/contrastive_bm25_hard_negative_top100_random15_valid.json`
- `TASK1_PROCESSED_DIR/`
- `TASK1_QUERY_DIR/`
- `TASK1_TRAIN_QID_PATH`
- `TASK1_VALID_QID_PATH`
- `TASK1_SCOPE_PATH`
- `TASK1_BASE_ENCODER_DIR`
- `TASK1_MODEL_ROOT_DIR`

test retrieval 額外需要：

- `TASK1_TEST_RAW_DIR`
- `TASK1_TEST_LABELS_PATH`

## 主要輸出

訓練：

- `TASK1_MODEL_ROOT_DIR/`
- `TASK1_MODEL_ROOT_DIR/tb/`

embedding：

- `processed/processed_document_<MODEL_NAME>_embeddings*.pkl`
- `processed_new/processed_new_document_<MODEL_NAME>_embeddings*.pkl`
- `processed_test/processed_test_document_<MODEL_NAME>_embeddings*.pkl`
- `processed_test/processed_test_query_<MODEL_NAME>_embeddings*.pkl`

ranking：

- `TASK1_MODEL_RESULTS_DIR/output_<MODEL_NAME>_dot_train.tsv`
- `TASK1_MODEL_RESULTS_DIR/output_<MODEL_NAME>_dot_valid.tsv`
- `TASK1_MODEL_RESULTS_DIR/output_<MODEL_NAME>_dot_test.tsv`
- `TASK1_MODEL_RESULTS_DIR/output_<MODEL_NAME>_cos_train.tsv`
- `TASK1_MODEL_RESULTS_DIR/output_<MODEL_NAME>_cos_valid.tsv`

## 常用變數

checkpoint 與 backbone：

- `TASK1_MODEL_ROOT_DIR`
- `TASK1_BASE_ENCODER_DIR`
- `TASK1_CHECKPOINT_METRIC`
- `TASK1_CHECKPOINT_MODE`

訓練：

- `TASK1_FINETUNE_DATA_DIR`
- `TASK1_SCOPE_FILTER`
- `TASK1_QUICK_TEST`
- `TASK1_QUICK_TEST_CAND_K`
- `TASK1_QUICK_TEST_QUERY_K`
- `TASK1_RETRIEVAL_BATCH_SIZE`
- `TASK1_RETRIEVAL_MAX_LENGTH`
- `TASK1_INIT_TEMPERATURE`
- `TASK1_AUTO_RESUME`
- `TASK1_RESUME_FROM_CHECKPOINT`

推論與 ranking：

- `TASK1_CANDIDATE_DIR`
- `TASK1_QUERY_DIR`
- `TASK1_CANDIDATE_EMBEDDINGS_OUTPUT`
- `TASK1_QUERY_EMBEDDINGS_OUTPUT`
- `TASK1_TEST_CANDIDATE_EMBEDDINGS_PATH`
- `TASK1_TEST_QUERY_EMBEDDINGS_PATH`
- `TASK1_OUTPUT_DOT_TRAIN_PATH`
- `TASK1_OUTPUT_DOT_VALID_PATH`
- `TASK1_OUTPUT_DOT_TEST_PATH`
- `LCR_QUERY_CANDIDATE_SCOPE_JSON`

## 直接覆寫範例

```bash
TASK1_MODEL_ROOT_DIR=./models/my_task1_model \
TASK1_BASE_ENCODER_DIR=./models/my_encoder/checkpoint-29000 \
TASK1_RETRIEVAL_MODEL_NAME=my_task1_model \
bash run_test_retrieval.sh
```

repo root `.env` 會提供預設值，shell 變數可做單次覆寫。
