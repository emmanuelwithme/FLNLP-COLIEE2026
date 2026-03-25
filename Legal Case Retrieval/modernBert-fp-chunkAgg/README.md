# modernBert-fp-chunkAgg

[English](README.en.md) | [Task 1 總覽](../README.md)

這個目錄提供 Task 1 的 chunk aggregation 版本。文件會先切成多個 chunk，各 chunk 表徵再融合成單一文件向量。

## 目錄內容

- `fine_tune/fine_tune.py`: chunkAgg 對比式訓練。
- `inference.py`: 產生 chunkAgg embeddings。
- `similarity_and_rank.py`: 讀取 embeddings，輸出 ranking。
- `fine_tune/modernbert_contrastive_model.py`: 模型定義。
- `run_train.sh`: 啟動訓練。
- `run_infer.sh`: 產生 embeddings。
- `run_rank.sh`: 產生 ranking。
- `.env`: 本目錄的預設參數檔。

## 文件編碼方式

1. 對全文做 tokenization。
2. 每個 chunk 保留 `TASK1_DOCUMENT_CHUNK_LENGTH` tokens。
3. 優先在句尾附近切分。
4. 每篇文件最多保留 `TASK1_MAX_DOCUMENT_CHUNKS` 個 chunks。
5. 每個 chunk 取 ModernBERT `[CLS]` 向量。
6. 以 learnable `[DOC]` token 與 chunk position embedding 做融合。
7. 輸出單一文件向量並做 L2 normalize。

## 使用方式

建議從 repo root 執行。

### 1. 設定參數

可編輯 `Legal Case Retrieval/modernBert-fp-chunkAgg/.env`，或在 shell 直接覆寫同名變數。shell 值優先。

常用變數：

- `TASK1_CHUNKAGG_MODEL_NAME`
- `TASK1_CHUNKAGG_BASE_ENCODER_DIR`
- `TASK1_CHUNKAGG_OUTPUT_DIR`
- `TASK1_CHUNKAGG_FINETUNE_DATA_DIR`
- `TASK1_MAX_DOCUMENT_CHUNKS`
- `TASK1_DOCUMENT_CHUNK_LENGTH`
- `TASK1_CHUNK_MICROBATCH_SIZE`
- `TASK1_RETRIEVAL_BATCH_SIZE`

### 2. 訓練

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_train.sh"
```

### 3. 產生 embeddings

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_infer.sh"
```

### 4. 產生 ranking

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_rank.sh"
```

若要切換 query embeddings 來源，可直接覆寫：

```bash
TASK1_CHUNKAGG_QUERY_EMB_SOURCE=processed_new \
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_rank.sh"
```

## 常見輸出

- `TASK1_CHUNKAGG_OUTPUT_DIR/`
- `processed/processed_document_<MODEL_NAME>_embeddings*.pkl`
- `processed_new/processed_new_document_<MODEL_NAME>_embeddings*.pkl`
- ranking TREC 檔

這個目錄的流程獨立於 repo root 的 Task 1 wrapper；若要走主流程，請參考 [../modernBert-fp/README.md](../modernBert-fp/README.md)。
