# modernBert-fp-chunkAgg

[English](README.en.md) | [Task 1 總覽](../README.md)

這個目錄是 ModernBERT 的 chunk aggregation 版本，用來把原本單段 `4096` tokens 的文件編碼，改成最多 `3` 個 chunk，再用額外的聚合層把 chunk-level 表徵合成單一文件向量。

它是保留中的實驗路線，不是目前 Task 1 submission 的預設主流程；預設主流程仍以 `../modernBert-fp/` 為主。

## 目錄內容

- `fine_tune/fine_tune.py`
  對比式 fine-tune 入口，query / positive / negative 都共用同一套 chunk encoder。
- `inference.py`
  讀最佳 checkpoint 產生 embeddings。
- `similarity_and_rank.py`
  讀 embeddings 後輸出 train / valid 排名。
- `fine_tune/modernbert_contrastive_model.py`
  chunkAgg 模型定義。
- `.env`
  這個目錄自己的主要設定檔。
- `run_train.sh`
  從 repo root 啟動 fine-tune。
- `run_infer.sh`
  產生 embeddings。
- `run_rank.sh`
  產生排名。

## 文件編碼邏輯

文件與 query 都使用同一套 chunk 編碼流程：

1. 先對全文做 tokenizer tokenization
2. 每個 chunk 最多保留 `TASK1_DOCUMENT_CHUNK_LENGTH` tokens
3. 切分時會優先往回找句尾邊界
4. 每篇文件最多保留 `TASK1_MAX_DOCUMENT_CHUNKS` 個 chunks
5. 超過總長度上限時，後段內容直接截斷
6. 每個 chunk 取 ModernBERT 的 `[CLS]` 向量後先過 projector
7. 再加入 learnable `[DOC]` token 與 chunk position embedding
8. 經過 1-layer pre-norm transformer block 融合
9. 最後取 `[DOC]` token 作為文件向量並做 L2 normalize

## 預設行為

- `inference.py` 會同時產生 `processed` 與 `processed_new` embeddings
- `similarity_and_rank.py` 預設 query / candidate 都使用 `processed`
- 若要切回舊 THUIR-style query，可以把 query embeddings 改成 `processed_new`

## `.env` 載入規則

這個目錄下的 `.env` 會被以下程式自動讀取：

- `fine_tune/fine_tune.py`
- `inference.py`
- `similarity_and_rank.py`
- `fine_tune/modernbert_contrastive_model.py`

規則如下：

- shell 內若已經有同名環境變數，shell 值優先
- `.env` 只補尚未設定的值
- 不需要手動 `source .env`

## 重要變數

### chunk encoder

- `TASK1_MAX_DOCUMENT_CHUNKS`
- `TASK1_DOCUMENT_CHUNK_LENGTH`
- `TASK1_CHUNK_MICROBATCH_SIZE`
- `TASK1_CHUNKAGG_ENABLE_TF32`
- `TASK1_CHUNKAGG_TEXT_CACHE_SIZE`
- `TASK1_CHUNKAGG_CHUNK_CACHE_SIZE`
- `TASK1_CHUNKAGG_PIN_MEMORY`
- `TASK1_CHUNKAGG_PERSISTENT_WORKERS`
- `TASK1_INIT_TEMPERATURE`
- `TASK1_RETRIEVAL_BATCH_SIZE`

### training

- `TASK1_CHUNKAGG_QUICK_TEST`
- `TASK1_CHUNKAGG_SCOPE_FILTER`
- `TASK1_CHUNKAGG_TRAIN_BATCH_SIZE`
- `TASK1_CHUNKAGG_GRAD_ACCUM_STEPS`
- `TASK1_CHUNKAGG_NUM_EPOCHS`
- `TASK1_CHUNKAGG_ENCODER_LR`
- `TASK1_CHUNKAGG_FUSION_LR`
- `TASK1_CHUNKAGG_TEMPERATURE_LR`
- `TASK1_CHUNKAGG_SAMPLING_TEMPERATURE`
- `TASK1_CHUNKAGG_UPDATE_FREQUENCY`

### inference / ranking

- `TASK1_CHUNKAGG_MODEL_NAME`
- `TASK1_CHUNKAGG_CANDIDATE_DIR`
- `TASK1_CHUNKAGG_QUERY_DIR`
- `TASK1_CHUNKAGG_CAND_EMB_SOURCE`
- `TASK1_CHUNKAGG_QUERY_EMB_SOURCE`
- `TASK1_CHUNKAGG_CAND_EMB_PATH`
- `TASK1_CHUNKAGG_QUERY_EMB_PATH`
- `TASK1_CHUNKAGG_SCOPE_PATH`

### path override

- `TASK1_CHUNKAGG_BASE_ENCODER_DIR`
- `TASK1_CHUNKAGG_OUTPUT_DIR`
- `TASK1_CHUNKAGG_FINETUNE_DATA_DIR`

## 使用方式

建議從 repo root 執行。

### 1. fine-tune

先確認 `modernBert-fp-chunkAgg/.env` 至少以下值正確：

- `TASK1_CHUNKAGG_OUTPUT_DIR`
- `TASK1_CHUNKAGG_MODEL_NAME`
- `TASK1_CHUNK_MICROBATCH_SIZE`
- `TASK1_RETRIEVAL_BATCH_SIZE`

然後執行：

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_train.sh"
```

### 2. 產生 embeddings

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_infer.sh"
```

### 3. 產生排名

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_rank.sh"
```

若要把 query 改回 `processed_new`：

```bash
TASK1_CHUNKAGG_QUERY_EMB_SOURCE=processed_new \
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_rank.sh"
```

## 建議調整順序

1. 先確認輸出目錄與實驗命名
2. 顯存不足時先調小 `TASK1_CHUNK_MICROBATCH_SIZE`
3. 若 retrieval 太慢，再調整 `TASK1_RETRIEVAL_BATCH_SIZE`
4. GPU 使用率偏低時，可開啟 `TASK1_CHUNKAGG_ENABLE_TF32=1` 並增加 cache 參數
5. 若訓練不穩，再調整 fusion / temperature 相關學習率

## 備註

- 句尾切分目前是 heuristic，不是完整法律 sentence segmenter
- 訓練、adaptive negative sampling、retrieval evaluation、inference 都共用同一套 chunk encoder
- loss 目前只保留 document-level InfoNCE / CrossEntropy
