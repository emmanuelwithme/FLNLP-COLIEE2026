# modernBert-fp-chunkAgg

這個目錄提供 ModernBERT 的 chunk aggregation 版本，目標是把原本單段 `4096` tokens 的 document encoder，改成最多 `3` 個 `4096-token` chunks，並在 chunk-level 用一層 transformer block 聚合成單一文件向量。

目前包含：
- `fine_tune/fine_tune.py`：對比式 fine-tune，query / positive / negative 全部共用同一套 3-chunk encoder
- `inference.py`：用最佳 checkpoint 產生 candidate / query embeddings
- `similarity_and_rank.py`：讀取 embeddings 後輸出 train / valid 的 dot、cos 排名
- `fine_tune/modernbert_contrastive_model.py`：chunkAgg 模型載入器，供推論共用
- `.env`：本目錄的主要設定檔
- `run_train.sh`：從 repo root 正確啟動 fine-tune
- `run_infer.sh`：從最佳 checkpoint 產生 embeddings
- `run_rank.sh`：讀 embeddings 後輸出 train / valid 排名

## 編碼邏輯
文件與 query 都使用相同編碼流程：
1. 整篇文本先做一次 tokenizer tokenization
2. 每個 chunk 最多保留 `TASK1_DOCUMENT_CHUNK_LENGTH` tokens
3. 優先往回找最近句尾 token 邊界切分
4. 每篇文件最多保留 `TASK1_MAX_DOCUMENT_CHUNKS` 個 chunks
5. 超過總長度上限時，後面內容直接截斷
6. 每個 chunk 取 ModernBERT 的 `[CLS]`，先經過 projector
7. 加上 learnable `[DOC]` token 與 chunk position embedding
8. 經過 1-layer pre-norm transformer block 融合
9. 取 `[DOC]` token 當最終文件向量並做 L2 normalize

目前預設行為：
- `inference.py` 會同時產生 `processed` 與 `processed_new` 兩份 embeddings
- `similarity_and_rank.py` 預設 query / candidate 都使用 `processed`
- 若要切回 THUIR-style query，可把 query embeddings 改成 `processed_new`

## `.env`
這個目錄下的 `.env` 會被以下程式自動讀取：
- `fine_tune/fine_tune.py`
- `inference.py`
- `similarity_and_rank.py`
- `fine_tune/modernbert_contrastive_model.py`

行為規則：
- 如果 shell 內已經存在同名環境變數，shell 內的值優先
- `.env` 只補未設定的值
- 不需要手動 `source .env`

## 重要變數
常用變數如下，詳細可直接看 `.env` 內的中文註解。

### Chunk encoder
- `TASK1_MAX_DOCUMENT_CHUNKS`：每篇文件最多保留幾個 chunks
- `TASK1_DOCUMENT_CHUNK_LENGTH`：每個 chunk 的 token 上限
- `TASK1_CHUNK_MICROBATCH_SIZE`：encoder 每次前向送幾個 chunks，顯存不足時優先調小
- `TASK1_CHUNKAGG_ENABLE_TF32`：是否開啟 TF32 加速 matmul
- `TASK1_CHUNKAGG_TEXT_CACHE_SIZE`：原始文本快取數量，降低重複讀檔成本
- `TASK1_CHUNKAGG_CHUNK_CACHE_SIZE`：chunk tokenization 快取數量，降低 CPU tokenizer 開銷
- `TASK1_CHUNKAGG_PIN_MEMORY`：dataloader 是否使用 pinned memory
- `TASK1_CHUNKAGG_PERSISTENT_WORKERS`：dataloader workers 是否常駐
- `TASK1_INIT_TEMPERATURE`：對比式學習初始溫度
- `TASK1_RETRIEVAL_BATCH_SIZE`：retrieval / inference 的文件 batch size

### Training
- `TASK1_CHUNKAGG_QUICK_TEST`：是否只跑極少量資料
- `TASK1_CHUNKAGG_SCOPE_FILTER`：是否使用 query-specific candidate scope
- `TASK1_CHUNKAGG_TRAIN_BATCH_SIZE`
- `TASK1_CHUNKAGG_GRAD_ACCUM_STEPS`
- `TASK1_CHUNKAGG_NUM_EPOCHS`
- `TASK1_CHUNKAGG_ENCODER_LR`
- `TASK1_CHUNKAGG_FUSION_LR`
- `TASK1_CHUNKAGG_TEMPERATURE_LR`
- `TASK1_CHUNKAGG_SAMPLING_TEMPERATURE`
- `TASK1_CHUNKAGG_UPDATE_FREQUENCY`

### Inference / ranking
- `TASK1_CHUNKAGG_MODEL_NAME`：embeddings 檔名與 ranking 輸出資料夾名稱
- `TASK1_CHUNKAGG_CANDIDATE_DIR`
- `TASK1_CHUNKAGG_QUERY_DIR`
- `TASK1_CHUNKAGG_CAND_EMB_SOURCE`：`processed` 或 `processed_new`，ranking 預設 `processed`
- `TASK1_CHUNKAGG_QUERY_EMB_SOURCE`：`processed` 或 `processed_new`，ranking 預設 `processed`
- `TASK1_CHUNKAGG_CAND_EMB_PATH`
- `TASK1_CHUNKAGG_QUERY_EMB_PATH`
- `TASK1_CHUNKAGG_SCOPE_PATH`

### Path override
- `TASK1_CHUNKAGG_BASE_ENCODER_DIR`：continued pretraining 後的 backbone checkpoint
- `TASK1_CHUNKAGG_OUTPUT_DIR`：Trainer 輸出 checkpoint 目錄
- `TASK1_CHUNKAGG_FINETUNE_DATA_DIR`：adaptive negatives 與 retrieval artifacts 目錄

## 使用方式
建議從 repo root 執行。

### 1. 直接 fine-tune
先檢查並修改 `modernBert-fp-chunkAgg/.env`，至少確認以下幾個值：
- `TASK1_CHUNKAGG_OUTPUT_DIR`
- `TASK1_CHUNKAGG_MODEL_NAME`
- `TASK1_CHUNK_MICROBATCH_SIZE`
- `TASK1_RETRIEVAL_BATCH_SIZE`

然後直接執行：
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

## 建議調參順序
1. 先確認 `.env` 內的 `TASK1_CHUNKAGG_OUTPUT_DIR`、`TASK1_CHUNKAGG_MODEL_NAME` 是否符合你的實驗命名
2. 顯存不夠時，先調小 `TASK1_CHUNK_MICROBATCH_SIZE`
3. 如果 retrieval 太慢，再調整 `TASK1_RETRIEVAL_BATCH_SIZE`
4. GPU 使用率偏低時，可確認 `TASK1_CHUNKAGG_ENABLE_TF32=1`，並逐步增加 `TASK1_CHUNKAGG_TEXT_CACHE_SIZE` / `TASK1_CHUNKAGG_CHUNK_CACHE_SIZE`
5. 若訓練不穩，再調整 `TASK1_CHUNKAGG_FUSION_LR` 與 `TASK1_INIT_TEMPERATURE`

## 備註
- 目前句尾切分是 heuristic，不是完整法律 sentence segmenter
- 訓練、adaptive negative sampling、retrieval evaluation、inference 全都共用同一套 chunk encoder
- loss 只保留 document-level InfoNCE / CrossEntropy，不含 auxiliary loss
