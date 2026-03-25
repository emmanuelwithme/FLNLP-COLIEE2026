# Legal Case Entailment by Mou

[English](README.en.md) | [回到根目錄](../README.md)

這個目錄是目前維護中的 COLIEE Task 2 流程，也是 repo 內唯一建議使用的 Task 2 主流程。

`Legal Case Entailment/` 是保留的舊資料夾，不屬於目前維護範圍，本 README 也不依賴那個目錄。

## 任務形式

這套流程把 Task 2 整理成 paragraph-level retrieval / matching 問題：

- query:
  `cases/<qid>/entailed_fragment.txt`
- candidates:
  `cases/<qid>/paragraphs/*.txt`
- positives:
  來自 `task2_train_labels_<YEAR>.json`

資料展平後的 ID 規則：

- query id 使用三位數 case id，例如 `001`
- candidate id 由 `case_id + paragraph_id` 組成，例如 `001003`

## 建議使用方式

### 一鍵流程

從 repo root 執行：

```bash
bash run_task2_finetune.sh
```

這是目前最建議的入口，因為它會：

- 讀取 repo root `.env`
- 啟用 `CONDA_ENV_NAME`
- 檢查 Task 2 原始資料是否存在
- 執行 paragraph-level 資料準備
- 視設定產生資料統計
- 執行 ModernBERT fine-tune

### 手動分步流程

```bash
python "Legal Case Entailment by Mou/prepare_task2_paragraph_data.py"
python "Legal Case Entailment by Mou/analyze_task2_stats.py"
python "Legal Case Entailment by Mou/fine_tune_task2.py"
```

第二步 statistics 可選擇跳過。

## 輸入資料格式

預設原始資料位置：

```text
coliee_dataset/task2/task2_train_files_<YEAR>/
```

必要內容：

- `cases/`
- `task2_train_labels_<YEAR>.json`

每個 case 目錄預期結構：

```text
cases/<qid>/
  entailed_fragment.txt
  paragraphs/
    001.txt
    002.txt
    ...
```

## `prepare_task2_paragraph_data.py` 會做什麼

這一步會把原始 case-level 結構整理成訓練可直接使用的 paragraph-level 資料集。

主要輸出目錄預設是：

```text
Legal Case Entailment by Mou/data/task2_<YEAR>_prepared/
```

常見輸出：

- `processed_queries/`
- `processed_candidates/`
- `query_candidates_map.json`
- `task2_train_labels_<YEAR>_flat.json`
- `task2_train_labels_<YEAR>_flat_train.json`
- `task2_train_labels_<YEAR>_flat_valid.json`
- `train_qid.tsv`
- `valid_qid.tsv`
- `finetune_data/contrastive_task2_random15_valid.json`
- `prepare_stats.json`

資料切分規則：

- train / valid 預設比例 `0.8 / 0.2`
- 預設 split seed `42`
- validation negatives 預設每筆抽 `15` 個

## `analyze_task2_stats.py` 會做什麼

這一步會計算 prepared dataset 的統計資訊，使用 `answerdotai/ModernBERT-base` tokenizer。

常見輸出：

- `stats/relevant_count_distribution.csv`
- `stats/query_token_length_distribution.csv`
- `stats/candidate_token_length_distribution.csv`
- `stats/relevant_count_distribution.png`
- `stats/query_token_length_hist.png`
- `stats/candidate_token_length_hist.png`
- `stats/summary.json`

## `fine_tune_task2.py` 會做什麼

這是維護中的 paragraph-level ModernBERT fine-tune 入口。

主要特性：

- query 是 `entailed_fragment`
- candidate 是同 case 下的 paragraphs
- negatives 由模型相似度做 adaptive sampling
- validation retrieval 會計算 global F1 / precision / recall
- best checkpoint selection 仍以 validation top-1 F1 為主

常見輸出：

- `modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_2026_para/` 或你在 `.env` 指定的 `TASK2_OUTPUT_DIR`
- `tb/`
- 每個 epoch 的 similarity scores
- adaptive negative JSON

## 常用環境變數

### 基本設定

- `CONDA_ENV_NAME`
- `COLIEE_TASK2_YEAR`
- `COLIEE_TASK2_ROOT`
- `COLIEE_TASK2_DIR`
- `COLIEE_TASK2_PREPARED_DIR`

### 初始化模型與輸出

- `TASK2_INIT_MODEL_ROOT`
- `TASK2_INIT_CHECKPOINT`
- `TASK2_INIT_METRIC`
- `TASK2_INIT_METRIC_MODE`
- `TASK2_OUTPUT_DIR`
- `TASK2_RESUME_CHECKPOINT`

### 訓練與評估

- `TASK2_EVAL_TOPK`
- `TASK2_NUM_TRAIN_EPOCHS`
- `TASK2_MAX_STEPS`
- `TASK2_LOGGING_STEPS`
- `TASK2_SAVE_TOTAL_LIMIT`
- `TASK2_EARLY_STOPPING_PATIENCE`
- `TASK2_TRAIN_BATCH_SIZE`
- `TASK2_EVAL_BATCH_SIZE`
- `TASK2_GRAD_ACCUM_STEPS`
- `TASK2_RETRIEVAL_BATCH_SIZE`
- `TASK2_RETRIEVAL_MAX_LENGTH`

### 效能與資料載入

- `TASK2_ENABLE_TF32`
- `TASK2_GRADIENT_CHECKPOINTING`
- `TASK2_CACHE_TEXTS`
- `TASK2_DATALOADER_NUM_WORKERS`
- `TASK2_DATALOADER_PIN_MEMORY`
- `TASK2_DATALOADER_PERSISTENT_WORKERS`

### 控制 wrapper 行為

- `TASK2_SKIP_STATS`
- `TASK2_MODE`

## Test mode

把 `.env` 或 shell 設成：

```bash
TASK2_MODE=test
```

效果：

- 只取較小的 train / valid query 子集
- 仍保留 adaptive negative sampling
- 輸出目錄會自動加上 `_test`，避免覆蓋 full run

常用 test mode 參數：

- `TASK2_TEST_SEED`
- `TASK2_TEST_TRAIN_QUERY_LIMIT`
- `TASK2_TEST_VALID_QUERY_LIMIT`
- `TASK2_TEST_NUM_TRAIN_EPOCHS`
- `TASK2_TEST_MAX_STEPS`
- `TASK2_TEST_LOGGING_STEPS`
- `TASK2_TEST_SAVE_TOTAL_LIMIT`
- `TASK2_TEST_EARLY_STOPPING_PATIENCE`

要切回完整模式：

```bash
TASK2_MODE=full
```

## 重要補充

### 1. 這個 wrapper 會讀 `.env`

`run_task2_finetune.sh` 會先讀 repo root `.env`，再保留呼叫者事先傳入的 shell 環境變數覆寫。所以 Task 2 的 wrapper 與 Task 1 的 wrapper 行為不同。

### 2. `TASK2_EVAL_TOPK` 目前實作上仍以 top-1 選最佳模型

程式雖然會報告 top-1 與 top-2 retrieval metrics，但最佳 checkpoint 與 early stopping 仍以 validation top-1 F1 為主。

### 3. 不要把 `Legal Case Entailment/` 當成目前流程的一部分

目前 Task 2 維護中流程只看這個資料夾，不要混用舊目錄內的腳本。
