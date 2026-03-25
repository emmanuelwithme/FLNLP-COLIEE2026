# Legal Case Entailment by Mou

[English](README.en.md) | [回到根目錄](../README.md)

這個目錄提供 Task 2 的 paragraph-level 資料準備、統計與 ModernBERT fine-tune。這部分作為補充程式碼保留，repo 的主要重點仍是 Task 1。

## 任務格式

- query: `cases/<qid>/entailed_fragment.txt`
- candidates: `cases/<qid>/paragraphs/*.txt`
- labels: `task2_train_labels_<YEAR>.json`

資料準備後：

- 每個 query 會寫成一個 `processed_queries/<qid>.txt`
- 每個候選段落會寫成一個 `processed_candidates/<caseid><paragraphid>.txt`
- label 會攤平成 paragraph-level 正例

## 一鍵流程

```bash
bash run_task2_finetune.sh
```

這支 wrapper 會：

- 讀取 repo root `.env`
- 啟用 `CONDA_ENV_NAME`
- 建立 paragraph-level 資料
- 視設定產生統計
- 執行 `fine_tune_task2.py`

## 手動分步流程

### 1. 準備資料

```bash
python "Legal Case Entailment by Mou/prepare_task2_paragraph_data.py"
```

主要輸入：

- `COLIEE_TASK2_DIR/cases/`
- `COLIEE_TASK2_DIR/TASK2_LABELS_FILENAME`

主要輸出：

- `COLIEE_TASK2_PREPARED_DIR/processed_queries/`
- `COLIEE_TASK2_PREPARED_DIR/processed_candidates/`
- `COLIEE_TASK2_PREPARED_DIR/query_candidates_map.json`
- `COLIEE_TASK2_PREPARED_DIR/train_qid.tsv`
- `COLIEE_TASK2_PREPARED_DIR/valid_qid.tsv`
- `COLIEE_TASK2_PREPARED_DIR/finetune_data/`

### 2. 產生統計

```bash
python "Legal Case Entailment by Mou/analyze_task2_stats.py"
```

輸出：

- `stats/summary.json`
- `stats/relevant_count_distribution.csv`
- `stats/query_token_length_hist.png`
- `stats/candidate_token_length_hist.png`

### 3. 訓練模型

```bash
python "Legal Case Entailment by Mou/fine_tune_task2.py"
```

輸出：

- `TASK2_OUTPUT_DIR/`

## 常用變數

資料：

- `COLIEE_TASK2_YEAR`
- `COLIEE_TASK2_ROOT`
- `COLIEE_TASK2_DIR`
- `COLIEE_TASK2_PREPARED_DIR`
- `TASK2_LABELS_FILENAME`

初始化與輸出：

- `TASK2_INIT_MODEL_ROOT`
- `TASK2_INIT_CHECKPOINT`
- `TASK2_INIT_METRIC`
- `TASK2_INIT_METRIC_MODE`
- `TASK2_OUTPUT_DIR`
- `TASK2_RESUME_CHECKPOINT`

訓練：

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

wrapper 控制：

- `TASK2_MODE`
- `TASK2_SKIP_STATS`

## 測試模式

```bash
TASK2_MODE=test bash run_task2_finetune.sh
```

測試模式會使用較小的 query 子集，並在輸出目錄名稱加上 `_test`。

repo root `.env` 會提供預設值，shell 變數可做單次覆寫。
