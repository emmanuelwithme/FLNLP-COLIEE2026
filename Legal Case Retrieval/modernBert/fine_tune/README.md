# modernBert/fine_tune

[English](README.en.md) | [Task 1 總覽](../../README.md)

這個目錄包含 BM25 hard-negative 產生器與一組額外的對比式訓練工具。

## 主要用途

repo root 的 Task 1 前處理流程會呼叫：

```bash
python "Legal Case Retrieval/modernBert/fine_tune/create_bm25_hard_negative_data_top100_random15.py"
```

這支腳本會把 BM25 train / valid 結果轉成 dense fine-tune 需要的對比式資料。

## 主要檔案

- `create_bm25_hard_negative_data_top100_random15.py`: 從 BM25 top-100 抽樣 15 個 negatives。
- `create_bm25_hard_negative_data.py`: 另一個 hard-negative 產生器。
- `create_config.py`: 訓練設定檔工具。
- `fine_tune.py`: 對比式訓練入口。
- `fine_tune_noprojector.py`: 不含 projector 的訓練入口。
- `modernbert_contrastive_model.py`: 模型定義。

## hard-negative 產生器輸入

- BM25 train ranking
- BM25 valid ranking
- train labels
- valid labels

repo wrapper 的對應檔案來自：

- `TASK1_BM25_DIR/output_bm25_train.tsv`
- `TASK1_BM25_DIR/output_bm25_valid.tsv`
- `TASK1_TRAIN_SPLIT_LABELS_PATH`
- `TASK1_VALID_SPLIT_LABELS_PATH`

## hard-negative 產生器輸出

- `TASK1_FINETUNE_DATA_DIR/contrastive_bm25_hard_negative_top100_random15_train.json`
- `TASK1_FINETUNE_DATA_DIR/contrastive_bm25_hard_negative_top100_random15_valid.json`

常用參數：

- `--top-k`
- `--max-negatives`
- `--random-seed`

repo root wrapper 對應的環境變數：

- `TASK1_HARD_NEG_TOPK`
- `TASK1_HARD_NEG_MAX_NEGATIVES`
- `TASK1_HARD_NEG_SEED`

## 直接執行範例

```bash
python "Legal Case Retrieval/modernBert/fine_tune/create_bm25_hard_negative_data_top100_random15.py" \
  --bm25-train-path "./coliee_dataset/task1/2026/lht_process/BM25/output_bm25_train.tsv" \
  --bm25-valid-path "./coliee_dataset/task1/2026/lht_process/BM25/output_bm25_valid.tsv" \
  --train-labels-path "./coliee_dataset/task1/2026/task1_train_labels_2026_train.json" \
  --valid-labels-path "./coliee_dataset/task1/2026/task1_train_labels_2026_valid.json" \
  --train-output-path "./coliee_dataset/task1/2026/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_train.json" \
  --valid-output-path "./coliee_dataset/task1/2026/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json" \
  --top-k 100 \
  --max-negatives 15 \
  --random-seed 289
```

Task 1 的主要 dense 訓練與推論流程請參考：

- [../../modernBert-fp/README.md](../../modernBert-fp/README.md)
- [../../lightgbm/README.md](../../lightgbm/README.md)
