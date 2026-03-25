# FLNLP-COLIEE2026

[入口頁](README.md) | [English](README.en.md)

本專案是 FLNLP 針對 COLIEE 2026 Task 1 與 Task 2 的工作程式碼庫。目前有兩條維護中的主流程：

- Task 1: `Legal Case Retrieval/`
- Task 2: `Legal Case Entailment by Mou/`

以下目錄不屬於目前維護流程：

- `Legal Case Entailment/`
  這是保留的舊程式碼與舊 README，僅供參考，不視為可直接執行的流程。

## 文件導覽

- 環境設定: [ENVIRONMENT.md](ENVIRONMENT.md)
- Task 1 總覽: [Legal Case Retrieval/README.md](Legal%20Case%20Retrieval/README.md)
- Task 1 dense encoder 主流程: [Legal Case Retrieval/modernBert-fp/README.md](Legal%20Case%20Retrieval/modernBert-fp/README.md)
- Task 1 LightGBM rerank / submission 主流程: [Legal Case Retrieval/lightgbm/README.md](Legal%20Case%20Retrieval/lightgbm/README.md)
- Task 1 chunk aggregation 實驗: [Legal Case Retrieval/modernBert-fp-chunkAgg/README.md](Legal%20Case%20Retrieval/modernBert-fp-chunkAgg/README.md)
- Task 1 舊版 modernBERT fine-tune 與 BM25 hard negative 工具: [Legal Case Retrieval/modernBert/fine_tune/README.md](Legal%20Case%20Retrieval/modernBert/fine_tune/README.md)
- Task 2 維護中流程: [Legal Case Entailment by Mou/README.md](Legal%20Case%20Entailment%20by%20Mou/README.md)

## 專案結構

- `Legal Case Retrieval/`
  COLIEE Task 1 的前處理、dense retrieval、LightGBM rerank、scope filter 與 submission 匯出流程。
- `Legal Case Entailment by Mou/`
  COLIEE Task 2 維護中的 paragraph-level ModernBERT fine-tune 流程。
- `coliee_dataset/`
  資料集根目錄。Task 1 與 Task 2 原始資料與中間產物預設放在這裡。
- `run_*.sh`
  根目錄的工作流程腳本。Task 1 主要是 2026 年度專用 wrapper；Task 2 wrapper 會讀 `.env`。
- `environment.frozen.yml`
  維護中流程的 conda + pip 凍結環境記錄。

## 建議開始方式

### 1. 建立環境

```bash
conda env create -f environment.frozen.yml
conda activate FLNLP-COLIEE2026-WSL
```

若環境已存在，要同步版本：

```bash
conda env update -n FLNLP-COLIEE2026-WSL -f environment.frozen.yml --prune
conda activate FLNLP-COLIEE2026-WSL
```

### 2. 準備資料

Task 1 預設資料根目錄：

```text
coliee_dataset/task1/2026/
```

常見內容：

- `task1_train_files_2026/`
- `task1_test_files_2026/`
- `task1_train_labels_2026.json`
- `task1_test_no_labels_2026.json`

Task 2 預設資料根目錄：

```text
coliee_dataset/task2/task2_train_files_2026/
```

常見內容：

- `cases/`
- `task2_train_labels_2026.json`

### 3. 依任務進入主流程

Task 1 建議順序：

1. 執行 `bash run_pre_finetune_2026.sh`
2. 依需要訓練或使用既有 dense model
3. 執行 `bash run_train_valid_inference_eval_2026.sh`
4. 若只要 dense retrieval test submission，執行 `bash run_test_retrieval_2026.sh`
5. 若要完整 LightGBM rerank submission，執行 `bash run_ltr_feature_train_valid_test_2026.sh`
6. 若只想重做 cutoff grid search，執行 `bash run_ltr_cutoff_postprocess_2026.sh`

Task 2 建議順序：

1. 設定 `.env`
2. 執行 `bash run_task2_finetune.sh`

## 重要設定說明

### `.env` 與 wrapper 腳本不是完全同一層

- `Legal Case Retrieval/lcr/task1_paths.py` 會自動讀 repo root 的 `.env`
- 但多數 Task 1 根目錄 wrapper 是 2026 專用腳本，且不會主動 `source .env`
- 也就是說，Task 1 Python 模組會看 `.env`，但 `run_pre_finetune_2026.sh`、`run_train_valid_inference_eval_2026.sh`、`run_test_retrieval_2026.sh` 這些 wrapper 仍以腳本內設定為主
- Task 2 的 `run_task2_finetune.sh` 會讀 `.env`

### 維護中與 legacy 的邊界

- Task 1 維護中主流程以 `Legal Case Retrieval/modernBert-fp/` 與 `Legal Case Retrieval/lightgbm/` 為主
- `Legal Case Retrieval/modernBert/` 保留了舊版訓練程式與仍在使用的 BM25 hard-negative 生成工具
- Task 2 只維護 `Legal Case Entailment by Mou/`

## Upstream Reference

本專案最初源自 THUIR COLIEE 2023 公開程式碼，之後已針對訓練、推論、資料流程與整體結構進行大量重構。

保留作為上游參考的公開資源：

- THUIR COLIEE 2023 repository: <https://github.com/CSHaitao/THUIR-COLIEE2023>
- SAILER: Structure-aware Pre-trained Language Model for Legal Case Retrieval
- THUIR@COLIEE 2023: Incorporating Structural Knowledge into Pre-trained Language Models for Legal Case Retrieval
- THUIR@COLIEE 2023: More Parameters and Legal Knowledge for Legal Case Entailment
