# FLNLP-COLIEE2026

[入口頁](README.md) | [English](README.en.md)

這個 repository 以 FLNLP 的 COLIEE Task 1 流程為主。

- Task 1 是這個 repository 的主要任務。
- 作者只參加 Task 1。
- Task 2 程式碼作為補充保留，不是主要重點。

## 快速開始

1. 建立環境。

```bash
conda env create -f environment.frozen.yml
conda activate FLNLP-COLIEE2026-WSL
```

2. 編輯 repo root 的 `.env`。

- 將資料路徑、模型路徑與執行參數放在 `.env`。
- 若只想單次覆寫，可在命令前加 shell 環境變數，優先於 `.env`。
- 共用模型目錄放在 `models/`。

3. 選擇任務流程。

- Task 1 前處理: `bash run_pre_finetune.sh`
- Task 1 dense 訓練: `python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"`
- Task 1 train/valid retrieval 檢查: `bash run_train_valid_inference_eval.sh`
- Task 1 test retrieval 匯出: `bash run_test_retrieval.sh`
- Task 1 LTR feature 與 fixed top-k 匯出: `bash run_ltr_feature_train_valid_test.sh`
- Task 1 LTR cutoff search 與最終 submission 匯出: `bash run_ltr_cutoff_postprocess.sh`
- Task 2 補充流程: `bash run_task2_finetune.sh`

## 文件索引

- 環境設定: [ENVIRONMENT.md](ENVIRONMENT.md)
- Task 1 總覽: [Legal Case Retrieval/README.md](Legal%20Case%20Retrieval/README.md)
- Task 1 dense retrieval: [Legal Case Retrieval/modernBert-fp/README.md](Legal%20Case%20Retrieval/modernBert-fp/README.md)
- Task 1 LightGBM rerank: [Legal Case Retrieval/lightgbm/README.md](Legal%20Case%20Retrieval/lightgbm/README.md)
- Task 2 paragraph 流程: [Legal Case Entailment by Mou/README.md](Legal%20Case%20Entailment%20by%20Mou/README.md)
