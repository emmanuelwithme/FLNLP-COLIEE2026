# modernBert/fine_tune

[English](README.en.md) | [Task 1 總覽](../../README.md)

這個目錄保存的是較早期的 modernBERT fine-tune 程式與 BM25 hard-negative 工具。它不是目前維護中的 Task 1 主訓練流程；目前主流程已移到 `../../modernBert-fp/`。

不過，這個目錄仍然有一部分是現在 workflow 會用到的：

- `create_bm25_hard_negative_data_top100_random15.py`
  `run_pre_finetune_2026.sh` 目前就是用這支腳本生成 dense fine-tune 前置的 BM25 hard negatives。

## 這個目錄的定位

### 仍在使用的部分

- BM25 hard-negative 生成工具
- 舊版對比式訓練相關模組，供歷史流程參考

### 不建議當成目前主流程的部分

- `fine_tune.py`
- `fine_tune_noprojector.py`
- 這些舊訓練入口不應取代 `modernBert-fp/fine_tune/fine_tune.py`

## 檔案導覽

- `create_bm25_hard_negative_data.py`
  舊版 BM25 hard-negative 生成腳本。
- `create_bm25_hard_negative_data_top100_random15.py`
  目前維護流程仍在使用的版本，會從 BM25 top-100 中隨機抽 15 個 negatives。
- `create_config.py`
  舊版設定檔生成工具。
- `fine_tune.py`
  舊版對比式訓練腳本。
- `fine_tune_noprojector.py`
  舊版不含 projector 的對比式訓練腳本。
- `modernbert_contrastive_model.py`
  舊版對比模型定義。

## 目前實際會怎麼用到這個目錄

如果你走 repo root 的 Task 1 主流程：

```bash
bash run_pre_finetune_2026.sh
```

其中有一步就是：

```bash
python "Legal Case Retrieval/modernBert/fine_tune/create_bm25_hard_negative_data_top100_random15.py"
```

這支腳本會在以下位置寫出前置檔案：

- `coliee_dataset/task1/<YEAR>/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_train.json`
- `coliee_dataset/task1/<YEAR>/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json`

這兩個檔案之後會被 `modernBert-fp/fine_tune/fine_tune.py` 當作 validation 與對照資料的一部分使用。

## 如果你只是想跑目前維護中的 Task 1

請直接看：

- [../../modernBert-fp/README.md](../../modernBert-fp/README.md)
- [../../lightgbm/README.md](../../lightgbm/README.md)

不需要直接操作這個目錄下的大部分舊訓練腳本。

## 補充說明

- 這個目錄的文件過去混合了舊版 modernBERT 流程與後來 `modernBert-fp` 的部分概念
- 現在重新整理後，原則上把它視為「歷史實驗 + 仍在使用的 BM25 hard-negative 工具」
- 若你要做新的主流程實驗，請以 `modernBert-fp/` 為準
