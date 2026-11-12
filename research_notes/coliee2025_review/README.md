# COLIEE／ModernBERT 技術稽核與驗證紀錄

導覽更新：2026-11-12（Asia/Taipei）。這裡保留可查核的研究摘要與歷史證據；目前程式設定及操作以 [ModernBERT BM25 指南](../../Legal%20Case%20Retrieval/modernBert-bm25/README.md) 為準。

## 閱讀順序

| 紀錄 | 用途與限制 |
| --- | --- |
| [現有實驗稽核工具](audit_existing_runs.py) | 讀取保存的 samples、Trainer state、TensorBoard 與抽樣分布；不執行模型訓練 |
| [2025 保存實驗摘要](evidence/audit.json) | 靜態／動態模型與資料的歷史稽核快照，不能視為今天重新推論的結果 |
| [Loss／F1 圖 PNG](evidence/coliee2025_saved_curves.png)、[PDF](evidence/coliee2025_saved_curves.pdf) | 僅繪製實際保存的指標；不存在的逐 epoch F1 不補畫 |
| [2026 Random／Adaptive 控制變因稽核](CONTROL_AUDIT_2026-11-03.md) | Scope、取樣、評分分母、初始化與續訓差異，以及歷史計數重算 |
| [SAILER／GradCache v2 效率紀錄](sailer_efficiency_2026-11-03/README.md) | RTX 4080 SUPER 的歷史微基準、梯度與 GPU 續訓結果；不是 v3 新訓練績效 |
| [Shuffle＋repair v3 大量統計](shuffle_repair_2026-11-03/README.md) | 10000 輪新分組及資料流驗證；不能據此推論 valid F1 必然提高 |
| [2026-11-06 整理驗證](evidence/cleanup_validation_2026-11-06.json) | 本次 CPU 測試、資料 preflight、CLI 與短程工具流程；未重跑 GPU 訓練或歷史大量統計 |
| [2026-11-12 輕量訓練快照](../../reports/modernbert-bm25/training_records/README.md) | 三個 BM25 run 的主要 TensorBoard、逐輪指標、設定及取樣稽核摘要；不含權重 |

日期資料夾記錄當時的測試範圍、實作版本與限制，不代表目前有相同訓練程序正在執行。涉及歷史 run 的絕對路徑與模型目錄需在本機實際存在，才能重新查核。

## 重跑工具

在 repository 根目錄、既有 `FLNLP-COLIEE2026-WSL` 環境執行：

```bash
python -m unittest discover -s "Legal Case Retrieval/modernBert-bm25/tests" -v
python research_notes/coliee2025_review/audit_existing_runs.py --plots --check-batch
python -u "Legal Case Retrieval/modernBert-bm25/validate_grouping.py"
```

第一個指令預設只跑一般測試，GPU 測試需要額外 opt-in。第二個指令會更新 `evidence/audit.json` 及曲線圖，需要本機的歷史模型與 TensorBoard 紀錄。第三個指令使用真實 pairs 做 CPU 統計；新結果寫入 `reports/modernbert-bm25/grouping/<UTC-run-id>/`，不覆寫本資料夾的歷史統計。

`validate_grouping.py --flow-only` 另需 legacy run 的 mask manifest、設定與 samples；GPU benchmark 及完整模型測試的條件見 [測試說明](../../Legal%20Case%20Retrieval/modernBert-bm25/tests/README.md)。2026 Random／Adaptive 的 JSON 是歷史專項稽核快照，沒有宣稱可由一般 `audit_existing_runs.py` 完整重建。

## 版本控制範圍

依「保留必要訓練紀錄」的用途，Git 保留 `.env`、JSON 設定、程式、技術說明、整理後的稽核／統計摘要、曲線圖，以及 `reports/modernbert-bm25/training_records/` 的輕量訓練快照。

以下保留本機，但由 Git 忽略：

- `reports/modernbert-bm25/grouping/`、`benchmark/` 中的重複診斷輸出。
- `shuffle_repair_*/new_policy.json`、`legacy_policy.json` 的逐輪大量明細，以及本研究目錄的原始測試 log。
- 個人 `AI_HANDOFF_*.md`、`finish_plan_60_days.md`、`meeting_handbook.md` 草稿。
- Python／測試快取、未完成的 `summary.partial.json` 及診斷暫存檔。

原有資料集與 `models/` 等忽略規則保留。2026-11-12 已另行保存三個 run 的主要 TensorBoard 與小型 JSON；未匯出的原文、checkpoint、完整抽樣、排名及 embeddings 不會隨 Git clone 提供。
