# 輕量訓練紀錄

快照匯出日期：2026-11-12（Asia/Taipei）。這裡保存三個既有 BM25 run 的摘要與主要訓練曲線，總計約 0.84 MiB，不包含模型權重。匯出日期不代表模型在當天重新訓練，快照也不會自動隨原始 run 更新。

| 模式 | 保存目錄 |
| --- | --- |
| Pairwise，1 query／16 candidates | [BM25 epoch resampling](modernBERT_bm25_top100_epoch_resample15_2025/) |
| SAILER-style，legacy v2 分組 | [GradCache v2](modernBERT_bm25_top100_epoch_resample15_sailer_gc_2025/) |
| SAILER-style，shuffle＋repair v3 | [GradCache v3](modernBERT_bm25_top100_epoch_resample15_sailer_gc_shuffle_2025/) |

每個目錄保存：

- `summary.json`：最佳 valid F1、epoch、steps、AMP 更新摘要及資料 hash。
- `metrics_history.json`：epoch 0 與逐 epoch valid loss、F1、precision、recall。
- `run_config.json`：當次實際解析的設定，包含來源機器的絕對路徑；重新訓練仍應使用專案 `configs/` 中的設定。
- `run_manifest.json`：程式、模型版本、套件及執行環境資訊。
- `sampling_audits.json`：各 epoch 的取樣 hash 與覆蓋率等稽核摘要，不含完整 negatives 明細。
- `tb/`：原 run 主要 TensorBoard events，可查看 train loss、valid loss／F1 及 learning rate。
- `snapshot.json`：匯出日期、來源與逐檔 SHA-256，供確認快照內容。

在 repository 根目錄啟動：

```bash
tensorboard --logdir "reports/modernbert-bm25/training_records" --port 6006
```

各 run 的大量逐參數 gradient extras、完整抽樣資料、排名、embeddings、debug、optimizer／scheduler／scaler／RNG 均未複製。這些快照供查閱與備份，不能單獨用來推論或恢復訓練；續訓仍需原本 `models/` 下的完整 checkpoint。

往後要保存其他訓練，請在其完成或停止寫入後，另建有日期／來源的快照；不要用不同內容覆蓋這次匯出而仍沿用原來的 SHA-256。
