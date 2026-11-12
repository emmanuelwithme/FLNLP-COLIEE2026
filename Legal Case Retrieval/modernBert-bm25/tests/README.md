# 測試與 debug 的用途

此資料夾存放程式正確性測試，不是 COLIEE 官方 test split。平常訓練不會讀取此資料夾。

日常試跑使用訓練入口的 `--debug`：完整 ModernBERT-base、100 個真實 train pairs、20 個 valid queries、1000 candidates、2 epochs，與正式訓練共用程式。它用於檢查實際資料與 GPU 流程；單元測試則驗證不容易靠 loss 曲線看出的標籤、梯度及續訓契約。

## 一般測試

在 repository 根目錄、既有 Conda 環境執行：

```bash
python -m unittest discover -s "Legal Case Retrieval/modernBert-bm25/tests" -v
```

截至 2026-11-06，發現 72 項測試，68 項 CPU 測試通過，4 項 GPU 測試預設跳過。一般測試不下載權重、不啟動完整 encoder 訓練；數值測試使用小型張量或 stub 驗證運算與序列化。

| 測試檔 | 驗證內容 |
| --- | --- |
| `test_flow.py` | Top-100 邊界、取樣、worker 資料對齊、micro F1、早停與 checkpoint 防護 |
| `test_training.py` | 設定、debug 相容性、optimizer groups、非有限數值與資料接收檢查 |
| `test_sailer.py` | 4×64 loss、Label Mask、輸入去重、GradCache／完整圖梯度及 RNG 重播 |
| `test_batch_grouping.py` | 全部 pairs 保留、padding、不同 query 分組、修復終止及 legacy 相容性 |

大量真實資料的分組統計另由 [validate_grouping.py](../validate_grouping.py) 執行，規模與指令見 [GROUPING.md](../GROUPING.md)。

## 真實 ModernBERT-base GPU 測試

確認 GPU 可供測試使用後，依要檢查的路徑明確啟用：

```bash
BM25_RUN_GPU_TESTS=1 python -m unittest discover \
  -s "Legal Case Retrieval/modernBert-bm25/tests" -p "test_gpu_amp.py" -v
BM25_RUN_INTEGRATION=1 python -m unittest discover \
  -s "Legal Case Retrieval/modernBert-bm25/tests" -p "test_integration.py" -v
BM25_RUN_SAILER_GPU=1 python -m unittest discover \
  -s "Legal Case Retrieval/modernBert-bm25/tests" -p "test_sailer_gpu.py" -v
```

前兩個檔案各有一項測試，分別檢查原生 AMP 溢位恢復及 pairwise 完整模型續訓；SAILER 檔案有兩項測試，檢查真實 encoder 的梯度等價及 4096 長度的訓練／續訓／獨立重載。使用既有官方 base 權重與 RTX 4080 SUPER；產物隔離在對應 run 的 `_debug/`，不計入正式模型績效。

歷史 GPU 結果見 [pairwise 驗證紀錄](../VERIFICATION.md) 與 [SAILER 驗證紀錄](../SAILER_GRADCACHE_VERIFICATION.md)。2026-11-06 的整理僅重跑 CPU 測試與命令／資料檢查，未重新執行 GPU 訓練。
