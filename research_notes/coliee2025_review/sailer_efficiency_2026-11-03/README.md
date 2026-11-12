# SAILER／GradCache 效率更新與重跑紀錄

日期：2026-11-03（Asia/Taipei）。實作識別：`modernBert-sailer-gradcache-v2`。

本檔為 v2 當時的歷史紀錄。2026-11-06 整理後，訓練入口預設使用 v3 分組，操作以 [SAILER 指南](../../../Legal%20Case%20Retrieval/modernBert-bm25/SAILER_GRADCACHE.md) 為準；新 benchmark 報告預設寫至 `reports/modernbert-bm25/benchmark/<UTC-run-id>/`，不覆寫本資料夾。

使用者中斷新SAILER／GradCache訓練後，要求增加效率、刪除剛產生的訓練產物，再以一個指令重新開始。本次未啟動正式長訓練，已完成真實模型基準、程式測試、文件與指定目錄清除。

## 最後採用的設定

```text
Logical batch：4個不同Query，4×64 masked InfoNCE
GradCache chunk：2
gradient_checkpointing：false
deduplicate_inputs：true
reference_compile：false
```

使用原始ModernBERT-base、4096長度上限、FP32weights＋FP16AMP、FlashAttention2；全部5421positivepairs、BM25top100每epoch抽15、LabelMask、full-validTop5microF1與早停規則都維持原定義。

本次沒有盲目把chunk調大。GradCache已限制同時保留的encoder圖，關閉另一層gradientcheckpointing可省去它額外的反向重算。小chunk仍控制顯存，在本機比大chunk更快。

另外新增：同batch相同截斷token輸入只編碼一次，以indexmapping還原完整68個邏輯輸入位置，再計算共同loss；梯度由autograd加總回共享向量。需要encoder／projector dropout為0，否則報錯。跨optimizer更新不重用舊向量。全部unique文字先一次tokenize，再分段padding；重複ID也只讀檔一次。

## RTX4080SUPER16GB 微基準

使用真正`answerdotai/ModernBERT-base`，每profile獨立程序；一輪warmup後，測兩個真實logicalbatches的完整GPU更新核心（cacheforward、loss、replay／backward、unscale、clip、fusedAdamWstep）。時間不含資料準備、完整valid、checkpoint寫檔；樣本數少，不能保證每epoch都有相同幅度改善。

另以68份不同真實案例的滿4096token輸入測顯存上限。該壓力測試只量tensor形狀成本，不是有相關性標註的訓練／績效實驗。

第一輪（gradientcheckpointing開啟）：

| Chunk | 輸入去重 | 更新秒數中位數 | 滿4096peakallocatedGiB | 結果 |
| --- | --- | ---: | ---: | --- |
| 2 | 關 | 6.034 | 3.141 | 基準 |
| 2 | 開 | 5.914 | 3.141 | 安全 |
| 4 | 開 | 6.137 | 3.848 | 安全，沒有較快 |
| 8 | 開 | 6.254 | 5.422 | 安全，沒有較快 |
| 16 | 開 | 6.130 | 8.562 | 安全，沒有較快 |
| 24 | 開 | 6.141 | 11.707 | 超過11.5GiB安全預算，不採用 |
| 32 | 開 | 未完成 | 未完成 | CUDA driver device-not-ready錯誤，不採用 |

Chunk32當次未完成，不能報其峰值或把錯誤確診成OOM。其後CUDA健康檢查與最終真實模型GPU整合測試均通過。正式config只允許1–16，預設benchmark也不會測24／32。

第二輪在相同流程重測基準，再比較關閉gradientcheckpointing：

| 設定 | 去重 | 更新秒數中位數 | 滿4096peakallocatedGiB |
| --- | --- | ---: | ---: |
| Chunk2、checkpointing開 | 關 | 5.994 | 3.141 |
| **Chunk2、checkpointing關** | **開** | **4.759** | **6.681** |
| Chunk4、checkpointing關 | 開 | 4.920 | 10.887 |

選擇chunk2／關checkpointing：本輪更新核心耗時減少約20.6%，更新吞吐約增加26%。這是相對原本chunk2＋checkpointing的GC版本，不是相對最早單queryBM25程式的同條件結論。PyTorchallocated不含桌面／驅動等全部GPU用量。

原始JSON見[第一輪](benchmark.json)、[第二輪](benchmark_no_checkpoint.json)及同目錄各profile。可重跑：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/benchmark_sailer.py" --without-checkpointing
```

它不儲存模型權重；會使用GPU、更新記憶體中的測試模型並保存微基準JSON，請不要與正式訓練同卡並跑。

## 最終驗證

- 一般測試：發現59項，55項一般測試全部通過，4項GPU依opt-in跳過。
- 新模式2項真實ModernBERT-baseGPU測試：全部通過，約119秒。
- 完整計算圖／GradCache：139個參數tensor，梯度relativeL2約8.26386e-8、loss相同。
- 新增去重測試：完整68位置重建正確，query與passage共享梯度加總與未去重計算相符。
- 真實4096長度、9原始pairs每輪補3，3logicalsteps／epoch，epoch0／1／2評估完整。
- Epoch1checkpoint-3中斷→epoch2checkpoint-6，optimizer／scheduler3→6，RNG恢復核對通過。
- Encoder／projector／temperature均更新；6attempts／6updates／0AMPskips。
- 獨立checkpoint重載之metrics及每query有序Top5與訓練內評估完全相同。
- 該短程訓練peakallocated約6.68070GiB，無OOM。

保存的機器可讀證據在[validation/](validation/)：`gradient_equivalence.json`、`sailer_integration_result.json`、`rng_verified.json`、debug 的 `summary.json`／`metrics_history.json`／`run_config.json`，以及原先 20 輪分組檢查快照。依 2026-11-12 的精簡備份規則，`cpu_tests.log`、`gpu_tests.log` 原始執行 log 只保留本機，由 Git 忽略；上述 JSON 與本頁摘要仍保留。`run_config.json` 中的舊 debug 權重路徑已清除；這些是驗證紀錄，不是可拿來續訓的 checkpoint。

## 清除與重新開始

已確認無訓練程序仍執行，並刪除：

```text
models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_2025/
```

包含該run的模型、訓練／驗證紀錄、TensorBoard、sampling、prepared與debug產物，合計14,478,108,573bytes。未備份舊訓練權重。新的benchmark／驗證報告保存在本資料夾，詳見[清除紀錄](cleanup.json)。

原本`models/modernBERT_bm25_top100_epoch_resample15_2025/`六輪模型保留；其summarySHA256在刪除前後相同。原始資料與官方base權重快取未動。

當時在 repository 根目錄、啟用 `FLNLP-COLIEE2026-WSL` 環境後使用以下命令；該入口現在預設對應 v3，不能把目前重跑結果稱為此處 v2 的相同分組實驗：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune_sailer.py" --no-debug
```

不加`--resume-from-checkpoint`。程式會重新建立prepared、執行epoch0，再正式訓練。此次執行速度優化不代表已驗證正式F1會提升。
