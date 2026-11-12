# SAILER-style GradCache 驗證紀錄

最新 v3 的四 query 隨機分組及大量 CPU 統計見 [GROUPING.md](GROUPING.md)。本檔保留 v1／v2 歷史 GPU 結果；2026-11-03 的分組更新與 2026-11-06 的整理均未重跑 GPU 測試。2026-11-06 完整測試集為 68 項 CPU 通過、4 項 GPU 按 opt-in 跳過。

下列為最初v1驗證的歷史紀錄。後續已於2026-11-03完成v2效率更新，並依使用者要求清除本模式舊訓練目錄（含下列debug權重）；最新設定、基準與獨立保存的驗證結果見[效率更新紀錄](../../research_notes/coliee2025_review/sailer_efficiency_2026-11-03/README.md)。請勿把下列3.14GiB或gradientcheckpointing設定當成v2現況。

日期：2026-11-03（Asia/Taipei）。實作：`modernBert-sailer-gradcache-v1`。

此文件記錄新功能實際完成的檢查；正式20epoch訓練尚未啟動。舊六epochBM25模型與原始資料保留。操作方式見 [SAILER_GRADCACHE.md](SAILER_GRADCACHE.md)。

## 真實資料與全部20輪抽樣檢查

使用現有2025processed快照：1341trainqueries、336validqueries、5421／1421uniquepositivepairs、7349candidates，資料hash：

```text
dada68cdbf9d094ea8e771bae16982c9a18b110e747d5bc1397c1682eb974662
```

完整7349篇已建立4096截斷tokenfingerprints，label-maskmanifesthash：

```text
cc6779f1e0d4856a44db29f6492af2c017189f654ada478434d8905101205fb0
```

已逐輪生成、檢查20epochs的分組與mask：每輪5421原始pairs完整保留、補3pairs、共5424pairinstances／1356logicalbatches；每批4個不同queryIDs、各1positive＋15negatives。每批mask均保有target與合法negative。Epoch1共遮罩7747個query–passagecells，之後依抽樣與分組變化。

可機器讀取的完整結果：

```text
models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_2025/prepared/sailer_preflight.json
models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_2025/prepared/label_mask_manifest.json
```

此處是驗證20輪**取樣／分組／mask**，不是宣稱已訓練20epochs。沒有更動資料切分、processed、BM25index或原始TREC。

## 一般測試

```bash
python -m unittest discover -s "Legal Case Retrieval/modernBert-bm25/tests" -v
```

結果：發現57項，**53項一般測試全部通過**，4項GPU測試依opt-in規則跳過。包含既有取樣／評估／早停／續訓防護，以及新增：

- 全部原始pairs保留、distinctquery分組、最後一批padding與可重現重抽。
- 多worker實際讀到各輪新資料，68段輸入順序與passagegroup位置一致。
- 4×64共同分母、targets `[0,16,32,48]`、maskedcolumn零梯度。
- 所有knownpositive／inputaliases、self／aliases、duplicate／invalid遮罩，target優先。
- GradCache與完整計算圖全部參數梯度一致；含dropout、TorchRNG重播及temperature。
- AMPscale只套一次，不對temperature或encoder重複scale。
- Config拒絕錯誤logicalbatch、chunk、accumulation；checkpoint保存／恢復compilepolicy。

代數／serialization單元測試採小型張量與stub驗證數學和檔案契約；以下GPU訓練與梯度驗證使用真正ModernBERT-base，沒有用tinyencoder替代。

## 真正 ModernBERT-base GPU 驗證

GPU：RTX4080SUPER16GB；官方base revision `8949b909ec900327062f0ebf497f51aef5e6f0c8`。
Torch2.6.0+cu126／Transformers4.52.4／Accelerate1.6.0／FlashAttention2.7.4.post1。

```bash
BM25_RUN_SAILER_GPU=1 python -u -m unittest discover \
  -s "Legal Case Retrieval/modernBert-bm25/tests" -p "test_sailer_gpu.py" -v
```

結果：**2項全部通過**，約130秒。本輪另兩項既有GPUAMP／舊pairwise整合測試未重跑；不能稱57項都在同次命令無skip執行。

### 1. 完整計算圖與 GradCache 梯度比對

使用相同原始ModernBERT-base、相同68份真實輸入、相同mask與4×64loss；為讓完整referencegraph也容納於16GB，此項輸入截斷128tokens，chunk=2，FP16autocast＋FP32weights，gradientscale128。

| 項目 | 實測 |
| --- | --- |
| 參數tensor比對數 | 139，包含encoder／projector／log_temperature |
| loss | 4.1415324211120605，兩路相同 |
| 全參數梯度relativeL2 | 8.26386011050019e-8 |
| 單參數最大relativeL2 | 約5.74e-7 |
| reference_compile | False |

Reference保留全部chunk計算圖後共同backward；GradCache只保留embeddings，再逐段重算backward。使用同樣AMPchunk邊界；CPU測試另以dropout檢查RNG一致。

結果檔：`models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_2025/_debug/gradient_equivalence.json`。

### 2. 4096長度、AMP、checkpoint、續訓與重載

成功的最終整合run：

```text
models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_2025/_debug/20261102T195640227078Z/
```

時間戳為UTC，對應台北2026-11-03。此整合測試使用9個真實trainpairs、4validqueries／256candidates；每輪補3pairs組成3logicalbatches。與正式共用`run_training`與GradCacheTrainer，保留4096上限、15negatives、FP16／FlashAttention2、同optimizer與worker流程。

| 項目 | 實測 |
| --- | --- |
| 實際觀察到的輸入長度 | 4096 |
| epoch0／1／2評估 | 全部保存 |
| epoch1中斷→續訓 | checkpoint-3→checkpoint-6 |
| Schedulersteps | 3→6，未重置 |
| Optimizerupdates | 3→6，所有參數步數一致 |
| AMP | 6attempts、6updates、0skips |
| Encoder／projector／temperature | epoch1→2權重均有改變 |
| 每輪received／planned | 完全一致，各12pairinstances |
| Epochsamplehash | 兩輪不同 |
| TorchCPU／CUDA、Python、NumPyRNG | 原生restore後逐項核對相符 |
| Patience | 0→0；本次epoch2F1改善，符合規則 |
| 重載bestcheckpoint | metrics與每query有序Top5均與訓練內評估相同 |
| 訓練步驟PyTorchpeakallocated | 3.1410GiB |
| 訓練步驟PyTorchpeakreserved | 3.2227GiB |

記憶體數字是本次短程**訓練步驟**的PyTorch統計，不含桌面／驅動等全部GPU占用，也不是宣稱整個驗證過程的全卡總峰值。沒有OOM；真實16GBGPU完成上述全流程。

本debugscope的F1從0到0.06451613（TP1），只驗證流程；不是正式336validqueries的效能，也不能推論正式模型必然提升。

主要證據：該run的`sailer_integration_result.json`、`rng_verified.json`、`summary.json`、`metrics_history.json`、`sampling/`、`checkpoint-*/gradcache_state.json`與`standalone/`。

## 測試找到並修正的問題

1. 本機ModernBERT可選referencecompiler在cache no-grad與grad路徑產生數值差：早期完整圖／GradCache整體梯度相差約2.45%。新模式關閉`reference_compile`後，誤差降至8.3e-8，保留FlashAttention2與gradientcheckpointing。
2. HF`ModernBertConfig.to_dict()`省略`reference_compile`，獨立重載原本會恢復auto-compile，使近同分排序變動。現於`contrastive_config.json`明確保存／恢復，獨立推論也依新模式固定False；最終完整重載測試通過。
3. Cache no-grad階段不需activationcheckpointing，現暫停後於replay恢復，避免無意義的requires_grad警告；不取消真正replay時的梯度與checkpointing。

以上修正在本專案實作，未修改Conda套件原始碼。

## 交付邊界

正式入口、debug、resume、checkpoint評估與文件已完成；正式輸出根目錄目前只有prepared衍生資料，沒有正式run_config／TB／checkpoint，使用者可直接依README的一個指令啟動。

本實驗同時新增in-batchloss、labelmask與GradCache執行方式，並固定compilepolicy；與先前BM25結果的差異不應全部歸因於單一機制。Mask能處理已知標註與完全相同的模型輸入，不能自動辨識所有未標註相關案例。
