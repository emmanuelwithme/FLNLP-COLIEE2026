# 重構版驗證紀錄

2026-11-06 更新：目前完整測試集發現 72 項，68 項 CPU 測試全部通過，4 項 GPU 測試依 opt-in 跳過；本次沒有重新執行模型 GPU 訓練。SAILER-style 與分組的規格另見 [SAILER_GRADCACHE.md](SAILER_GRADCACHE.md) 及 [GROUPING.md](GROUPING.md)。以下保留 pairwise 重構當時的驗證紀錄。

驗證日期：2026-10-20（Asia/Taipei）。
實作識別：`modernBert-fp-bm25-v2`，config version 2。

此文件記錄本次實際執行的檢查，不代表正式 20-epoch 訓練或正式 valid F1 已完成。
原本 `modernBert-fp/`、2025 輸入資料與 Conda 套件程式碼均未修改。
重構前的多階段 debug／smoke 執行器已移除，日常 debug 直接走正式訓練函式。

## 測試結果

### Debug 資料量更新

一般 debug 預設已改為 100 train pairs、20 valid queries、1000 candidates、2 epochs（共 50 steps）。
正式 config 不變；小範圍 GPU 續訓整合測試明確保留 9 pairs／4 queries／256 candidates，
與日常 debug 的預設資料量分開。舊 debug 續訓不會改變原資料範圍。
本次擴量以一般測試與真實資料範圍檢查驗證，尚未重跑擴量後的 GPU 訓練。

### 重構當時的完整 GPU 測試

完整執行以下指令，**42 項測試全部通過，沒有 skipped**：

```bash
BM25_RUN_GPU_TESTS=1 BM25_RUN_INTEGRATION=1 python -u -m unittest discover -s "Legal Case Retrieval/modernBert-bm25/tests" -v
```

當時未設定這兩個環境變數時，40 項一般測試執行，2 項 GPU／完整模型整合測試明確跳過。
當時擴量更新另新增舊 debug 續訓相容性與範圍欄位驗證測試，該階段一般測試為 42 項、2 項 GPU 測試跳過。
測試涵蓋：

- 原始 top-100 邊界、無補位、去重、排除 self／全部 positives。
- 每 pair 每 epoch 抽樣重現與重抽；多 worker DataLoader／collator 對齊。
- 手算 micro F1、重複 labels、空 predictions、自身排除與同分排序。
- Epoch 0 診斷、epoch 1 建立 best、patience 累加與改善歸零。
- Config 型別與範圍、2026 環境不覆蓋設定、資料變更拒絕續訓。
- 三組 optimizer LR、AMP overflow 跳過後恢復、連續異常停止。
- Debug／正式共用函式，不由日常 debug 啟動子程序。
- Checkpoint、scheduler、scaler、patience、RNG 恢復與獨立重載。

## 真實 2025 資料

| 檢查 | 結果 |
| --- | --- |
| Train / valid queries | 1341 / 336，無交集 |
| Train / valid unique pairs | 5421 / 1421 |
| Candidates | 7349 |
| Train / valid pool 篇數 | 83～99 / 87～99 |
| 文件、labels、qid、BM25 pool 完整性 | 通過 |
| 資料 hash | `dada68cdbf9d094ea8e771bae16982c9a18b110e747d5bc1397c1682eb974662` |

與重構前已核對的資料 hash 相同；沒有重新切分資料、重建 BM25 或更動資料快照。

## 完整 ModernBERT-base 實跑

使用 RTX 4080 SUPER、官方原始 ModernBERT-base revision
`8949b909ec900327062f0ebf497f51aef5e6f0c8`，不是 tiny model。
Torch 2.6.0+cu126、Transformers 4.52.4、Accelerate 1.6.0、Flash Attention 2.7.4.post1。

一般 `--debug` 已實際連續跑完 8 pairs × 2 epochs、epoch 0／1／2 評估與 4 次成功更新；
loss 有限，每輪實際收到的資料與該輪計畫取樣相符。

重構當時的整合測試產物（不是擴量後的 100-pair run）：

```text
models/modernBERT_bm25_top100_epoch_resample15_2025/_debug/20261019T230738962456Z/
  integration_result.json
  rng_restore_verified.json
  fp_reference_comparison.json
  run_manifest.json
```

該測試使用 9 個真實 pairs，確認不足完整 accumulation group 的最後一筆也確實送入模型。

| 項目 | 實測 |
| --- | --- |
| 模型／max length／negatives | 完整 base／實際 4096／15 |
| 精度／attention | FP32 weights + FP16 AMP／Flash Attention 2 |
| Valid 範圍 | 4 queries / 256 candidates，含全部選定 positives 與 queries |
| Epoch 0、1、2 | 全部保存評估、Top-5、TensorBoard |
| 模型實際收到的取樣資料 | 兩輪皆與 planned samples 完全相符 |
| 跨 epoch 重抽 | sample hashes 不同 |
| 中斷／續訓 | epoch 1 checkpoint-3 → epoch 2 checkpoint-6 |
| Scheduler last_epoch | 3 → 6，三組 LR 未重設 |
| AdamW 實際步數 | 3 → 6 |
| AMP | 6 次成功、0 次跳過；另有獨立的人工 overflow 恢復測試 |
| RNG | Python、NumPy、CPU、CUDA 皆在原生 restore 後逐一比對相符 |
| Patience | 0 → 1，符合兩輪 F1 持平 |
| 參數更新 | encoder、projector、temperature 的兩輪 checkpoint 均不同 |
| 重載最佳 checkpoint | F1、每 query 排名與 TP／FP／FN 完全等同訓練內評估 |
| TensorBoard | loss steps 1～6，valid F1 steps 0、3、6 |

另外實跑 `inference.py` 保存 embeddings，再透過 `similarity_and_rank.py --embeddings` 排名；
與直接重載評估的所有指標及每 query 結果一致。權重／資料／config hash 均核對通過。

## 與成功版本的數值比對

測試從 `modernBert-fp/fine_tune/fine_tune.py` 擷取原模型 class，不執行其全域環境初始化。
使用同一組 encoder、projector、temperature 權重，以及一筆真實 query／positive／15 negatives，
在 eval + FP16 autocast 下比較兩個版本：

- 真實序列長度 4096。
- 16 個 logits 完全相等（rtol=0、atol=0）。
- Cross-entropy loss 完全相等（rtol=0、atol=0）。

這證明此輸入下模型 forward 與成功版本一致，不宣稱不同負採樣／資料／初始化的整段訓練軌跡都相同。
該次整合測試 manifest 記錄的全部 Python 原始碼與 reference hashes，當時已核對一致；
本次 debug 擴量後的檔案已有變更，不宣稱舊 GPU 報告涵蓋新資料量。

## 文件與執行入口

Python 編譯檢查、git diff 空白檢查通過。
四個主要入口的 `--help` 已核對；README 的本機連結與 Bash 範例皆檢查通過。
Bash 範例只做語法檢查，未執行清除或正式長訓練指令。

Debug F1 為 0，是少量資料的流程測試結果，不是正式 336-query 的績效。
尚未啟動正式 20-epoch 訓練，也未重跑歷史 0911 的完整檢索評估。
短程成功不能保證所有長訓練 steps 永不 overflow 或保證最終 F1 提升。
