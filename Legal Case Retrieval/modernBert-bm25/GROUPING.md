# 四個不同 Query 的隨機分組與大量驗證

統計日期：2026-11-03（Asia/Taipei）；操作說明更新：2026-11-06。策略：`shuffle_repair_v1`；新訓練實作識別：`modernBert-sailer-gradcache-v3`。

核心在[batch_grouping.py](fine_tune/batch_grouping.py)的`shuffle_repair_rows`／`_repair_distinct_queries`；[sailer.py](fine_tune/sailer.py)負責接上既有negative抽樣、Trainer與groupreceipt；[validate_grouping.py](validate_grouping.py)是可重跑的大量CPU統計工具。

## 現在的流程

1. 保留全部5421個原始query-positivepairs，先按既有SHA-256派生seed各抽15個BM25negatives。
2. 用獨立paddingseed補3筆已有pairs；本資料有足夠不同Query，因此補3個不同Query並記錄其IDs。
3. 用獨立shuffle seed打亂全部5424筆，再每4筆形成logicalbatch。
4. 同批出現重複Query時，以**整筆row**和其他批交換。Query、Positive、15negatives及其他payload一起移動，不拆開或重配標籤。
5. 每次修復必須降低全體重複Query的slot數；隨機嘗試64次仍找不到，就完整搜尋合法交換位置，不丟樣本、不改成舊優先排序。
6. 核對原始pairs與補齊multiset、每批4個不同Query，保存hash；DataLoader另將完整batches的出場順序打亂。

不再使用「剩餘pairs最多的Query先搭配」。沒有加入跨epoch禁止搭檔的規則，也不強制三人／四人組永遠不能重現。局部交換是近似隨機修復，**不能宣稱對所有合法分組嚴格均勻**。

所有亂數由samplerseed與epoch派生，negative、padding、shuffle、repair使用分開的亂數流。相同資料／seed／epoch／軟體版本可重現，不消耗Python全域RNG。同epoch分組的變更不影響每個原始pair抽到的15negativeIDs。

## 可行性與修復終止

設最後有M個batches，每批4個不同Query，則任何Query最多出現M次。超過就直接報錯。本資料最大Query有34pairs，M=1356，距離上限很遠。

若某批有重複A，而A總次數不超過M，一定存在一批沒有A。從那批選B交換：

- 若B不在原批，原批不同Query數增加，對方不新增重複。
- 若對方只有原批已有的Query，其中必有重複B；交換能先改善對方，原批不惡化。

所以不能要求「一次交換立刻讓兩批都完全合法」，否則會錯拒可行資料。程式以全體重複slot數為單調下降量，每次減少至少1；交換次數上限是初始重複slot數。隨機提案失敗時有完整搜尋，沒有無限重試。

一般化極端資料如Query次數`[4,4,4,1]`，要補3筆只能補第四個Query三次；程式會在不突破M的前提下處理。這是測試用邊界，本COLIEE資料每輪都能補3個不同Query。

## 模型與 loss 保持既有設定

每logicalbatch仍有4個Query、各1positive＋15negatives，使用4×64maskedInfoNCE。原始5421pairs全部保留，只有明確記錄的3個paddinginstances；共1356logicalsteps／epoch。Padding在全shuffle之前加入，不固定集中在最後一批。

原始ModernBERT-base、CLS/projector、temperature、4096長度、FP16／FlashAttention2、GradCachechunk2、inputdedup、gradientcheckpointing=false、optimizer、scheduler、full-validTop5microF1與早停均沿用v2。只更換分組／padding安排與相關audit，沒有改資料前處理或BM25排名。

## 新舊版本與輸出

2026-11-03 更新分組時沒有中斷、刪除或重新啟動當時的 v2 run。舊規則原樣保留，供既有 run 續訓；以下配置對照不代表目前程序仍在執行。

| 版本 | 設定 | 輸出 |
| --- | --- | --- |
| 新v3 | `configs/coliee2025_sailer_gradcache.json`，`batch_grouping=shuffle_repair_v1` | `models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_shuffle_2025/` |
| 舊v2 | `configs/coliee2025_sailer_gradcache_legacy.json`，缺省即legacy策略 | 原`models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_2025/` |

新舊run不得混用checkpoint／optimizer／patience。舊設定不插入新欄位，以維持保存config完全相符；已核對目前保存的epoch1、2資料能由legacy規則逐筆重現，舊resume設定檢查也通過。

既有訓練結束或自行停止後，在repository根目錄、原Conda環境啟動新run：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune_sailer.py" --no-debug
```

不要指向舊v2checkpoint。Debug使用同入口`--debug`，會在新目錄`_debug/`隔離產物。本次未自動啟動任何新模型訓練。

## 保存與續訓檢查

- `sampling/epoch_NN_train.json`：完整plannedrows。
- `epoch_NN_train.audit.json`：策略、paddingpairs、初始衝突、修復次數、fallback次數、groupinghash與samplehash。
- `epoch_NN_received.json`：模型實際收到的全部pairs／negatives與planned一致。
- `epoch_NN_groups_received.json`：模型收到的**整個四人組**multiset與planned一致，允許batch出場順序改變。
- `gradcache_state.json`：保存分組策略；config、implementation或checkpoint策略不符時拒絕續訓。

分組與negative抽樣依epoch重新推導，續訓不用依賴之前消耗了多少全域random值。CPU測試已比對連續執行與epochboundary重新建立sampler，下一輪plannedrows一致。

## 大量測試結果

完整數字、原始JSON與log見[統計報告](../../research_notes/coliee2025_review/shuffle_repair_2026-11-03/README.md)。本次執行：

- 一般測試：發現72項，68項CPU測試全部通過，4項GPU測試按opt-in跳過。
- 新分組：500seeds×20epochs＝10000輪真實pair分組；1356萬logicalbatches、5424萬pairinstances，全部約束檢查通過。
- 舊規則對照：100seeds×20epochs＝2000輪；另以相同100組seeds做配對比較。
- 極端分布：20000組可行輸入全部通過、20000組不可行輸入全部正確拒絕；其中2000組強制走完整搜尋fallback。
- 完整negative流程：200輪、1084200個原始pairinstances、16263000個原始negativeslots，負樣本未被分組修改、排除self／knownpositives／pool外IDs、重現性檢查全部通過。
- 真實LabelMask：20輪、27120logicalbatches、6942720個scorecells，target／self／knownpositives／inputaliases／duplicates等契約全部通過。

相同100組seed的對照結果：

| 指標 | 舊規則 | 新規則 |
| --- | ---: | ---: |
| 單輪相同四人組再次出現比例 | 10.71% | 本次0 |
| 每Query單輪不同搭檔平均數 | 6.73 | 12.01 |
| 跨輪至少2個Query同組重合 | 12.17% | 15.13% |
| 跨輪至少3個Query同組重合 | 1.574% | 0.0305% |
| 跨輪4個Query完全重合 | 0.5115% | 此100組seed本次0 |

擴大至500seeds的新規則中，12882000個跨相鄰epoch的batch比較位置有1次四人組重現。以上都是觀測值，不是禁止重複的保證。至少2人重合較高不代表更差：新規則讓上一輪出現過的搭檔集合更廣，應一起看三／四人組重合與搭檔多樣性。

這些是 CPU 資料流與隨機分組統計，不是模型訓練 F1。模型數值與 GPU 路徑沿用 v2；2026-11-03 的分組驗證未重跑 GPU 測試，2026-11-06 的整理也僅重跑 CPU 測試及工具流程。

## 重跑驗證

```bash
python -m unittest discover -s "Legal Case Retrieval/modernBert-bm25/tests" -v
nice -n 10 python -u "Legal Case Retrieval/modernBert-bm25/validate_grouping.py"
nice -n 10 python -u "Legal Case Retrieval/modernBert-bm25/validate_grouping.py" --flow-only
```

第一個統計指令預設執行 10000 輪新分組、2000 輪 legacy、20000 組壓測與 200 輪完整 negative 流程；第二個執行 20 輪真實 Mask 及舊 run 相容性檢查。

新報告分別寫入 `reports/modernbert-bm25/grouping/<UTC-run-id>/`，啟動時會列印路徑；可用 `--output-dir` 明確指定同一目的地。不寫入訓練目錄，也不預設覆寫 2026-11-03 的歷史摘要。依 2026-11-12 的精簡備份規則，這些重複輸出、逐輪 `new_policy.json`／`legacy_policy.json` 與測試 log 只留本機；Git 保留歷史整理後的摘要與獨立的輕量訓練快照。

`--flow-only` 需要舊 run 現存的 label-mask manifest、保存設定及 samples；若已搬走，應先提供可核對的相同資料，不能把舊報告當成本次執行結果。
