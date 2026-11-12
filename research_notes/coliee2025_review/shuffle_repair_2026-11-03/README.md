# Shuffle＋Query衝突修復：實作與大量統計

日期：2026-11-03（Asia/Taipei）。策略`shuffle_repair_v1`，訓練識別`modernBert-sailer-gradcache-v3`。

本次依使用者要求實作並大量驗證四Query分組。全部是CPU資料／取樣測試，沒有停止正在執行的v2GPU訓練，也沒有新開GPU訓練。原始資料、BM25排名與現有模型產物未修改或刪除。

操作與可行性／終止說明見[GROUPING.md](../../../Legal%20Case%20Retrieval/modernBert-bm25/GROUPING.md)。新程式將全部pairs先shuffle，再交換修復同批Query重複；不按剩餘positive數量排序，也不強制跨epoch禁止搭檔。

## 測試規模與約束

輸入為現存COLIEE2025的1341個trainqueries、5421個去重positivepairs；每輪補3筆至5424，形成1356個四Querylogicalbatches。

| 項目 | 實際規模 | 結果 |
| --- | --- | --- |
| CPU一般測試 | 68項；另外4項GPU依opt-in跳過 | 全部通過 |
| 新規則真實ID分組 | 500seeds×20輪＝10000輪 | 全部約束通過 |
| 新規則累計batch／pair位置 | 13560000／54240000 | 無遺漏、非預期增加或同批Query重複 |
| Legacy對照 | 100seeds×20輪＝2000輪 | 全部原始pairs保留 |
| 人工可行極端分布 | 20000組 | 全部通過 |
| 人工不可行分布 | 20000組 | 全部明確拒絕 |
| 強制完整搜尋fallback | 2000組極端資料 | 全部終止且資料完整 |
| 完整抽15negatives流程 | 200輪、1084200個原始pair位置、16263000個原始negative位置 | 原始negative內容與各項合法性不變；重現性通過 |
| 真實LabelMask資料流 | 20輪、27120logicalbatches、6942720scorecells | target與合法性契約通過 |

分組大規模階段用真實query-positiveIDs，不重算negatives；完整negative階段另外測試真正的15negative抽樣與payload保留。以上位置數是重複試驗的累計，不是數千萬筆不同案例。

壓測涵蓋：只有4個Query、Query次數恰好等於batch數、非常偏斜的頻率、需連續補同一Query、原批與對方都含重複Query、隨機提案失敗後完整搜尋等。合計修復1048049次，含強制情況的fallback搜尋548148次；每次修復都必須降低全體衝突數，沒有無限重試。

## 相同seed的配對比較

新規則跑500組seeds、舊規則跑100組；下表取兩者共同的100組seed，每組20輪。原始5421pairs相同、每輪各補3筆，paddingIDs依策略可不同。

| 指標 | Legacy平均 | 新規則平均 |
| --- | ---: | ---: |
| 同輪四人組再次出現比例 | 10.708% | 本次0 |
| 每Query單輪不同搭檔 | 6.730 | 12.009 |
| 跨相鄰輪至少2Query同組重合 | 12.173% | 15.127% |
| 跨相鄰輪至少3Query同組重合 | 1.5739% | 0.03047% |
| 跨相鄰輪完整四人組重合 | 0.51149% | 此100組seed本次0 |

100組配對seed中，每組的新規則都提高不同搭檔數，並降低三人組／四人組重合。搭檔數每組平均增加5.252–5.308個。這是配對試驗的描述統計，不是正式F1提升或對所有合法分組均勻性的證明。

「至少2人」重合不宜單獨最小化。新規則在一輪內涵蓋更多不同搭檔，因此下一輪更容易與某一個舊搭檔相遇；目前舊規則的問題主要是搭檔集中與三／四人群組留存。

## 新規則全部10000輪

- 每輪不同搭檔平均12.0085；單輪均值範圍11.9523–12.0582。
- 最高positive數Query`099187`單輪平均遇到96.68個不同搭檔，舊規則平均28.91個。
- 每輪平均修復9.2268次，最多22次；相對5424筆資料，只有少量位置交換。
- 同輪完整四人組重複在這10000輪中未觀察到，並非程式禁止重複。
- 9500組相鄰輪比較、12882000個第二輪batch位置中，完整四人組重現1次，約0.00000776%。
- 跨輪至少3人同組平均0.03038%；至少2人平均15.131%。

此法是全shuffle後的局部修復，會有條件偏差，不能稱「對所有合法分組嚴格均勻」。未加入任何跨epoch排斥或強制零重合規則。

## 完整資料流與舊run保護

`flow_validation.json`檢查20輪的實際BM25pools與LabelMask：每列指定positive位置正確，合法negative均排除self／knownpositive／相同輸入aliases，保留候選沒有重複tokenidentity，且每Query至少有一個合法negative。

已核對現有v2的epoch1、2保存樣本與legacy重建逐筆一致，legacy設定及implementation與保存run相符。新規則另有epochboundary重新建立sampler與groupreceipt的CPU測試；四人組被交換但row總集合未變的錯誤也會被groupreceipt攔下。

新輸出：`models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_shuffle_2025/`。舊`..._sailer_gc_2025/`仍在訓練，沒有在同一run中途改規則。新舊checkpoint不能混用策略。

本次分組檢查不執行encoderforward／backward，不能把v2歷史GPU驗證標成本次v3結果。模型、loss、GradCache數值路徑未改；本次沒有建立tiny模型，也沒有與執行中的GPU訓練搶跑另一模型。

## 證據檔與重跑

- [summary.json](summary.json)：完整總結、來源labelhash、配對比較、壓測與negative流程。
- `new_policy.json`：10000 輪逐輪／逐 seed 統計、位置計數與 hash，保留本機，Git 忽略。
- `legacy_policy.json`：2000 輪對照明細，保留本機，Git 忽略。
- [flow_validation.json](flow_validation.json)：真實20輪Mask與舊run相容性。
- `unit_tests.log`、`statistical_tests.log`、`flow_tests.log`：原始執行 log，保留本機，Git 忽略；本頁及 JSON 摘要記錄驗證範圍與結果。

在repo根目錄與原Conda環境執行：

```bash
python -m unittest discover -s "Legal Case Retrieval/modernBert-bm25/tests" -v
nice -n 10 python -u "Legal Case Retrieval/modernBert-bm25/validate_grouping.py"
nice -n 10 python -u "Legal Case Retrieval/modernBert-bm25/validate_grouping.py" --flow-only
```

新報告預設寫入 `reports/modernbert-bm25/grouping/<UTC-run-id>/`，不覆寫本資料夾的歷史摘要，也不寫入模型訓練目錄。可用 `--output-dir` 明確指定目的地。`--flow-only` 會讀 legacy run 的既有 label-mask manifest、設定及 samples；若搬機或刪除該資料，需先提供相同快照，不能把舊報告當成新執行結果。
