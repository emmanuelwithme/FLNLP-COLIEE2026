# 2026 Random／Adaptive negative 控制變因稽核

稽核日期：2026-11-03；說明整理：2026-11-06（Asia/Taipei）。完整機器可讀結果見 [random_adaptive_2026_control_audit.json](evidence/random_adaptive_2026_control_audit.json)。本項使用 CPU 讀取歷史產物，沒有重新訓練或執行 checkpoint encoder 推論。

比較的保存目錄：

```text
models/modernBERT_contrastive_random_fp_fp16_scopeFilteredRaw_plus0_emptyExclude_2026/
models/modernBERT_contrastive_adaptive_fp_fp16_scopeFilteredRaw_plus0_emptyExclude_2026/
```

## 已核對相同的條件

兩者現存產物符合 `query_candidate_scope_raw_plus0_empty_exclude.json`。Raw 指從原文抽取年份；plus0 指 candidate 年份不晚於 query；年份不明的 query 給空 scope，年份不明的 candidate 排除。`emptyExclude` 不代表內容空白過濾，也不代表 encoder 使用 raw。

- 每份 samples 有 1594 個 train queries、6573 個唯一 query–positive pairs；兩者 pair 集合相同。
- 每 pair 有 15 negatives；Adaptive 15 份與 Random 20 份檔案中，scope 外、self、known-positive 及同筆重複 negatives 均為 0。
- Random 20 份以現有候選池、scope、seed 289＋epoch 重建，逐筆完全相同。
- Adaptive epoch 0 的 7961237 筆計分資料，其 1594 個 query 的完整候選集合與 scope 相符。
- 兩者保留 311 個 positive 位於 scope 外的訓練 pairs；6 個空 scope train queries 的 22 個 gold pairs 未建立訓練 row。
- Tokenizer SHA-256 相同；保存的 TrainingArguments 除輸出與 logging 名稱外相同，包括 batch 1、accumulation 4、LR 5e-6、temperature LR 5e-4、FP16、linear scheduler、warmup 0.1、seed 42、20 epochs 上限及 `eval_global_f1` 選模型。

這支持 Scope 與主要訓練設定對齊；缺少逐 run 不可變資料快照，仍不能把今天的檔案雜湊當成所有歷史 epoch 的來源證明，也不能只靠保存 samples 證明當時所有 workers 實際收到相同內容。

## 已找到的評分差異

完整 valid 有 401 queries、1656 個 gold pairs。Query `004467`、`050666` 是空 scope，各有 2 個 gold；非空 scope 的 399 queries 有 1652 個 gold。

| 最佳 checkpoint 的保存數值 | Random | Adaptive |
| --- | ---: | ---: |
| Checkpoint／epoch | 16440／10 | 11508／7 |
| TP，依保存 P／R 還原 | 290 | 300 |
| Predictions | 1995 | 1995 |
| 保存評分的 gold 分母 | 1656 | 1652 |
| 保存 F1 | 0.15886058614078336 | 0.16451878256100905 |
| 統一完整 valid gold 後的計數重算 F1 | 0.15886058614078336 | 0.16433853738701726 |

Adaptive 的分母對應排除兩個空 scope query，與舊評估只納入有 prediction 的 query 的行為相符。Random 保存的最佳 epoch 10 TREC 已直接核對完整候選範圍、Top-5 與 TP=290；Adaptive 的同名 valid TREC 已被後續 Random 執行覆寫，因此 Adaptive 的上表是保存指標與目前 labels 的整數計數重建，**不是重新載入 checkpoint 得到的績效**。

## 其他控制缺口

- 兩者初期 patience 都為 5；Adaptive 從 epoch 13 改為 10，Random 始終為 5。兩者最佳值在修改前已出現，不能把最佳 F1 差距直接歸因於後來的 patience 調整；完整續訓流程仍不一致。
- 最新保存狀態為 Adaptive epoch 14、Random epoch 20。Adaptive 最後 state 的 `should_training_stop=false`，不能宣稱它已按照相同早停流程結束。
- 兩者 pair 集合相同，原始 row 順序不同。相同 Trainer seed 不代表實際資料順序相同。
- 目前保存版本的程式先初始化 projector，再建立會設定 seed 的 Trainer，沒有在模型建立前明確固定 Torch seed；歷史 encoder／projector 的初始雜湊也未保存，不能證明初始化完全相同。
- Adaptive 保存的 `output_dir` 與今天的資料夾名稱不同，另缺少完整歷史 config、code revision 與資料內容 manifest。名稱相近不能替代來源證據。

這兩個 run 可以作為條件接近的歷史比較；尚不足以將全部 F1 差距歸因於模型分數加權，亦不能拿 2026 數值直接解釋 2025 原始 ModernBERT-base 的結果。優先統一既有 checkpoint 評分可修正分母差異，但補不回歷史初始化的控制證據。
