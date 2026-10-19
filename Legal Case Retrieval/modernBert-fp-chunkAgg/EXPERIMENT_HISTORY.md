# ChunkAgg 版本演進與實驗紀錄

[English](EXPERIMENT_HISTORY.en.md) | [v5 說明](README.md) | [v6 說明](../modernBert-fp-chunkAgg-v6/README.md)

本文件記錄截至 2026-10-20 可由 Git、checkpoint `trainer_state.json`、每輪 adaptive-negative JSON 與現行程式碼交叉驗證的 ChunkAgg 演進。無法由 artifact 證實的版本與數字不補猜。

## 版本名稱與比較口徑

Git commit 標題中的 `[Ver 4.4]`、`[Ver 5.0]` 是整個 repository 的 release 標籤；本目錄所說的 ChunkAgg v3/v4/v5/v6 則是模型 architecture/input contract。兩者不是同一套版本號。

- v1、v2：目前沒有可辨識的程式快照、獨立 checkpoint 或評估紀錄，無法可靠還原。
- early legacy：第一個可驗證的已訓練 ChunkAgg run。其輸入具有後來稱為 v3 的 special-token 問題，但 checkpoint 本身沒有 architecture version metadata，因此以下以 `legacy（v3-era）` 稱呼，不把它冒充成正式標記的 v3。
- v4、v5、v6：都有名稱明確的獨立模型與 adaptive-negative 目錄。

結果欄位是各 run 的 `trainer_state.json` 所記錄、每 epoch 重新執行 full scoped valid retrieval 的 micro top-5 F1。v4 至 v6 的 fused/mean/first 是同一 epoch、同一 valid set 的 aggregation ablation；legacy 沒有保存這三種 ablation。這些是訓練期歷史結果，不是重新跑過目前 `eval.py` 的 canonical offline 結果；在最終論文或正式報表引用前，仍應用目前 inference/ranking/eval 流程重算。

## 結果總表

| Run | 主要差異 | 每輪訓練列數 | 最佳 epoch / checkpoint | Fused F1 | Mean F1 | First F1 | 相對前版 |
|---|---|---:|---|---:|---:|---:|---:|
| legacy（v3-era） | 初代 1–3 chunk + learned `[DOC]` fusion | 6573 | 5 / `checkpoint-8220` | 0.065735 | 未記錄 | 未記錄 | — |
| v4 | 修正 chunk 輸入；加入 residual/evidence/ESS，但 fusion residual 仍無界 | 6262 | 4 / `checkpoint-6264` | 0.092577 | 0.119419 | 0.122706 | +0.026842（+40.83%） |
| v5 | 在 v4 基礎加入 bounded/zero-init fusion contract 與 content filter | 6257 | 7 / `checkpoint-10955` | **0.145365** | 0.138782 | 0.140428 | **+0.052787（+57.02%）** |
| v6 | 保留 v5 網路/loss；固定 `softmax(score/1.0)`；恢復 scope 外合格 positive | 6568 | 10 / `checkpoint-16420` | 0.129457 | 0.119583 | 0.124520 | -0.015908（-10.94%） |
| single-chunk adaptive emptyExclude | 供參考，不屬於 ChunkAgg 版本序列 | 6573 | 7 / `checkpoint-11508` | 0.164519 | 不適用 | 不適用 | 比 v5 高 0.019154 |

目前結論是 v5 為已保存 ChunkAgg run 中的最佳結果，但仍落後 single-chunk adaptive emptyExclude 0.019154 F1（相對 single-chunk -11.64%）。v6 比 v5 下降 0.015908；不過 v6 同時改了 sampling policy 與 positive eligibility，所以不能只把下降歸因於 `temperature=1.0` 或只歸因於加入 scope 外 positives。

各 run 的 `eval_global_f1` 曲線如下，方便辨認 peak 與過度訓練：

```text
legacy: e1 .015338, e2 .018077, e3 .039441, e4 .060805, e5 .065735,
        e6 .053136, e7 .054780, e8 .053136, e9 .056971, e10 .052588
v4:     e1 .079430, e2 .067927, e3 .072857, e4 .092577, e5 .073405
v5:     e1 .087767, e2 .103675, e3 .124520, e4 .132748, e5 .130554,
        e6 .132748, e7 .145365, e8 .138234, e9 .143171, e10 .130554,
        e11 .134394, e12 .127811
v6:     e1 .087219, e2 .101481, e3 .105321, e4 .115195, e5 .124520,
        e6 .121229, e7 .114646, e8 .125069, e9 .121777, e10 .129457,
        e11 .115743, e12 .115743, e13 .124520, e14 .119035, e15 .119583
```

## 每次調整與結論

### legacy（v3-era）：先證明 end-to-end multi-chunk 可訓練

初代實作把每篇文件切成最多 3 個 4096-token chunks，取各 chunk 的 `[CLS]`，經一般 MLP projector 後，與一個獨立 learned `[DOC]` token 一起送進單層 transformer fusion。超過三段時只保留最前面的三段；training objective 只有 document-level contrastive loss。負樣本使用 full-scope similarity 的 `softmax(score / 1.0)` 抽樣。

後來確認 ModernBERT fast tokenizer 的 Python `build_inputs_with_special_tokens()` 在這條路徑可能不實際插入 `[CLS]/[SEP]`，因此該 run 的 chunk input contract 有缺陷。即使如此，它證明 multi-chunk pipeline 能端到端訓練；但最佳 F1 只有 0.065735，且 epoch 5 後下降。

結論：只增加 chunks 與 fusion capacity 不會自然改善 retrieval；token contract 必須先被測試固定，且 learned `[DOC]` 從無內容狀態學起太脆弱。

### v4：輸入修正有效，但 fusion 反而破壞表示

v4 明確修正每個 chunk 的 `[CLS] + body + [SEP]` 組成，也建立 architecture version 邊界。由 checkpoint keys 與 trainer logs 可確認 v4 已經有 residual projector、fusion output、chunk-evidence loss、target-ESS sampling，以及 fused/mean/first ablation；這些不是 v5 才加入。不過 v4 的 fusion residual 沒有 norm cap，可以大幅覆寫穩定基底；positive 也改成必須位於 query scope 內，留下 6262 組訓練 pairs，333 組 scope 外 positive 被排除。

Fused F1 從 legacy 的 0.065735 升至 0.092577，但同一 checkpoint 的 mean F1 為 0.119419、first F1 為 0.122706。換句話說，不使用 learned fusion 反而分別高出 0.026842 與 0.030129。

結論：special-token 修正與較可靠的資料契約有正面效果，但 v4 的 learned fusion 在 valid set 上是負貢獻。之後不能只報 fused 分數，必須固定同報 mean/first ablation。

### v5：限制 fusion 破壞力，首次讓融合勝過簡單基線

v5 的重點不是再堆一個更大的聚合器，而是在保留 v4 evidence loss 與 ESS curriculum 的前提下，限制 fusion 分支的破壞力：

- 把 residual projector、normalized masked-mean base 與 mean-seeded `[DOC]` 固定成 v5 architecture contract；
- projector/fusion 新分支使用保守的 zero-output initialization；
- fusion applied residual norm 以 0.5 smooth cap 限制，並新增 raw-residual penalty；
- 保留同組 positive/negatives 的 chunk-evidence InfoNCE 與先廣後難 target-ESS curriculum；
- 加入至少 100 個非空白字元的 content filter；
- positive 仍必須在 scope 內，因此 6257 列 = v4 的 6262 列再少 5 組內容不合格 pairs。

最佳 fused F1 達 0.145365，比 v4 增加 0.052787，而且首次高於同 checkpoint 的 mean（+0.006583）與 first（+0.004937）。這是目前最重要的正面證據：受限、保守初始化的 fusion 確實能學到額外訊息。

結論：v5 的 bounded residual 方向應保留。不過 v5 同時正式化初始化契約並加入 content filter，不能把全部增益只歸因於 cap。成長到 epoch 7 才達 peak，但之後震盪並下降，顯示 early stopping、negative difficulty 與 learning dynamics 仍需監控；不宜用最後 epoch 代替最佳 checkpoint。

### v6：恢復 baseline 抽樣與 positive 語義，但總分退步

v6 完全保留 v5 網路、loss、content filter 與 15 個 without-replacement negatives，將 dynamic-negative sampling 固定為 single-chunk baseline 的 `softmax(score / 1.0)`，不再做 ESS calibration。同時恢復 baseline positive eligibility：positive 只要存在於 `processed` 且通過 100 字元門檻，即使不在 query scope 也建立訓練列；scope 只限制 negatives 與 ranking。因此每 epoch 有 6568 列，其中 311 組是 scope 外但內容合格的 positives。

最佳 fused F1 為 0.129457，比 v5 下降 0.015908；但仍高於 v6 自己的 mean（+0.009874）與 first（+0.004937），所以 bounded fusion 本身仍有正貢獻。它在 epoch 10 才達 peak，之後也沒有恢復。

結論：固定 `temperature=1.0` 的整體訓練契約沒有超過 v5 ESS curriculum，但此 run 不是嚴格單一變因，因為 positive eligibility 也同時改變。應保留 scope 外合格 positives 的語義，下一個實驗則把它與 v5 ESS sampling 組合，才能分離兩個因素。

## 100 字元 content filter 是否有效

這個 filter **確實有執行並改變資料流**，但目前沒有單一變因實驗能證明它提高 F1。v5 與 v6 都啟用相同門檻；v4 沒有門檻但同時有不同 fusion contract，single-chunk baseline 也有不同模型與訓練流程，因此不能從現有版本分數分離出 filter 的因果效果。

目前可量化的實際作用如下：

- 7708 篇 `processed` 文件中只排除 5 篇（0.065%）：`015738(0)`、`039404(0)`、`058076(10)`、`092745(10)`、`085084(11)`；括號是非空白字元數。
- training 移除 5 / 6573 組原本可建立的 pairs（0.076%）：短 query `085084` 的 3 組 positives，以及以 `039404`、`058076` 為 positive 的各 1 組。
- 這 5 篇文件原本合計出現在 1445 個 query scopes 的 6533 個 candidate 位置。雖然文件數很少，排除它們仍可防止空白／近空白文件產生退化 embedding，進而污染大量 query 的 adaptive-negative sampling 或 top-5 ranking。
- valid gold 中受影響的是 3 / 1656 組（0.181%）：空白 query `015738` 的 2 組 gold，以及近空白 candidate `092745` 的 1 組 gold。filter 會把它們明確當成不可預測，但這些文件本身幾乎沒有可供模型使用的內容。

因此現階段應把 100 字門檻定位為**資料完整性與訓練穩定性的 guardrail**，而不是已證實的增分功能。它很可能避免 pathological embedding，但 v5 的 +0.052787 不能歸功於它。若要衡量績效因果，應固定 v5 的 checkpoint initialization、sampling、seed 與其他設定，只比較 `min_chars=0`、`1`、`100`；同時記錄被過濾的 gold 數、negative-sampling 命中率及最終 canonical F1。

## `emptyExclude` 與 scope 外 positive

`scopeFilteredRaw_plus0_emptyExclude` 的名稱只描述如何建立 query-candidate scope：

- `Raw`：年份由 raw 文件抽取；
- `plus0`：known-year query 只允許年份不晚於 query 的 candidates；
- `empty`：無法取得年份的 query 得到空 scope；
- `Exclude`：年份未知的 candidate 不放入 known-year query 的 scope。

它不代表「把 scope 外 positive 一律刪除」。positive 是否必須在 scope 內，是各訓練版本另外決定的 policy。

| Pipeline | scope 外、內容合格 positive | 空 scope query | 實際保留 pairs |
|---|---|---|---:|
| single-chunk adaptive emptyExclude | **保留** | 因沒有 retrieval scores / negatives 而略過 | 6573 / 6595 |
| legacy ChunkAgg | **保留** | 略過 | 6573 / 6595 |
| ChunkAgg v4 | 排除 | 略過 | 6262 / 6595 |
| ChunkAgg v5 | 排除，另套 content filter | 略過 | 6257 / 6595 |
| ChunkAgg v6 | **保留，另套 content filter** | 略過 | 6568 / 6595 |

所以答案是：**Single-chunk adaptive emptyExclude 確實也會保留 scope 外 positive。** 實際 `adaptive_negative_epoch7_train.json` 有 6573 組唯一 `(query, positive)`，其中 311 組 positive 位於 scope 外。原始 train labels 共 6595 組，缺少的 22 組全部屬於 6 個 empty-scope queries；不是因 positive 本身位於 scope 外，而是該 query 沒有任何 scoped candidate 可供計分及抽負樣本。

推薦長期固定以下語義：

```text
gold label + 文件存在 + 通過 content filter => eligible positive
scope - self - all labeled positives              => negative/ranking pool
empty scope                                        => 無法建立 contrastive row，但 gold 仍留在評分分母
```

## 後續改良原則

1. 下一個最有資訊量的實驗應是「v5 network/loss + v5 ESS curriculum + v6 positive eligibility」，也就是只改 positive policy；不要同時改 sampling temperature。
2. scope 外 gold positive 不應被當成 negative，也不應只因 scope heuristic 排除而失去監督訊號；資料品質門檻與 scope 年份規則要分開處理。
3. 每個正式 run 只改一個主要因素，並把 architecture、positive eligibility、negative sampling、corpus source 各自寫入 manifest。
4. 每輪固定同報 fused、mean、first，只有 fused 穩定高於兩個簡單基線時，才能主張 aggregation 有效。
5. 以完整 valid denominator 的 retrieval F1 選 checkpoint，不以 contrastive `eval_loss/acc` 或最後 epoch 選模。
6. v4、v5、v6 都在 peak 後下降；保留 early stopping，並將 epoch 4、7、10 附近視為後續 ablation 的重要觀察點。
7. 至少重跑多個 seeds，並用目前 inference → ranking → `eval.py` 產生 canonical 結果；目前單一 run 的差距只能作工程決策，還不足以宣稱統計顯著。

## 可稽核 artifact

```text
models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026/
models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v4/
models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v5/
models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v6/
models/modernBERT_contrastive_adaptive_fp_fp16_scopeFilteredRaw_plus0_emptyExclude_2026/
coliee_dataset/task1/2026/lht_process/modernBert-chunkAgg_scopeFilteredRaw_plus0_emptyExclude*/finetune_data/
coliee_dataset/task1/2026/lht_process/modernBert_scopeFilteredRaw_plus0_emptyExclude/finetune_data/
```
