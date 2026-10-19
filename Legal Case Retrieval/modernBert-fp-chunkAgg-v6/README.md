# modernBert-fp-chunkAgg-v6

[English](README.en.md) | [版本與實驗紀錄](../modernBert-fp-chunkAgg/EXPERIMENT_HISTORY.md) | [Task 1 總覽](../README.md)

v6 是隔離的後續實驗：從 continued-pretraining ModernBERT backbone 初始化端到端 multi-chunk retriever，保留 v5 的網路與 loss，把動態負樣本固定成 single-chunk baseline 的 `softmax(score / 1.0)`，並恢復「scope 外但內容合格的 positive 仍可訓練」的 baseline 語義。由於 sampling policy 與 positive eligibility 同時改變，它不是嚴格的單一變因 ablation；結果解讀與後續建議見[版本與實驗紀錄](../modernBert-fp-chunkAgg/EXPERIMENT_HISTORY.md)。它不載入 single-chunk 或舊 chunkAgg checkpoint，也不讀舊 `TASK1_CHUNKAGG_*` / `TASK1_AUTO_RESUME` 設定。

沒有 Stage 1/2/3、grouped multi-positive LSEPair、warm-start gate 或 chunk fusion gate。整個 run 只訓練 single-positive InfoNCE 目標。

## 固定實驗契約

- 語料：query、positive、negative、candidate 全部使用同一個 `processed`。
- Split：既有 `train_qid.tsv` 與 `valid_qid.tsv`。
- Scope：`query_candidate_scope_raw_plus0_empty_exclude.json`。
- Content filter：所有實際送入模型的文件至少有 100 個非空白字元。
- Positive：內容合格即可作為訓練目標，不要求位於 scope；這重現 baseline 語義。
- Negative：只能來自該 query 的合格 scoped candidates，且排除 self 與該 query 的所有標註 positives。
- 每個 `(query, positive)` 是一列，各自重抽 15 個不重複 negatives。同 query 的其他 positive 不共用固定抽樣結果。
- 不同列或 epoch 可以再次抽到相同 negative。
- Valid ranking 先套 scope、content filter 與 self exclusion，再取 top-5。
- Valid F1 永遠以完整 401 queries / 1656 gold pairs 計分；短 query、空 scope 或缺少 prediction 都是空答案，不會從分母移除。

正式 preflight 鎖定下列目前資料統計，任一數字漂移就停止：

```text
processed documents       7708
eligible documents        7703
train qids                1600
trainable qids            1593
epoch training rows       6568
minimum negative pool       50
scope-external positives   311
valid qids                 401
valid gold pairs          1656
```

被 content filter 排除的是 `015738(0)`、`039404(0)`、`058076(10)`、`092745(10)`、`085084(11)`；括號是非空白字元數。短 query 仍保留在全域評分分母。

## 網路與資料形狀

1. 文件先完整 tokenization，再優先往回找到最近句界；找不到句界才依 token budget 硬切。
2. 每個有效 chunk 是 `[CLS] + 最多 4094 個正文 tokens + [SEP]`，總長最多 4096。
3. 超過三個 sentence-aware chunks 時，選 head/middle/tail，最多保留三個。
4. 不足三個時用 padding tensor 補齊固定形狀 `[3, 4096]`，`chunk_mask` 會完全遮掉空位置；沒有 learned empty-chunk token。
5. 每個 chunk 個別經同一個 ModernBERT encoder，取 chunk 的 `[CLS]` hidden state。
6. `[CLS]` 通過 residual projector：`LayerNorm(x + MLP(x))`。MLP 最後一層以零初始化。
7. 每個有效 chunk vector 先 L2 normalize，再做 masked mean 並再次 normalize，得到 `base`。
8. Fusion transformer 的 sequence 是四個 tokens：`[DOC] + chunk1 + chunk2 + chunk3`。learned position embeddings 是加到這四個向量上，不是第五個 token。padding chunks 由 attention mask 遮掉。
9. `[DOC]` 的初始內容是 `base + learned_doc_bias`。fusion 的輸出轉成 bounded residual，再加回 base：

```text
base = normalize(masked_mean(normalize(valid_chunk_vectors)))
raw_residual = fusion_output(fusion_transformer([DOC=base+bias], chunks)[DOC])
fusion_residual = raw_residual / (1 + ||raw_residual|| / 0.5)
document = normalize(base + fusion_residual)
```

`fusion_output` 起始為零，因此初始 `fused == mean`。這裡沒有 gate，也不把最終向量除以 token 數；masked mean 已依有效 chunk 數平均，最後再 L2 normalize，所以 3 chunks 不會只因數量較多而放大向量 norm。

每列 collator 輸出：

```text
anchor_input   [1, 3, 4096]
positive_input [1, 3, 4096]
negative_input [15, 3, 4096]
negative_counts [15]
labels [0]
```

Document logits 固定為 16-way：一個 positive 加 15 negatives。

這是「邏輯上的 mini-batch」：共 17 篇文件，最多對應 51 個有效 chunks。`TASK1_CHUNKAGG_V6_CHUNK_MICROBATCH_SIZE=1` 會讓 encoder 分小批處理有效 chunks，因此不是把 51 個 4096-token sequences 同時留在 GPU。`gradient_accumulation_steps=4` 則累積四筆上述 mini-batches 後才做一次 optimizer update，不會把它們合併成同一組 61 個 negatives。

## Loss

```text
L = L_document_InfoNCE
    + 0.2 * L_chunk_evidence_InfoNCE
    + 0.01 * mean(relu(||raw_fusion_residual|| - 0.5)^2)

e(q, d) = tau_e * (
    logsumexp_j(q dot chunk_j / tau_e) - log(number_of_valid_chunks)
)
tau_e = 0.1
```

Document 與 evidence loss 使用同一個 positive 及同一組 15 negatives。training temperature 是可學參數，初始 `0.55555`、限制在 `[0.05, 2.0]`；`tau_e=0.1` 只用於 chunk evidence smooth pooling；dynamic-negative sampling temperature 則固定為 `1.0`。三者用途不同。

## 動態負樣本

每個 epoch 開始前都用當時模型重新編碼完整合格語料並計算該 query 完整 scope 的 dot-product scores：

```text
p(d|q) = exp(score(q,d) - max_score)
         / sum_d' exp(score(q,d') - max_score)
```

這就是穩定版 `softmax(score / 1.0)`。沒有 ESS calibration、temperature schedule、top-k 截斷、random/BM25 混合或 pool 太小時 replacement：

- Epoch 1 的資料由尚未更新的初始 v6 模型產生。
- Epoch N 的資料由 epoch N-1 結束後的模型產生。
- 每個 positive 都呼叫一次 without-replacement sampling。
- pool 小於 15 會直接報錯；正式資料的最小 pool 是 50。
- ESS、entropy、top-100 probability mass 只寫入 TensorBoard 作觀察，不改變 probabilities。
- 每輪只保存小型 `adaptive_negative_for_train_epoch_NN.json`，不保存大型 similarity TREC TSV，也不重用舊 JSON。

既有 valid contrastive JSON 的檔名含 `bm25`，但它只提供輔助 `eval_loss/acc`；不會供應訓練 negatives。選模使用的是當前模型每 epoch 完整重做的 scoped retrieval `eval_global_f1`。

## 預設超參數

設定都位於本目錄可追蹤的 `.env`，shell 中同名的 `TASK1_CHUNKAGG_V6_*` 值優先。

```text
backbone              checkpoint-29000 continued pretraining
epochs                20
early stopping        patience=5, threshold=0, metric=eval_global_f1
batch / grad accum    1 / 4
expected rows/epoch   6568
expected steps/epoch  ceil(6568/4) = 1642
encoder LR            5e-6
projector/fusion LR   1e-5
temperature LR        1e-4
optimizer             adamw_torch_fused
scheduler             linear
warmup ratio          0.1
fp16 / bf16 / tf32    on / off / off
retrieval batch       2
dataloader workers    2
persistent workers    off
logging               first step and every 50 optimizer steps
```

`persistent_workers` 固定關閉，因為每個 epoch 會替換 dataset；worker 若持續存在可能保留上一輪資料副本。`PYTORCH_CUDA_ALLOC_CONF` 預設使用 `expandable_segments:True`。文字與 chunk cache 只減少 CPU/SSD 重讀及 tokenization，不會改變 encoder 一次看到的 token 數。

## 驗證

先執行不載模型的完整資料/sampler preflight：

```bash
/home/lab/miniconda3/envs/FLNLP-COLIEE2026-WSL/bin/python \
  "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/preflight.py" --skip-model
```

再執行包含真實 tokenizer、4096-token chunk contract、16-way forward 與 backward 的完整 preflight：

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/lab/miniconda3/envs/FLNLP-COLIEE2026-WSL/bin/python \
  "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/preflight.py"
```

單元測試：

```bash
/home/lab/miniconda3/envs/FLNLP-COLIEE2026-WSL/bin/python -m unittest discover \
  -s "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/tests" -v
```

Quick smoke test 使用獨立 output/data directory，避免污染正式 run：

```bash
CUDA_VISIBLE_DEVICES=0 \
TASK1_CHUNKAGG_V6_QUICK_TEST=1 \
TASK1_CHUNKAGG_V6_NUM_EPOCHS=1 \
TASK1_CHUNKAGG_V6_AUTO_RESUME=0 \
TASK1_CHUNKAGG_V6_OUTPUT_DIR=./models/chunkagg_v6_smoke \
TASK1_CHUNKAGG_V6_FINETUNE_DATA_DIR=./coliee_dataset/task1/2026/lht_process/chunkagg_v6_smoke/finetune_data \
python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/fine_tune/fine_tune.py"
```

## 正式訓練與續訓

新 run 的預設 output directory 必須不存在或為空。直接執行即可使用 `.env` 正式設定：

```bash
cd /home/lab/FLNLP-COLIEE2026
python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/fine_tune/fine_tune.py"
```

不指定 `CUDA_VISIBLE_DEVICES` 時，PyTorch 可看到所有 GPU，Trainer 通常使用第一張可見 GPU。單卡正式實驗建議明確指定：

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/fine_tune/fine_tune.py"
```

`TASK1_CHUNKAGG_V6_AUTO_RESUME=1` 是預設值。中斷後執行相同指令會找到 v6 output 下最新 checkpoint，恢復 model、optimizer、scheduler、FP16 scaler、epoch/global step，再用恢復後模型重算下一輪 negatives。若 output 有 TensorBoard events 卻沒有可續訓 checkpoint，程式會拒絕混接曲線。

指定 checkpoint：

```bash
TASK1_CHUNKAGG_V6_RESUME_FROM_CHECKPOINT=./models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v6/checkpoint-N \
python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/fine_tune/fine_tune.py"
```

在 VS Code WSL terminal 內用 tmux：

```bash
tmux new -s chunkagg-v6
cd /home/lab/FLNLP-COLIEE2026
CUDA_VISIBLE_DEVICES=0 python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/fine_tune/fine_tune.py"
```

Detach：`Ctrl-b` 再按 `d`；重連：`tmux attach -t chunkagg-v6`。不需要用 `tee` 或 redirect 保存整份 terminal log。

## TensorBoard

訓練只寫入一個 Trainer run：

```text
models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v6/tb
```

使用 `tbfix`，`--port 0` 由作業系統分配空 port；terminal 會印出實際網址：

```bash
conda run -n tbfix tensorboard \
  --logdir models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v6/tb \
  --port 0
```

不需要 `--logdir_spec`，也不會出現 stage/startup/by_stage 曲線。Step 1 就會有第一筆 train log，之後每 50 optimizer steps 一筆；retrieval 指標每個 epoch 一筆。

主要 tags：

```text
train/loss, train/grad_norm, train/temperature, train/lr_group_*
train/diagnostics/loss_document
train/diagnostics/loss_evidence
train/diagnostics/loss_fusion_residual_penalty
train/diagnostics/positive_score
train/diagnostics/hardest_negative_score
train/diagnostics/score_margin
train/diagnostics/positive_evidence_score
train/diagnostics/evidence_dilution_index
train/diagnostics/fusion_raw_residual_norm
train/diagnostics/fusion_residual_norm
train/diagnostics/fusion_residual_cap_utilization
train/diagnostics/base_fused_cosine
train/diagnostics/fusion_attention_*
train/sampling/effective_sample_size_{mean,median}
train/sampling/entropy_{mean,median}
train/sampling/top100_probability_mass_{mean,median}
train/sampling/max_probability_{mean,median}
train/sampling/pool_size_{mean,median}
train/sampling/base_temperature       # 永遠 1.0
train/sampling/source_is_model        # 永遠 1.0
train/sampling/row_count              # 正式 run 永遠 6568
eval/loss, eval/acc1, eval/acc5, eval/global_f1
eval/retrieval_valid_top5_{f1,precision,recall}
eval/retrieval_valid_{fused,mean,first}_top5_{f1,precision,recall}
```

## 推論、Ranking 與績效表

不需要 `run_infer.sh`、`run_rank.py` 或 `run_train.py`。訓練完成後依序執行：

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/inference.py"

python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/similarity_and_rank.py"

TASK1_RETRIEVAL_MODEL_NAME=modernBert_fp_chunkAgg_v6_fp16 \
TASK1_MODEL_RESULTS_DIR=./coliee_dataset/task1/2026/lht_process/modernBert_fp_chunkAgg_v6_fp16 \
TASK1_SCOPE_PATH=./coliee_dataset/task1/2026/lht_process/scope_compare/query_candidate_scope_raw_plus0_empty_exclude.json \
python "Legal Case Retrieval/utils/eval.py"
```

`inference.py` 依 `eval_global_f1` 找最佳 v6 checkpoint，只編碼一次合格 `processed`，存到：

```text
coliee_dataset/task1/2026/lht_process/modernBert_fp_chunkAgg_v6_fp16/embeddings/processed.pkl
```

`similarity_and_rank.py` 對 train/valid 輸出 dot/cos top-5 TREC TSV。`utils/eval.py` 會把缺少 prediction 的 labeled qid 當空答案，並追加績效到 `Legal Case Retrieval/results/experiment_results.csv`。

## Canonical Baseline 比較

歷史 `0.164518` 只作參考。要以相同 `processed`、scope、100 字元規則、top-5 與 401/1656 分母重算 baseline checkpoint-11508 和 v6 best：

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/compare_canonical.py"
```

結果寫到：

```text
coliee_dataset/task1/2026/lht_process/modernBert_fp_chunkAgg_v6_fp16/canonical_comparison.json
```

## 清理 v6 產物

以下命令只刪除預設 v6 checkpoint/TensorBoard、動態負樣本、embeddings/rankings，不會刪除 backbone、single-chunk baseline、舊 chunkAgg 或原始 `processed`：

```bash
rm -rf -- \
  models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v6 \
  coliee_dataset/task1/2026/lht_process/modernBert-chunkAgg_scopeFilteredRaw_plus0_emptyExclude_v6 \
  coliee_dataset/task1/2026/lht_process/modernBert_fp_chunkAgg_v6_fp16
```

Runtime checkpoint、TensorBoard、embeddings、adaptive-negative JSON、`__pycache__` 均由 Git 忽略；v6 Python、README、tests 與 `.env` 應納入版本控制。
