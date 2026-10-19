# modernBert-fp-chunkAgg

[English](README.en.md) | [版本與實驗紀錄](EXPERIMENT_HISTORY.md) | [Task 1 總覽](../README.md)

這個目錄提供 Task 1 的端到端多 chunk dense retriever。每篇文件最多選 3 個 4096-token chunks，各 chunk 個別經 ModernBERT 編碼，再由 4-token fusion transformer 產生一個文件向量。每筆資料仍只有一個 positive；loss 是 document InfoNCE 加上同一組 positive/negatives 的 chunk-evidence InfoNCE，不使用 grouped multi-positive loss。

## 入口程式

- `fine_tune/fine_tune.py`: 訓練、動態負樣本與完整 valid retrieval 評估。
- `fine_tune/modernbert_contrastive_model.py`: 訓練與推論共用的唯一模型定義。
- `inference.py`: 載入 valid F1 最佳 checkpoint 並輸出 embeddings。
- `similarity_and_rank.py`: 使用同一份 scope 輸出 train/valid ranking。
- `.env`: 預設超參數與路徑。

可直接依序執行三個 Python 程式，不需要 shell wrapper 或額外的 `run_*.py`。

## 文件向量

1. 對全文 tokenization，每個有效 chunk 明確組成 `[CLS] + 最多 4094 個正文 tokens + [SEP]`，總長最多 `4096` tokens；最多送入 `3` chunks。這裡不能依賴 `PreTrainedTokenizerFast.build_inputs_with_special_tokens()`，因為 ModernBERT 的 template 位於 backend post-processor，該 Python method 可能不加入 special tokens。
2. 優先回退到最近句界切分；沒有句界時才按 token budget 硬切。
3. 文件切出超過 3 chunks 時，預設從完整句界 chunks 選 `first / middle / last`，不再直接丟掉尾端。
4. 不存在的 chunk 位置以 padding 填入，並由 `chunk_mask` 完全遮掉，不是可學習的 empty chunk token。
5. `[CLS]` 經 residual projector：`LayerNorm(CLS + MLP(CLS))`；MLP 輸出層初始為零，因此起訓時保留 continued-pretraining 表示。
6. 每個有效 chunk 先做 L2 normalization，再將 masked mean 正規化成穩定基底。
7. `[DOC]` 不是獨立的無內容 token；它以 `base + learned_doc_bias` 初始化，再和 `chunk1 + chunk2 + chunk3` 加上位置向量，形成固定最多 4 tokens 的 pre-norm fusion transformer 輸入。
8. fusion 輸出經 zero-initialized residual projection，再做 smooth norm cap，最終為：

```text
base = normalize(masked_mean(normalized_chunks))
doc_seed = base + learned_doc_bias
raw_residual = fusion_transformer(doc_seed, chunks)[DOC]
fusion_residual = raw_residual / (1 + ||raw_residual|| / 0.5)
doc = normalize(base + fusion_residual)
```

這個設計沒有 gate。applied fusion residual 的 norm 永遠小於 `0.5`，因此新 fusion 分支不能像 v4 一樣把穩定的 chunk mean 完全蓋掉；另以小權重懲罰 raw residual 超過 `0.5`，避免分支長期卡在 cap 飽和區。projector 與 fusion 都使用保守 residual 初始化，起訓時 `fused == mean`。

訓練 loss：

```text
L = L_document_InfoNCE + 0.2 * L_chunk_evidence_InfoNCE
    + 0.01 * mean(relu(||raw_fusion_residual|| - 0.5)^2)
e(q,d) = tau_e * (logsumexp_j(q dot chunk_j / tau_e) - log(valid_chunks))
```

兩項 loss 都使用同一個 positive 與同一批顯式 negatives。`tau_e=0.1` 只控制 chunk evidence 的 smooth-max；它不是負樣本 sampling temperature，也不是 document InfoNCE training temperature。

設計依據：PARADE 顯示 passage representation aggregation 對需要跨段落證據的文件排序有幫助，但也指出 relevance 集中時簡單 aggregation 可能更好，因此本模型保留 mean/first retrieval ablation，而不是假設 fusion 必勝。ReZero 與 LayerScale 類工作顯示深層 residual branch 從很小的貢獻開始通常較穩定；v5 的 zero initialization 與 norm cap 是針對本次 v4 residual 爆增現象採用的保守實作。ANCE 支持以當前 retriever 挖掘全域 negatives；RocketQA 與 ANCE-Tele 同時說明過難或反覆更新的 negatives 可能帶來 false negatives、訓練震盪或遺忘，因此 ESS curriculum 是本專案根據這些風險與本地曲線做出的工程推論，不是論文原封不動的 schedule。

- [PARADE: Passage Representation Aggregation for Document Reranking](https://arxiv.org/abs/2008.09093)
- [ReZero is All You Need](https://arxiv.org/abs/2003.04887)
- [Going Deeper With Image Transformers (LayerScale)](https://arxiv.org/abs/2103.17239)
- [ANCE: Approximate Nearest Neighbor Negative Contrastive Learning](https://arxiv.org/abs/2007.00808)
- [RocketQA](https://aclanthology.org/2021.naacl-main.466/)
- [ANCE-Tele](https://aclanthology.org/2022.emnlp-main.445/)

## Scope 與標註

訓練、完整驗證與離線 ranking 都強制使用：

```text
coliee_dataset/task1/2026/lht_process/scope_compare/query_candidate_scope_raw_plus0_empty_exclude.json
```

- 訓練 positive 必須在該 query 的 scope 內且存在於 candidate retrieval scores。
- query、positive、negative 與 retrieval candidates 預設都必須至少有 `100` 個非空白字元。此條件由 `TASK1_CHUNKAGG_MIN_DOCUMENT_CHARS` 控制，Dataset 讀檔時還會再檢查一次。
- 目前 `processed` 的 7708 篇中保留 7703 篇，排除 `015738(0)`、`039404(0)`、`058076(10)`、`092745(10)`、`085084(11)`；括號內是非空白字元數。其中 `085084` 是 train query、`015738` 是 valid query，`058076` 與 `039404` 各是 train positive，`092745` 只出現在 candidate corpus。
- 不在 scope 內的 positive pair 不建立訓練樣本，但所有 labeled positives 仍一律從 negative pool 排除。
- negative pool 為完整 scoped candidates，排除 self 與所有 labeled positives。
- epoch 0 在任何參數更新前，先用初始 chunkAgg 模型對完整 scope 重新編碼與計分；不讀 BM25 或預先產生的 training negatives。
- epoch 1 起固定在每個 epoch 開始前重挖一次。此時模型權重就是上一個 epoch 結束的權重，因此抽樣機率來自上一個 epoch 結束模型的 full-scope similarity。
- 不提供跳過某些 epoch 的更新頻率開關，確保每輪訓練資料都依最新模型重抽。
- 抽樣機率仍為完整 pool 的 `softmax(similarity / sampling_temperature)`，不做 top-k 截斷，也不混 random pool。
- 預設每個 query 依模型分數尺度自動校準 sampling temperature，使用先廣後難的 ESS curriculum：epoch `0/2/4/6/8` 起分別為 `4000/2000/1000/500/300`，epoch 8 後維持 300。目標大於該 query 的 pool 時會自動限制為 pool size。
- 每一筆 `(query, positive)` 都獨立重抽 negatives；同 query 的不同 positive 不共用固定 negative set。
- 同一筆樣本使用 without-replacement；pool 小於上限時直接使用全部，不補抽。
- 不同 batch 或 epoch 可以再次抽到相同 negative。
- 正式流程固定不加入跨 step late-cache negatives，確保 loss 內所有 negatives 都是該 epoch 由模型 similarity distribution 抽出的顯式 negatives；舊的同名 shell 環境變數也不會改變這項行為。
- valid ranking 只包含同時通過 content filter 與 scope 的 candidates，且在取 top-5 前先排除 self；最終 F1 仍以完整 labeled valid query set 為分母，內容不足、scope 空或缺少 prediction 的 query 都以空答案計分。

`TASK1_CHUNKAGG_VALID_JSON_PATH` 的既有檔名含有 `bm25`，但它只供固定的輔助 `eval_loss/acc` 使用，不會提供訓練 negatives，也不參與任何 epoch 的動態抽樣。選模使用的 `eval_global_f1` 仍由當前模型完整重做 scoped retrieval。

動態取樣採 streaming score consumer，不會把全部 query-candidate 分數與大型 TREC ranking 同時留在記憶體或寫入 SSD。語料文字按 encoding batch 才讀取；chunkAgg 會明確啟用 embedding reuse，因此 query 與 candidate 同為 `processed` 時直接重用已算好的 candidate embedding。共用 retrieval API 預設不啟用 reuse，以維持舊 single-chunk 模型分別編碼 query 與 candidate 的行為。每個 epoch 只保留實際使用的 `adaptive_negative_epochN_train.json`。

retrieval、epoch-end 評估與離線 inference 都遵守 `TASK1_CHUNKAGG_FP16` / `TASK1_CHUNKAGG_BF16`；兩者皆為 `0` 時使用 float32，不會再固定偷用 fp16。

## 正式訓練設定

預設沿用成功 single-chunk 訓練的 temperature、optimizer 與 backbone LR 策略，但對新 fusion 分支使用較保守的 LR、bounded residual 與負樣本難度 curriculum：

```bash
TASK1_INIT_TEMPERATURE=0.55555
TASK1_CHUNKAGG_TEMPERATURE_MIN=0.05
TASK1_CHUNKAGG_TEMPERATURE_MAX=2.0
TASK1_CHUNKAGG_NUM_EPOCHS=20
TASK1_CHUNKAGG_EARLY_STOPPING_PATIENCE=5
TASK1_CHUNKAGG_ENCODER_LR=5e-6
TASK1_CHUNKAGG_PROJECTOR_LR=1e-5
TASK1_CHUNKAGG_FUSION_LR=1e-5
TASK1_CHUNKAGG_TEMPERATURE_LR=1e-4
TASK1_CHUNKAGG_WARMUP_RATIO=0.1
TASK1_CHUNKAGG_LR_SCHEDULER_TYPE=linear
TASK1_CHUNKAGG_TRAIN_BATCH_SIZE=1
TASK1_CHUNKAGG_GRAD_ACCUM_STEPS=4
TASK1_CHUNKAGG_MAX_NEGATIVES=15
TASK1_CHUNKAGG_EVIDENCE_LOSS_WEIGHT=0.2
TASK1_CHUNKAGG_EVIDENCE_POOL_TEMPERATURE=0.1
TASK1_CHUNKAGG_FUSION_RESIDUAL_MAX_NORM=0.5
TASK1_CHUNKAGG_FUSION_RESIDUAL_PENALTY_WEIGHT=0.01
TASK1_CHUNKAGG_MIN_DOCUMENT_CHARS=100
TASK1_CHUNKAGG_SAMPLING_TARGET_ESS_SCHEDULE=0:4000,2:2000,4:1000,6:500,8:300
TASK1_CHUNKAGG_SAMPLING_TARGET_ESS=300
TASK1_CHUNKAGG_SAMPLING_TEMPERATURE_SCHEDULE=0:0.20,3:0.15,5:0.10
```

精度與資源設定：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TASK1_CHUNKAGG_FP16=1
TASK1_CHUNKAGG_BF16=0
TASK1_CHUNKAGG_ENABLE_TF32=0
TASK1_RETRIEVAL_BATCH_SIZE=2
TASK1_CHUNKAGG_NUM_WORKERS=2
TASK1_CHUNKAGG_PERSISTENT_WORKERS=0
TASK1_CHUNKAGG_TEXT_CACHE_SIZE=4096
TASK1_CHUNKAGG_CHUNK_CACHE_SIZE=1024
```

`persistent_workers` 必須關閉。訓練資料集會在每個 epoch 開始前替換 negatives；persistent worker process 可能保留舊 dataset copy，導致模型重算了 negatives 卻仍訓練舊資料。程式即使讀到 `=1` 也會強制使用 `False`。

Optimizer 使用 fused AdamW 與 linear scheduler。參數依 encoder、projector、fusion、temperature 分組，bias 與 normalization 參數放在 no-decay group。TensorBoard 的 `train/lr_group_*` 對應：

```text
encoder_decay / encoder_no_decay
projector_decay / projector_no_decay
fusion_decay / fusion_no_decay
log_temperature
```

## 執行流程

從 repo root 執行：

```bash
cd /home/lab/FLNLP-COLIEE2026
```

### 1. 訓練

`.env` 已包含正式預設值：

```bash
python "Legal Case Retrieval/modernBert-fp-chunkAgg/fine_tune/fine_tune.py"
```

明確指定 GPU 並從空 output directory 開始：

```bash
CUDA_VISIBLE_DEVICES=0 \
TASK1_CHUNKAGG_QUICK_TEST=0 \
TASK1_AUTO_RESUME=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/fine_tune/fine_tune.py"
```

中斷後續訓：

```bash
CUDA_VISIBLE_DEVICES=0 \
TASK1_AUTO_RESUME=1 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/fine_tune/fine_tune.py"
```

`TASK1_AUTO_RESUME` 預設為 `1`。續訓會恢復 model、optimizer、scheduler、scaler、global step 與 TensorBoard step，並用恢復後的模型重新推論當輪動態 negatives。

目前 checkpoint architecture/input-contract version 為 `5`。v4 使用未受限 fusion residual，v3 使用缺少 `[CLS]/[SEP]` 的舊 chunk 輸入，因此 v4 與更舊 checkpoint 都會被明確拒絕；`.env` 已改用全新的 `_v5` output/data directory，舊模型不會被刪除或混入新曲線。

### 2. 產生 embeddings

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/inference.py"
```

`inference.py` 以 `eval_global_f1` 選最佳 checkpoint，並輸出 `processed` 與 `processed_new` embeddings。

### 3. 產生 ranking

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/similarity_and_rank.py"
```

預設 query/candidate 都選 `processed` embeddings。改用 `processed_new` query：

```bash
TASK1_CHUNKAGG_QUERY_EMB_SOURCE=processed_new \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/similarity_and_rank.py"
```

### 4. 產出績效表

```bash
TASK1_RETRIEVAL_MODEL_NAME=modernBert_fp_chunkAgg_v5_fp16 \
TASK1_MODEL_RESULTS_DIR=./coliee_dataset/task1/2026/lht_process/modernBert_fp_chunkAgg_v5_fp16 \
TASK1_SCOPE_PATH=./coliee_dataset/task1/2026/lht_process/scope_compare/query_candidate_scope_raw_plus0_empty_exclude.json \
python "Legal Case Retrieval/utils/eval.py"
```

結果會追加到 `Legal Case Retrieval/results/experiment_results.csv`。

## TensorBoard

程式只使用 Hugging Face Trainer 的單一 event stream：

```text
TASK1_CHUNKAGG_OUTPUT_DIR/tb/
```

啟動方式：

```bash
cd /home/lab/FLNLP-COLIEE2026
conda activate tbfix
tensorboard --logdir models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v5/tb
```

預設在 optimizer step 1 與每 `TASK1_CHUNKAGG_LOGGING_STEPS=50` steps 記錄。完整 retrieval 每個 epoch 記錄一次。常看 tags：

- `train/loss`
- `train/temperature`
- `train/diagnostics/loss_document`
- `train/diagnostics/loss_evidence`
- `train/diagnostics/positive_score`
- `train/diagnostics/hardest_negative_score`
- `train/diagnostics/score_margin`
- `train/diagnostics/evidence_dilution_index`
- `train/diagnostics/chunk_mean_pre_norm`
- `train/diagnostics/fusion_raw_residual_norm`
- `train/diagnostics/fusion_residual_norm`
- `train/diagnostics/fusion_residual_cap_utilization`
- `train/diagnostics/base_fused_cosine`
- `train/diagnostics/loss_fusion_residual_penalty`
- `train/diagnostics/fusion_attention_entropy`
- `train/diagnostics/fusion_attention_self_mass`
- `train/diagnostics/fusion_attention_chunk_mass`
- `train/cache/current_size`
- `train/sampling/effective_sample_size_median`
- `train/sampling/temperature_median`
- `train/sampling/top100_probability_mass_median`
- `train/sampling/source_is_model`（每個 epoch 都應為 `1`）
- `train/sampling/target_ess`
- `train/lr_group_*`
- `eval/loss`
- `eval/global_f1`
- `eval/retrieval_valid_top5_f1`
- `eval/retrieval_valid_top5_precision`
- `eval/retrieval_valid_top5_recall`
- `eval/retrieval_valid_fused_top5_f1`
- `eval/retrieval_valid_mean_top5_f1`
- `eval/retrieval_valid_first_top5_f1`

三條 aggregation F1 使用同一次 chunk encoding，只額外做三次低成本相似度 ranking。它們可直接判斷 fusion 是否真的優於 mean/first，而不會把 encoder 重新跑三次。

v5 中 `base` 為單位向量且 applied residual norm 小於 `0.5`，所以理論上 `train/diagnostics/base_fused_cosine` 不應低於約 `sqrt(1-0.5^2)=0.866`。若 `fusion_residual_cap_utilization` 很早就長期接近 `1`，同時 penalty 上升，代表 raw fusion 仍在嘗試撞上限；若 fused F1 仍低於 mean F1，應優先降低 fusion LR 或提高 penalty，而不是放寬 cap。

## Quick Test

正式訓練前可先跑只讀 preflight。它會列出 content filter、labels/scope、全域 F1 分母、ESS curriculum、model-only dynamic-negative 契約與真實模型打分抽樣 probe、Dataset、collator、4096-token chunks、真實 v5 bounded fusion、InfoNCE 與 gradient 的資料結構；不建立 checkpoint、TensorBoard event 或 adaptive-negative JSON：

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/preflight.py"
```

最後看到 `Stage 9: result` 與 `PASS` 才表示所有契約通過。

```bash
QT_RUN_ID=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 \
TASK1_CHUNKAGG_QUICK_TEST=1 \
TASK1_AUTO_RESUME=0 \
TASK1_CHUNKAGG_OUTPUT_DIR="./models/chunkagg_quick_${QT_RUN_ID}" \
TASK1_CHUNKAGG_FINETUNE_DATA_DIR="./coliee_dataset/task1/2026/lht_process/chunkagg_quick_${QT_RUN_ID}/finetune_data" \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/fine_tune/fine_tune.py"
```

Quick test 會挑選至少具有一個 scoped positive 與 negative 的 query，避免隨機小候選集導致 0 筆訓練資料。完整 retrieval F1 仍以全域 labels 為分母，因此數值可能接近 0；重點是驗證 chunking、forward/backward、動態抽樣、評估與 checkpoint 流程。
