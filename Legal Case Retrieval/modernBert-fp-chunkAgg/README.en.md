# modernBert-fp-chunkAgg

[中文](README.md) | [Version and experiment history](EXPERIMENT_HISTORY.en.md) | [Task 1 Overview](../README.en.md)

This directory contains the end-to-end multi-chunk dense retriever for Task 1. A document contributes at most three 4096-token chunks, each chunk is encoded independently by ModernBERT, and a four-token fusion transformer produces one document vector. Every row still has one positive; training combines document InfoNCE and chunk-evidence InfoNCE over that same positive and negative set.

## Entrypoints

- `fine_tune/fine_tune.py`: training, dynamic negatives, and full valid retrieval evaluation.
- `fine_tune/modernbert_contrastive_model.py`: the single shared training/inference model definition.
- `inference.py`: loads the best valid-F1 checkpoint and writes embeddings.
- `similarity_and_rank.py`: writes scoped train/valid rankings.
- `.env`: default paths and hyperparameters.

The Python scripts can be run directly; no wrapper or extra `run_*.py` entrypoint is required.

## Document Vector

1. Tokenize the full text and explicitly form every valid chunk as `[CLS] + at most 4094 content tokens + [SEP]`, for a total length of at most 4096 tokens and at most three model inputs per document. Do not rely on `PreTrainedTokenizerFast.build_inputs_with_special_tokens()` here: ModernBERT keeps its template in the backend post-processor, so that Python method may not insert special tokens.
2. Prefer the nearest sentence boundary; use a hard token-budget split only as a fallback.
3. If a document produces more than three chunks, select deterministic first/middle/last chunks after complete sentence-aware splitting.
4. Missing chunk positions are padding and are fully hidden by `chunk_mask`.
5. Use a zero-output residual projector, `LayerNorm(CLS + MLP(CLS))`, so initialization preserves the continued-pretraining representation.
6. L2-normalize every projected chunk, then normalize their masked mean into the base vector.
7. `[DOC]` is not a standalone content-free token. Seed it with `base + learned_doc_bias`, add position embeddings to `[DOC] + chunk1 + chunk2 + chunk3`, and run one pre-norm fusion transformer block.
8. Apply a zero-initialized residual projection followed by a smooth norm cap:

```text
base = normalize(masked_mean(normalized_chunks))
doc_seed = base + learned_doc_bias
raw_residual = fusion_transformer(doc_seed, chunks)[DOC]
fusion_residual = raw_residual / (1 + ||raw_residual|| / 0.5)
doc = normalize(base + fusion_residual)
```

There is no gate. The applied fusion residual norm is always below `0.5`, so the new branch cannot overwrite the stable chunk mean. A small penalty discourages the raw branch from remaining above the cap. Initialization has `fused == mean`.

```text
L = L_document_InfoNCE + 0.2 * L_chunk_evidence_InfoNCE
    + 0.01 * mean(relu(||raw_fusion_residual|| - 0.5)^2)
e(q,d) = tau_e * (logsumexp_j(q dot chunk_j / tau_e) - log(valid_chunks))
```

Both losses remain single-positive. `tau_e=0.1` is separate from negative-sampling temperature and document InfoNCE temperature.

The design is informed by [PARADE](https://arxiv.org/abs/2008.09093), which shows benefits from passage-representation aggregation but also reports cases where simpler aggregation is preferable; this is why mean and first-chunk retrieval remain explicit ablations. [ReZero](https://arxiv.org/abs/2003.04887) and [LayerScale](https://arxiv.org/abs/2103.17239) motivate starting residual branches with a small contribution. [ANCE](https://arxiv.org/abs/2007.00808) supports model-mined global negatives, while [RocketQA](https://aclanthology.org/2021.naacl-main.466/) and [ANCE-Tele](https://aclanthology.org/2022.emnlp-main.445/) document false-negative and training-stability risks. The exact ESS curriculum is a local engineering inference from those risks and the v4 curves, not a schedule copied from those papers.

## Scope And Labels

Training, full validation, and offline ranking require the same scope:

```text
coliee_dataset/task1/2026/lht_process/scope_compare/query_candidate_scope_raw_plus0_empty_exclude.json
```

- A training positive must be in the query scope and present in candidate retrieval scores.
- Queries, positives, negatives, and retrieval candidates must contain at least 100 non-whitespace characters by default. `TASK1_CHUNKAGG_MIN_DOCUMENT_CHARS` controls the threshold, and Dataset reads enforce it again.
- The current `processed` corpus keeps 7703 of 7708 documents and rejects `015738(0)`, `039404(0)`, `058076(10)`, `092745(10)`, and `085084(11)`; parentheses show non-whitespace character counts. `085084` is a train query, `015738` is a valid query, `058076` and `039404` are train positives, and `092745` appears only as a corpus candidate.
- Out-of-scope positive pairs do not become training rows, but every labeled positive is still excluded from negatives.
- The negative pool is the complete scoped candidate set minus self and all labeled positives.
- Before any epoch-0 update, the initial chunkAgg model re-encodes and scores the complete scope. No BM25 or pre-generated training negatives are read.
- From epoch 1 onward, mining runs before every epoch. The weights at that point are the weights left by the previous epoch, so the new probabilities come from the previous epoch-end model.
- There is no update-frequency switch that can skip an epoch.
- Sampling remains full-pool softmax with no top-k cutoff or random mixture. Per-query temperature follows an easy-to-hard ESS curriculum: epochs `0/2/4/6/8` start targets `4000/2000/1000/500/300`, then remain at 300. A target larger than the query pool is clamped to the pool size.
- Every `(query, positive)` independently resamples negatives.
- Sampling is without replacement within a row. A smaller pool is used as-is.
- The same negative may reappear in another batch or epoch.
- Formal training always disables cross-step late-cache negatives, so every negative in the loss is explicitly sampled from that epoch's model-similarity distribution. A stale shell variable with the former cache setting cannot change this behavior.
- Valid rankings include only content-eligible scoped candidates and remove self before selecting top 5. Final F1 still uses all labeled valid queries as the denominator; an ineligible query or missing prediction counts as empty.

The existing `TASK1_CHUNKAGG_VALID_JSON_PATH` filename contains `bm25`, but that file is used only for auxiliary fixed-set `eval_loss/acc`. It never supplies training negatives and never participates in epoch mining. Model selection via `eval_global_f1` still performs fresh full scoped retrieval with the current model.

Dynamic sampling uses a streaming score consumer. It does not retain all query-candidate Python dictionaries or write a large per-epoch TREC ranking. Corpus text is loaded one encoding batch at a time. ChunkAgg explicitly enables embedding reuse, so query embeddings are reused from candidate embeddings when both use `processed`; the shared retrieval API keeps reuse disabled by default to preserve legacy single-chunk query/candidate encoding behavior. Only the sampled `adaptive_negative_epochN_train.json` is retained.

Retrieval, epoch-end evaluation, and offline inference honor `TASK1_CHUNKAGG_FP16` / `TASK1_CHUNKAGG_BF16`. When both are `0`, they use float32 instead of silently forcing fp16.

## Formal Defaults

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

Precision and resource defaults:

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

Persistent workers must remain disabled because the dataset replaces negatives at every epoch. A persistent worker can keep an obsolete dataset copy. The code forces this setting to `False` even if the environment requests `1`.

The optimizer is fused AdamW with a linear scheduler. Encoder, projector, fusion, and temperature have separate LR groups; bias and normalization parameters use no-decay groups.

## Workflow

Run from the repository root:

```bash
cd /home/lab/FLNLP-COLIEE2026
```

Train with `.env` defaults:

```bash
python "Legal Case Retrieval/modernBert-fp-chunkAgg/fine_tune/fine_tune.py"
```

Start from an empty output directory:

```bash
CUDA_VISIBLE_DEVICES=0 \
TASK1_CHUNKAGG_QUICK_TEST=0 \
TASK1_AUTO_RESUME=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/fine_tune/fine_tune.py"
```

Resume an interrupted run:

```bash
CUDA_VISIBLE_DEVICES=0 \
TASK1_AUTO_RESUME=1 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/fine_tune/fine_tune.py"
```

`TASK1_AUTO_RESUME` defaults to `1`. Resume restores model, optimizer, scheduler, scaler, global step, and TensorBoard step, then regenerates the current epoch's negatives with the restored model.

The current checkpoint architecture/input-contract version is `5`. Version 4 used an unbounded fusion residual, while version 3 used the old chunk input without `[CLS]/[SEP]`; both are rejected explicitly. `.env` points to fresh `_v5` output/data directories, preserving old models without mixing checkpoints or curves.

Generate embeddings:

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/inference.py"
```

Generate rankings:

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/similarity_and_rank.py"
```

Evaluate:

```bash
TASK1_RETRIEVAL_MODEL_NAME=modernBert_fp_chunkAgg_v5_fp16 \
TASK1_MODEL_RESULTS_DIR=./coliee_dataset/task1/2026/lht_process/modernBert_fp_chunkAgg_v5_fp16 \
TASK1_SCOPE_PATH=./coliee_dataset/task1/2026/lht_process/scope_compare/query_candidate_scope_raw_plus0_empty_exclude.json \
python "Legal Case Retrieval/utils/eval.py"
```

## TensorBoard

The Trainer writes one event stream under `TASK1_CHUNKAGG_OUTPUT_DIR/tb/`.

```bash
cd /home/lab/FLNLP-COLIEE2026
conda activate tbfix
tensorboard --logdir models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v5/tb
```

Training scalars are written at optimizer step 1 and every `TASK1_CHUNKAGG_LOGGING_STEPS=50` steps. Full retrieval metrics are written once per epoch. Useful tags include:

- `train/loss`
- `train/temperature`
- `train/diagnostics/loss_document`
- `train/diagnostics/loss_evidence`
- `train/diagnostics/score_margin`
- `train/diagnostics/evidence_dilution_index`
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
- `train/sampling/source_is_model` (must remain `1` at every epoch)
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

All three aggregation curves reuse one chunk-encoding pass.

In v5, `base` is unit-normalized and the applied residual norm is below `0.5`, so `train/diagnostics/base_fused_cosine` should not fall below approximately `sqrt(1-0.5^2)=0.866`. If `fusion_residual_cap_utilization` stays near `1` early while the penalty rises and fused F1 remains below mean F1, lower the fusion LR or raise the penalty before considering a wider cap.

## Quick Test

Run the read-only preflight before formal training. It prints content filtering, labels/scope, the global metric denominator, the ESS curriculum, the model-only dynamic-negative contract and a real initial-model scoring/sampling probe, Dataset, collator, 4096-token chunks, the real v5 bounded-fusion model, InfoNCE, and gradients. It writes no checkpoint, TensorBoard event, or adaptive-negative JSON:

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/preflight.py"
```

Proceed only after the final `Stage 9: result` reports `PASS`.

```bash
QT_RUN_ID=$(date +%Y%m%d_%H%M%S)
CUDA_VISIBLE_DEVICES=0 \
TASK1_CHUNKAGG_QUICK_TEST=1 \
TASK1_AUTO_RESUME=0 \
TASK1_CHUNKAGG_OUTPUT_DIR="./models/chunkagg_quick_${QT_RUN_ID}" \
TASK1_CHUNKAGG_FINETUNE_DATA_DIR="./coliee_dataset/task1/2026/lht_process/chunkagg_quick_${QT_RUN_ID}/finetune_data" \
python "Legal Case Retrieval/modernBert-fp-chunkAgg/fine_tune/fine_tune.py"
```

Quick test selects queries with at least one scoped positive and negative, avoiding an empty training set caused by an unrelated random candidate subset. Global retrieval F1 can still be near zero because the metric denominator remains global.
