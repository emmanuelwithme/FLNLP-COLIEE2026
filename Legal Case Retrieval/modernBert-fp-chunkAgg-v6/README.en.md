# modernBert-fp-chunkAgg-v6

[中文](README.md) | [Version and experiment history](../modernBert-fp-chunkAgg/EXPERIMENT_HISTORY.en.md) | [Task 1 overview](../README.en.md)

v6 is an isolated follow-up experiment. It trains the v5 multi-chunk network and loss from the continued-pretraining ModernBERT backbone, fixes the dynamic-negative policy to the single-chunk baseline formula `softmax(score / 1.0)`, and restores the baseline rule that a content-eligible positive may remain trainable outside scope. Because sampling policy and positive eligibility changed together, this is not a strict one-variable ablation; see the [version and experiment history](../modernBert-fp-chunkAgg/EXPERIMENT_HISTORY.en.md) for interpretation and the next recommended experiment. It never warm-starts from a single-chunk or older chunkAgg retrieval checkpoint and reads only `TASK1_CHUNKAGG_V6_*` settings.

There are no stages, grouped multi-positive LSEPair loss, warm-start gate, or fusion gate. The complete run uses one single-positive objective.

## Experiment Contract

- Queries, positives, negatives, and candidates all come from `processed`.
- Existing `train_qid.tsv`, `valid_qid.tsv`, and `query_candidate_scope_raw_plus0_empty_exclude.json` are fixed.
- Every document passed to the model must contain at least 100 non-whitespace characters.
- A labeled positive may be outside the query scope if it is content-eligible. Scope constrains only negatives and ranking.
- The negative pool is the complete eligible scope, minus self and every labeled positive for the query.
- Each `(query, positive)` row independently samples exactly 15 unique negatives without replacement.
- Different rows and epochs may sample the same negative again.
- Ranking applies scope, the content filter, and self exclusion before selecting top-5.
- Global validation always includes all 401 queries and 1656 gold pairs. Missing/short/empty-scope queries receive an empty prediction.

Formal preflight locks the current corpus to 7703 eligible documents, 1593 trainable qids, 6568 rows per epoch, a minimum pool of 50, 311 eligible scope-external positives, and the 401/1656 validation denominator.

## Model and Loss

Sentence-aware chunking produces at most three sequences of `[CLS] + up to 4094 body tokens + [SEP]`. Documents with more chunks use head/middle/tail coverage. Missing chunk slots are padding tensors masked by `chunk_mask`, not learned empty tokens.

Each chunk is independently encoded. Its `[CLS]` state passes through a zero-initialized residual projector. Normalized valid chunks form a normalized masked-mean base. A four-token transformer processes `[DOC] + chunk1 + chunk2 + chunk3`; learned positions are added to those four vectors and are not a fifth token.

```text
base = normalize(masked_mean(normalize(valid_chunk_vectors)))
raw_residual = fusion_output(fusion_transformer([DOC=base+bias], chunks)[DOC])
fusion_residual = raw_residual / (1 + ||raw_residual|| / 0.5)
document = normalize(base + fusion_residual)
```

The fusion output projection starts at zero, so `fused == mean` at initialization. There is no gate. Final L2 normalization prevents a three-chunk document from gaining magnitude merely because it has more chunks.

```text
L = L_document_InfoNCE
    + 0.2 * L_chunk_evidence_InfoNCE
    + 0.01 * mean(relu(||raw_fusion_residual|| - 0.5)^2)

e(q,d) = 0.1 * (logsumexp_j(q dot chunk_j / 0.1) - log(valid_chunks))
```

Both contrastive losses use the same positive and 15 negatives. Learnable training temperature starts at `0.55555` and is clamped to `[0.05, 2.0]`. Evidence pooling temperature is `0.1`. Dynamic-negative sampling temperature is a separate fixed value of `1.0`.

One logical mini-batch therefore contains 17 documents: one query, one positive, and 15 negatives, or at most 51 valid chunks. `TASK1_CHUNKAGG_V6_CHUNK_MICROBATCH_SIZE=1` lets the encoder process valid chunks in small groups instead of retaining all 51 4096-token sequences on the GPU at once. `gradient_accumulation_steps=4` accumulates four such mini-batches before one optimizer update; it does not combine them into one 61-negative contrastive row.

## Dynamic Negatives

Before every epoch, the current model re-encodes the eligible corpus and scores every candidate in each query's complete scope:

```text
p(d|q) = exp(score(q,d) - max_score) / sum_d' exp(score(q,d') - max_score)
```

This is the numerically stable form of `softmax(score / 1.0)`. There is no ESS calibration, schedule, top-k truncation, random/BM25 mixture, or replacement fallback. Epoch 1 uses the initial v6 model; epoch N uses the model at the end of epoch N-1. ESS, entropy, top-100 mass, and maximum probability are diagnostics only.

Each epoch writes only `adaptive_negative_for_train_epoch_NN.json`. It does not write a large similarity TREC TSV and never reuses an old JSON when starting or resuming.

## Defaults

The tracked `.env` contains the formal defaults. Same-name shell values take precedence.

```text
epochs / early stopping  20 / patience 5, threshold 0
batch / accumulation     1 / 4
rows / steps per epoch   6568 / 1642
encoder LR               5e-6
projector/fusion LR      1e-5
temperature LR           1e-4
optimizer / scheduler    fused AdamW / linear
warmup                    0.1
fp16 / bf16 / tf32       on / off / off
retrieval batch           2
workers / persistent      2 / off
logging                   step 1, then every 50 optimizer steps
best metric               eval_global_f1
```

## Validation and Training

From the repository root:

```bash
/home/lab/miniconda3/envs/FLNLP-COLIEE2026-WSL/bin/python \
  "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/preflight.py" --skip-model

CUDA_VISIBLE_DEVICES=0 \
/home/lab/miniconda3/envs/FLNLP-COLIEE2026-WSL/bin/python \
  "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/preflight.py"

/home/lab/miniconda3/envs/FLNLP-COLIEE2026-WSL/bin/python -m unittest discover \
  -s "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/tests" -v
```

Quick smoke test, isolated from the formal run:

```bash
CUDA_VISIBLE_DEVICES=0 \
TASK1_CHUNKAGG_V6_QUICK_TEST=1 \
TASK1_CHUNKAGG_V6_NUM_EPOCHS=1 \
TASK1_CHUNKAGG_V6_AUTO_RESUME=0 \
TASK1_CHUNKAGG_V6_OUTPUT_DIR=./models/chunkagg_v6_smoke \
TASK1_CHUNKAGG_V6_FINETUNE_DATA_DIR=./coliee_dataset/task1/2026/lht_process/chunkagg_v6_smoke/finetune_data \
python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/fine_tune/fine_tune.py"
```

Formal training uses `.env` directly:

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/fine_tune/fine_tune.py"
```

`TASK1_CHUNKAGG_V6_AUTO_RESUME=1` is the default. Re-running the command restores the newest v6 checkpoint, optimizer, scheduler, FP16 scaler, epoch/global step, and then recomputes the next epoch's negatives. Checkpoint contract version 6 rejects v5, older gate models, unversioned single-chunk checkpoints, and partial loads.

For a tmux run:

```bash
tmux new -s chunkagg-v6
cd /home/lab/FLNLP-COLIEE2026
CUDA_VISIBLE_DEVICES=0 python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/fine_tune/fine_tune.py"
```

Detach with `Ctrl-b d`; reconnect with `tmux attach -t chunkagg-v6`.

## TensorBoard

The trainer writes one run under the v6 output directory. Use the `tbfix` environment; port zero requests an available port:

```bash
conda run -n tbfix tensorboard \
  --logdir models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v6/tb \
  --port 0
```

No `--logdir_spec` or stage directories are needed. Training appears at step 1 and every 50 optimizer steps. Full retrieval metrics appear once per epoch. Important tags include train loss/temperature/LR groups, document/evidence/fusion diagnostics, observational sampling diagnostics, `eval/global_f1`, and fused/mean/first validation F1, precision, and recall.

## Inference, Ranking, and Evaluation

No `run_*.sh` or `run_*.py` wrappers are needed:

```bash
CUDA_VISIBLE_DEVICES=0 python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/inference.py"
python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/similarity_and_rank.py"

TASK1_RETRIEVAL_MODEL_NAME=modernBert_fp_chunkAgg_v6_fp16 \
TASK1_MODEL_RESULTS_DIR=./coliee_dataset/task1/2026/lht_process/modernBert_fp_chunkAgg_v6_fp16 \
TASK1_SCOPE_PATH=./coliee_dataset/task1/2026/lht_process/scope_compare/query_candidate_scope_raw_plus0_empty_exclude.json \
python "Legal Case Retrieval/utils/eval.py"
```

Inference selects the best `eval_global_f1` checkpoint and encodes eligible `processed` once. Query and candidate roles share `embeddings/processed.pkl`. Ranking writes scoped top-5 dot/cos TREC files.

Canonical same-condition baseline comparison:

```bash
CUDA_VISIBLE_DEVICES=0 \
python "Legal Case Retrieval/modernBert-fp-chunkAgg-v6/compare_canonical.py"
```

The historical baseline F1 `0.164518` remains reference-only; the generated `canonical_comparison.json` is the controlled result.

## Cleanup

This removes only default v6 runtime artifacts:

```bash
rm -rf -- \
  models/modernBERT_contrastive_adaptive_fp_fp16_chunkAgg_scopeFilteredRaw_plus0_emptyExclude_2026_v6 \
  coliee_dataset/task1/2026/lht_process/modernBert-chunkAgg_scopeFilteredRaw_plus0_emptyExclude_v6 \
  coliee_dataset/task1/2026/lht_process/modernBert_fp_chunkAgg_v6_fp16
```

Python sources, READMEs, tests, and `.env` are tracked. Checkpoints, TensorBoard events, embeddings, adaptive-negative JSON, and caches remain ignored.
