# modernBert-fp

[中文](README.md) | [Task 1 Overview](../README.en.md)

This directory is the maintained dense retrieval pipeline for Task 1. It is responsible for:

- contrastive fine-tuning of the ModernBERT encoder
- generating embeddings from the best checkpoint
- writing train / valid / test similarity rankings
- feeding both the retrieval-only submission path and the LightGBM rerank pipeline

If your current goal is experimental test-set prediction, this directory is the main dense-encoder workflow for Task 1.

## Files You Will Actually Use

- `fine_tune/fine_tune.py`
  The maintained contrastive fine-tuning entrypoint. It recomputes similarities and resamples hard negatives every epoch.
- `inference.py`
  Automatically selects the checkpoint with the best `eval_global_f1` and generates embeddings for `processed` / `processed_new` or `processed_test`.
- `similarity_and_rank.py`
  Loads embeddings and writes dot / cosine TREC rankings.
- `find_best_model.py`
  Selects the best checkpoint from a checkpoint directory using a chosen metric.
- `train_modernbert_caselaw_fp.py`
  Continued pretraining for the backbone checkpoint used downstream, currently `modernbert-caselaw-accsteps-fp/checkpoint-29000`.

Files still present but not part of the main maintained path:

- `inference-noSFT.py`
- `inference-test-noSFT.py`
- `similarity_and_rank_noSFT.py`

## Two Common Usage Scenarios

### Scenario A: train a new dense model

1. Run `bash run_pre_finetune_2026.sh` from repo root
2. Run `python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"`
3. For train / valid verification, run `bash run_train_valid_inference_eval_2026.sh`
4. For retrieval-only test submission, run `bash run_test_retrieval_2026.sh`
5. For LightGBM reranking, make sure the downstream stage is reading the checkpoint you actually want, then run `bash run_ltr_feature_train_valid_test_2026.sh`

### Scenario B: skip retraining and use an existing checkpoint for test prediction

1. Make sure the checkpoint tree and backbone checkpoint are already available
2. Run `bash run_test_retrieval_2026.sh`
3. If you want LTR reranking, continue with `bash run_ltr_feature_train_valid_test_2026.sh`
4. If you only want to rerun cutoff search, run `bash run_ltr_cutoff_postprocess_2026.sh`

## Main Input Requirements

### For training

- `coliee_dataset/task1/2026/processed/`
- `coliee_dataset/task1/2026/train_qid.tsv`
- `coliee_dataset/task1/2026/valid_qid.tsv`
- `coliee_dataset/task1/2026/task1_train_labels_2026.json`
- `coliee_dataset/task1/2026/task1_train_labels_2026_train.json`
- `coliee_dataset/task1/2026/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json`
- `coliee_dataset/task1/2026/lht_process/modernBert/query_candidate_scope.json`

### For inference

- `modernbert-caselaw-accsteps-fp/checkpoint-29000`
- the target checkpoint root
- `processed/` or `processed_test/`

### Additional requirements for the retrieval-only test pipeline

- `coliee_dataset/task1/2026/task1_test_files_2026/`
- `coliee_dataset/task1/2026/task1_test_no_labels_2026.json`

## What It Writes

### 1. Fine-tune outputs

`fine_tune.py` writes:

- checkpoint directories
- TensorBoard logs
- per-epoch similarity files and adaptive-negative JSONs

The current built-in naming convention is:

- model output: `./modernBERT_contrastive_adaptive_fp_fp16_scopeFilteredRaw_<YEAR>`
- fine-tune artifacts: `coliee_dataset/task1/<YEAR>/lht_process/modernBert/finetune_data/`

Common files:

- `similarity_scores_epoch<E>.tsv`
- `adaptive_negative_epoch<E>_train.json`
- `similarity_scores_<epoch>_eval_train.tsv`
- `similarity_scores_<epoch>_eval_valid.tsv`

### 2. Embedding outputs

Train / valid mode:

- `processed/processed_document_modernBert_fp_fp16_embeddings.pkl`
- `processed_new/processed_new_document_modernBert_fp_fp16_embeddings.pkl`

Test mode:

- `processed_test/processed_test_document_modernBert_fp_fp16_embeddings.pkl`
- `processed_test/processed_test_query_modernBert_fp_fp16_embeddings.pkl`

### 3. Ranking outputs

Train / valid:

- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_train.tsv`
- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_train.tsv`
- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_valid.tsv`
- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_valid.tsv`

Test:

- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_dot_test.tsv`
- `lht_process/modernBert_fp_fp16/output_modernBert_fp_fp16_cos_test.tsv`

### 4. Retrieval-only submission outputs

`run_test_retrieval_2026.sh` also writes:

- `lht_process/submission/task1_FLNLPBM25.txt`
- `lht_process/submission/task1_FLNLPEMBED.txt`

## Recommended Operating Order

### 1. Build the preparation artifacts

```bash
bash run_pre_finetune_2026.sh
```

This step lives outside the directory, but it prepares the actual inputs required by training here.

### 2. Start fine-tuning

```bash
python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
```

Key current behaviors of `fine_tune.py`:

- query / positive / negative share one encoder
- the loss is document-level InfoNCE / CrossEntropy
- `TASK1_RETRIEVAL_BATCH_SIZE` affects retrieval evaluation and adaptive sampling
- `TASK1_INIT_TEMPERATURE` sets the initial value of the learnable temperature
- validation retrieval F1 is used for best-model selection
- `TASK1_AUTO_RESUME=1` automatically resumes from the latest checkpoint
- `TASK1_RESUME_FROM_CHECKPOINT` can explicitly point to a checkpoint

### 3. Generate train / valid embeddings and rankings

```bash
bash run_train_valid_inference_eval_2026.sh
```

Useful environment switches:

- `FORCE_REENCODE=1`
  ignore existing embeddings and rerun `inference.py`
- `SKIP_BM25=1`
  skip BM25 valid/train retrieval
- `RUN_FULL_EVAL=1`
  additionally run `Legal Case Retrieval/utils/eval.py`

### 4. Generate the test retrieval-only submission

```bash
bash run_test_retrieval_2026.sh
```

Useful environment switch:

- `SUBMISSION_TOPK`
  defaults to 5 and controls how many predictions are exported

This step performs the full test-side preparation:

- raw test files -> `processed_test/`
- write `test_qid.tsv`
- build the `BM25_test` index
- build the test scope JSON
- write BM25 and dense retrieval submissions

## Embedding Usage Convention

The maintained Task 1 dense ranking defaults to:

- query from `processed`
- candidate from `processed`

So even though `inference.py` still writes `processed_new` embeddings, `similarity_and_rank.py` does not use them for query ranking by default.

To switch sources:

```bash
export LCR_QUERY_EMBED_SOURCE=processed_new
export LCR_CANDIDATE_EMBED_SOURCE=processed
python "Legal Case Retrieval/modernBert-fp/similarity_and_rank.py"
```

You can also point directly to custom files:

```bash
export LCR_QUERY_EMBEDDINGS_PATH=/abs/path/query_embeddings.pkl
export LCR_CANDIDATE_EMBEDDINGS_PATH=/abs/path/candidate_embeddings.pkl
```

## Common Environment Variables

### Paths and year

- `COLIEE_TASK1_YEAR`
- `COLIEE_TASK1_ROOT`
- `COLIEE_TASK1_DIR`

### Training and resume

- `TASK1_RETRIEVAL_BATCH_SIZE`
- `TASK1_INIT_TEMPERATURE`
- `TASK1_AUTO_RESUME`
- `TASK1_RESUME_FROM_CHECKPOINT`

### Inference and ranking

- `LCR_TEST_MODE`
- `LCR_QUERY_CANDIDATE_SCOPE_JSON`
- `LCR_QUERY_EMBED_SOURCE`
- `LCR_CANDIDATE_EMBED_SOURCE`
- `LCR_QUERY_EMBEDDINGS_PATH`
- `LCR_CANDIDATE_EMBEDDINGS_PATH`

## Important Current Caveats

### 1. The checkpoint root naming is inconsistent

There is a real mismatch in the current code:

- `fine_tune.py` saves to `modernBERT_contrastive_adaptive_fp_fp16_scopeFilteredRaw_<YEAR>`
- `inference.py` defaults to `modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_<YEAR>`

So if you just trained a new `scopeFilteredRaw` experiment, `inference.py` and some repo-root wrappers will not automatically switch to it. You need to align the checkpoint path yourself.

### 2. `inference.py` does not expose model-root override via CLI

Unlike `lightgbm/ltr_feature_pipeline.py`, `inference.py` hardcodes its checkpoint root inside the script. That is why the naming mismatch above matters enough to document explicitly.

### 3. The backbone checkpoint is a hard dependency

`inference.py`, `run_train_valid_inference_eval_2026.sh`, and `run_test_retrieval_2026.sh` all require:

```text
modernbert-caselaw-accsteps-fp/checkpoint-29000
```

If that directory is missing, the downstream stages will not start.
