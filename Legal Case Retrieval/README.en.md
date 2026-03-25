# Legal Case Retrieval

[中文](README.md) | [Back to Root](../README.md)

This directory contains the maintained COLIEE Task 1 workflow, from case preprocessing to dense retrieval, LightGBM reranking, and final submission export.

## Primary Subdirectories

- `modernBert-fp/`
  The maintained dense retrieval pipeline. ModernBERT contrastive fine-tuning, embedding inference, and similarity ranking live here.
- `lightgbm/`
  The maintained LightGBM learning-to-rank and submission post-processing pipeline.
- `pre-process/`
  Utilities for raw-case preprocessing, test preprocessing, and scope JSON generation.
- `lcr/`
  Shared Task 1 utilities, including dataset path resolution, embedding selection, similarity scoring, metrics, and retrieval helpers.
- `lexical models/`
  BM25 and QLD related scripts plus Pyserini index/search wrappers.

## Still Present but Not the Main Path

- `modernBert-fp-chunkAgg/`
  Chunk aggregation variant of the ModernBERT retriever.
- `modernBert/`
  Older or historical training code. `run_pre_finetune_2026.sh` still uses its BM25 hard-negative generator.
- `SAILER/`
  Preserved upstream experiment directory, not the current submission path.

## Dataset and Artifact Conventions

Default Task 1 directory:

```text
coliee_dataset/task1/2026/
```

Typical inputs:

- `task1_train_files_2026/`
- `task1_test_files_2026/`
- `task1_train_labels_2026.json`
- `task1_test_no_labels_2026.json`

Typical generated artifacts:

- `summary/`
  Case summaries from `pre-process/summary.py`.
- `processed/`
  Cleaned train/valid corpus.
- `processed_new/`
  Alternate query-side text source retained for older THUIR-style query experiments.
- `processed_test/`
  Cleaned test corpus.
- `lht_process/BM25/`
  Train/valid BM25 index, topics, and retrieval outputs.
- `lht_process/BM25_test/`
  Test BM25 index, topics, and retrieval outputs.
- `lht_process/modernBert/`
  Dense-retrieval preparation artifacts such as `finetune_data` and `query_candidate_scope.json`.
- `lht_process/modernBert_fp_fp16/`
  Dense-retrieval train/valid/test ranking outputs.
- `lht_process/lightgbm_ltr_scope_raw/`
  LightGBM features, model file, raw rerank outputs, fixed top-k exports, and cutoff-search outputs.

## Recommended Execution Order

The following order is the closest thing to a maintained end-to-end Task 1 path from raw data to submission.

### 1. Preprocessing and fine-tune preparation

Run from repo root:

```bash
bash run_pre_finetune_2026.sh
```

This stage creates:

- `summary/`
- `processed/`
- `task1_train_labels_2026_train.json`
- `task1_train_labels_2026_valid.json`
- `train_qid.tsv`
- `valid_qid.tsv`
- `lht_process/BM25/` index and train/valid retrieval outputs
- `lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_{train,valid}.json`
- `lht_process/modernBert/query_candidate_scope.json`

That `query_candidate_scope.json` is produced by the built-in default mode of `build_query_candidate_scope.py`:

- query and candidate text both come from `processed/`
- year extraction comes from raw `task1_train_files_<YEAR>/`
- train and valid qids are merged
- `year_slack=1`
- self is not excluded

### 2. Dense retrieval training or reuse of an existing checkpoint

Main documentation:

- [modernBert-fp/README.en.md](./modernBert-fp/README.en.md)

Direct training entrypoint:

```bash
python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
```

If you only need test-time prediction, you may reuse an existing checkpoint and skip retraining.

### 3. Train/valid retrieval check

```bash
bash run_train_valid_inference_eval_2026.sh
```

This stage can:

- regenerate `processed` embeddings when needed
- write dot/cosine TREC rankings for train and valid
- optionally rerun BM25 retrieval on valid/train
- print focused metrics and qid-coverage sanity checks

### 4. Retrieval-only test prediction

```bash
bash run_test_retrieval_2026.sh
```

This stage builds or refreshes:

- `processed_test/`
- `test_qid.tsv`
- `lht_process/BM25_test/`
- `lht_process/modernBert/query_candidate_scope_test_raw.json`
- `lht_process/submission/task1_FLNLPBM25.txt`
- `lht_process/submission/task1_FLNLPEMBED.txt`

If you want a dense-retrieval-only submission, this is already a complete path.

### 5. LightGBM rerank submission pipeline

Main documentation:

- [lightgbm/README.en.md](./lightgbm/README.en.md)

Run from repo root:

```bash
bash run_ltr_feature_train_valid_test_2026.sh
```

This stage:

- rebuilds train / valid / test feature CSVs
- trains the LightGBM ranker
- writes raw and scope-filtered prediction CSVs
- immediately exports a fixed top-k test submission

The current wrapper intentionally adds `--skip-cutoff-search`, so this step only produces the fixed top-k baseline and does not run cutoff grid search.

Default fixed top-k outputs:

- `coliee_dataset/task1/2026/lht_process/lightgbm_ltr_scope_raw/fixed_top5/test_submission_fixed_topk.txt`
- repo-root copy: `task1_FLNLPLTRTOP5.txt`

### 6. Cutoff-only postprocess

```bash
bash run_ltr_cutoff_postprocess_2026.sh
```

This step reuses existing rerank outputs without rebuilding features or retraining LightGBM. It:

- applies legal filters
- compares fixed top-k, ratio cutoff, and largest-gap cutoff on validation
- selects the best mode
- applies the selected cutoff to test exactly once

Default final submission:

- `coliee_dataset/task1/2026/lht_process/lightgbm_ltr_scope_raw/cutoff_search/best_overall/test_submission_best_mode.txt`
- repo-root copy: `task1_FLNLPLTR.txt`

## Important Conventions and Caveats

### 1. Most Task 1 wrappers are 2026-oriented

- `run_pre_finetune_2026.sh`
- `run_train_valid_inference_eval_2026.sh`
- `run_test_retrieval_2026.sh`

These scripts are not general year-switch wrappers driven purely by `.env`; they are written around the 2026 workflow.

### 2. `lcr.task1_paths` reads `.env`

Many Task 1 Python scripts resolve paths through `lcr/task1_paths.py`, which reads:

- `COLIEE_TASK1_YEAR`
- `COLIEE_TASK1_ROOT`
- `COLIEE_TASK1_DIR`

But whether a repo-root wrapper actively reads `.env` depends on that shell script itself. Do not assume Python-side path resolution and shell-side defaults are the same layer.

### 3. The maintained embedding default is `processed`

Current Task 1 behavior:

- inference still writes both `processed` and `processed_new` embeddings
- ranking defaults to `processed` for both query and candidate
- to switch back to a THUIR-style query setup:

```bash
export LCR_QUERY_EMBED_SOURCE=processed_new
export LCR_CANDIDATE_EMBED_SOURCE=processed
```

### 4. There are multiple scope files, and they are not interchangeable by default

Common scope files:

- `lht_process/modernBert/query_candidate_scope.json`
  commonly used for dense train/valid retrieval
- `lht_process/modernBert/query_candidate_scope_test_raw.json`
  test scope built by `run_test_retrieval_2026.sh`
- `lht_process/scope_compare/query_candidate_scope_raw_plus0.json`
  the current default valid-scope path expected by the LightGBM wrapper

That last validation-scope file is a LightGBM wrapper default, not something automatically produced by `run_pre_finetune_2026.sh`. If your scope file uses a different name or policy, override it explicitly via environment variables or CLI arguments.

### 5. There is a checkpoint directory naming mismatch in the current codebase

One caveat is important enough to call out directly:

- `modernBert-fp/fine_tune/fine_tune.py` currently saves to `modernBERT_contrastive_adaptive_fp_fp16_scopeFilteredRaw_<YEAR>`
- `modernBert-fp/inference.py`, `run_train_valid_inference_eval_2026.sh`, `run_test_retrieval_2026.sh`, and `lightgbm/ltr_feature_pipeline.py` still default to `modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_<YEAR>`

So if you train a new `scopeFilteredRaw` run, the downstream wrappers may not pick it up automatically. You need to align the checkpoint path manually, adjust the script, or override the model-root path where the script supports it.

## Related Documentation

- Dense retrieval main flow: [modernBert-fp/README.en.md](./modernBert-fp/README.en.md)
- LightGBM rerank main flow: [lightgbm/README.en.md](./lightgbm/README.en.md)
- Chunk aggregation experiment: [modernBert-fp-chunkAgg/README.en.md](./modernBert-fp-chunkAgg/README.en.md)
- Legacy modernBERT fine-tune and BM25 hard negatives: [modernBert/fine_tune/README.en.md](./modernBert/fine_tune/README.en.md)
