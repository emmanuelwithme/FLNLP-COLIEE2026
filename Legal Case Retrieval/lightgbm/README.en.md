# lightgbm

[中文](README.md) | [Task 1 Overview](../README.en.md)

This directory contains the maintained Task 1 reranking and submission pipeline. If your goal is to turn test data into the final reranked submission, this is the main workflow.

## What This Directory Handles

- building train / valid / test LTR features
- training `LightGBM LGBMRanker`
- writing raw rerank prediction CSVs
- applying scope and self-removal legal filters
- exporting a fixed top-k baseline submission
- running validation-side cutoff grid search and applying the selected mode to test

## Main Files

- `ltr_feature_pipeline.py`
  Full feature building + LightGBM training + raw prediction export + fixed top-k / cutoff integration entrypoint.
- `cutoff_postprocess.py`
  Consumes existing rerank prediction CSVs only. It does not rebuild features or retrain the model; it only compares cutoff modes and exports submission files.
- `fixed_topk_postprocess.py`
  Small wrapper that directly applies a fixed top-k export to test rerank outputs.

## Recommended End-to-End Test Submission Order

If you want the maintained repo path from raw data to the final submission, the recommended order is:

1. Finish `bash run_pre_finetune_2026.sh`
2. Prepare or reuse a dense checkpoint
3. Run `bash run_test_retrieval_2026.sh`
   this prepares `processed_test/`, `test_qid.tsv`, the `BM25_test` index, the test scope, and test embeddings
4. Run `bash run_ltr_feature_train_valid_test_2026.sh`
   this writes the fixed top-k baseline submission
5. Run `bash run_ltr_cutoff_postprocess_2026.sh`
   this performs the only cutoff grid search and writes the final submission

## Prerequisites This Pipeline Does Not Create for You

`run_ltr_feature_train_valid_test_2026.sh` and `ltr_feature_pipeline.py` do not create the following prerequisites automatically:

- `processed/`
- `processed_test/`
- `train_qid.tsv`
- `valid_qid.tsv`
- `test_qid.tsv`
- `processed/processed_document_modernBert_fp_fp16_embeddings.pkl`
- `processed_test/processed_test_document_modernBert_fp_fp16_embeddings.pkl`
- `lht_process/BM25/index`
- `lht_process/BM25_test/index`
- `modernbert-caselaw-accsteps-fp/checkpoint-29000`
- the target dense checkpoint root

If the test-side prerequisites such as `processed_test`, `test_qid.tsv`, `BM25_test/index`, or test embeddings are missing, the simplest preparation step is:

```bash
bash run_test_retrieval_2026.sh
```

## Feature Set

`ltr_feature_pipeline.py` currently combines:

- lexical scores
  `bm25_score`, `qld_score`, `bm25_ngram_score`
- dense retrieval score
  `dense_score`
- rank features
  `bm25_rank`, `dense_rank`
- length features
  `query_length`, `doc_length`, `len_ratio`, `len_diff`
- placeholder features
  citation / reference / fragment counts and ratios
- year features
  `query_year`, `doc_year`, `year_diff`
- chunk similarity aggregation
  `chunk_sim_max`, `chunk_sim_mean`, `chunk_sim_top2_mean`

So the LTR stage is not just consuming one dense score. It combines lexical, dense, length, placeholder, year, and chunk-level similarity signals.

## `run_ltr_feature_train_valid_test_2026.sh`

This wrapper runs the full LTR training path and exports a fixed top-k baseline.

```bash
bash run_ltr_feature_train_valid_test_2026.sh
```

Default behavior:

- calls `Legal Case Retrieval/lightgbm/ltr_feature_pipeline.py`
- uses `coliee_dataset/task1/<YEAR>/lht_process/lightgbm_ltr_scope_raw/` as the output root
- writes `train_features.csv`, `valid_features.csv`, `test_features.csv`
- trains `lgbm_ranker_scope_raw.txt`
- writes `valid_predictions_raw.csv` and `test_predictions_raw.csv`
- also writes scope-filtered `valid_predictions.csv` and `test_predictions.csv`
- runs a fixed top-k export
- intentionally skips cutoff search

Main outputs:

- `train_features.csv`
- `valid_features.csv`
- `test_features.csv`
- `lgbm_ranker_scope_raw.txt`
- `valid_predictions_raw.csv`
- `valid_predictions.csv`
- `test_predictions_raw.csv`
- `test_predictions.csv`
- `fixed_top5/test_submission_fixed_topk.txt`
- `fixed_top5/fixed_topk_summary.json`

Repo-root copy:

- `task1_FLNLPLTRTOP5.txt`

### Common environment variables for this wrapper

- `COLIEE_TASK1_YEAR`
- `COLIEE_TASK1_ROOT`
- `COLIEE_TASK1_DIR`
- `COLIEE_LTR_OUTPUT_DIR`
- `COLIEE_LTR_NUM_WORKERS`
- `COLIEE_LTR_DENSE_BATCH_SIZE`
- `COLIEE_LTR_FEATURE_SCORE_BATCH_SIZE`
- `COLIEE_LTR_LGBM_DEVICE`
- `COLIEE_LTR_FIXED_TOPK`
- `COLIEE_LTR_FIXED_TOPK_RUN_TAG`
- `COLIEE_LTR_FIXED_TOPK_FINAL_SUBMISSION_PATH`

This wrapper does not `source .env` by itself. If you want to override the year or output directory, pass shell variables explicitly, for example:

```bash
COLIEE_TASK1_YEAR=2026 \
COLIEE_LTR_OUTPUT_DIR=./coliee_dataset/task1/2026/lht_process/lightgbm_ltr_scope_raw \
bash run_ltr_feature_train_valid_test_2026.sh
```

## `run_ltr_cutoff_postprocess_2026.sh`

This wrapper only runs cutoff post-processing. It does not rebuild features and does not retrain the model.

```bash
bash run_ltr_cutoff_postprocess_2026.sh
```

Default flow:

1. read `valid_predictions_raw.csv`
2. read `test_predictions_raw.csv`
3. apply valid / test scope filters and self-removal
4. compare three cutoff modes on validation
5. select the best mode and parameters
6. apply the selected cutoff to test once
7. write `cutoff_summary.json` and the final submission

Current cutoff modes:

- fixed top-k
- ratio cutoff
- largest-gap adaptive cutoff

Main outputs:

- `cutoff_search/valid_predictions_legal_filtered.csv`
- `cutoff_search/test_predictions_legal_filtered.csv`
- `cutoff_search/validation_mode_comparison.csv`
- `cutoff_search/fixed_topk/`
- `cutoff_search/ratio_cutoff/`
- `cutoff_search/largest_gap_cutoff/`
- `cutoff_search/best_overall/test_submission_best_mode.txt`
- `cutoff_search/cutoff_summary.json`

Repo-root copy:

- `task1_FLNLPLTR.txt`

### Common environment variables for this wrapper

- `COLIEE_LTR_OUTPUT_DIR`
- `COLIEE_LTR_VALID_PRED_PATH`
- `COLIEE_LTR_TEST_PRED_PATH`
- `COLIEE_LTR_VALID_SCOPE_PATH`
- `COLIEE_LTR_TEST_SCOPE_PATH`
- `COLIEE_LTR_CUTOFF_OUTPUT_DIR`
- `COLIEE_LTR_CUTOFF_CONFIG_JSON`
- `COLIEE_LTR_SUBMISSION_RUN_TAG`
- `COLIEE_LTR_FINAL_SUBMISSION_PATH`

If you only want to swap the cutoff grid without retraining LightGBM:

```bash
COLIEE_LTR_CUTOFF_CONFIG_JSON=/path/to/cutoff_config.json \
bash run_ltr_cutoff_postprocess_2026.sh
```

## Pay Attention to the Default Scope Paths

The current pipeline uses two different default scope paths:

- valid scope:
  `coliee_dataset/task1/<YEAR>/lht_process/scope_compare/query_candidate_scope_raw_plus0.json`
- test scope:
  `coliee_dataset/task1/<YEAR>/lht_process/modernBert/query_candidate_scope_test_raw.json`

The test scope can usually be generated by `run_test_retrieval_2026.sh`.

But the validation scope `query_candidate_scope_raw_plus0.json` is only the current wrapper's default filename. It is not an artifact automatically created by `run_pre_finetune_2026.sh`. If your validation scope lives somewhere else, override it explicitly.

## Cutoff Search Scoring Logic

`cutoff_postprocess.py` records multiple validation metrics, including:

- F1
- Precision
- Recall
- nDCG@10
- P@5
- R@5

Mode selection is driven primarily by validation F1, then by recall, precision, and related tie-breaks.

## The Three Limitations You Most Need to Know

### 1. The LTR wrapper does not generate test embeddings for you

If test embeddings are missing, `ltr_feature_pipeline.py` will fail directly. It will not call `modernBert-fp/inference.py` on your behalf. That is why `run_test_retrieval_2026.sh` is the recommended preparation step.

### 2. The default dense checkpoint root still points to `scopeFiltered`

`ltr_feature_pipeline.py` defaults `--model-root-dir` to:

```text
modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_<YEAR>
```

If your latest model actually lives under `scopeFilteredRaw_<YEAR>`, you must override `--model-root-dir` or adjust the wrapper.

### 3. `run_ltr_feature_train_valid_test_2026.sh` intentionally skips cutoff search

This is deliberate, not an omission. The current design keeps the cutoff grid search in exactly one place: `run_ltr_cutoff_postprocess_2026.sh`, to avoid repeated searches and accidental submission overwrites.
