# lightgbm

[中文](README.md) | [Task 1 Overview](../README.en.md)

This directory provides Task 1 reranking, fixed top-k export, and cutoff post-processing.

## Main Files

- `ltr_feature_pipeline.py`: builds features, trains `LGBMRanker`, writes prediction CSVs, and exports fixed top-k.
- `cutoff_postprocess.py`: reads prediction CSVs, compares cutoff modes, and writes the final submission.
- `fixed_topk_postprocess.py`: exports a fixed top-k submission directly from test predictions.

## Prerequisites

Before running LTR, you will usually need:

- `processed/`
- `processed_test/`
- `train_qid.tsv`
- `valid_qid.tsv`
- `test_qid.tsv`
- train / valid dense embeddings
- test dense embeddings
- `lht_process/BM25/index`
- `lht_process/BM25_test/index`
- `TASK1_MODEL_ROOT_DIR`
- `TASK1_BASE_ENCODER_DIR`

The most direct preparation order is:

```bash
bash run_pre_finetune.sh
python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
bash run_test_retrieval.sh
```

## Step 1: Features, training, and fixed top-k

```bash
bash run_ltr_feature_train_valid_test.sh
```

This step writes:

- `train_features.csv`
- `valid_features.csv`
- `test_features.csv`
- `valid_predictions_raw.csv`
- `test_predictions_raw.csv`
- `fixed_top<k>/fixed_topk_summary.json`
- fixed top-k submission

Common variables:

- `COLIEE_LTR_OUTPUT_DIR`
- `COLIEE_LTR_NUM_WORKERS`
- `COLIEE_LTR_DENSE_BATCH_SIZE`
- `COLIEE_LTR_FEATURE_SCORE_BATCH_SIZE`
- `COLIEE_LTR_LGBM_DEVICE`
- `COLIEE_LTR_FIXED_TOPK`
- `COLIEE_LTR_FIXED_TOPK_RUN_TAG`
- `COLIEE_LTR_FIXED_TOPK_FINAL_SUBMISSION_PATH`

## Step 2: Cutoff search and final submission

```bash
bash run_ltr_cutoff_postprocess.sh
```

This step only reads existing prediction CSVs. It does not rebuild features or retrain the model.

Outputs:

- `cutoff_search/cutoff_summary.json`
- `cutoff_search/best_overall/test_submission_best_mode.txt`
- `COLIEE_LTR_FINAL_SUBMISSION_PATH`

Common variables:

- `COLIEE_LTR_OUTPUT_DIR`
- `COLIEE_LTR_VALID_SCOPE_PATH`
- `COLIEE_LTR_TEST_SCOPE_PATH`
- `COLIEE_LTR_CUTOFF_CONFIG_JSON`
- `COLIEE_LTR_SUBMISSION_RUN_TAG`
- `COLIEE_LTR_FINAL_SUBMISSION_PATH`

## Direct Python CLI Usage

```bash
python "Legal Case Retrieval/lightgbm/ltr_feature_pipeline.py" --help
python "Legal Case Retrieval/lightgbm/cutoff_postprocess.py" --help
```

`ltr_feature_pipeline.py` reads default dataset paths, dense embeddings, model directories, and output directories from the repo-root `.env`. Shell variables can override them for a single run.
