# modernBert-fp

[中文](README.md) | [Task 1 Overview](../README.en.md)

This directory provides Task 1 dense encoder training, embedding generation, and ranking export.

## Main Files

- `fine_tune/fine_tune.py`: contrastive fine-tuning entry point.
- `inference.py`: selects a checkpoint from `TASK1_MODEL_ROOT_DIR` and writes embeddings.
- `similarity_and_rank.py`: reads embeddings and writes dot / cosine rankings.
- `find_best_model.py`: selects the best checkpoint by metric.
- `train_modernbert_caselaw_fp.py`: backbone continued pretraining.

## Typical Flows

### Train a new dense model

```bash
bash run_pre_finetune.sh
python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
bash run_train_valid_inference_eval.sh
bash run_test_retrieval.sh
```

### Reuse an existing checkpoint

Set:

- `TASK1_MODEL_ROOT_DIR`
- `TASK1_BASE_ENCODER_DIR`
- `TASK1_RETRIEVAL_MODEL_NAME`

Then run:

```bash
bash run_train_valid_inference_eval.sh
bash run_test_retrieval.sh
```

If you want LTR afterward:

```bash
bash run_ltr_feature_train_valid_test.sh
bash run_ltr_cutoff_postprocess.sh
```

## Main Inputs

- `TASK1_FINETUNE_DATA_DIR/contrastive_bm25_hard_negative_top100_random15_train.json`
- `TASK1_FINETUNE_DATA_DIR/contrastive_bm25_hard_negative_top100_random15_valid.json`
- `TASK1_PROCESSED_DIR/`
- `TASK1_QUERY_DIR/`
- `TASK1_TRAIN_QID_PATH`
- `TASK1_VALID_QID_PATH`
- `TASK1_SCOPE_PATH`
- `TASK1_BASE_ENCODER_DIR`
- `TASK1_MODEL_ROOT_DIR`

Additional inputs for test retrieval:

- `TASK1_TEST_RAW_DIR`
- `TASK1_TEST_LABELS_PATH`

## Main Outputs

Training:

- `TASK1_MODEL_ROOT_DIR/`
- `TASK1_MODEL_ROOT_DIR/tb/`

Embeddings:

- `processed/processed_document_<MODEL_NAME>_embeddings*.pkl`
- `processed_new/processed_new_document_<MODEL_NAME>_embeddings*.pkl`
- `processed_test/processed_test_document_<MODEL_NAME>_embeddings*.pkl`
- `processed_test/processed_test_query_<MODEL_NAME>_embeddings*.pkl`

Rankings:

- `TASK1_MODEL_RESULTS_DIR/output_<MODEL_NAME>_dot_train.tsv`
- `TASK1_MODEL_RESULTS_DIR/output_<MODEL_NAME>_dot_valid.tsv`
- `TASK1_MODEL_RESULTS_DIR/output_<MODEL_NAME>_dot_test.tsv`
- `TASK1_MODEL_RESULTS_DIR/output_<MODEL_NAME>_cos_train.tsv`
- `TASK1_MODEL_RESULTS_DIR/output_<MODEL_NAME>_cos_valid.tsv`

## Common Variables

Checkpoint and backbone:

- `TASK1_MODEL_ROOT_DIR`
- `TASK1_BASE_ENCODER_DIR`
- `TASK1_CHECKPOINT_METRIC`
- `TASK1_CHECKPOINT_MODE`

Training:

- `TASK1_FINETUNE_DATA_DIR`
- `TASK1_SCOPE_FILTER`
- `TASK1_QUICK_TEST`
- `TASK1_QUICK_TEST_CAND_K`
- `TASK1_QUICK_TEST_QUERY_K`
- `TASK1_RETRIEVAL_BATCH_SIZE`
- `TASK1_RETRIEVAL_MAX_LENGTH`
- `TASK1_INIT_TEMPERATURE`
- `TASK1_AUTO_RESUME`
- `TASK1_RESUME_FROM_CHECKPOINT`

Inference and ranking:

- `TASK1_CANDIDATE_DIR`
- `TASK1_QUERY_DIR`
- `TASK1_CANDIDATE_EMBEDDINGS_OUTPUT`
- `TASK1_QUERY_EMBEDDINGS_OUTPUT`
- `TASK1_TEST_CANDIDATE_EMBEDDINGS_PATH`
- `TASK1_TEST_QUERY_EMBEDDINGS_PATH`
- `TASK1_OUTPUT_DOT_TRAIN_PATH`
- `TASK1_OUTPUT_DOT_VALID_PATH`
- `TASK1_OUTPUT_DOT_TEST_PATH`
- `LCR_QUERY_CANDIDATE_SCOPE_JSON`

## Override Example

```bash
TASK1_MODEL_ROOT_DIR=./models/my_task1_model \
TASK1_BASE_ENCODER_DIR=./models/my_encoder/checkpoint-29000 \
TASK1_RETRIEVAL_MODEL_NAME=my_task1_model \
bash run_test_retrieval.sh
```

The repo-root `.env` provides defaults, and shell variables can override them for a single run.
