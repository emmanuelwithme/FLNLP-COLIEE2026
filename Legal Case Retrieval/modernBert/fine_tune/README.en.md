# modernBert/fine_tune

[中文](README.md) | [Task 1 Overview](../../README.en.md)

This directory contains BM25 hard-negative generators and a set of additional contrastive training utilities.

## Main Role

The repo-root Task 1 preprocessing flow calls:

```bash
python "Legal Case Retrieval/modernBert/fine_tune/create_bm25_hard_negative_data_top100_random15.py"
```

This script converts BM25 train / valid rankings into contrastive data for dense fine-tuning.

## Main Files

- `create_bm25_hard_negative_data_top100_random15.py`: samples 15 negatives from BM25 top-100.
- `create_bm25_hard_negative_data.py`: another hard-negative generator.
- `create_config.py`: training config utility.
- `fine_tune.py`: contrastive training entry point.
- `fine_tune_noprojector.py`: training entry point without a projector.
- `modernbert_contrastive_model.py`: model definition.

## Hard-Negative Generator Inputs

- BM25 train ranking
- BM25 valid ranking
- train labels
- valid labels

The repo wrapper passes:

- `TASK1_BM25_DIR/output_bm25_train.tsv`
- `TASK1_BM25_DIR/output_bm25_valid.tsv`
- `TASK1_TRAIN_SPLIT_LABELS_PATH`
- `TASK1_VALID_SPLIT_LABELS_PATH`

## Hard-Negative Generator Outputs

- `TASK1_FINETUNE_DATA_DIR/contrastive_bm25_hard_negative_top100_random15_train.json`
- `TASK1_FINETUNE_DATA_DIR/contrastive_bm25_hard_negative_top100_random15_valid.json`

Common arguments:

- `--top-k`
- `--max-negatives`
- `--random-seed`

Matching repo-root environment variables:

- `TASK1_HARD_NEG_TOPK`
- `TASK1_HARD_NEG_MAX_NEGATIVES`
- `TASK1_HARD_NEG_SEED`

## Direct Example

```bash
python "Legal Case Retrieval/modernBert/fine_tune/create_bm25_hard_negative_data_top100_random15.py" \
  --bm25-train-path "./coliee_dataset/task1/2026/lht_process/BM25/output_bm25_train.tsv" \
  --bm25-valid-path "./coliee_dataset/task1/2026/lht_process/BM25/output_bm25_valid.tsv" \
  --train-labels-path "./coliee_dataset/task1/2026/task1_train_labels_2026_train.json" \
  --valid-labels-path "./coliee_dataset/task1/2026/task1_train_labels_2026_valid.json" \
  --train-output-path "./coliee_dataset/task1/2026/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_train.json" \
  --valid-output-path "./coliee_dataset/task1/2026/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json" \
  --top-k 100 \
  --max-negatives 15 \
  --random-seed 289
```

For the main dense training and inference flow, see:

- [../../modernBert-fp/README.en.md](../../modernBert-fp/README.en.md)
- [../../lightgbm/README.en.md](../../lightgbm/README.en.md)
