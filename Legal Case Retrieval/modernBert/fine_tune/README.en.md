# modernBert/fine_tune

[中文](README.md) | [Task 1 Overview](../../README.en.md)

This directory keeps older modernBERT fine-tuning code and BM25 hard-negative utilities. It is not the maintained main training path for Task 1; the maintained workflow has moved to `../../modernBert-fp/`.

However, part of this directory is still used by the current workflow:

- `create_bm25_hard_negative_data_top100_random15.py`
  `run_pre_finetune_2026.sh` still uses this script to generate BM25 hard negatives before dense fine-tuning.

## Current Role of This Directory

### Still actively used

- BM25 hard-negative generation utilities
- older contrastive-training modules kept for historical reference

### Not recommended as the current main path

- `fine_tune.py`
- `fine_tune_noprojector.py`
- these older training entrypoints should not replace `modernBert-fp/fine_tune/fine_tune.py`

## File Guide

- `create_bm25_hard_negative_data.py`
  Older BM25 hard-negative generator.
- `create_bm25_hard_negative_data_top100_random15.py`
  The version still used by the maintained workflow. It samples 15 negatives from BM25 top-100.
- `create_config.py`
  Older config-generation utility.
- `fine_tune.py`
  Older contrastive training script.
- `fine_tune_noprojector.py`
  Older contrastive training script without a projector.
- `modernbert_contrastive_model.py`
  Older contrastive model definition.

## How This Directory Is Used Today

If you follow the repo-root Task 1 main flow:

```bash
bash run_pre_finetune_2026.sh
```

one of its steps is:

```bash
python "Legal Case Retrieval/modernBert/fine_tune/create_bm25_hard_negative_data_top100_random15.py"
```

That script writes:

- `coliee_dataset/task1/<YEAR>/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_train.json`
- `coliee_dataset/task1/<YEAR>/lht_process/modernBert/finetune_data/contrastive_bm25_hard_negative_top100_random15_valid.json`

Those files are then consumed downstream by `modernBert-fp/fine_tune/fine_tune.py` as part of the maintained Task 1 preparation flow.

## If You Only Want the Maintained Task 1 Workflow

Go directly to:

- [../../modernBert-fp/README.en.md](../../modernBert-fp/README.en.md)
- [../../lightgbm/README.en.md](../../lightgbm/README.en.md)

You do not need to operate most of the older training scripts in this directory directly.

## Additional Note

- Historically, this directory's documentation mixed older modernBERT training notes with later `modernBert-fp` concepts
- after reorganization, it should be treated mainly as "historical experiments plus the still-used BM25 hard-negative utility"
- for new main-path experiments, use `modernBert-fp/` as the source of truth
