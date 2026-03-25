# Legal Case Entailment by Mou

[中文](README.md) | [Back to Root](../README.md)

This directory is the maintained COLIEE Task 2 workflow and the only Task 2 path recommended in this repository.

`Legal Case Entailment/` is a preserved legacy directory. It is outside the maintained scope, and this README does not depend on it.

## Task Formulation

This workflow reformulates Task 2 as a paragraph-level retrieval / matching problem:

- query:
  `cases/<qid>/entailed_fragment.txt`
- candidates:
  `cases/<qid>/paragraphs/*.txt`
- positives:
  from `task2_train_labels_<YEAR>.json`

Flattened ID convention:

- query id uses the three-digit case id, for example `001`
- candidate id is `case_id + paragraph_id`, for example `001003`

## Recommended Usage

### One-command workflow

Run from repo root:

```bash
bash run_task2_finetune.sh
```

This is the recommended entrypoint because it:

- reads the repo-root `.env`
- activates `CONDA_ENV_NAME`
- checks that the raw Task 2 data exists
- prepares the paragraph-level dataset
- optionally generates dataset statistics
- runs ModernBERT fine-tuning

### Manual step-by-step workflow

```bash
python "Legal Case Entailment by Mou/prepare_task2_paragraph_data.py"
python "Legal Case Entailment by Mou/analyze_task2_stats.py"
python "Legal Case Entailment by Mou/fine_tune_task2.py"
```

The statistics step is optional.

## Raw Input Layout

Default raw-data location:

```text
coliee_dataset/task2/task2_train_files_<YEAR>/
```

Required contents:

- `cases/`
- `task2_train_labels_<YEAR>.json`

Expected structure for each case:

```text
cases/<qid>/
  entailed_fragment.txt
  paragraphs/
    001.txt
    002.txt
    ...
```

## What `prepare_task2_paragraph_data.py` Does

This step converts the raw case-level layout into a paragraph-level dataset directly usable for training.

Default output directory:

```text
Legal Case Entailment by Mou/data/task2_<YEAR>_prepared/
```

Common outputs:

- `processed_queries/`
- `processed_candidates/`
- `query_candidates_map.json`
- `task2_train_labels_<YEAR>_flat.json`
- `task2_train_labels_<YEAR>_flat_train.json`
- `task2_train_labels_<YEAR>_flat_valid.json`
- `train_qid.tsv`
- `valid_qid.tsv`
- `finetune_data/contrastive_task2_random15_valid.json`
- `prepare_stats.json`

Split defaults:

- train / valid ratio `0.8 / 0.2`
- split seed `42`
- validation negatives per sample: `15`

## What `analyze_task2_stats.py` Does

This step computes statistics on the prepared dataset using the `answerdotai/ModernBERT-base` tokenizer.

Common outputs:

- `stats/relevant_count_distribution.csv`
- `stats/query_token_length_distribution.csv`
- `stats/candidate_token_length_distribution.csv`
- `stats/relevant_count_distribution.png`
- `stats/query_token_length_hist.png`
- `stats/candidate_token_length_hist.png`
- `stats/summary.json`

## What `fine_tune_task2.py` Does

This is the maintained paragraph-level ModernBERT fine-tuning entrypoint.

Key characteristics:

- the query is the `entailed_fragment`
- the candidates are the paragraphs from the same case directory
- negatives are adaptively sampled from model similarities
- validation retrieval reports global F1 / precision / recall
- best-checkpoint selection still uses validation top-1 F1

Common outputs:

- `modernBERT_contrastive_adaptive_fp_fp16_scopeFiltered_2026_para/` or the path set by `TASK2_OUTPUT_DIR`
- `tb/`
- per-epoch similarity-score files
- adaptive-negative JSONs

## Common Environment Variables

### Base settings

- `CONDA_ENV_NAME`
- `COLIEE_TASK2_YEAR`
- `COLIEE_TASK2_ROOT`
- `COLIEE_TASK2_DIR`
- `COLIEE_TASK2_PREPARED_DIR`

### Initialization model and output

- `TASK2_INIT_MODEL_ROOT`
- `TASK2_INIT_CHECKPOINT`
- `TASK2_INIT_METRIC`
- `TASK2_INIT_METRIC_MODE`
- `TASK2_OUTPUT_DIR`
- `TASK2_RESUME_CHECKPOINT`

### Training and evaluation

- `TASK2_EVAL_TOPK`
- `TASK2_NUM_TRAIN_EPOCHS`
- `TASK2_MAX_STEPS`
- `TASK2_LOGGING_STEPS`
- `TASK2_SAVE_TOTAL_LIMIT`
- `TASK2_EARLY_STOPPING_PATIENCE`
- `TASK2_TRAIN_BATCH_SIZE`
- `TASK2_EVAL_BATCH_SIZE`
- `TASK2_GRAD_ACCUM_STEPS`
- `TASK2_RETRIEVAL_BATCH_SIZE`
- `TASK2_RETRIEVAL_MAX_LENGTH`

### Performance and data loading

- `TASK2_ENABLE_TF32`
- `TASK2_GRADIENT_CHECKPOINTING`
- `TASK2_CACHE_TEXTS`
- `TASK2_DATALOADER_NUM_WORKERS`
- `TASK2_DATALOADER_PIN_MEMORY`
- `TASK2_DATALOADER_PERSISTENT_WORKERS`

### Wrapper behavior

- `TASK2_SKIP_STATS`
- `TASK2_MODE`

## Test Mode

Set in `.env` or in the shell:

```bash
TASK2_MODE=test
```

Effect:

- uses only a smaller subset of train / valid queries
- still keeps adaptive negative sampling
- automatically appends `_test` to the output directory so the full run is not overwritten

Common test-mode controls:

- `TASK2_TEST_SEED`
- `TASK2_TEST_TRAIN_QUERY_LIMIT`
- `TASK2_TEST_VALID_QUERY_LIMIT`
- `TASK2_TEST_NUM_TRAIN_EPOCHS`
- `TASK2_TEST_MAX_STEPS`
- `TASK2_TEST_LOGGING_STEPS`
- `TASK2_TEST_SAVE_TOTAL_LIMIT`
- `TASK2_TEST_EARLY_STOPPING_PATIENCE`

To switch back to the full workflow:

```bash
TASK2_MODE=full
```

## Important Notes

### 1. This wrapper does read `.env`

`run_task2_finetune.sh` reads the repo-root `.env` first, then reapplies shell variables already set by the caller. That makes its behavior different from the Task 1 repo-root wrappers.

### 2. `TASK2_EVAL_TOPK` does not change the best-model criterion

The code reports both top-1 and top-2 retrieval metrics, but best-checkpoint selection and early stopping still follow validation top-1 F1.

### 3. Do not treat `Legal Case Entailment/` as part of the maintained workflow

The maintained Task 2 workflow is only this directory. Do not mix it with scripts from the old directory.
