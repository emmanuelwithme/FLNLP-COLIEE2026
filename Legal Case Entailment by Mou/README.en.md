# Legal Case Entailment by Mou

[中文](README.md) | [Back to Root](../README.md)

This directory provides Task 2 paragraph-level data preparation, statistics, and ModernBERT fine-tuning. It is kept as supplementary code; the main focus of the repository is Task 1.

## Task Format

- query: `cases/<qid>/entailed_fragment.txt`
- candidates: `cases/<qid>/paragraphs/*.txt`
- labels: `task2_train_labels_<YEAR>.json`

After preparation:

- each query is written to `processed_queries/<qid>.txt`
- each candidate paragraph is written to `processed_candidates/<caseid><paragraphid>.txt`
- labels are flattened into paragraph-level positives

## One-Command Flow

```bash
bash run_task2_finetune.sh
```

This wrapper will:

- read the repo-root `.env`
- activate `CONDA_ENV_NAME`
- build paragraph-level data
- optionally generate statistics
- run `fine_tune_task2.py`

## Manual Step-by-Step Flow

### 1. Prepare data

```bash
python "Legal Case Entailment by Mou/prepare_task2_paragraph_data.py"
```

Main inputs:

- `COLIEE_TASK2_DIR/cases/`
- `COLIEE_TASK2_DIR/TASK2_LABELS_FILENAME`

Main outputs:

- `COLIEE_TASK2_PREPARED_DIR/processed_queries/`
- `COLIEE_TASK2_PREPARED_DIR/processed_candidates/`
- `COLIEE_TASK2_PREPARED_DIR/query_candidates_map.json`
- `COLIEE_TASK2_PREPARED_DIR/train_qid.tsv`
- `COLIEE_TASK2_PREPARED_DIR/valid_qid.tsv`
- `COLIEE_TASK2_PREPARED_DIR/finetune_data/`

### 2. Generate statistics

```bash
python "Legal Case Entailment by Mou/analyze_task2_stats.py"
```

Outputs:

- `stats/summary.json`
- `stats/relevant_count_distribution.csv`
- `stats/query_token_length_hist.png`
- `stats/candidate_token_length_hist.png`

### 3. Train the model

```bash
python "Legal Case Entailment by Mou/fine_tune_task2.py"
```

Output:

- `TASK2_OUTPUT_DIR/`

## Common Variables

Data:

- `COLIEE_TASK2_YEAR`
- `COLIEE_TASK2_ROOT`
- `COLIEE_TASK2_DIR`
- `COLIEE_TASK2_PREPARED_DIR`
- `TASK2_LABELS_FILENAME`

Initialization and output:

- `TASK2_INIT_MODEL_ROOT`
- `TASK2_INIT_CHECKPOINT`
- `TASK2_INIT_METRIC`
- `TASK2_INIT_METRIC_MODE`
- `TASK2_OUTPUT_DIR`
- `TASK2_RESUME_CHECKPOINT`

Training:

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

Wrapper controls:

- `TASK2_MODE`
- `TASK2_SKIP_STATS`

## Test Mode

```bash
TASK2_MODE=test bash run_task2_finetune.sh
```

Test mode uses a smaller query subset and appends `_test` to the output directory name.

The repo-root `.env` provides defaults, and shell variables can override them for a single run.
