# Task2 Paragraph-Level Fine-Tuning

This workflow keeps the `modernBert-fp` fine-tune style, but targets COLIEE task2 paragraph retrieval:

- query: `cases/<qid>/entailed_fragment.txt`
- candidates: all files in `cases/<qid>/paragraphs/*.txt`
- positives: from `task2_train_labels_2026.json`
- training tuple: `query : positive : negatives = 1 : 1 : 15`
- negatives are sampled by model similarity probability from previous epoch
- retrieval evaluation uses micro-average over all queries

## 1) Prepare data

```bash
python "Legal Case Entailment by Mou/prepare_task2_paragraph_data.py"
```

Default output directory:

`Legal Case Entailment by Mou/data/task2_2026_prepared`

## 2) Train

```bash
python "Legal Case Entailment by Mou/fine_tune_task2.py"
```

## 3) Dataset statistics

```bash
python "Legal Case Entailment by Mou/analyze_task2_stats.py"
```

Outputs are saved to:

`Legal Case Entailment by Mou/data/task2_2026_prepared/stats`

## One-command run

```bash
./run_task2_finetune.sh
```

The script automatically:

- loads `.env`
- activates `CONDA_ENV_NAME`
- runs preprocess
- runs training

Edit `.env` once, then run without extra CLI arguments.

## Test mode

Set in `.env`:

```bash
TASK2_MODE=test
```

Behavior:

- uses a subset of train/valid queries for a fast smoke test
- keeps adaptive negative sampling, but only on test subset
- writes to `TASK2_OUTPUT_DIR` with `_test` suffix to avoid overriding full runs

Switch back:

```bash
TASK2_MODE=full
```

## Environment variables

- `COLIEE_TASK2_YEAR` (default: `2026`)
- `CONDA_ENV_NAME` (default: `FLNLP-COLIEE2026-WSL`)
- `COLIEE_TASK2_PREPARED_DIR`
- `TASK2_INIT_MODEL_ROOT`
- `TASK2_INIT_CHECKPOINT`
- `TASK2_INIT_METRIC` (default: `eval_global_f1`)
- `TASK2_INIT_METRIC_MODE` (default: `max`)
- `TASK2_OUTPUT_DIR`
- `TASK2_RESUME_CHECKPOINT`
- `TASK2_EVAL_TOPK` (default: `1`)
- `TASK2_MODE` (`full` or `test`)
- `TASK2_SKIP_STATS` (`1` to skip stats in `run_task2_finetune.sh`)
- `TASK2_TEST_TRAIN_QUERY_LIMIT` / `TASK2_TEST_VALID_QUERY_LIMIT`
- `TASK2_TEST_NUM_TRAIN_EPOCHS` / `TASK2_TEST_MAX_STEPS`
- `TASK2_TRAIN_BATCH_SIZE` / `TASK2_EVAL_BATCH_SIZE` / `TASK2_GRAD_ACCUM_STEPS`
- `TASK2_RETRIEVAL_BATCH_SIZE` / `TASK2_RETRIEVAL_MAX_LENGTH`
- `TASK2_ENABLE_TF32` / `TASK2_GRADIENT_CHECKPOINTING` / `TASK2_CACHE_TEXTS`

Notes:

- Retrieval metrics now report both top-1 and top-2.
- Early stopping and best-checkpoint selection still use validation top-1 F1 (`eval_global_f1`).
