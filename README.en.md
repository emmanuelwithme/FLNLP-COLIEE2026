# FLNLP-COLIEE2026

[Landing](README.md) | [中文](README.zh.md)

This repository is the FLNLP working codebase for COLIEE 2026 Task 1 and Task 2. Two workflows are currently maintained:

- Task 1: `Legal Case Retrieval/`
- Task 2: `Legal Case Entailment by Mou/`

The following directory is not part of the maintained workflow:

- `Legal Case Entailment/`
  This is legacy code kept for reference only. Its old README is intentionally left untouched and should not be treated as an actively supported pipeline.

## Documentation Map

- Environment setup: [ENVIRONMENT.en.md](ENVIRONMENT.en.md)
- Task 1 overview: [Legal Case Retrieval/README.en.md](Legal%20Case%20Retrieval/README.en.md)
- Task 1 dense encoder pipeline: [Legal Case Retrieval/modernBert-fp/README.en.md](Legal%20Case%20Retrieval/modernBert-fp/README.en.md)
- Task 1 LightGBM rerank and submission pipeline: [Legal Case Retrieval/lightgbm/README.en.md](Legal%20Case%20Retrieval/lightgbm/README.en.md)
- Task 1 chunk aggregation experiment: [Legal Case Retrieval/modernBert-fp-chunkAgg/README.en.md](Legal%20Case%20Retrieval/modernBert-fp-chunkAgg/README.en.md)
- Task 1 legacy modernBERT fine-tune and BM25 hard-negative utilities: [Legal Case Retrieval/modernBert/fine_tune/README.en.md](Legal%20Case%20Retrieval/modernBert/fine_tune/README.en.md)
- Task 2 maintained workflow: [Legal Case Entailment by Mou/README.en.md](Legal%20Case%20Entailment%20by%20Mou/README.en.md)

## Repository Layout

- `Legal Case Retrieval/`
  Task 1 preprocessing, dense retrieval, LightGBM reranking, scope filtering, and submission export.
- `Legal Case Entailment by Mou/`
  The maintained paragraph-level ModernBERT fine-tuning workflow for Task 2.
- `coliee_dataset/`
  Default dataset root for raw data and generated artifacts.
- `run_*.sh`
  Workflow wrappers at repo root. Task 1 wrappers are mainly year-specific 2026 entrypoints. Task 2 wrapper reads `.env`.
- `environment.frozen.yml`
  Frozen conda + pip environment record for the maintained workflows.

## Recommended Starting Point

### 1. Create the environment

```bash
conda env create -f environment.frozen.yml
conda activate FLNLP-COLIEE2026-WSL
```

If the environment already exists and you want to sync to the recorded versions:

```bash
conda env update -n FLNLP-COLIEE2026-WSL -f environment.frozen.yml --prune
conda activate FLNLP-COLIEE2026-WSL
```

### 2. Prepare the datasets

Default Task 1 root:

```text
coliee_dataset/task1/2026/
```

Typical contents:

- `task1_train_files_2026/`
- `task1_test_files_2026/`
- `task1_train_labels_2026.json`
- `task1_test_no_labels_2026.json`

Default Task 2 root:

```text
coliee_dataset/task2/task2_train_files_2026/
```

Typical contents:

- `cases/`
- `task2_train_labels_2026.json`

### 3. Enter the task-specific workflow

Recommended Task 1 order:

1. Run `bash run_pre_finetune_2026.sh`
2. Train a dense model or reuse an existing checkpoint
3. Run `bash run_train_valid_inference_eval_2026.sh`
4. For retrieval-only test prediction, run `bash run_test_retrieval_2026.sh`
5. For the full LightGBM rerank submission path, run `bash run_ltr_feature_train_valid_test_2026.sh`
6. To rerun only the cutoff grid search, run `bash run_ltr_cutoff_postprocess_2026.sh`

Recommended Task 2 order:

1. Configure `.env`
2. Run `bash run_task2_finetune.sh`

## Important Configuration Notes

### `.env` and the repo-root wrappers are not exactly the same layer

- `Legal Case Retrieval/lcr/task1_paths.py` automatically reads the repo-root `.env`
- most Task 1 repo-root wrappers are year-specific scripts and do not actively `source .env`
- in practice, Task 1 Python modules can read `.env`, but wrappers such as `run_pre_finetune_2026.sh`, `run_train_valid_inference_eval_2026.sh`, and `run_test_retrieval_2026.sh` still follow their own shell-side defaults
- Task 2's `run_task2_finetune.sh` does read `.env`

### Maintained vs legacy boundary

- The maintained Task 1 workflow is centered on `Legal Case Retrieval/modernBert-fp/` and `Legal Case Retrieval/lightgbm/`
- `Legal Case Retrieval/modernBert/` still contains older training code and a BM25 hard-negative generator that is still used in preprocessing
- Task 2 is maintained only under `Legal Case Entailment by Mou/`

## Upstream Reference

This repository started from the public THUIR COLIEE 2023 release and has since been heavily refactored across training, inference, data flow, and overall structure.

Public upstream references retained as background:

- THUIR COLIEE 2023 repository: <https://github.com/CSHaitao/THUIR-COLIEE2023>
- SAILER: Structure-aware Pre-trained Language Model for Legal Case Retrieval
- THUIR@COLIEE 2023: Incorporating Structural Knowledge into Pre-trained Language Models for Legal Case Retrieval
- THUIR@COLIEE 2023: More Parameters and Legal Knowledge for Legal Case Entailment
