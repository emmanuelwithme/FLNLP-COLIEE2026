# FLNLP-COLIEE2026

[中文總覽](README.zh.md) | [English Overview](README.en.md)

FLNLP working codebase for COLIEE 2026 Task 1 and Task 2.

## Maintained Workflows

- Task 1: `Legal Case Retrieval/`
- Task 2: `Legal Case Entailment by Mou/`

`Legal Case Entailment/` is kept as legacy reference only and is not part of the maintained workflow.

## Quick Links

- Environment: [中文](ENVIRONMENT.md) | [English](ENVIRONMENT.en.md)
- Task 1 overview: [中文](Legal%20Case%20Retrieval/README.md) | [English](Legal%20Case%20Retrieval/README.en.md)
- Task 1 dense retrieval pipeline: [中文](Legal%20Case%20Retrieval/modernBert-fp/README.md) | [English](Legal%20Case%20Retrieval/modernBert-fp/README.en.md)
- Task 1 LightGBM rerank pipeline: [中文](Legal%20Case%20Retrieval/lightgbm/README.md) | [English](Legal%20Case%20Retrieval/lightgbm/README.en.md)
- Task 2 maintained workflow: [中文](Legal%20Case%20Entailment%20by%20Mou/README.md) | [English](Legal%20Case%20Entailment%20by%20Mou/README.en.md)

## Recommended Entry Points

- Task 1 preprocessing: `bash run_pre_finetune_2026.sh`
- Task 1 train/valid retrieval check: `bash run_train_valid_inference_eval_2026.sh`
- Task 1 retrieval-only test prediction: `bash run_test_retrieval_2026.sh`
- Task 1 LightGBM submission pipeline: `bash run_ltr_feature_train_valid_test_2026.sh`
- Task 1 cutoff search only: `bash run_ltr_cutoff_postprocess_2026.sh`
- Task 2 one-command run: `bash run_task2_finetune.sh`
