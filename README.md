# FLNLP-COLIEE2026

[中文說明](README.zh.md) | [English Guide](README.en.md)

FLNLP codebase centered on COLIEE Task 1.

- The main workflow in this repository is Task 1.
- The author participated in Task 1 only.
- Task 2 code is kept as supplementary material and is not a main focus.
- Edit paths, model locations, and run parameters in `.env`.
- For one-off runs, shell environment variables override `.env`.
- Shared model directories are stored under `models/`.

## Entry Points

- Task 1 preprocessing: `bash run_pre_finetune.sh`
- Task 1 dense training: `python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"`
- Task 1 train/valid retrieval check: `bash run_train_valid_inference_eval.sh`
- Task 1 test retrieval export: `bash run_test_retrieval.sh`
- Task 1 LTR feature and fixed top-k export: `bash run_ltr_feature_train_valid_test.sh`
- Task 1 LTR cutoff search and final submission export: `bash run_ltr_cutoff_postprocess.sh`
- Task 2 supplementary workflow: `bash run_task2_finetune.sh`

## Documentation

- Environment: [中文](ENVIRONMENT.md) | [English](ENVIRONMENT.en.md)
- Task 1 overview: [中文](Legal%20Case%20Retrieval/README.md) | [English](Legal%20Case%20Retrieval/README.en.md)
- Task 1 dense retrieval: [中文](Legal%20Case%20Retrieval/modernBert-fp/README.md) | [English](Legal%20Case%20Retrieval/modernBert-fp/README.en.md)
- Task 1 LightGBM rerank: [中文](Legal%20Case%20Retrieval/lightgbm/README.md) | [English](Legal%20Case%20Retrieval/lightgbm/README.en.md)
- Task 2 paragraph workflow: [中文](Legal%20Case%20Entailment%20by%20Mou/README.md) | [English](Legal%20Case%20Entailment%20by%20Mou/README.en.md)
