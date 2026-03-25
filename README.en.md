# FLNLP-COLIEE2026

[Landing](README.md) | [中文](README.zh.md)

This repository is centered on the FLNLP workflow for COLIEE Task 1.

- Task 1 is the primary focus of this repository.
- The author participated in Task 1 only.
- Task 2 code is kept as supplementary material and is not a main focus.

## Quick Start

1. Create the environment.

```bash
conda env create -f environment.frozen.yml
conda activate FLNLP-COLIEE2026-WSL
```

2. Edit the repo-root `.env`.

- Put dataset paths, model paths, and run parameters there.
- For one-off runs, shell environment variables override `.env`.
- Shared model directories are stored under `models/`.

3. Choose a workflow.

- Task 1 preprocessing: `bash run_pre_finetune.sh`
- Task 1 dense training: `python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"`
- Task 1 train/valid retrieval check: `bash run_train_valid_inference_eval.sh`
- Task 1 test retrieval export: `bash run_test_retrieval.sh`
- Task 1 LTR feature and fixed top-k export: `bash run_ltr_feature_train_valid_test.sh`
- Task 1 LTR cutoff search and final submission export: `bash run_ltr_cutoff_postprocess.sh`
- Task 2 supplementary workflow: `bash run_task2_finetune.sh`

## Documentation Map

- Environment: [ENVIRONMENT.en.md](ENVIRONMENT.en.md)
- Task 1 overview: [Legal Case Retrieval/README.en.md](Legal%20Case%20Retrieval/README.en.md)
- Task 1 dense retrieval: [Legal Case Retrieval/modernBert-fp/README.en.md](Legal%20Case%20Retrieval/modernBert-fp/README.en.md)
- Task 1 LightGBM rerank: [Legal Case Retrieval/lightgbm/README.en.md](Legal%20Case%20Retrieval/lightgbm/README.en.md)
- Task 2 paragraph workflow: [Legal Case Entailment by Mou/README.en.md](Legal%20Case%20Entailment%20by%20Mou/README.en.md)
