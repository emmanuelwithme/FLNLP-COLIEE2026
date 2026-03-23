# FLNLP-COLIEE2026

This repository is the FLNLP working codebase for COLIEE 2026 Task 1 and Task 2. The project was bootstrapped from a public upstream 2023 codebase and has been updated for the current repo name, paths, scripts, and 2026 workflows.

## Task 1: Legal Case Retrieval

Task 1 focuses on retrieving supporting cases for a new case from the legal case corpus.

Main 2026 entrypoints:

- `python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"`
  - Task 1 ModernBERT-FP contrastive fine-tuning entrypoint.
  - Uses the Task 1 dataset configured in repo-root `.env`.

- `run_ltr_feature_train_valid_test_2026.sh`
  - Full LightGBM LTR pipeline.
  - Rebuilds train/valid/test features, retrains the ranker, writes rerank outputs, then runs cutoff search and submission generation.

- `run_ltr_cutoff_postprocess_2026.sh`
  - Post-process only.
  - Reuses existing rerank outputs and only reruns scope filtering, self-removal, cutoff search, and submission export.

Default submission filename:

- `task1_FLNLPLTR.txt`

Task 1 embedding convention:

- inference scripts keep generating both `processed` and `processed_new` document embeddings
- similarity scripts now default to `processed` for both query and candidate
- the old THUIR setting (`processed_new` as query) is still available via embedding-selection env vars documented in `Legal Case Retrieval/README.md`

Detailed Task 1 documentation:

- `Legal Case Retrieval/README.md`

## Task 2: Legal Case Entailment

Task 2 identifies the paragraph in a relevant case that entails the decision of a new case.

Status note:

- `Legal Case Entailment/` is a legacy upstream folder.
- It has not been fully migrated or repaired for the current repo, so code under that folder should be treated as not runnable as-is.
- The current maintained Task 2 workflow is under `Legal Case Entailment by Mou/` and the repo-root `run_task2_finetune.sh`.

Main 2026 entrypoints:

- `run_task2_finetune.sh`
  - Loads repo-root `.env`
  - Activates the configured conda environment
  - Prepares paragraph-level Task 2 data
  - Optionally generates dataset statistics
  - Runs ModernBERT fine-tuning

Detailed Task 2 documentation:

- `Legal Case Entailment/README.md`
- `Legal Case Entailment by Mou/README.md`

## Dataset

Use the official COLIEE dataset access process for the corresponding competition year. Keep Task 1 and Task 2 raw data under `./coliee_dataset/` or override the dataset roots in `.env`.

## Upstream References

The following papers are the original upstream references retained from the public THUIR code release:

## Citation
```
@misc{li2023sailer,
      title={SAILER: Structure-aware Pre-trained Language Model for Legal Case Retrieval}, 
      author={Haitao Li and Qingyao Ai and Jia Chen and Qian Dong and Yueyue Wu and Yiqun Liu and Chong Chen and Qi Tian},
      year={2023},
      eprint={2304.11370},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

```
@misc{li2023thuircoliee,
      title={THUIR@COLIEE 2023: Incorporating Structural Knowledge into Pre-trained Language Models for Legal Case Retrieval}, 
      author={Haitao Li and Weihang Su and Changyue Wang and Yueyue Wu and Qingyao Ai and Yiqun Liu},
      year={2023},
      eprint={2305.06812},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```

```
@misc{li2023thuircoliee,
      title={THUIR@COLIEE 2023: More Parameters and Legal Knowledge for Legal Case Entailment}, 
      author={Haitao Li and Changyue Wang and Weihang Su and Yueyue Wu and Qingyao Ai and Yiqun Liu},
      year={2023},
      eprint={2305.06817},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
