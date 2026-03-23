# FLNLP-COLIEE2026

This repository is the FLNLP working codebase for COLIEE 2026 Task 1 and Task 2. The project was bootstrapped from a public upstream 2023 codebase and has been updated for the current repo name, paths, scripts, and 2026 workflows.

## Task 1: Legal Case Retrieval

Task 1 focuses on retrieving supporting cases for a new case from the legal case corpus.

Main 2026 entrypoints:

- `python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"`
  - Task 1 ModernBERT-FP contrastive fine-tuning entrypoint.
  - Uses the Task 1 dataset configured in repo-root `.env`.

- `run_ltr_feature_train_valid_test_2026.sh`
  - LightGBM LTR pipeline.
  - Rebuilds train/valid/test features, retrains the ranker, writes rerank outputs, then exports a fixed top-5 test submission.
  - This wrapper now skips cutoff grid search on purpose, so you can keep the only cutoff grid search in `run_ltr_cutoff_postprocess_2026.sh`.

- `run_ltr_cutoff_postprocess_2026.sh`
  - Post-process only.
  - Reuses existing rerank outputs and only reruns scope filtering, self-removal, cutoff search, and submission export.
  - It also writes the test submission file and, by default, copies the best submission to repo root as `task1_FLNLPLTR.txt`.

Default submission filename:

- fixed top-5 baseline from step 6: `task1_FLNLPLTRTOP5.txt`
- cutoff grid-search final submission from step 7: `task1_FLNLPLTR.txt`

Task 1 embedding convention:

- inference scripts keep generating both `processed` and `processed_new` document embeddings
- similarity scripts now default to `processed` for both query and candidate
- the old THUIR setting (`processed_new` as query) is still available via embedding-selection env vars documented in `Legal Case Retrieval/README.md`

Detailed Task 1 documentation:

- `Legal Case Retrieval/README.md`

### 建議執行順序

Task 1 若要從原始資料一路跑到 submission，建議依下列順序進行：

1. 先把 Task 1 原始資料放到 `coliee_dataset/task1/<YEAR>/`，並在 repo-root `.env` 設定 `COLIEE_TASK1_YEAR`、資料路徑與必要環境變數。
2. 執行 fine-tune 前置流程，建立 `summary`、`processed`、BM25、train/valid split、hard negatives 與 scope：
   `bash run_pre_finetune_2026.sh`
3. 訓練 dense encoder：
   `python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"`
4. 跑 train/valid 端的 retrieval 驗證與 smoke check：
   `bash run_train_valid_inference_eval_2026.sh`
5. 若要先產出 retrieval-only 的 test submission，可執行：
   `bash run_test_retrieval_2026.sh`
6. 若要跑完整的 LightGBM LTR submission pipeline，執行：
   `bash run_ltr_feature_train_valid_test_2026.sh`
   這一步現在只做 LTR，並在 test 端輸出 fixed top-5 submission，不會做 cutoff grid search。
   預設會另外複製一份 baseline submission 到 repo root：`task1_FLNLPLTRTOP5.txt`
7. 若只是想重搜 cutoff，不重跑 features / LTR，執行：
   `bash run_ltr_cutoff_postprocess_2026.sh`
   這一步會重用既有 rerank 輸出，做唯一一次 cutoff grid search，選出最佳 mode，並產生最後的 test submission。

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

### 建議執行順序

Task 2 若要從原始資料開始，請使用 `Legal Case Entailment by Mou/` 這套流程：

1. 先把 Task 2 原始資料放到 `coliee_dataset/task2/task2_train_files_<YEAR>/`，並在 repo-root `.env` 設定 `COLIEE_TASK2_YEAR`、資料路徑與 `CONDA_ENV_NAME`。
2. 建議直接執行一鍵流程：
   `bash run_task2_finetune.sh`
3. 若要分步執行，順序為：
   `python "Legal Case Entailment by Mou/prepare_task2_paragraph_data.py"`
   `python "Legal Case Entailment by Mou/analyze_task2_stats.py"`（可選）
   `python "Legal Case Entailment by Mou/fine_tune_task2.py"`
4. `Legal Case Entailment/` 目前不要當成可直接執行的主流程使用。

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
