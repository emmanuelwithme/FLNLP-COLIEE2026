
# Legal Case Retrieval

To be specific, we design structure-aware pre-trained language models to enhance the understanding of legal cases. Furthermore, we propose heuristic pre-processing and post-processing approaches to reduce the influence of irrelevant messages. In the end, learning-to-rank methods are employed to merge features with different dimensions.


## Pre-processing

`process.py` file is used for preprocessing legal cases, including removing certain symbols and tags, removing French, concatenating abstracts(連接摘要), and more. 

`reference.py` file only retains sentences containing special tags(特殊標記) and adjacent sentences(相鄰的句子). 

`summary.py` file is used for extracting the summary from an unprocessed file. 

`build_query_candidate_scope.py` (in `pre-process/`) builds a per-query candidate scope JSON (query -> allowed candidate ids), so retrieval can filter out future cases before similarity computation.

### Dataset year switch (`.env`)

Task1 paths are controlled by repo-root `.env`:

```env
COLIEE_TASK1_YEAR=2025
COLIEE_TASK1_ROOT=./coliee_dataset/task1
```

Scripts read this setting and resolve Task1 data directory as:

`$COLIEE_TASK1_ROOT/$COLIEE_TASK1_YEAR`

Example:
- `2025` -> `./coliee_dataset/task1/2025`
- `2026` -> `./coliee_dataset/task1/2026`

## Embedding convention

For the current FLNLP Task 1 setup:

- inference scripts still encode and save both `processed` and `processed_new`
- similarity / ranking scripts default to `processed` for both query and candidate
- the original THUIR paper setting used `processed_new` as query embeddings
- this repo now defaults to `processed` as query embeddings, while candidates also default to `processed`

Default embedding files:

- `processed`: `${TASK1_DIR}/processed/processed_document_<MODEL_NAME>_embeddings.pkl`
- `processed_new`: `${TASK1_DIR}/processed_new/processed_new_document_<MODEL_NAME>_embeddings.pkl`

For most similarity scripts, you can switch the source without editing code:

```bash
export LCR_QUERY_EMBED_SOURCE=processed_new
export LCR_CANDIDATE_EMBED_SOURCE=processed
python "Legal Case Retrieval/modernBert/similarity_and_rank.py"
```

Accepted values:

- `LCR_QUERY_EMBED_SOURCE`: `processed` or `processed_new`
- `LCR_CANDIDATE_EMBED_SOURCE`: `processed` or `processed_new`

If you need a completely custom file, you can override the path directly:

```bash
export LCR_QUERY_EMBEDDINGS_PATH=/abs/path/to/query_embeddings.pkl
export LCR_CANDIDATE_EMBEDDINGS_PATH=/abs/path/to/candidate_embeddings.pkl
```

## 建議執行順序（2026）

如果要從 Task 1 原始資料一路跑到最終 submission，建議流程如下：

1. 將 Task 1 原始資料與 labels 放到 `coliee_dataset/task1/<YEAR>/`。
2. 設定 repo-root `.env`，確認 `COLIEE_TASK1_YEAR` 與 `COLIEE_TASK1_ROOT`。
3. 執行前處理與 fine-tune 前置流程：
   `bash run_pre_finetune_2026.sh`
   這一步會依序建立 `summary`、`processed`、train/valid split、BM25 index / ranking、BM25 hard negatives，以及 `query_candidate_scope.json`。
4. 訓練 dense encoder：
   `python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"`
5. 跑 train/valid 端檢查與排名輸出：
   `bash run_train_valid_inference_eval_2026.sh`
6. 若要先做 retrieval-only test submission：
   `bash run_test_retrieval_2026.sh`
7. 若要跑完整 LightGBM rerank + submission：
   `bash run_ltr_feature_train_valid_test_2026.sh`
   這一步現在只做 LTR，並在 test 端輸出 fixed top-5 submission，不會做 cutoff grid search。
   預設會另外複製一份 baseline submission 到 repo root：`task1_FLNLPLTRTOP5.txt`
8. 若已有 rerank 輸出，只想重做 cutoff 後處理：
   `bash run_ltr_cutoff_postprocess_2026.sh`
   這一步會重用既有的 `valid_predictions_raw.csv` / `test_predictions_raw.csv`，做唯一一次 cutoff grid search，並輸出最後的 test submission。


## Traditional Lexical Matching Models

We implement BM25 and QLD with the [Pyserini](https://github.com/castorini/pyserini).

`lexical models` directory provides an example for reproduction.

## SAILER

SAILER stands for Structure-aware Pre-trained Language Model for Legal Case Retrieval, which has been accepted by SIGIR2023.

The implementation and checkpoint of SAILER can be viewed at [SAILER](https://github.com/lihaitao18375278/SAILER)



## Learning to rank

`lightgbm` directory provides the implementation of Lightgbm.

### 2026 LightGBM LTR pipeline

For the current Task 1 workflow, the LightGBM rerank pipeline is implemented in:

- `Legal Case Retrieval/lightgbm/ltr_feature_pipeline.py`

This pipeline now supports:

- accelerated feature building with multi-core CPU and batched GPU inference
- explicit raw-scope or processed-scope filtering
- post-rerank legal filtering and self-removal
- validation-only cutoff search
- final test-time application of the selected cutoff

Two repo-root shell scripts are provided:

- `run_ltr_feature_train_valid_test_2026.sh`
  - LightGBM LTR pipeline.
  - Rebuilds train/valid/test features, retrains LightGBM, writes rerank outputs, then exports a fixed top-k test submission.
  - The repo-root wrapper `run_ltr_feature_train_valid_test_2026.sh` currently fixes this to top-5 and skips cutoff grid search on purpose.
  - This is the slow path because it repeats feature generation.

- `run_ltr_cutoff_postprocess_2026.sh`
  - Post-process only.
  - Reuses existing `valid_predictions_raw.csv` and `test_predictions_raw.csv`.
  - Runs scope filtering, self-removal, cutoff search, best-mode selection, and submission export.
  - It also writes `best_overall/test_submission_best_mode.txt` and, by default, copies that file to repo root as `task1_FLNLPLTR.txt`.
  - Use this when you want the only cutoff grid search in the workflow and the final test submission.

Typical usage:

```bash
bash run_ltr_feature_train_valid_test_2026.sh
```

This writes a fixed top-5 baseline submission to:

- `coliee_dataset/task1/<YEAR>/lht_process/lightgbm_ltr_scope_raw/fixed_top5/test_submission_fixed_topk.txt`
- repo root copy: `task1_FLNLPLTRTOP5.txt`

```bash
bash run_ltr_cutoff_postprocess_2026.sh
```

This writes the final cutoff-searched submission to:

- `coliee_dataset/task1/<YEAR>/lht_process/lightgbm_ltr_scope_raw/cutoff_search/best_overall/test_submission_best_mode.txt`
- repo root copy: `task1_FLNLPLTR.txt`

If you want to override the cutoff grid without retraining:

```bash
COLIEE_LTR_CUTOFF_CONFIG_JSON=/path/to/cutoff_config.json \
bash run_ltr_cutoff_postprocess_2026.sh
```

By default, `run_ltr_cutoff_postprocess_2026.sh` uses:

- valid scope: `coliee_dataset/task1/<YEAR>/lht_process/scope_compare/query_candidate_scope_raw_plus0.json`
- test scope: `coliee_dataset/task1/<YEAR>/lht_process/modernBert/query_candidate_scope_test_raw.json`
- cutoff config: prefer `coliee_dataset/task1/<YEAR>/lht_process/lightgbm_ltr_scope_raw/cutoff_search_expanded_config.json` if that file exists; otherwise use the built-in defaults in `cutoff_postprocess.py`
- submission filename: `task1_FLNLPLTR.txt`

The cutoff post-processing module is implemented in:

- `Legal Case Retrieval/lightgbm/cutoff_postprocess.py`

It compares three per-query cutoff modes on the same rerank output:

- fixed top-k
- ratio cutoff `(p, l, h)`
- largest-gap adaptive cutoff `(N, buffer, l, h)`

The workflow is:

1. read existing validation/test rerank results
2. apply scope-based filtering and self-removal
3. search cutoff parameters on validation only
4. select the best mode and parameters by validation metrics
5. apply the selected cutoff to test once
6. write validation comparison tables, test predictions, candidate lists, and submission text


## utils

`eval.py` is the evaluation code we provided. The input result should be in trec format.

`train_qid` and `valid_qid` reflect the ids of our training and validation sets.
