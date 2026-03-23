
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


## Traditional Lexical Matching Models

We implement BM25 and QLD with the [Pyserini](https://github.com/castorini/pyserini).

`lexical models` directory provides an example for reproduction.

## SAILER

SAILER stands for Structure-aware Pre-trained Language Model for Legal Case Retrieval, which has been accepted by SIGIR2023.

The implementation and checkpoint of SAILER can be viewed at [SAILER](https://github.com/lihaitao18375278/SAILER)



## Learning to rank

`lightgbm` directory provides the implementation of Lightgbm.

python lgb_ltr.py -process process feauture data to feat.txt and group.txt

python lgb_ltr.py -train

python lgb_ltr.py -predict

The format of feauture data (like ranklib):
0 qid:10002 1:0.007477 2:0.000000 ... 45:0.000000 46:0.007042 

Reference: [Link](https://github.com/jiangnanboy/learning_to_rank)

### 2026 LightGBM LTR pipeline

For the current Task 1 workflow, the LightGBM rerank pipeline is implemented in:

- `Legal Case Retrieval/lightgbm/src/trees/ltr_feature_pipeline.py`

This pipeline now supports:

- accelerated feature building with multi-core CPU and batched GPU inference
- explicit raw-scope or processed-scope filtering
- post-rerank legal filtering and self-removal
- validation-only cutoff search
- final test-time application of the selected cutoff

Two repo-root shell scripts are provided:

- `run_ltr_feature_train_valid_test_2026.sh`
  - Full pipeline.
  - Rebuilds train/valid/test features, retrains LightGBM, writes rerank outputs, then runs cutoff post-processing.
  - This is the slow path because it repeats feature generation.

- `run_ltr_cutoff_postprocess_2026.sh`
  - Post-process only.
  - Reuses existing `valid_predictions_raw.csv` and `test_predictions_raw.csv`.
  - Runs scope filtering, self-removal, cutoff search, best-mode selection, and submission export.
  - Use this when you only want to re-search cutoff settings.

Typical usage:

```bash
bash run_ltr_feature_train_valid_test_2026.sh
```

```bash
bash run_ltr_cutoff_postprocess_2026.sh
```

If you want to override the cutoff grid without retraining:

```bash
COLIEE_LTR_CUTOFF_CONFIG_JSON=/path/to/cutoff_config.json \
bash run_ltr_cutoff_postprocess_2026.sh
```

By default, `run_ltr_cutoff_postprocess_2026.sh` uses:

- valid scope: `coliee_dataset/task1/<YEAR>/lht_process/scope_compare/query_candidate_scope_raw_plus0.json`
- test scope: `coliee_dataset/task1/<YEAR>/lht_process/modernBert/query_candidate_scope_test_raw.json`
- submission filename: `task1_FLNLPLTR.txt`

The cutoff post-processing module is implemented in:

- `Legal Case Retrieval/lightgbm/src/trees/cutoff_postprocess.py`

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

## Post-processing

`year.py` file is used for generating the JSON file filtering by trial date. Note that this file uses raw documents, which are not included in this folder. Please make sure the necessary documents are provided before running this script.

For modern `lcr` pipelines, you can move this logic into pre-filtered retrieval by setting env `LCR_QUERY_CANDIDATE_SCOPE_JSON=/path/to/query_candidate_scope.json`. Then both training-time retrieval (`generate_similarity_artifacts`) and inference-time ranking (`compute_similarity_and_save`) only score candidates in each query's scope.

`grid_search.py` file is used for searching the hyperparameters (p, l, h), where `p` denotes the truncation percentage relative to the highest score, `l` denotes the minimum number of answers, and `h` denotes the maximum number of answers. This file requires two JSON files: one for year filtering and another for score results.

`inference.py` file is used for generating the JSON file of test set results with customizable hyperparameters. Like `grid_search.py`, this file requires two JSON files for year filtering and score results.

Files with prefix "score" in the folder are score result files in the following format: 

```json
{
	"query_id1": {
		"doc_id1": {
			"score": _score1,
			"rank": _rank1
		}, 
		"doc_id2": {
			"score": _score2,
			"rank": _rank2
		}, 
		...
	},
	...
}
```


## utils

`eval.py` is the evaluation code we provided. The input result should be in trec format.

`train_qid` and `valid_qid` reflect the ids of our training and validation sets.
