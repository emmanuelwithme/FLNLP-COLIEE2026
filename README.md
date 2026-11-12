# FLNLP-COLIEE2026

[中文說明](README.zh.md) | [English Guide](README.en.md)

FLNLP codebase centered on COLIEE Task 1.

- The main workflow in this repository is Task 1.
- The author participated in Task 1 only.
- Task 2 code is kept as supplementary material and is not a main focus.
- Edit paths, model locations, and run parameters in `.env`.
- For one-off runs, shell environment variables override `.env`.
- Shared model directories are stored under `models/`.
- The isolated `modernBert-bm25` 2025 experiment uses its own JSON config rather than the root `.env`.

## Entry Points

- Task 1 preprocessing: `bash run_pre_finetune.sh`
- Task 1 dense training: `python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"`
- Task 1 BM25 epoch-resampling experiment: `python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune.py" --no-debug`
- Task 1 BM25 with SAILER-style in-batch negatives and GradCache: `python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune_sailer.py" --no-debug`
- Task 1 dense embeddings: `python "Legal Case Retrieval/modernBert-fp/inference.py"`
- Task 1 train/valid ranking: `python "Legal Case Retrieval/modernBert-fp/similarity_and_rank.py"`
- Task 1 evaluation: `python "Legal Case Retrieval/utils/eval.py"`
- Task 1 LTR feature and fixed top-k export: `bash run_ltr_feature_train_valid_test.sh`
- Task 1 LTR cutoff search and final submission export: `bash run_ltr_cutoff_postprocess.sh`
- Task 2 supplementary workflow: run the preparation, statistics, and training Python entry points documented below.

## Documentation

- Environment: [中文](ENVIRONMENT.md) | [English](ENVIRONMENT.en.md)
- Task 1 overview: [中文](Legal%20Case%20Retrieval/README.md) | [English](Legal%20Case%20Retrieval/README.en.md)
- Task 1 dense retrieval: [中文](Legal%20Case%20Retrieval/modernBert-fp/README.md) | [English](Legal%20Case%20Retrieval/modernBert-fp/README.en.md)
- Task 1 original ModernBERT-base, BM25 top-100 / epoch random-15: [Guide](Legal%20Case%20Retrieval/modernBert-bm25/README.md) | [SAILER-style GradCache](Legal%20Case%20Retrieval/modernBert-bm25/SAILER_GRADCACHE.md) | [Grouping validation](Legal%20Case%20Retrieval/modernBert-bm25/GROUPING.md)
- COLIEE experiment audits and historical validation: [Research records](research_notes/coliee2025_review/README.md)
- Task 1 ChunkAgg v5: [中文](Legal%20Case%20Retrieval/modernBert-fp-chunkAgg/README.md) | [English](Legal%20Case%20Retrieval/modernBert-fp-chunkAgg/README.en.md)
- Task 1 ChunkAgg v6: [中文](Legal%20Case%20Retrieval/modernBert-fp-chunkAgg-v6/README.md) | [English](Legal%20Case%20Retrieval/modernBert-fp-chunkAgg-v6/README.en.md)
- Task 1 LightGBM rerank: [中文](Legal%20Case%20Retrieval/lightgbm/README.md) | [English](Legal%20Case%20Retrieval/lightgbm/README.en.md)
- Task 2 paragraph workflow: [中文](Legal%20Case%20Entailment%20by%20Mou/README.md) | [English](Legal%20Case%20Entailment%20by%20Mou/README.en.md)
