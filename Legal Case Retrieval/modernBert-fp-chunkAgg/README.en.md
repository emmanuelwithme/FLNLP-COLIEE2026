# modernBert-fp-chunkAgg

[中文](README.md) | [Task 1 Overview](../README.en.md)

This directory contains the chunk-aggregation variant of ModernBERT. Instead of encoding a document as one single `4096`-token segment, it keeps up to `3` chunks and aggregates chunk-level representations into one document vector.

It is a preserved experimental path, not the default submission workflow. The maintained default path for Task 1 is still `../modernBert-fp/`.

## Directory Contents

- `fine_tune/fine_tune.py`
  Contrastive fine-tuning entrypoint. Query / positive / negative all share the same chunk encoder.
- `inference.py`
  Loads the best checkpoint and generates embeddings.
- `similarity_and_rank.py`
  Reads embeddings and writes train / valid rankings.
- `fine_tune/modernbert_contrastive_model.py`
  ChunkAgg model definition.
- `.env`
  Main configuration file for this directory.
- `run_train.sh`
  Starts fine-tuning from repo root.
- `run_infer.sh`
  Generates embeddings.
- `run_rank.sh`
  Generates rankings.

## Document Encoding Logic

Documents and queries share the same chunk encoding logic:

1. tokenize the full text
2. keep at most `TASK1_DOCUMENT_CHUNK_LENGTH` tokens per chunk
3. backtrack to a sentence-end boundary when possible
4. keep at most `TASK1_MAX_DOCUMENT_CHUNKS` chunks per document
5. truncate the tail if the full text still exceeds the total limit
6. take ModernBERT `[CLS]` for each chunk and pass it through the projector
7. add a learnable `[DOC]` token and chunk position embeddings
8. fuse them with a 1-layer pre-norm transformer block
9. use the `[DOC]` token as the final document vector and apply L2 normalization

## Default Behavior

- `inference.py` writes both `processed` and `processed_new` embeddings
- `similarity_and_rank.py` defaults to `processed` for both query and candidate
- if you want the older THUIR-style query setup, switch the query embeddings to `processed_new`

## `.env` Loading Rules

This directory-level `.env` is automatically read by:

- `fine_tune/fine_tune.py`
- `inference.py`
- `similarity_and_rank.py`
- `fine_tune/modernbert_contrastive_model.py`

Rules:

- if the shell already defines a variable with the same name, the shell value wins
- `.env` only fills variables that are still unset
- no manual `source .env` is required

## Important Variables

### chunk encoder

- `TASK1_MAX_DOCUMENT_CHUNKS`
- `TASK1_DOCUMENT_CHUNK_LENGTH`
- `TASK1_CHUNK_MICROBATCH_SIZE`
- `TASK1_CHUNKAGG_ENABLE_TF32`
- `TASK1_CHUNKAGG_TEXT_CACHE_SIZE`
- `TASK1_CHUNKAGG_CHUNK_CACHE_SIZE`
- `TASK1_CHUNKAGG_PIN_MEMORY`
- `TASK1_CHUNKAGG_PERSISTENT_WORKERS`
- `TASK1_INIT_TEMPERATURE`
- `TASK1_RETRIEVAL_BATCH_SIZE`

### training

- `TASK1_CHUNKAGG_QUICK_TEST`
- `TASK1_CHUNKAGG_SCOPE_FILTER`
- `TASK1_CHUNKAGG_TRAIN_BATCH_SIZE`
- `TASK1_CHUNKAGG_GRAD_ACCUM_STEPS`
- `TASK1_CHUNKAGG_NUM_EPOCHS`
- `TASK1_CHUNKAGG_ENCODER_LR`
- `TASK1_CHUNKAGG_FUSION_LR`
- `TASK1_CHUNKAGG_TEMPERATURE_LR`
- `TASK1_CHUNKAGG_SAMPLING_TEMPERATURE`
- `TASK1_CHUNKAGG_UPDATE_FREQUENCY`

### inference / ranking

- `TASK1_CHUNKAGG_MODEL_NAME`
- `TASK1_CHUNKAGG_CANDIDATE_DIR`
- `TASK1_CHUNKAGG_QUERY_DIR`
- `TASK1_CHUNKAGG_CAND_EMB_SOURCE`
- `TASK1_CHUNKAGG_QUERY_EMB_SOURCE`
- `TASK1_CHUNKAGG_CAND_EMB_PATH`
- `TASK1_CHUNKAGG_QUERY_EMB_PATH`
- `TASK1_CHUNKAGG_SCOPE_PATH`

### path override

- `TASK1_CHUNKAGG_BASE_ENCODER_DIR`
- `TASK1_CHUNKAGG_OUTPUT_DIR`
- `TASK1_CHUNKAGG_FINETUNE_DATA_DIR`

## Usage

Running from repo root is recommended.

### 1. Fine-tune

Before starting, verify at least these values in `modernBert-fp-chunkAgg/.env`:

- `TASK1_CHUNKAGG_OUTPUT_DIR`
- `TASK1_CHUNKAGG_MODEL_NAME`
- `TASK1_CHUNK_MICROBATCH_SIZE`
- `TASK1_RETRIEVAL_BATCH_SIZE`

Then run:

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_train.sh"
```

### 2. Generate embeddings

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_infer.sh"
```

### 3. Generate rankings

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_rank.sh"
```

To switch query embeddings back to `processed_new`:

```bash
TASK1_CHUNKAGG_QUERY_EMB_SOURCE=processed_new \
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_rank.sh"
```

## Recommended Tuning Order

1. confirm output directory and experiment naming
2. if VRAM is insufficient, reduce `TASK1_CHUNK_MICROBATCH_SIZE` first
3. if retrieval is too slow, tune `TASK1_RETRIEVAL_BATCH_SIZE`
4. if GPU utilization is low, enable `TASK1_CHUNKAGG_ENABLE_TF32=1` and increase cache-related settings
5. if training is unstable, tune the fusion and temperature learning rates

## Notes

- sentence-end splitting is currently heuristic, not a full legal sentence segmenter
- training, adaptive negative sampling, retrieval evaluation, and inference all share the same chunk encoder
- the loss is currently document-level InfoNCE / CrossEntropy only
