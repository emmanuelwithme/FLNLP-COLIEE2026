# modernBert-fp-chunkAgg

[中文](README.md) | [Task 1 Overview](../README.en.md)

This directory provides the Task 1 chunk-aggregation variant. Each document is split into multiple chunks, and chunk representations are fused into a single document vector.

## Directory Contents

- `fine_tune/fine_tune.py`: chunkAgg contrastive training.
- `inference.py`: generates chunkAgg embeddings.
- `similarity_and_rank.py`: reads embeddings and writes rankings.
- `fine_tune/modernbert_contrastive_model.py`: model definition.
- `run_train.sh`: starts training.
- `run_infer.sh`: generates embeddings.
- `run_rank.sh`: generates rankings.
- `.env`: default configuration for this directory.

## Encoding Pipeline

1. Tokenize the full text.
2. Keep `TASK1_DOCUMENT_CHUNK_LENGTH` tokens per chunk.
3. Split near sentence boundaries when possible.
4. Keep at most `TASK1_MAX_DOCUMENT_CHUNKS` chunks per document.
5. Take the ModernBERT `[CLS]` vector for each chunk.
6. Fuse chunks with a learnable `[DOC]` token and chunk position embeddings.
7. Output one document vector and apply L2 normalization.

## Usage

Running from the repo root is recommended.

### 1. Configure variables

Edit `Legal Case Retrieval/modernBert-fp-chunkAgg/.env`, or override the same variables in the shell. Shell values take priority.

Common variables:

- `TASK1_CHUNKAGG_MODEL_NAME`
- `TASK1_CHUNKAGG_BASE_ENCODER_DIR`
- `TASK1_CHUNKAGG_OUTPUT_DIR`
- `TASK1_CHUNKAGG_FINETUNE_DATA_DIR`
- `TASK1_MAX_DOCUMENT_CHUNKS`
- `TASK1_DOCUMENT_CHUNK_LENGTH`
- `TASK1_CHUNK_MICROBATCH_SIZE`
- `TASK1_RETRIEVAL_BATCH_SIZE`

### 2. Train

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_train.sh"
```

### 3. Generate embeddings

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_infer.sh"
```

### 4. Generate rankings

```bash
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_rank.sh"
```

To switch the query embedding source:

```bash
TASK1_CHUNKAGG_QUERY_EMB_SOURCE=processed_new \
bash "Legal Case Retrieval/modernBert-fp-chunkAgg/run_rank.sh"
```

## Common Outputs

- `TASK1_CHUNKAGG_OUTPUT_DIR/`
- `processed/processed_document_<MODEL_NAME>_embeddings*.pkl`
- `processed_new/processed_new_document_<MODEL_NAME>_embeddings*.pkl`
- ranking TREC files

This directory runs independently from the repo-root Task 1 wrappers. For the main Task 1 flow, see [../modernBert-fp/README.en.md](../modernBert-fp/README.en.md).
