# Environment Setup

[中文](ENVIRONMENT.md) | [Back to Landing](README.md)

The maintained workflows in this repository use the conda environment `FLNLP-COLIEE2026-WSL`.

## What This Covers

This environment record is intended only for the maintained workflows:

- `Legal Case Retrieval/`
- `Legal Case Entailment by Mou/`
- maintained repo-root shell wrappers

Excluded:

- `Legal Case Entailment/` legacy code

## Environment Record File

- `environment.frozen.yml`
  Full frozen conda + pip export

## Create the Environment

```bash
conda env create -f environment.frozen.yml
conda activate FLNLP-COLIEE2026-WSL
```

If the environment already exists and you want to sync it to the recorded versions:

```bash
conda env update -n FLNLP-COLIEE2026-WSL -f environment.frozen.yml --prune
conda activate FLNLP-COLIEE2026-WSL
```

## `.env` vs the Conda Environment

- `environment.frozen.yml` records the Python and package environment
- the repo-root `.env` records dataset paths, year settings, and some workflow parameters
- they serve different purposes, and most runs need both

## System-Level Requirements

The following are not fully installed or reproduced by `conda env create -f environment.frozen.yml`:

- `conda` `25.3.0`
- `bash` `5.2.21(1)-release`
- OpenJDK runtime `21.0.10`
- NVIDIA driver `591.86`
- GPU used during development: `NVIDIA GeForce RTX 4080 SUPER`

## Notes

- Pyserini and BM25-related workflows require Java
- GPU training and inference require a compatible NVIDIA driver and CUDA runtime
- CPU-only execution is still possible for some stages, but it is substantially slower
