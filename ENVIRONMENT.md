# Environment Setup

This repository uses the conda environment `FLNLP-COLIEE2026-WSL`.

## Files

- `environment.frozen.yml`: full frozen conda + pip environment export

## Scope

This environment record covers the maintained workflows only:

- `Legal Case Retrieval/`
- `Legal Case Entailment by Mou/`
- repo-root 2026 shell scripts

Excluded:

- `Legal Case Entailment/` (legacy code)

## Recreate The Environment

```bash
conda env create -f environment.frozen.yml
conda activate FLNLP-COLIEE2026-WSL
```

If the environment already exists:

```bash
conda env update -n FLNLP-COLIEE2026-WSL -f environment.frozen.yml --prune
conda activate FLNLP-COLIEE2026-WSL
```

## Notes

- `environment.frozen.yml` records the required Python / package dependencies for the maintained workflows.
- Some system-level requirements are still separate, especially Java for Pyserini/BM25 and compatible NVIDIA drivers for GPU execution.

## Not Recorded In `environment.frozen.yml`

The following machine-level programs are not fully installed or reproduced by `conda env create -f environment.frozen.yml`:

- `conda` `25.3.0`
- `bash` `5.2.21(1)-release`
- OpenJDK runtime `21.0.10`
- NVIDIA driver `591.86`
- GPU used during development: `NVIDIA GeForce RTX 4080 SUPER`

These are system-level requirements outside the frozen Python package environment.
