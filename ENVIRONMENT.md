# 環境設定

[English](ENVIRONMENT.en.md) | [回到入口頁](README.md)

本專案目前維護中的流程使用 conda 環境 `FLNLP-COLIEE2026-WSL`。

## 這份環境文件涵蓋什麼

這份環境記錄只保證以下流程：

- `Legal Case Retrieval/`
- `Legal Case Entailment by Mou/`
- 根目錄維護中的 shell wrappers

不包含：

- `Legal Case Entailment/` legacy 程式碼

## 版本記錄檔

- `environment.frozen.yml`
  完整的 conda + pip 凍結匯出

## 建立環境

```bash
conda env create -f environment.frozen.yml
conda activate FLNLP-COLIEE2026-WSL
```

若環境已存在，要同步到記錄版本：

```bash
conda env update -n FLNLP-COLIEE2026-WSL -f environment.frozen.yml --prune
conda activate FLNLP-COLIEE2026-WSL
```

## `.env` 與 conda 環境的關係

- `environment.frozen.yml` 記錄的是 Python 與套件環境
- repo root 的 `.env` 則記錄資料路徑、年份與部分 workflow 參數
- 兩者用途不同，通常都需要

## 系統層需求

以下需求不會由 `conda env create -f environment.frozen.yml` 自動完整安裝：

- `conda` `25.3.0`
- `bash` `5.2.21(1)-release`
- OpenJDK runtime `21.0.10`
- NVIDIA driver `591.86`
- 開發時使用的 GPU: `NVIDIA GeForce RTX 4080 SUPER`

## 補充說明

- Pyserini / BM25 相關流程需要 Java
- GPU 訓練與推論需要相容的 NVIDIA driver 與 CUDA 執行環境
- 若只跑 CPU 路徑，部分流程仍可執行，但會明顯較慢
