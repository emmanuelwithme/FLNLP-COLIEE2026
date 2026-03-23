# ModernBERT 對比式訓練與動態負樣本抽樣使用說明（年份可切換版）

本文件整合舊版 `ADAPTIVE_SAMPLING_README.md`、`EVALUATION_SUMMARY.md` 等內容，同時補上目前程式碼重構後的實際流程。現在以 `Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py` 與共享模組 `Legal Case Retrieval/lcr/` 為準。

---

## 1. 目的與核心概念

* 訓練 ModernBERT 對比式檢索模型，正樣本來自官方標註（`task1_train_labels_${COLIEE_TASK1_YEAR}_train.json`），負樣本使用「模型當前相似度分數」動態抽樣。
* 每個 epoch：
  * 開頭：重新計算 query ↔ candidate 的相似度，產生新的負樣本 JSON。
  * 結尾：跑完整個語料（train/valid）的檢索評估，產生 TREC 檔並寫入 TensorBoard。
* 所有訓練過程需要從專案根目錄執行（`~/FLNLP-COLIEE2026`），必要的 `sys.path` 會由腳本自動加入。

---

## 2. 執行訓練

```bash
cd ~/FLNLP-COLIEE2026
python "Legal Case Retrieval/modernBert-fp/fine_tune/fine_tune.py"
```

主要依賴：

* `lcr.device.get_device()`：列印並回傳訓練使用的裝置（GPU / CPU）。`get_device()` 也會在 `similarity_and_rank.py` 等腳本中使用，確保整個專案的裝置偵測一致。
* `lcr.data.EmbeddingsData`、`lcr.retrieval.generate_similarity_artifacts`、`lcr.metrics.*` 等工具模組負責資料載入、相似度計算與指標運算。

### 必要路徑

* `.env` 切換：
  * `COLIEE_TASK1_YEAR=2025` 或 `2026`
  * `COLIEE_TASK1_ROOT=./coliee_dataset/task1`
  * 實際資料目錄 = `./coliee_dataset/task1/<year>`
* 訓練資料（由 `.env` 自動解析）：
  * Candidate：`${TASK1_DIR}/processed`
  * Query：`${TASK1_DIR}/processed` 或 `${TASK1_DIR}/processed_new`（可在 `fine_tune.py` 中調整 `doc_folder` / `query_dataset_path`）
  * 標註：`${TASK1_DIR}/task1_train_labels_${COLIEE_TASK1_YEAR}*.json`
* 訓練輸出：
  * 模型 checkpoint：`./modernBERT_contrastive_adaptive_fp_fp16..._${COLIEE_TASK1_YEAR}`
  * TensorBoard：`./modernBERT_contrastive_adaptive_fp_fp16..._${COLIEE_TASK1_YEAR}/tb`
  * 動態負樣本 / 檢索評估：`${TASK1_DIR}/lht_process/modernBert/finetune_data`

若要在「相似度計算前」就過濾掉每個 query 的未來案例，可先用 `Legal Case Retrieval/pre-process/build_query_candidate_scope.py` 生成 scope JSON，再設定：

`export LCR_QUERY_CANDIDATE_SCOPE_JSON=/path/to/query_candidate_scope.json`

`lcr.retrieval.generate_similarity_artifacts()` 與 `lcr.similarity.compute_similarity_and_save()` 會自動套用此 scope。

---

## 3. 設定與可調參數

* `QUICK_TEST`（`fine_tune.py` 頂部）：開啟後，每個流程（訓練、抽樣、評估）都會取較小的資料子集，以快速檢查流程。關閉 (`False`) 即使用完整資料。
* `QT_CAND_K` / `QT_QUERY_K`：在 QUICK_TEST 模式下限制候選 / query 的數量。
* `update_frequency`（`AdaptiveNegativeSamplingTrainer` 初始化時的參數）：控制每隔幾個 epoch 重新抽樣負樣本（預設 1 = 每個 epoch 都更新）。
* `sampling_temperature`：負樣本抽樣的 softmax 溫度（見 §5）。
* `TrainingArguments`：
  * `output_dir` / `logging_dir` 會自動在 QUICK_TEST 時加上 `_test` 後綴（`base_output_dir` 及 `finetune_data_dir` 也同步加後綴），以免覆蓋正式訓練結果。
  * `metric_for_best_model="eval_global_f1"`，並啟用 `load_best_model_at_end=True`、`EarlyStoppingCallback(patience=5)`，與重構前設計一致。

---

## 4. 主要程式碼節點

**裝置設定與資料匯入**
* `get_device()`：`fine_tune.py` / `fine_tune_noprojector.py` / `similarity_and_rank.py` / `inference.py` 皆統一使用，執行時會印出 GPU / CPU 型號。
* 讀取訓練資料集使用 `ContrastiveDataset`（在 `fine_tune.py` 相同目錄），其 `update_data()` 方法會在抽樣後被呼叫。

**評估流程**
* `evaluate_model_retrieval()`：
  * 透過 `lcr.retrieval.generate_similarity_artifacts()` 生成 query × candidate 相似度，並產出 TREC (`similarity_scores_{tag}.tsv`)。
  * `trec_file_to_dict()` + `my_classification_report()` 計算 F1 / Precision / Recall。
  * 結果寫入 TensorBoard（由 `TensorBoardExtras` Callback）及 Trainer 的 metrics（`eval_global_f1` 用於最佳模型與早停）。

**動態負樣本抽樣**
* `generate_similarity_artifacts()`（`lcr/retrieval.py`）：
  * 以批次張量運算（`score_matrix = query @ candidate.T`）計算相似度，可在 GPU 上平行處理整個 query 集合。
  * 透過 `torch.sort(..., stable=True)` 將分數寫成 TREC 格式。
  * 回傳 `SimilarityArtifacts(scores=..., trec_path=...)`，其中 `scores` 是 `{qid: {docid: score}}`，供抽樣使用。
* `generate_adaptive_negative_samples()`（`fine_tune.py`）：
  * 根據分數以 `softmax(score / temperature)` 生成機率。
  * 固定每個 (query, positive) 取 `max_negatives` 個負樣本，使用 with-replacement 對應候選不足的情況。
* `AdaptiveNegativeSamplingCallback`：
  * `on_epoch_begin()` 根據 `update_frequency` 觸發抽樣流程，並寫出 `adaptive_negative_epoch{E}_train.json`。

**TensorBoard 與 Logging**
* 兩種 Callback：
  * `TensorBoardExtras`：寫入訓練 loss 與溫度 (`log_temperature`)。
  * `EvaluationCallback`：於每個 eval epoch 將 train/valid 的 top‑5 指標寫入 `retrieval/*` 指標群。

---

## 5. 檔案輸出與命名

重構後的檔案命名延續舊版風格，細節如下：

| 節點 | 主要檔案 | 說明 |
| ---- | -------- | ---- |
| Epoch 開始前 | `similarity_scores_epoch{E}.tsv` | 完整 corpus 的排名結果，供抽樣負樣本使用。 |
| Epoch 開始前 | `adaptive_negative_epoch{E}_train.json` | 每筆訓練樣本包含 `query_id`, `positive_id`, `negative_ids`（固定長度）。 |
| Epoch 評估 | `similarity_scores_epoch{E}_eval_train.tsv` | 訓練查詢的檢索結果，供指標計算與除錯。 |
| Epoch 評估 | `similarity_scores_epoch{E}_eval_valid.tsv` | 驗證查詢的檢索結果，產生 `eval_global_f1` 等指標。 |
| TensorBoard | `./modernBERT_contrastive_adaptive..._${COLIEE_TASK1_YEAR}/tb` | 可用 `tensorboard --logdir ...` 查看訓練與檢索指標。 |

Quick Test 模式下會在 `output_dir`, `finetune_data_dir` 追加 `_test` 後綴，避免污染正式結果。

---

## 6. 負樣本抽樣的深入說明

1. **來源**：透過 `generate_similarity_artifacts()` 計算出每個 query 對候選語料的 dot/cos 相似度。ModernBERT 模型本身在 forward 時會輸出 L2-normalized 的向量，因此 dot product ≈ cosine。
2. **排除**：抽樣時會跳過正樣本與與查詢相同 ID 的候選（`doc_id not in pos_set and doc_id != qid`）。
3. **機率**：score → `torch.tensor(scores) / temperature` → softmax → numpy array。`temperature < 1` 會更集中在高分候選（更難的負樣本）。
4. **with-replacement**：若候選負樣本數量不夠，允許抽中重複 ID；程式會自動處理，確保 JSON 中每筆樣本擁有相同長度的 negative 清單。
5. **儲存**：所有抽樣資料寫入 `JSON`，檔案名稱包含 epoch 編號，方便追蹤。

---

## 7. 檢索評估與指標

* `fine_tune.py` 中的 `evaluate_model_retrieval()` 每個 epoch 評估 train / valid：
  * 生成 TREC 檔 → `trec_file_to_dict()` → `my_classification_report()` → F1 / Precision / Recall。
  * `results['valid']['f1']` 會被回傳給 Trainer 當作 `eval_global_f1`。
* `lcr/metrics.py` 內建的 random baseline 等輔助函式可用於 sanity check。
* 若需要額外報表，可使用 `Legal Case Retrieval/utils/eval.py`，該腳本目前也引用 `lcr.metrics` 與 `lcr.results`，會把結果寫入 `results/experiment_results.csv`。

---

## 8. Quick Test（流程縮小）與最佳化建議

* `QUICK_TEST = True` 時會：
  * 限縮候選檔案與 query 清單（透過 `_QT_CANDIDATE_FILES`、`_QT_TRAIN_QIDS`、`_QT_VALID_QIDS`）。
  * 評估時只跑部分 `eval_dataset`，大幅縮短時間。
* 建議：
  * 調整 `batch_size`（目前為 1）可配合 GPU 記憶體情況調整，但需注意長文本會造成大量記憶體使用。
  * 大型資料時建議使用 NVMe SSD 或 RAM Disk，以減少讀寫 TREC/JSON 的 I/O 成本。
  * 若長時間訓練建議監控 GPU 記憶體與溫度。`pynvml` 已於程式中啟動，可根據 nvml 回傳資訊自訂監控。

---

## 9. 相關腳本與模組

* `Legal Case Retrieval/modernBert/inference.py`：以 `get_device()` + `process_directory_to_embeddings()` 產生 candidate / query embeddings (`.pkl`)。
  * 目前固定同時輸出 `processed` 與 `processed_new` 兩份 embeddings。
* `Legal Case Retrieval/modernBert/similarity_and_rank.py`、`Legal Case Retrieval/SAILER/similarity_and_rank.py`：產生 dot/cos TREC 檔。
  * 預設 query / candidate 都使用 `processed`。
  * 若要重現 THUIR 論文習慣，可設定 `LCR_QUERY_EMBED_SOURCE=processed_new`。
* `Legal Case Retrieval/lcr/`：
  * `data.py`：`EmbeddingsData`、`load_query_ids()` 等基礎資料結構。
  * `embeddings.py`：批次 encode 文字、整合儲存結果。
  * `similarity.py`：批量計算相似度（vectorized），並以穩定排序輸出 TREC。
  * `retrieval.py`：高階 API `generate_similarity_artifacts()`，同時產生 TREC 檔與 `{qid: score_dict}`。
  * `metrics.py` / `results.py`：評估指標與實驗結果記錄。
  * `device.py`：裝置偵測與列印。

---

## 10. 常見問題

| 問題 | 排除方式 |
| ---- | -------- |
| `ModuleNotFoundError: lcr` | 確保從專案根執行；所有腳本在開頭會自動將 `Legal Case Retrieval/` 加入 `sys.path`。 |
| 想切回 THUIR 的 query 設定 | similarity 腳本預設 query / candidate 都是 `processed`；若要改回 THUIR 論文常用設定，可先 `export LCR_QUERY_EMBED_SOURCE=processed_new` 後再執行。 |
| dot 與 cos 指標不一致 | 先確認 query / candidate 實際載入的是哪份 embeddings。腳本現在會印出 `processed`、`processed_new` 與最終選擇的路徑。 |
| Random Guess baseline 不為 0 | 這是修復型別不一致的結果（過去因 `int` vs `str`比較，顯示 0，而非真實隨機表現）。 |
| 評估非常慢 | 使用 QUICK_TEST，或在 `${TASK1_DIR}/lht_process/modernBert/finetune_data` 清理舊檔只保留必要檔案，避免 I/O 混亂。 |

---

## 11. 參考流程（時間軸示意）

1. **Epoch 0 前**  
   `similarity_scores_epoch0.tsv` + `adaptive_negative_epoch0_train.json`
2. **Epoch 0 Eval**  
   `similarity_scores_epoch1_eval_train.tsv`、`similarity_scores_epoch1_eval_valid.tsv`
3. **Epoch 1 前**  
   `similarity_scores_epoch1.tsv` + `adaptive_negative_epoch1_train.json`
4. **Epoch 1 Eval**  
   `similarity_scores_epoch2_eval_train.tsv`、`similarity_scores_epoch2_eval_valid.tsv`
5. **Epoch 2 前**  
   `similarity_scores_epoch2.tsv` + `adaptive_negative_epoch2_train.json`

若早停於第 2 個 epoch，則不會產生 epoch3 的訓練檔案。

---

## 12. 建議修改點

* 調整 `generate_similarity_artifacts()` 的 `batch_size` / `max_length` 或改寫 `encode_batch()`，可對長文本 encoding 進一步最佳化。
* 若要增加新的評估指標，可在 `evaluate_model_retrieval()` 內直接延伸 `results[split]`。
* 若需要整合其他 embeddings（例如外部模型），將 `.pkl` 讀寫改成相同格式即可（`EmbeddingsData(ids, torch.Tensor)`）。

更多細節請直接查閱程式碼，所有關鍵函數與參數都已有註解。建議從 `generate_similarity_artifacts()`、`generate_adaptive_negative_samples()` 和 `AdaptiveNegativeSamplingCallback` 開始逐步閱讀。祝順利！ 🎯
