# ModernBERT-base：BM25 top-100 每 epoch 重抽 15 個負樣本

新增 **SAILER-style In-batch Negatives＋Label Mask＋GradCache** 模式，使用 4 個不同 query 的共同 4×64 loss，適用 RTX 4080 SUPER 16GB。完整規格、測試與操作見 [SAILER_GRADCACHE.md](SAILER_GRADCACHE.md)。正式啟動：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune_sailer.py" --no-debug
```

目前 SAILER-style 模式為 v3，將全部 pairs 打亂後，以交換方式修復同批 query 重複；詳見[分組與大量統計驗證](GROUPING.md)。以下第 2～7 節的訓練指令與超參數描述 pairwise BM25 模式；SAILER-style 的 debug、續訓及評估請使用[專用指南](SAILER_GRADCACHE.md)。

本專案依照 `modernBert-fp/` 的微調、推論、排名與最佳模型搜尋方式重構。
使用原始 `answerdotai/ModernBERT-base`，**沒有繼續預訓練**；不修改、不 import 原本成功的 `modernBert-fp`。
設定來自所選的 JSON config，不讀根目錄的 2026 `.env`。

| 模式 | 設定檔 | 預設正式輸出（位於 `models/`） |
| --- | --- | --- |
| Pairwise BM25，1 query／16 candidates | [coliee2025.json](configs/coliee2025.json) | `modernBERT_bm25_top100_epoch_resample15_2025/` |
| SAILER-style v3，4 queries／64 candidates | [coliee2025_sailer_gradcache.json](configs/coliee2025_sailer_gradcache.json) | `modernBERT_bm25_top100_epoch_resample15_sailer_gc_shuffle_2025/` |
| 舊 SAILER-style v2，供既有 run 續訓 | [coliee2025_sailer_gradcache_legacy.json](configs/coliee2025_sailer_gradcache_legacy.json) | `modernBERT_bm25_top100_epoch_resample15_sailer_gc_2025/` |

SAILER-style 的分組實作 v3 與 JSON schema 的 `version=2` 是不同識別。不同模式的 checkpoint、optimizer 與設定不能混用。
已完成的測試範圍見 [VERIFICATION.md](VERIFICATION.md) 與 [SAILER_GRADCACHE_VERIFICATION.md](SAILER_GRADCACHE_VERIFICATION.md)。歷史 GPU 測試及統計摘要不代表目前 run 的正式 F1。

## 1. 檔案與流程

| 檔案 | 功能／與 modernBert-fp 的對應 |
| --- | --- |
| [fine_tune/fine_tune.py](fine_tune/fine_tune.py) | 同名訓練入口：設定 Trainer、每 epoch 抽樣、驗證、早停、儲存 |
| [fine_tune/fine_tune_sailer.py](fine_tune/fine_tune_sailer.py) | SAILER-style 入口，指定專用 config 後共用 `run_training` |
| [fine_tune/sailer.py](fine_tune/sailer.py) | Logical batch、Label Mask、輸入去重與 GradCache Trainer |
| [fine_tune/gradcache.py](fine_tune/gradcache.py) | Masked InfoNCE 與分段 gradient cache／RNG 重播 |
| [fine_tune/modernbert_contrastive_model.py](fine_tune/modernbert_contrastive_model.py) | 唯一模型定義，訓練／推論共用；CLS、projector、InfoNCE、完整權重載入 |
| [fine_tune/data.py](fine_tune/data.py) | 從舊訓練主程式分離 Dataset、collator；新增 BM25 pool 與 epoch 抽樣 |
| [fine_tune/batch_grouping.py](fine_tune/batch_grouping.py) | v3全pairs shuffle、同批Query衝突修復；保留legacy規則供舊run續訓 |
| [validate_grouping.py](validate_grouping.py) | 真實pairs的大量CPU統計、極端資料與完整negative／Mask資料流驗證 |
| [benchmark_sailer.py](benchmark_sailer.py) | 真實 ModernBERT-base 更新速度與滿長度顯存微基準 |
| [inference.py](inference.py) | 同名推論入口，將 processed 文件轉成向量 |
| [similarity_and_rank.py](similarity_and_rank.py) | 同名排名入口；共用 Top-5、micro F1 與訓練內評估 |
| [find_best_model.py](find_best_model.py) | 同名最佳 checkpoint 搜尋，只看本次 run 直接子目錄 |
| [prepare_data.py](prepare_data.py) | 檢查既有 2025 輸入、產生 pools 與資料 manifest |
| [config.py](config.py) | JSON 設定、debug 設定及基本 JSON／hash 工具 |
| [tests/](tests/README.md) | 單元測試、CPU 資料流與明確 opt-in 的完整模型 GPU 測試 |

主流程：
讀設定 → 檢查資料 → 初始化模型 → epoch 0 評估 → Trainer 訓練 →
每 epoch 重抽／評估／存 checkpoint → 早停並載入最佳訓練 checkpoint。

日常少量訓練統一使用訓練入口的 `--debug`；獨立評估使用 `similarity_and_rank.py`，歷史檢查使用 `find_best_model.py --history`。兩種新模式均固定使用 BM25 候選池，沒有模型分數動態加權或年份 scope 分支。

## 2. 環境、資料檢查與一般測試

以下指令在 repository 根目錄執行：

```bash
cd /home/lab/FLNLP-COLIEE2026
conda activate FLNLP-COLIEE2026-WSL

python "Legal Case Retrieval/modernBert-bm25/prepare_data.py" --check-only
python -m unittest discover -s "Legal Case Retrieval/modernBert-bm25/tests" -v
```

換電腦時更換 `cd` 路徑。`--check-only` 不寫檔。去掉該參數可先產生 `prepared/`；
正式與 debug 訓練啟動時也會自動準備，不必每次手動做一次。

預期資料：

- Train / valid queries：1341 / 336，兩者無交集。
- 去重後 train / valid pairs：5421 / 1421；原始 links 的 33 / 4 筆重複會去除。
- Query、positive、negative 與檢索 candidates 都讀 `2025/processed/`；candidate 共 7349 篇。
- 原始 BM25 rank ≤ 100 排除 self 與該 query 所有已知 positives 後，train pools 為 83～99、valid pools 為 87～99。
- 不讀舊版預先固定 15 negatives 的 train JSON，不改原有 train / valid 切分。

此步驟只讀現有 processed、labels、qid 與 BM25 TREC，**不會重建原文前處理、BM25 index 或排名**。
若輸入缺檔、pool 不足 15、split 重疊或數量不符，直接報錯；不會偷偷補資料。
重複準備時會核對既有產物，避免改 seed 或資料後覆寫同一 run。

模型 revision 固定為 `8949b909ec900327062f0ebf497f51aef5e6f0c8`，預設 `local_files_only=true`。
換機器須先準備該官方 snapshot；不可改指向繼續預訓練的 FP checkpoint。

## 3. DEBUG：同一訓練流程、少量真實資料

可直接修改訓練入口的 `DEBUG = True / False`，或用指令覆寫：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune.py" --debug
```

DEBUG 使用完整 ModernBERT-base，100 個真實 train pairs、20 個 valid queries、1000 candidates，
包含選定 valid queries 的全部 positives 與 query 文件。執行 epoch 0、再連續訓練 2 epochs，
每輪 25 個 HF optimizer-step attempts，兩輪合計 50 個。每輪的 train negatives 仍從完整 BM25 pool 抽取。
參考 `modernBert-fp` 的縮小真實 query／candidate 流程；不沿用其目前每 split 5 queries、20 candidates 的較小設定。

模型、4096 長度上限、15 negatives、FP16 AMP、Flash Attention 2、gradient checkpointing、
optimizer、LR、batch size、gradient accumulation、workers、F1／早停規則都沿用正式流程。
只縮小資料量與 epoch 上限，並改成每 step logging。
輸出隔離在正式目錄的 `_debug/<UTC-run-id>/`，不會被當成正式最佳模型。
新 run 會在 `run_config.json` 保存 debug 資料量。舊 debug 的續訓保留原來的資料範圍，
不會自動從 4 queries／256 candidates 擴大；要使用新版資料量請直接另開一次 `--debug`。

想多跑一些真實 batches 可調整：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune.py" --debug --debug-train-pairs 200
```

一般 debug **不自動中斷或另啟程序**。它的目的只是試跑同一套程式；
少量 F1 不作為論文績效，也不要求幾個 steps 的 loss 單調下降。
發生錯誤會直接顯示完整 traceback；訓練／驗證內的錯誤也保存到該 run 的 `failure.json`。

續訓及 CUDA 溢位恢復是選用測試，不必每次 debug 都執行：

```bash
BM25_RUN_GPU_TESTS=1 python -m unittest discover -s "Legal Case Retrieval/modernBert-bm25/tests" -p "test_gpu_amp.py" -v
BM25_RUN_INTEGRATION=1 python -m unittest discover -s "Legal Case Retrieval/modernBert-bm25/tests" -p "test_integration.py" -v
```

整合測試使用真正 ModernBERT-base、9 pairs，驗證最後不足 4 筆的累積組、
epoch 1 中斷後續訓 epoch 2、scheduler／scaler／patience 延續及完整 checkpoint 重載。
它將獨立測試產物放在 `_debug/`，不是另一套使用者 smoke 模式。

## 4. 正式訓練

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune.py" --no-debug
```

正式使用全部 train pairs、336 valid queries 與 7349 candidates，最多 20 epochs。
每 epoch 5421 batches，gradient accumulation 4，預期 1356 HF optimizer-step attempts；
20 epochs 上限 27120。AMP 溢位時可跳過實際更新，因此 HF global_step 不一定等於成功更新數；
`amp/updates_applied` 與 `amp/skipped_updates` 另行記錄。

核心設定：

| 設定 | 值 |
| --- | --- |
| 表示／loss | CLS → Linear → ReLU → Linear → L2；cosine InfoNCE |
| Train / eval / retrieval batch | 1 / 1 / 1 |
| Max length / negatives / accumulation | 4096 / 15 / 4 |
| Encoder、projector LR / temperature LR | 5e-6 / 5e-4 |
| Initial temperature / forward clamp | 0.55555 / [0.001, 2.0] |
| 精度與 attention | FP32 weights + FP16 AMP、Flash Attention 2 |
| Optimizer / weight decay | AdamW fused 三組參數 / 0 |
| Scheduler / warmup | linear / 0.1，各 optimizer group 都套用 |
| Trainer seed / sampler seed | 42（初始化前設定）/ 289 |
| Epochs / patience / logging | 20 / 5 / 每 50 steps |
| DataLoader | 8 workers，persistent workers 關閉 |

使用原生 Trainer／Accelerate AMP 更新，不修改梯度、不自行寫 backward。
單次 AMP overflow 由 GradScaler 處理並降低 scale；連續 8 次 overflow、非有限 forward loss、
非有限權重會報錯，不能把大量跳過更新當成訓練正常。
checkpoint 保存 encoder、projector、temperature、模型設定、tokenizer，以及原生 optimizer／scheduler／scaler／RNG／Trainer 狀態。

最多保留 20 個完整 checkpoints，約需 34 GiB，另預留寫入中、測試與其他紀錄空間。
不要在同一張 GPU 同時跑正式訓練與另一個推論／debug。

### 每 epoch 如何抽樣

1. 讀 BM25 原始 TREC 的 rank ≤ 100，去重。
2. 排除 query 自身及該 query 所有已知 positives，不從第 101 名後補位。
3. 每個 `(query, positive)` 各自均勻、不放回抽 15 個；不同 positives 不是從「上一筆剩下的」繼續抽。
4. 使用 SHA-256(`[289, epoch_index, query_id, positive_id]`) 派生種子；同 epoch 可重現，換 epoch 重抽。
5. 跨 pair、跨 epoch 允許重疊；沒有 BM25 分數加權或模型分數加權。

`sampling/epoch_NN_train.json` 是計畫送入的資料；旁邊 audit 記錄 hash 與 pool 覆蓋率。
`epoch_NN_received.json` 核對模型在該輪實際收到的全部 pairs／negative IDs 是否與計畫一致。
若 worker 仍拿舊 negatives、漏筆或多筆，程式會報錯。

### Valid F1 與早停 A

- 訓練前 epoch 0：原始 encoder 加本次初始化的 projector；保存 sampled valid loss、full-valid F1、Top-5。
- Epoch 0 僅供診斷，不建立 best metric、不參與最佳訓練模型選擇。
- Epoch 1：建立第一個 best metric，patience=0。
- 之後 F1 嚴格增加才改善；相同／降低都加 1，連續 5 輪未改善停止。
- 結束後載入 valid F1 最高的訓練 checkpoint；同分選先出現者。

F1 是 **全部 336 個 valid queries** 對 **完整 7349 candidates** 的 Top-5 micro F1，
沒有合併 train queries 的分數。Candidates 可能同時出現在訓練資料，這不等於把 train F1 算進 valid。
排除 query 自身、同分依文件 ID 排序；labels 去重，缺 prediction 的 query 以空答案計入。

固定 sampled valid loss 只供觀察，不控制早停。它有 1421 個 pairs，每 pair 要編碼 query、positive、15 negatives，
因此會重複編碼文件，不能與 corpus encoding 的 doc/s 直接比較。
Query 與 candidate 都是同一份 processed，故正式檢索可直接重用 query 的 candidate 向量。

## 5. 續訓與重跑

只能明確指定本次 run 的最新 epoch-boundary checkpoint，不自動找模型續訓：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune.py" --no-debug \
  --resume-from-checkpoint "models/modernBERT_bm25_top100_epoch_resample15_2025/checkpoint-1356"
```

須替換成實際存在的 checkpoint。恢復模型、optimizer、scheduler、scaler、RNG、best metric 與 patience；
不重算 epoch 0，負樣本從完成 epoch 的下一輪對齊。資料 hash／config 改變會拒絕續訓。
Debug 續訓使用 `--debug --resume-from-checkpoint "實際debug run/checkpoint-N"`。

本次重構使用 config version 2 與新的實作識別，**不支援從重構前的 checkpoint 接續訓練**。
舊 0911 模型仍可嚴格載入做獨立評估；不能以舊 optimizer 狀態假裝延續本次實驗。

若尚未存出第一個 checkpoint 就中斷，無法恢復只在記憶體的更新，應重新開始。
已有 run 時不覆寫；可加 `--output-dir "models/modernBERT_bm25_top100_epoch_resample15_2025_run2"`。

如果確定不要預設目錄內任何產物，先停止相關訓練／推論／TensorBoard，再自行核對並執行：

```bash
ls -ld -- "/home/lab/FLNLP-COLIEE2026/models/modernBERT_bm25_top100_epoch_resample15_2025"
rm -rI -- "/home/lab/FLNLP-COLIEE2026/models/modernBERT_bm25_top100_epoch_resample15_2025"
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune.py" --no-debug
```

`rm` 需輸入 y 確認，會永久刪除此實驗的模型、TB、prepared、取樣、rankings 與 debug 產物，不能直接復原。
不包含原始 2025 資料、程式碼、官方模型快取或其他模型。程式不會自動執行這項刪除。

## 6. 最佳模型、獨立評估與 TensorBoard

```bash
python "Legal Case Retrieval/modernBert-bm25/find_best_model.py" --history
python -u "Legal Case Retrieval/modernBert-bm25/similarity_and_rank.py" \
  --run-dir "models/modernBERT_bm25_top100_epoch_resample15_2025" --checkpoint best
tensorboard --logdir "models/modernBERT_bm25_top100_epoch_resample15_2025/tb" --port 6006
```

瀏覽器開啟 http://localhost:6006 。關注 `train/loss`、`eval/loss`、
`eval/global_f1`、precision／recall、`eval/improvement_vs_epoch0`、
`train/model/temperature`、`train/optimizer/lr_*`、`train/amp/*`、`train/sampling/*`。
最佳模型搜尋只看選定 run 的直接子目錄，不搜尋 `_debug/`，且排除 epoch 0。

`similarity_and_rank.py` 預設完成「載入模型 → 編碼 → 排名 → F1」，
與訓練共用相同函式及 FP16 autocast／float32 相似度。
輸出 `standalone/checkpoint-N.tsv`、`.json`、`.manifest.json`。
Debug 檢查時將 `--run-dir` 指向實際 debug run；不能把 debug checkpoint 搭配正式 config。

若希望像原專案分開「推論編碼」與「排名」，可先存 embeddings：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/inference.py" \
  --run-dir "models/modernBERT_bm25_top100_epoch_resample15_2025" --checkpoint best \
  --destination "models/modernBERT_bm25_top100_epoch_resample15_2025/standalone/best"
python "Legal Case Retrieval/modernBert-bm25/similarity_and_rank.py" \
  --run-dir "models/modernBERT_bm25_top100_epoch_resample15_2025" --checkpoint best \
  --embeddings "models/modernBERT_bm25_top100_epoch_resample15_2025/standalone/best.embeddings.pt"
```

Embeddings 附有 checkpoint 權重 hash、資料 hash 與 config hash；混用快取時報錯。
正式評估以所有 valid queries 計分，不能只取有 embeddings 的 queries 來提高分數。

歷史 0911 checkpoint 重評（須實際存在）：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/similarity_and_rank.py" \
  --checkpoint "models/modernBERT_contrastive_0911/checkpoint-5424" --legacy \
  --destination "models/modernBERT_bm25_top100_epoch_resample15_2025/baseline_0911/valid"
```

嚴格載入 encoder、projector、temperature；缺權重不補隨機層，不啟動訓練。

## 7. 向教授現場說明的順序

1. 開 config：確認是 2025、原始 base、processed、完整 candidates，沒有 FP／scope。
2. 跑 `prepare_data.py --check-only`：核對 split、pairs、文件及 pool。
3. 看 `data.py::build_pools / sample_epoch`：原始 top-100、排除 positives／self、每 pair 每 epoch 抽 15。
4. 看模型的 `encode / forward`：17 篇合併編碼，positive 在第 0 類，cosine InfoNCE。
5. 跑 `--debug`：使用與正式相同的 `run_training`，觀察 loss 與 checkpoint。
6. 比較兩輪的 sampling JSON、audit、received 檔；再用 `find_best_model.py --history` 與 TensorBoard 解釋結果。

Loss 下降代表 sampled 對比目標有被最佳化，不保證 full-corpus F1 必然提升。
本次採現存 2025 快照且修正 self-negative；與舊固定 BM25 結果之差異，不能全歸因於 epoch 重抽。

## 8. 診斷報告與 Git 管理

`validate_grouping.py` 的新報告預設寫至 `reports/modernbert-bm25/grouping/<UTC-run-id>/`；
`benchmark_sailer.py` 則寫至 `reports/modernbert-bm25/benchmark/<UTC-run-id>/`。
兩者均在啟動時列印實際路徑，可分別以 `--output-dir`／`--output` 指定目的地，預設不覆寫歷史研究證據。
Benchmark 會使用 GPU；資料分組統計使用 CPU。完整指令與執行條件見 [GROUPING.md](GROUPING.md) 及 [SAILER_GRADCACHE.md](SAILER_GRADCACHE.md)。

Git 保留 `.env`、三份 JSON 設定、程式、測試、技術說明、驗證摘要及必要訓練紀錄。
2026-11-12 已將三個 run 的主要 TensorBoard、metrics history、summary、實際設定及取樣 audit 另存至 [輕量訓練快照](../../reports/modernbert-bm25/training_records/README.md)，可隨 Git 備份。
大量逐輪統計明細、重複 grouping／benchmark 輸出、原始測試 log、個人草稿與快取被忽略；`models/`、資料集等原有規則保留。
快照不含模型權重及 optimizer，不能替代推論／續訓所需的完整 checkpoint；往後訓練也不會自動更新這次保存的快照。
忽略規則只影響版本控制，不會刪除本機檔案，也不會替代續訓所需的 checkpoint。
研究紀錄的日期、適用版本及已知限制見 [research_notes 導覽](../../research_notes/coliee2025_review/README.md)。
