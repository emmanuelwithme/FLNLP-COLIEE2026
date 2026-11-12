# ModernBERT：SAILER-style in-batch negatives＋Label Mask＋GradCache

這個模式在現有 BM25 專案中新增共同分母的對比 loss，不覆寫先前六輪 BM25 實驗，也不修改 `modernBert-fp`。使用原始 ModernBERT-base、4096 長度上限、processed query／candidates，保留現有 CLS→兩層 MLP→L2 與可學習 temperature。它採用 SAILER 的跨 query 候選比較方式；不聲稱完整復刻 SAILER 的法律結構預訓練、raw-CLS/dot loss 或論文績效。

目前為 `modernBert-sailer-gradcache-v3`，使用 `shuffle_repair_v1` 分組：全部 pairs 先 shuffle，再交換修復同批 query 重複。分組規格、可行性說明及大量統計見 [GROUPING.md](GROUPING.md)。新模式與舊 v2 使用不同輸出目錄；v2 的續訓需明確指定 legacy config。

沿用2026-11-03的v2效率設定：chunk=2、**關閉額外的 gradient checkpointing**、相同輸入去重及一次批次tokenization。v2更新核心微基準約5.99→4.76秒，滿4096峰值allocated約6.68GiB；這是歷史微基準，不代表v3整體epoch耗時，完整結果見[效率測試紀錄](../../research_notes/coliee2025_review/sailer_efficiency_2026-11-03/README.md)。

## 一個指令啟動正式訓練

在 repository 根目錄並啟用既有 Conda 環境後：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune_sailer.py" --no-debug
```

完整環境準備：

```bash
cd /home/lab/FLNLP-COLIEE2026
conda activate FLNLP-COLIEE2026-WSL
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune_sailer.py" --no-debug
```

設定：`configs/coliee2025_sailer_gradcache.json`。
獨立輸出：`models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_shuffle_2025/`。
第一次啟動會自動檢查資料、建立 BM25 pools 與 token fingerprints，再執行 epoch 0 評估與正式訓練。只產生 `prepared/` 不算已開始訓練；若已有 checkpoint／run_config／TB events，程式要求明確 resume 或改新輸出目錄。

## 每個 logical batch 的資料與 loss

依使用者選擇，**每epoch保留全部5421個query-positivepairs**。先按原規則產生每pair的15negatives，再補3筆已存在的pairs、shuffle全部5424筆，切成四人組；同批若有重複Query，以完整pair交換修復。原始Query／Positive／15negatives的對應不變，沒有剩餘pair數優先排序。

本資料每輪補3個不同Query的pairs，補齊樣本也會一起shuffle，不固定塞在最後一批。所有補齊、修復與hash都保存；一般化極端資料在容量受限時可能需從同Query補多筆，仍不得突破可行的每Query出現上限。

每輪 5424 pair instances、1356 logical batches；所有原始 pairs 至少出現一次，只有 3 個額外 pair instances。Debug 也保留相同分組規則，但先跨 query 選少量 pairs，避免前幾個 pairs 都屬同一 query。

每個 query 配 1 positive＋15 BM25 negatives；負樣本仍對每個 pair 每 epoch 從原始 rank≤100 的合法 pool 均勻、不放回抽取，不從 rank101 之後補位。抽樣 seed 沿用 289 的穩定 SHA-256 規則。

```text
4 queries
64 passages = [q0的positive與15negatives, q1的16篇, q2的16篇, q3的16篇]
4×64 scores = normalized_query @ normalized_passage.T / temperature
targets = [0, 16, 32, 48]
loss = mean CrossEntropy(masked_scores, targets)
```

在 mask 前，每個 query 有 63 個負樣本，其中 48 篇來自其他三個 query。Mask 後合法負樣本數可能少於 63，不會為了湊足數目放回已遮罩文件。

DataLoader 的 `batch_size=1` 表示每次拿一個已打包的 logical batch，**裡面是 4 個不同 queries**。`gradient_accumulation_steps=1`；這與舊版 4 次互不相連的單 query loss 累積不同。最多 20 epochs／27120 logical update attempts，LR／warmup／FP16／Flash Attention 2 等沿用既有設定。

## Label Mask 的規則與範圍

Preflight 為完整候選 corpus 建立 stripped-text hash、實際 tokenizer 截斷至 4096 後的 token hash、有效性及所有 known positives。每個 query 獨立建立 mask：

1. 保留本列指定 positive 的唯一 target column。
2. 遮罩該 query 的其他全部 known positive IDs，以及與任一 positive 的模型輸入完全相同的 aliases。
3. 遮罩 query 自身 ID，以及與 query 模型輸入相同的 aliases（指定合法 positive 除外）。
4. 遮罩不在 corpus manifest 的 ID、空文字候選。
5. 相同 ID／相同截斷 token 輸入只保留一個合法代表；target 優先，即使 alias 出現在更早欄位。

空 query／gold 在 preflight 直接報錯，不會偷偷刪 gold 或改 valid 分母。文件簡短不自動視為非法，沒有未告知的字數門檻／年份 scope／新語言篩選。

Manifest 保存 train／valid 的原有 labels，但訓練每列只查該 train query 的 known positives；train／valid query IDs 無交集，不利用 valid query 的 relevance 作訓練目標。

這可避免已標註 positives／相同輸入 aliases 被當成 negatives；**未標註但實際相關、近重複而非完全相同的文件仍無法自動辨識**。不要把此 mask 描述成保證排除所有 false negatives。

## GradCache 與 16GB 記憶體

一個 logical batch 共有 68 個輸入位置（4 queries＋64 passages）。先按4096截斷tokenhash去重，相同輸入在該批只編碼一次；index mapping還原完整68位置與4×64loss，autograd把多個位置的梯度加總回共用向量。跨optimizer更新不沿用舊向量。

此去重只在encoder／projector dropout皆為0時啟用，否則直接報錯，避免把原本應獨立的隨機dropout表示合併。全部unique文字先一次tokenize，再依chunk分段padding，減少重複讀檔／tokenization。

預設 `gradcache_chunk_size=2`，每次只保留最多2段文件的encoder計算圖：

1. 用 no-grad 分段 forward，保存小型 embeddings 與每段 Torch CPU／CUDA RNG 狀態。
2. 用完整 4×64 masked InfoNCE 計算 embedding gradients；temperature 的梯度也在這一步計算一次。
3. 重播相同 RNG，對每個小段重新 forward，把 cached embedding gradients 傳回 encoder／projector。
4. 全部重播完成後，由原生 Trainer／Accelerate 完成 unscale、clipping、AdamW step、GradScaler update、scheduler。

Representation loss 僅 AMP scale 一次；replay gradients 已含 scale，不再呼叫第二次 `scaler.scale`。不是 detach passages 後只更新 queries，也不是用普通 gradient accumulation 假裝 in-batch loss。

FP32weights＋FP16autocast、FlashAttention2保留；目前 `gradient_checkpointing=false`。GradCache已限制每次保留的計算圖範圍，關閉另一層checkpointing能省去其反向重算；在RTX4080SUPER16GB上實測比單純放大chunk更有效。

Chunk設定允許1–16、預設2；**越大不一定越快**。關閉checkpointing時本機chunk4的滿長度峰值已約10.89GiB；請先實測再調高，不能因為設定範圍允許就假設一定放得下。改chunk／資料／設定後不能接續同一run，請另開輸出目錄。GradCache仍有一輪額外cacheforward，與舊單query版本的速度不能只由chunk推算。

新模式固定關閉ModernBERT的可選`reference_compile`。本機版本實測啟用時no-gradcache與gradreplay路徑有數值差，整體梯度相對誤差約2.45%；關閉後降至約8.3e-8。若另選啟用gradientcheckpointing，cache的no-grad階段會暫停它、replay前恢復；目前預設則全程不啟用該額外checkpointing。

Transformers 的 `ModernBertConfig.to_dict()` 會省略 `reference_compile`，因此本專案另外在 `contrastive_config.json` 保存／恢復該旗標，獨立推論亦明確沿用新模式政策；避免重載後默默改走不同計算路徑。

新模式預設 `save_total_limit=3`，保留 best 並輪替近期 checkpoints；sampling、metrics、rankings 仍逐輪保存。避免預留 20 份完整 optimizer checkpoint 導致磁碟不足。

## 驗證、debug、續訓與評估

與原本相同：epoch0先跑 fixed sampled valid loss＋完整 valid F1；epoch0只作診斷，epoch1建立best，之後嚴格提升才改善，patience=5。336validqueries對7349candidates的Top5microF1選模型。固定 sampled valid loss仍是舊16-way監控指標，**不是新訓練4×64 masked loss**，數值不能直接互比。

64類均勻logits的loss約`ln(64)=4.1589`，16類則約`ln(16)=2.7726`；新模式trainloss剛開始接近4.1並不表示比舊模式的2.7更差。

一般 debug：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune_sailer.py" --debug
```

預設100trainpairs、20validqueries、1000candidates、2epochs，輸出於新模式目錄的 `_debug/`，與正式共用同一 `run_training`／Trainer／GradCache。錯誤直接顯示 traceback；訓練／評估錯誤也寫入 `failure.json`。

續訓請指定最新完整 epoch-boundary checkpoint，將下列名稱改成實際存在者：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune_sailer.py" --no-debug \
  --resume-from-checkpoint "models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_shuffle_2025/checkpoint-1356"
```

原生optimizer、scheduler、scaler、RNG、bestmetric／patience都會恢復；不重跑epoch0。額外核對label-maskhash、logicalbatch／chunk與分組策略。v3只能接續同一策略的v3checkpoint；不能以舊v2的optimizer接續新分組實驗。

舊 v2 保持原有分組規則。續訓時使用以下指令，將 `checkpoint-N` 換成該 run 最新的完整 epoch checkpoint：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/fine_tune/fine_tune_sailer.py" --no-debug \
  --config "Legal Case Retrieval/modernBert-bm25/configs/coliee2025_sailer_gradcache_legacy.json" \
  --resume-from-checkpoint "models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_2025/checkpoint-N"
```

Legacy 設定刻意不插入新的分組欄位，以維持與既有 run 保存的 config 相符。執行另一個訓練或 GPU benchmark 前，應確認 GPU 可供使用。

檢查與獨立評估：

```bash
python "Legal Case Retrieval/modernBert-bm25/find_best_model.py" \
  --run-dir "models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_shuffle_2025" --history
python -u "Legal Case Retrieval/modernBert-bm25/similarity_and_rank.py" \
  --run-dir "models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_shuffle_2025" --checkpoint best
tensorboard --logdir "models/modernBERT_bm25_top100_epoch_resample15_sailer_gc_shuffle_2025/tb" --port 6006
```

新增TB指標：`train/inbatch/allowed_negatives_mean`、`train/inbatch/masked_columns`、`train/gradcache/peak_allocated_gib`、`train/gradcache/peak_reserved_gib`、`train/gradcache/unique_documents`與`train/gradcache/deduplicated_documents`；另有epochmaskstats。原本的loss、F1、temperature、LR、AMP指標保留。

保存`prepared/label_mask_manifest.json`、`sampling/epoch_NN_train.audit.json`的padding／repair／groupinghash、`epoch_NN_received.json`、`epoch_NN_groups_received.json`、`epoch_NN_label_mask.json`，以及checkpoint的`gradcache_state.json`。除了全部pair內容，也核對模型收到的每個四人組與planned完全一致，允許DataLoader重新排列整批出場順序。

## 測試指令與來源

```bash
python -m unittest discover -s "Legal Case Retrieval/modernBert-bm25/tests" -v
BM25_RUN_SAILER_GPU=1 python -u -m unittest discover \
  -s "Legal Case Retrieval/modernBert-bm25/tests" -p "test_sailer_gpu.py" -v
```

一般測試包含手算／矩陣 loss、masked column 零梯度、target優先、全knownpositive／self／alias／duplicate遮罩、完整pairs與padding、RNG／dropout重播與全部參數的梯度一致性、AMP scale只套一次。數學測試用小型張量驗證鏈式法則；正式GPU驗證使用真正ModernBERT-base。

GPU 測試程式支援目前新分組；歷史 v1／v2 結果見 [SAILER_GRADCACHE_VERIFICATION.md](SAILER_GRADCACHE_VERIFICATION.md)。2026-11-03 的 v3 更新以大量 CPU 真實資料驗證分組、audit 與策略續訓保護；2026-11-06 再確認 68 項 CPU 測試通過，4 項 GPU 測試跳過。歷史 GPU 數值不能標成本次重新執行的結果。

速度及顯存基準可另外執行：

```bash
python -u "Legal Case Retrieval/modernBert-bm25/benchmark_sailer.py" --without-checkpointing
```

此命令會使用真正的 ModernBERT-base 與 GPU；新報告寫入 `reports/modernbert-bm25/benchmark/<UTC-run-id>/`，不保存模型權重。時間只涵蓋 GPU 更新核心，不包含完整 valid、資料準備與 checkpoint 寫入。

演算法依據：[SAILER dense forward](https://github.com/CSHaitao/SAILER/blob/f5bfb4263b3f227fd800c3555debb6186f89fd75/src/dense/modeling.py#L106-L136)、[GradCache 原始論文](https://arxiv.org/abs/2101.06983)及[官方實作](https://github.com/luyug/GradCache)。本專案實作針對單GPU、共享encoder、既有temperature與HF AMP流程整合，未新增外部套件依賴。
