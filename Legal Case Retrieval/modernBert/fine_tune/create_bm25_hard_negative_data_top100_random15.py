import json
import sys
import os
import argparse
from typing import Dict, List, Set
from collections import defaultdict
import random
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_env_int, get_task1_dir, get_task1_year, resolve_repo_path

def read_bm25_output_trec(tsv_path: str, top_k: int = 100) -> Dict[str, List[str]]:
    """讀取 TREC 格式 BM25 檢索結果，補齊 query_id 和 doc_id 至 6 位數"""
    bm25_results: Dict[str, List[str]] = defaultdict(list)
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            query_id_raw, _, doc_id_raw, rank_str, score, _ = parts
            rank = int(rank_str)
            if rank <= top_k:
                query_id = query_id_raw.zfill(6)
                doc_id = doc_id_raw.zfill(6)
                bm25_results[query_id].append(doc_id)
    return bm25_results

def read_positive_pairs_from_json(json_path: str) -> Dict[str, Set[str]]:
    """從 JSON 讀取正樣本對映表，並去除 .txt 副檔名"""
    positives: Dict[str, Set[str]] = defaultdict(set)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for q_txt, pos_list in data.items():
        qid = q_txt.replace(".txt", "")
        for doc_txt in pos_list:
            doc_id = doc_txt.replace(".txt", "")
            positives[qid].add(doc_id)
    return positives

def generate_contrastive_json(
    bm25_path: str,
    json_positive_path: str,
    output_path: str,
    top_k: int = 100,
    max_negatives: int = 15,
    random_seed: int = None
) -> None:
    """
    產生 contrastive learning 格式的 JSON 檔，包含 query_id、positive_id、negative_ids
    此版本會從 BM25 top_k 中去除正樣本後，隨機抽出 max_negatives 個負樣本。

    - bm25_path: BM25 TREC 格式結果檔路徑
    - json_positive_path: 正樣本 JSON 檔（key、value 都帶 .txt）
    - output_path: 將產出的訓練/驗證資料寫到這個 JSON 路徑
    - top_k: 從 BM25 前 top_k 名中挑選負樣本候選
    - max_negatives: 從候選負樣本中要隨機抽幾個
    - random_seed: 如果指定，會在抽取負樣本時設置隨機種子以保證可重現
    """
    if random_seed is not None:
        random.seed(random_seed)

    bm25_results: Dict[str, List[str]] = read_bm25_output_trec(bm25_path, top_k=top_k)
    positives: Dict[str, Set[str]] = read_positive_pairs_from_json(json_positive_path)

    dataset: List[Dict[str, object]] = []
    skipped_pairs = 0

    for qid, pos_set in positives.items():
        # 如果 query 不在 BM25 結果中就跳過
        if qid not in bm25_results:
            continue
        bm25_docs = bm25_results[qid]  # 這是 BM25 排序後的 doc_id 列表

        for pos_id in pos_set:
            # 候選負樣本 = BM25 top_k 裡，但不在正樣本集合中的所有 doc_id
            all_neg_candidates = [doc_id for doc_id in bm25_docs if doc_id not in pos_set]
            # 確保候選負樣本至少要有 max_negatives 個
            if len(all_neg_candidates) >= max_negatives:
                # 隨機抽 max_negatives 個
                neg_sample = random.sample(all_neg_candidates, max_negatives)
                dataset.append({
                    "query_id": qid,
                    "positive_id": pos_id,
                    "negative_ids": neg_sample
                })
            else:
                skipped_pairs += 1

    # 將結果寫到 output_path
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"✅ 共產生 {len(dataset)} 筆對比學習資料，已儲存至 {output_path}")
    print(f"⚠️ 有 {skipped_pairs} 筆 (query_id, positive_id) 因候選負樣本不足 {max_negatives} 而被略過")


def parse_args() -> argparse.Namespace:
    task1_dir = Path(get_task1_dir())
    task1_year = get_task1_year()
    bm25_dir = resolve_repo_path(os.getenv("TASK1_BM25_DIR")) or (task1_dir / "lht_process" / "BM25")
    finetune_data_dir = resolve_repo_path(os.getenv("TASK1_FINETUNE_DATA_DIR")) or (
        task1_dir / "lht_process" / "modernBert" / "finetune_data"
    )
    train_labels_path = resolve_repo_path(os.getenv("TASK1_TRAIN_SPLIT_LABELS_PATH")) or (
        task1_dir / f"task1_train_labels_{task1_year}_train.json"
    )
    valid_labels_path = resolve_repo_path(os.getenv("TASK1_VALID_SPLIT_LABELS_PATH")) or (
        task1_dir / f"task1_train_labels_{task1_year}_valid.json"
    )
    parser = argparse.ArgumentParser(description="Generate BM25 hard-negative JSON files for Task 1 contrastive fine-tuning.")
    parser.add_argument("--bm25-train-path", type=Path, default=bm25_dir / "output_bm25_train.tsv")
    parser.add_argument("--bm25-valid-path", type=Path, default=bm25_dir / "output_bm25_valid.tsv")
    parser.add_argument("--train-labels-path", type=Path, default=train_labels_path)
    parser.add_argument("--valid-labels-path", type=Path, default=valid_labels_path)
    parser.add_argument(
        "--train-output-path",
        type=Path,
        default=finetune_data_dir / "contrastive_bm25_hard_negative_top100_random15_train.json",
    )
    parser.add_argument(
        "--valid-output-path",
        type=Path,
        default=finetune_data_dir / "contrastive_bm25_hard_negative_top100_random15_valid.json",
    )
    parser.add_argument("--top-k", type=int, default=get_env_int("TASK1_HARD_NEG_TOPK", 100))
    parser.add_argument("--max-negatives", type=int, default=get_env_int("TASK1_HARD_NEG_MAX_NEGATIVES", 15))
    parser.add_argument("--random-seed", type=int, default=get_env_int("TASK1_HARD_NEG_SEED", 289))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_contrastive_json(
        bm25_path=str(args.bm25_train_path),
        json_positive_path=str(args.train_labels_path),
        output_path=str(args.train_output_path),
        top_k=int(args.top_k),
        max_negatives=int(args.max_negatives),
        random_seed=int(args.random_seed),
    )

    generate_contrastive_json(
        bm25_path=str(args.bm25_valid_path),
        json_positive_path=str(args.valid_labels_path),
        output_path=str(args.valid_output_path),
        top_k=int(args.top_k),
        max_negatives=int(args.max_negatives),
        random_seed=int(args.random_seed),
    )
