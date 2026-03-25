import json
import random
import os
import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_env_float, get_env_int, get_task1_dir, get_task1_year, resolve_repo_path

TASK1_DIR = get_task1_dir()
TASK1_YEAR = get_task1_year()

def split_dataset(input_file, train_ratio=0.8, seed=42, output_dir=None):
    """
    將訓練資料分割成訓練集和驗證集
    
    Args:
        input_file: 輸入的標籤檔案路徑，JSON格式
        train_ratio: 訓練集比例，預設0.8
        seed: 隨機種子，預設42
        output_dir: 輸出目錄，預設為input_file所在目錄
    
    Returns:
        train_file: 訓練集檔案路徑
        valid_file: 驗證集檔案路徑
        valid_qid_file: 驗證集qid檔案路徑
    """
    # 設定隨機種子，確保結果可重現
    random.seed(seed)
    
    # 讀取輸入檔案
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 獲取所有qid
    qids = list(data.keys())
    
    # 隨機打亂qid
    random.shuffle(qids)
    
    # 計算訓練集大小
    train_size = int(len(qids) * train_ratio)
    
    # 分割qid列表
    train_qids = qids[:train_size]
    valid_qids = qids[train_size:]
    
    # 根據qid列表創建訓練集和驗證集
    train_data = {qid: data[qid] for qid in train_qids}
    valid_data = {qid: data[qid] for qid in valid_qids}
    
    # 設定輸出路徑
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    # 獲取檔案名稱（不含副檔名）
    file_name = os.path.basename(input_file).split('.')[0]
    
    # 設定輸出檔案路徑
    train_file = os.path.join(output_dir, f"{file_name}_train.json")
    valid_file = os.path.join(output_dir, f"{file_name}_valid.json")
    valid_qid_file = os.path.join(output_dir, "valid_qid.tsv")
    
    # 保存訓練集和驗證集
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
    
    with open(valid_file, 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=4)
    
    # 保存驗證集qid列表
    with open(valid_qid_file, 'w', encoding='utf-8') as f:
        for qid in valid_qids:
            # 移除.txt後綴，只保留數字部分
            qid_num = qid.split('.')[0]
            f.write(f"{qid_num}\n")
    
    print(f"訓練集大小: {len(train_qids)}")
    print(f"驗證集大小: {len(valid_qids)}")
    print(f"訓練集已保存至: {train_file}")
    print(f"驗證集已保存至: {valid_file}")
    print(f"驗證集qid列表已保存至: {valid_qid_file}")
    
    return train_file, valid_file, valid_qid_file

def parse_args() -> argparse.Namespace:
    input_default = resolve_repo_path(os.getenv("TASK1_TRAIN_LABELS_PATH")) or Path(TASK1_DIR) / f"task1_train_labels_{TASK1_YEAR}.json"
    output_default = resolve_repo_path(os.getenv("COLIEE_TASK1_DIR")) or Path(TASK1_DIR)
    parser = argparse.ArgumentParser(description="Split Task 1 labels into train/valid JSON files.")
    parser.add_argument("--input-file", type=Path, default=input_default)
    parser.add_argument("--train-ratio", type=float, default=get_env_float("TASK1_SPLIT_TRAIN_RATIO", 0.8))
    parser.add_argument("--seed", type=int, default=get_env_int("TASK1_SPLIT_SEED", 42))
    parser.add_argument("--output-dir", type=Path, default=output_default)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_dataset(str(args.input_file), float(args.train_ratio), int(args.seed), str(args.output_dir))
