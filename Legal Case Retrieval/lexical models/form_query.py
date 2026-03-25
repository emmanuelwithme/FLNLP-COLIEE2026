import json
import stat
from tqdm import tqdm
import os
import sys
import argparse
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_env_int, get_task1_dir, get_task1_year, resolve_repo_path


def parse_args() -> argparse.Namespace:
    task1_dir = Path(get_task1_dir())
    task1_year = get_task1_year()
    raw_path = resolve_repo_path(os.getenv("TASK1_PROCESSED_DIR")) or (task1_dir / "processed")
    output_dir = resolve_repo_path(os.getenv("TASK1_BM25_DIR")) or (task1_dir / "lht_process" / "BM25")
    label_path = resolve_repo_path(os.getenv("TASK1_TRAIN_LABELS_PATH")) or (
        task1_dir / f"task1_train_labels_{task1_year}.json"
    )
    valid_path = resolve_repo_path(os.getenv("TASK1_VALID_QID_PATH")) or (task1_dir / "valid_qid.tsv")
    train_path = resolve_repo_path(os.getenv("TASK1_TRAIN_QID_PATH")) or (task1_dir / "train_qid.tsv")
    parser = argparse.ArgumentParser(description="Generate BM25 query TSV files for Task 1.")
    parser.add_argument("--raw-dir", type=Path, default=raw_path)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument("--labels-path", type=Path, default=label_path)
    parser.add_argument("--valid-qid-path", type=Path, default=valid_path)
    parser.add_argument("--train-qid-path", type=Path, default=train_path)
    parser.add_argument(
        "--truncate-threshold",
        type=int,
        default=get_env_int("TASK1_LEXICAL_QUERY_TRUNCATE_THRESHOLD", 30000),
    )
    parser.add_argument(
        "--truncate-length",
        type=int,
        default=get_env_int("TASK1_LEXICAL_QUERY_TRUNCATE_LENGTH", 10000),
    )
    return parser.parse_args()


def read_qid_list(path: Path) -> list[str]:
    qids: list[str] = []
    with path.open("r", encoding="utf-8") as valid_file:
        for raw_line in valid_file:
            parts = raw_line.strip().split()
            if not parts:
                continue
            qids.append(parts[0])
    return qids


def build_query_text(path: Path, *, truncate_threshold: int, truncate_length: int) -> str:
    text_parts: list[str] = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            normalized = line.replace("\n", " ").replace("\t", " ").strip()
            if normalized:
                text_parts.append(normalized)
    text = "".join(text_parts).strip()
    if len(text) > truncate_threshold:
        return text[:truncate_length]
    return text


def main() -> None:
    args = parse_args()
    raw_path = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    query_valid_file = output_dir / "query_valid.tsv"
    query_train_file = output_dir / "query_train.tsv"
    label_path = args.labels_path.resolve()
    valid_path = args.valid_qid_path.resolve()
    train_path = args.train_qid_path.resolve()

    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(train_path.parent, exist_ok=True)
    os.makedirs(label_path.parent, exist_ok=True)

    os.chmod(raw_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(output_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(train_path.parent, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(label_path.parent, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    file_dir = os.listdir(raw_path)

    train_file: list[str] = []
    try:
        with label_path.open("r", encoding="utf-8") as label_file:
            label_dict = json.load(label_file)
        for key in label_dict.keys():
            train_file.append(key.split(".")[0])
    except Exception as e:
        print(f"讀取標籤文件時出錯: {e}")
        label_dict = {}

    try:
        valid_list = read_qid_list(valid_path)
    except Exception as e:
        print(f"讀取驗證集ID文件時出錯: {e}")
        valid_list = [f.split(".")[0] for f in file_dir]

    train_list = [pid for pid in train_file if pid not in valid_list]

    with train_path.open("w", encoding="utf-8") as f:
        for qid in train_list:
            f.write(f"{qid}\n")
    print(f"已成功創建訓練集ID文件: {train_path}")

    max_len_valid = 0
    skipped_valid_empty: list[str] = []
    print("處理驗證集查詢...")
    with query_valid_file.open("w", encoding="utf-8") as outfile_valid:
        for a_file in tqdm(file_dir):
            pid = a_file.split(".")[0]
            if pid not in valid_list:
                continue
            path = raw_path / a_file
            try:
                text_ = build_query_text(
                    path,
                    truncate_threshold=args.truncate_threshold,
                    truncate_length=args.truncate_length,
                )
                if len(text_) == 0:
                    skipped_valid_empty.append(pid)
                    continue
                max_len_valid = max(max_len_valid, len(text_))
                outfile_valid.write(f"{pid}\t{text_}\n")
            except Exception as e:
                print(f"處理文件 {a_file} 時出錯: {e}")

    print(f"最大驗證集文本長度: {max_len_valid}")
    print(f"已成功創建驗證集查詢文件: {query_valid_file}")
    if skipped_valid_empty:
        print(f"⚠️ 驗證集有 {len(skipped_valid_empty)} 筆空查詢已跳過，例如: {skipped_valid_empty[:10]}")

    max_len_train = 0
    skipped_train_empty: list[str] = []
    print("處理訓練集查詢...")
    with query_train_file.open("w", encoding="utf-8") as outfile_train:
        for a_file in tqdm(file_dir):
            pid = a_file.split(".")[0]
            if pid not in train_list:
                continue
            path = raw_path / a_file
            try:
                text_ = build_query_text(
                    path,
                    truncate_threshold=args.truncate_threshold,
                    truncate_length=args.truncate_length,
                )
                if len(text_) == 0:
                    skipped_train_empty.append(pid)
                    continue
                max_len_train = max(max_len_train, len(text_))
                outfile_train.write(f"{pid}\t{text_}\n")
            except Exception as e:
                print(f"處理文件 {a_file} 時出錯: {e}")

    print(f"最大訓練集文本長度: {max_len_train}")
    print(f"已成功創建訓練集查詢文件: {query_train_file}")
    print(f"訓練集查詢文件包含 {len(train_list)} 條記錄，驗證集查詢文件包含 {len(valid_list)} 條記錄")
    if skipped_train_empty:
        print(f"⚠️ 訓練集有 {len(skipped_train_empty)} 筆空查詢已跳過，例如: {skipped_train_empty[:10]}")


if __name__ == "__main__":
    main()
