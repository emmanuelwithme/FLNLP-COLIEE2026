import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
import stat
import argparse

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, resolve_repo_path


def parse_args() -> argparse.Namespace:
    task1_dir = Path(get_task1_dir())
    raw_path = resolve_repo_path(os.getenv("TASK1_PROCESSED_DIR")) or (task1_dir / "processed")
    output_dir = resolve_repo_path(os.getenv("TASK1_BM25_DIR")) or (task1_dir / "lht_process" / "BM25")
    parser = argparse.ArgumentParser(description="Generate BM25 corpus JSON from processed Task 1 cases.")
    parser.add_argument("--raw-dir", type=Path, default=raw_path)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_path = args.raw_dir.resolve()
    bm25_dir = args.output_dir.resolve()
    corpus_dir = bm25_dir / "corpus"
    file_dir = sorted([p for p in raw_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

    os.makedirs(corpus_dir, exist_ok=True)
    os.chmod(corpus_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    print(f"✅ 目標資料夾 {corpus_dir} 已建立並設定適當權限")

    written_count = 0
    skipped_count = 0
    output_path = corpus_dir / "corpus.json"
    with output_path.open("w", encoding="utf-8") as outfile:
        for file_path in tqdm(file_dir):
            pid = file_path.stem
            text_ = ""

            try:
                with open(file_path, encoding="utf-8") as fin:
                    for line in fin:
                        text_ += line.replace("\n", "")
            except UnicodeDecodeError as exc:
                skipped_count += 1
                print(f"⚠️ 跳過無法以 UTF-8 讀取的檔案: {file_path.name} ({exc})")
                continue

            save_dict = {"id": pid, "contents": text_}
            outline = json.dumps(save_dict, ensure_ascii=False) + "\n"
            outfile.write(outline)
            written_count += 1

    print(f"✅ JSON 檔案已成功寫入至 {output_path}")
    print(f"寫入文件數: {written_count}，跳過文件數: {skipped_count}")


if __name__ == "__main__":
    main()
