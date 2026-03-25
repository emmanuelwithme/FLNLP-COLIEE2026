import os
import sys
import argparse
from tqdm import tqdm
import multiprocessing
from functools import partial
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_env_int, get_task1_dir, get_task1_year, resolve_repo_path

def process_file(name, input_dir, output_dir):
    with open(f"{input_dir}/{name}", "r", encoding="utf-8") as f:
        txt = f.read()
        if "Summary:" in txt and "no summary" not in txt and "for this document are in preparation." not in txt:
            idx = txt.find("Summary:")
            end = txt.find("- Topic", idx) #從idx位置之後正面數來第一次出現"- Topic"字串的位置
            end2 = txt.rfind("\n", idx, end) #從最後面數來第一次出現換行的位置
            summ=txt[idx+8:end2].strip()
            if summ.count("\n") > 20:
                print(f"摘要換行大於20個: {name}")
                return False
            with open(f"{output_dir}/{name}", "w+", encoding="utf-8") as fp:
                if summ == "":
                    print(f"摘要為空: {name}")
                    return False
                fp.write(summ)
            return True
        else:
            # print(f"沒摘要: {name}")
            return False


def parse_args() -> argparse.Namespace:
    task1_dir = Path(get_task1_dir())
    task1_year = get_task1_year()
    input_dir = resolve_repo_path(os.getenv("TASK1_TRAIN_RAW_DIR")) or (
        task1_dir / f"task1_train_files_{task1_year}"
    )
    output_dir = resolve_repo_path(os.getenv("TASK1_SUMMARY_DIR")) or (task1_dir / "summary")
    parser = argparse.ArgumentParser(description="Extract summaries from raw Task 1 cases.")
    parser.add_argument("--input-dir", type=Path, default=input_dir)
    parser.add_argument("--output-dir", type=Path, default=output_dir)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=get_env_int("TASK1_SUMMARY_NUM_WORKERS", 0),
        help="0 => use multiprocessing.cpu_count().",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    os.makedirs(output_dir, exist_ok=True)

    names = os.listdir(input_dir)
    num_cores = args.num_workers if args.num_workers > 0 else multiprocessing.cpu_count()
    print(f"使用 {num_cores} 個CPU核心進行並行處理")

    process_func = partial(process_file, input_dir=input_dir, output_dir=output_dir)
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(process_func, names), total=len(names), desc="处理文件"))

    total_files = len(names)
    summaries_written = sum(results)

    print(f"\n總共處理 {total_files} 篇檔案")
    print(f"成功記錄下來摘要的有 {summaries_written} 篇")
    print(f"沒有記錄摘要的有 {total_files - summaries_written} 篇")
