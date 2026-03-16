from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path

from tqdm import tqdm

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, get_task1_year
from process import process_file

TASK1_DIR = Path(get_task1_dir())
TASK1_YEAR = get_task1_year()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed_test corpus from task1_test_files_<year> using the same cleaner as process.py."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=TASK1_DIR / f"task1_test_files_{TASK1_YEAR}",
        help="Raw test files directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TASK1_DIR / "processed_test",
        help="Output directory for cleaned test files.",
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=TASK1_DIR / "summary",
        help="Summary directory used by the cleaner; when absent it is ignored.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, multiprocessing.cpu_count()),
        help="Number of parallel workers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    summary_dir = args.summary_dir

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input test directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    have_sum = os.listdir(summary_dir) if summary_dir.is_dir() else []
    names = sorted([p.name for p in input_dir.glob("*.txt") if p.is_file()])

    print(f"Using input dir : {input_dir}")
    print(f"Using output dir: {output_dir}")
    print(f"Files to process: {len(names)}")
    print(f"Workers         : {args.workers}")

    process_func = partial(
        process_file,
        input_dir=str(input_dir),
        summary_dir=str(summary_dir),
        output_dir=str(output_dir),
        have_sum=have_sum,
    )

    with multiprocessing.Pool(processes=args.workers) as pool:
        list(tqdm(pool.imap(process_func, names), total=len(names), desc="Processing test files"))

    print(f"✅ processed_test generated: {output_dir}")


if __name__ == "__main__":
    main()
