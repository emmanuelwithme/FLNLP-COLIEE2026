from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from lcr.task1_paths import get_task1_dir, get_task1_year

TASK1_DIR = Path(get_task1_dir())
TASK1_YEAR = get_task1_year()


def normalize_case_id(raw_id: object) -> str:
    case_id = str(raw_id).strip()
    if case_id.endswith(".txt"):
        case_id = case_id[:-4]
    return case_id


def flatten_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def truncate_query_text(text: str, *, max_terms: int, max_chars: int) -> str:
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars]
    if max_terms > 0:
        terms = text.split()
        if len(terms) > max_terms:
            text = " ".join(terms[:max_terms])
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare test qid list, BM25 query tsv, and BM25 corpus jsonl from processed_test."
    )
    parser.add_argument(
        "--test-label-json",
        type=Path,
        default=TASK1_DIR / f"task1_test_no_labels_{TASK1_YEAR}.json",
        help="Path to test no-label JSON.",
    )
    parser.add_argument(
        "--processed-test-dir",
        type=Path,
        default=TASK1_DIR / "processed_test",
        help="Directory containing processed test documents.",
    )
    parser.add_argument(
        "--test-qid-path",
        type=Path,
        default=TASK1_DIR / "test_qid.tsv",
        help="Output test qid file (one id per line).",
    )
    parser.add_argument(
        "--bm25-query-path",
        type=Path,
        default=TASK1_DIR / "lht_process" / "BM25_test" / "query_test.tsv",
        help="Output BM25 test topics in TSV format.",
    )
    parser.add_argument(
        "--bm25-corpus-path",
        type=Path,
        default=TASK1_DIR / "lht_process" / "BM25_test" / "corpus" / "corpus.json",
        help="Output BM25 corpus jsonl path.",
    )
    parser.add_argument(
        "--max-query-terms",
        type=int,
        default=900,
        help="Cap BM25 query length by number of terms to avoid Lucene TooManyClauses.",
    )
    parser.add_argument(
        "--max-query-chars",
        type=int,
        default=10000,
        help="Cap BM25 query length by characters before term truncation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.test_label_json.is_file():
        raise FileNotFoundError(f"Test no-label JSON not found: {args.test_label_json}")
    if not args.processed_test_dir.is_dir():
        raise FileNotFoundError(f"Processed test directory not found: {args.processed_test_dir}")

    with args.test_label_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Test no-label JSON must be an object: {args.test_label_json}")

    ordered_qids = [normalize_case_id(qid) for qid in payload.keys()]
    dedup_qids: list[str] = []
    seen: set[str] = set()
    for qid in ordered_qids:
        if not qid or qid in seen:
            continue
        seen.add(qid)
        dedup_qids.append(qid)

    args.test_qid_path.parent.mkdir(parents=True, exist_ok=True)
    args.test_qid_path.write_text("\n".join(dedup_qids) + "\n", encoding="utf-8")

    missing_queries: list[str] = []
    args.bm25_query_path.parent.mkdir(parents=True, exist_ok=True)
    with args.bm25_query_path.open("w", encoding="utf-8") as fout:
        for qid in dedup_qids:
            query_path = args.processed_test_dir / f"{qid}.txt"
            if not query_path.exists():
                missing_queries.append(qid)
                continue
            text = flatten_text(query_path.read_text(encoding="utf-8", errors="ignore"))
            text = truncate_query_text(
                text,
                max_terms=args.max_query_terms,
                max_chars=args.max_query_chars,
            )
            if not text:
                continue
            fout.write(f"{qid}\t{text}\n")

    args.bm25_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    doc_count = 0
    with args.bm25_corpus_path.open("w", encoding="utf-8") as fout:
        for path in sorted(args.processed_test_dir.glob("*.txt")):
            text = flatten_text(path.read_text(encoding="utf-8", errors="ignore"))
            record = {"id": path.stem, "contents": text}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            doc_count += 1

    print(f"✅ test_qid.tsv written: {args.test_qid_path} ({len(dedup_qids)} qids)")
    print(f"✅ BM25 query TSV written: {args.bm25_query_path}")
    print(f"✅ BM25 corpus JSONL written: {args.bm25_corpus_path} ({doc_count} docs)")
    if missing_queries:
        print(f"⚠️ Missing processed query files: {len(missing_queries)} (example: {missing_queries[:10]})")


if __name__ == "__main__":
    main()
