from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def normalize_case_id(raw_id: object, *, zero_pad_width: int) -> str:
    case_id = str(raw_id).strip()
    if case_id.endswith(".txt"):
        case_id = case_id[:-4]
    if zero_pad_width > 0 and case_id.isdigit():
        case_id = case_id.zfill(zero_pad_width)
    return case_id


def load_qids(path: Path, *, zero_pad_width: int) -> List[str]:
    qids: List[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid = normalize_case_id(line.split()[0], zero_pad_width=zero_pad_width)
        if not qid or qid in seen:
            continue
        seen.add(qid)
        qids.append(qid)
    return qids


def load_scope(path: Path, *, zero_pad_width: int) -> Dict[str, List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Scope JSON must be an object: {path}")

    scope: Dict[str, List[str]] = {}
    for raw_qid, raw_candidates in payload.items():
        qid = normalize_case_id(raw_qid, zero_pad_width=zero_pad_width)
        if not isinstance(raw_candidates, Sequence) or isinstance(raw_candidates, (str, bytes)):
            raise ValueError(f"Scope candidates must be a sequence for query {qid}.")
        seen: set[str] = set()
        normalized: List[str] = []
        for raw_doc_id in raw_candidates:
            docid = normalize_case_id(raw_doc_id, zero_pad_width=zero_pad_width)
            if not docid or docid in seen:
                continue
            seen.add(docid)
            normalized.append(docid)
        scope[qid] = normalized
    return scope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter TREC ranking by query-specific candidate scope and rewrite sequential ranks."
    )
    parser.add_argument("--input-path", type=Path, required=True, help="Input TREC file path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output filtered TREC file path.")
    parser.add_argument("--scope-path", type=Path, required=True, help="Query->candidate scope JSON path.")
    parser.add_argument(
        "--qid-path",
        type=Path,
        default=None,
        help="Optional qid list; if provided output order follows this file.",
    )
    parser.add_argument("--skip-self", action="store_true", help="Drop qid==docid rows.")
    parser.add_argument(
        "--strict-scope",
        action="store_true",
        help="Raise error if qid is missing from scope.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=0,
        help="Keep at most top-k rows per query after filtering; 0 means keep all.",
    )
    parser.add_argument(
        "--zero-pad-width",
        type=int,
        default=6,
        help="Zero-pad numeric ids to this width. Use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_path.is_file():
        raise FileNotFoundError(f"TREC file not found: {args.input_path}")
    if not args.scope_path.is_file():
        raise FileNotFoundError(f"Scope file not found: {args.scope_path}")

    scope = load_scope(args.scope_path, zero_pad_width=args.zero_pad_width)
    scope_sets = {qid: set(docids) for qid, docids in scope.items()}

    qid_order: List[str] | None = None
    qid_set: set[str] | None = None
    if args.qid_path:
        if not args.qid_path.is_file():
            raise FileNotFoundError(f"QID file not found: {args.qid_path}")
        qid_order = load_qids(args.qid_path, zero_pad_width=args.zero_pad_width)
        qid_set = set(qid_order)

    kept: "OrderedDict[str, List[Tuple[str, str, str]]]" = OrderedDict()
    seen_rows = 0
    dropped_scope = 0
    dropped_self = 0
    dropped_qid = 0

    with args.input_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            seen_rows += 1
            qid = normalize_case_id(parts[0], zero_pad_width=args.zero_pad_width)
            docid = normalize_case_id(parts[2], zero_pad_width=args.zero_pad_width)
            score = parts[4]
            run_tag = parts[5]
            if not qid or not docid:
                continue

            if qid_set is not None and qid not in qid_set:
                dropped_qid += 1
                continue
            if args.skip_self and qid == docid:
                dropped_self += 1
                continue

            allowed = scope_sets.get(qid)
            if allowed is None:
                if args.strict_scope:
                    raise ValueError(f"Query {qid} missing from scope: {args.scope_path}")
                dropped_scope += 1
                continue
            if docid not in allowed:
                dropped_scope += 1
                continue

            kept.setdefault(qid, []).append((docid, score, run_tag))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    lines_written = 0
    with args.output_path.open("w", encoding="utf-8") as fout:
        if qid_order is None:
            out_qids = list(kept.keys())
        else:
            out_qids = qid_order
        for qid in out_qids:
            entries = kept.get(qid, [])
            if args.topk > 0:
                entries = entries[: args.topk]
            for rank, (docid, score, run_tag) in enumerate(entries, start=1):
                fout.write(f"{qid} Q0 {docid} {rank} {score} {run_tag}\n")
                lines_written += 1

    queries_with_rows = sum(1 for qid in (qid_order or kept.keys()) if kept.get(qid))
    print(f"✅ filtered trec written: {args.output_path}")
    print(f"Input rows: {seen_rows}")
    print(f"Dropped by qid filter: {dropped_qid}")
    print(f"Dropped by self filter: {dropped_self}")
    print(f"Dropped by scope filter: {dropped_scope}")
    print(f"Queries with rows: {queries_with_rows}")
    print(f"Output rows: {lines_written}")


if __name__ == "__main__":
    main()
