from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence


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
        for raw_docid in raw_candidates:
            docid = normalize_case_id(raw_docid, zero_pad_width=zero_pad_width)
            if not docid or docid in seen:
                continue
            seen.add(docid)
            normalized.append(docid)
        scope[qid] = normalized
    return scope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert TREC output into 3-column submission format: [query] [relevant] [run_tag]."
    )
    parser.add_argument("--trec-path", type=Path, required=True, help="Input TREC file path.")
    parser.add_argument("--output-path", type=Path, required=True, help="Output submission path.")
    parser.add_argument("--run-tag", type=str, required=True, help="Run tag to place in column 3.")
    parser.add_argument("--topk", type=int, default=5, help="Keep top-k docs per query after filtering.")
    parser.add_argument(
        "--scope-path",
        type=Path,
        default=None,
        help="Optional query->candidate scope JSON to enforce year filter.",
    )
    parser.add_argument(
        "--qid-path",
        type=Path,
        default=None,
        help="Optional qid list (one per line). Restrict output to these queries and keep this order.",
    )
    parser.add_argument(
        "--skip-self",
        action="store_true",
        help="Drop predictions where qid == docid.",
    )
    parser.add_argument(
        "--strict-scope",
        action="store_true",
        help="Raise error when a qid is missing from scope (only when --scope-path is provided).",
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

    if not args.trec_path.is_file():
        raise FileNotFoundError(f"TREC file not found: {args.trec_path}")

    scope = None
    if args.scope_path:
        if not args.scope_path.is_file():
            raise FileNotFoundError(f"Scope file not found: {args.scope_path}")
        scope = load_scope(args.scope_path, zero_pad_width=args.zero_pad_width)

    qid_order = None
    qid_set = None
    if args.qid_path:
        if not args.qid_path.is_file():
            raise FileNotFoundError(f"QID file not found: {args.qid_path}")
        qid_order = load_qids(args.qid_path, zero_pad_width=args.zero_pad_width)
        qid_set = set(qid_order)

    selected: Dict[str, List[str]] = {}

    with args.trec_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            qid = normalize_case_id(parts[0], zero_pad_width=args.zero_pad_width)
            docid = normalize_case_id(parts[2], zero_pad_width=args.zero_pad_width)
            if not qid or not docid:
                continue
            if qid_set is not None and qid not in qid_set:
                continue
            if args.skip_self and qid == docid:
                continue

            if scope is not None:
                allowed = scope.get(qid)
                if allowed is None:
                    if args.strict_scope:
                        raise ValueError(f"Query {qid} missing from scope {args.scope_path}")
                    allowed_set = None
                else:
                    allowed_set = set(allowed)
                if allowed_set is not None and docid not in allowed_set:
                    continue

            entries = selected.setdefault(qid, [])
            if len(entries) < args.topk:
                entries.append(docid)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output_path.open("w", encoding="utf-8") as fout:
        if qid_order is None:
            qids_to_write = sorted(selected.keys())
        else:
            qids_to_write = qid_order
        for qid in qids_to_write:
            for docid in selected.get(qid, []):
                fout.write(f"{qid} {docid} {args.run_tag}\n")
                written += 1

    print(f"✅ submission written: {args.output_path}")
    print(f"Queries with predictions: {sum(1 for q in (qid_order or selected.keys()) if selected.get(q))}")
    print(f"Lines written: {written}")


if __name__ == "__main__":
    main()
