from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cutoff_postprocess import load_scope, normalize_case_id, run_fixed_topk_postprocess


def _load_qids(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"QID file not found: {path}")

    qids: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid = normalize_case_id(line.split()[0])
        if not qid or qid in seen:
            continue
        seen.add(qid)
        qids.append(qid)
    return qids


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("fixed_topk_postprocess")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Apply a fixed top-k post-process to test rerank outputs and export submission files.")
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--test-scope", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-qid", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--keep-self", action="store_true", help="Disable query self-removal.")
    parser.add_argument("--no-submission", action="store_true", help="Skip submission file output.")
    parser.add_argument("--submission-run-tag", type=str, default="lgbm_top5")
    parser.add_argument("--final-submission-path", type=Path, default=None)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    logger = _build_logger()

    test_scope = load_scope(args.test_scope) if args.test_scope else None
    run_fixed_topk_postprocess(
        test_predictions_path=args.test_predictions,
        test_scope=test_scope,
        output_dir=args.output_dir,
        logger=logger,
        k=args.topk,
        test_query_ids=_load_qids(args.test_qid),
        remove_self=not args.keep_self,
        write_submission=not args.no_submission,
        submission_run_tag=args.submission_run_tag,
        final_submission_path=args.final_submission_path,
    )


if __name__ == "__main__":
    main()
