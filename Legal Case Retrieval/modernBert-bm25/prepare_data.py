"""Preflight real 2025 data; --check-only performs no writes."""
import argparse
import json

from fine_tune.data import prepare
from config import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = prepare(load_config(args.config, args.output_dir), write=not args.check_only)["manifest"]
    print(json.dumps({k: v for k, v in result.items() if k not in ("document_hashes", "candidate_ids")}, indent=2))


if __name__ == "__main__":
    main()
