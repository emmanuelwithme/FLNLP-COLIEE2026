from __future__ import annotations

import os
from pathlib import Path


def load_chunkagg_dotenv(start_path: str | Path | None = None) -> Path | None:
    """Load the nearest `.env` for modernBert-fp-chunkAgg if present."""
    base_path = Path(start_path).resolve() if start_path is not None else Path(__file__).resolve()
    if base_path.is_file():
        search_path = base_path.parent
    else:
        search_path = base_path

    for candidate_dir in [search_path, *search_path.parents]:
        env_path = candidate_dir / ".env"
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value and len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            os.environ.setdefault(key, value)
        return env_path

    return None
