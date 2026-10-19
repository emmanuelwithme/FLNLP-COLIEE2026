from __future__ import annotations

import os
from pathlib import Path


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1].strip()
    return value


def _read_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        values[key] = _strip_quotes(value)
    return values


def load_chunkagg_dotenv(start_path: str | Path | None = None) -> Path | None:
    """Load repo and chunkAgg `.env` files while preserving shell overrides."""
    base_path = Path(start_path).resolve() if start_path is not None else Path(__file__).resolve()
    if base_path.is_file():
        search_path = base_path.parent
    else:
        search_path = base_path

    search_dirs = [search_path, *search_path.parents]
    env_paths = [candidate_dir / ".env" for candidate_dir in reversed(search_dirs)]
    existing_env_paths = [env_path for env_path in env_paths if env_path.exists()]
    original_keys = set(os.environ)

    loaded_path: Path | None = None
    for env_path in existing_env_paths:
        loaded_path = env_path
        for key, value in _read_dotenv(env_path).items():
            if key in original_keys:
                continue
            os.environ[key] = value

    return loaded_path
