from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR_ENV_KEY = "COLIEE_MODELS_DIR"
_MISSING = object()


def get_repo_root() -> Path:
    return REPO_ROOT


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1].strip()
    return value


@lru_cache(maxsize=None)
def _load_dotenv_from(dotenv_path_str: str) -> None:
    dotenv_path = Path(dotenv_path_str)
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
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
        os.environ.setdefault(key, _strip_quotes(value))


def load_dotenv_if_present(repo_root: str | os.PathLike[str] | None = None) -> None:
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    _load_dotenv_from(str((root / ".env").resolve()))


def resolve_repo_path(
    raw_path: str | os.PathLike[str] | None,
    repo_root: str | os.PathLike[str] | None = None,
) -> Path | None:
    if raw_path is None:
        return None
    raw = str(raw_path).strip()
    if not raw:
        return None
    root = Path(repo_root) if repo_root is not None else REPO_ROOT
    path = Path(raw)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _get_env_value(
    name: str,
    *,
    default: object = _MISSING,
    required: bool = False,
    allow_empty: bool = False,
) -> str | None:
    load_dotenv_if_present()
    raw = os.getenv(name)
    if raw is None:
        if default is not _MISSING:
            return str(default)
        if required:
            raise KeyError(f"Required environment variable is not set: {name}")
        return None

    value = raw.strip()
    if value or allow_empty:
        return value
    if default is not _MISSING:
        return str(default)
    if required:
        raise ValueError(f"Required environment variable is empty: {name}")
    return None


def get_env(
    name: str,
    *,
    default: str | None | object = _MISSING,
    required: bool = False,
    allow_empty: bool = False,
) -> str | None:
    return _get_env_value(name, default=default, required=required, allow_empty=allow_empty)


def parse_env_bool(raw_value: str) -> bool:
    return raw_value.strip().lower() not in {"", "0", "false", "no", "off"}


def get_env_bool(
    name: str,
    *,
    default: bool | object = _MISSING,
    required: bool = False,
) -> bool:
    raw = _get_env_value(name, default=_MISSING, required=required)
    if raw is None:
        if default is _MISSING:
            raise KeyError(f"Required boolean environment variable is not set: {name}")
        return bool(default)
    return parse_env_bool(raw)


def get_env_int(
    name: str,
    default: int | object = _MISSING,
    *,
    required: bool = False,
) -> int:
    raw = _get_env_value(name, default=_MISSING, required=required)
    if raw is None:
        if default is _MISSING:
            raise KeyError(f"Required integer environment variable is not set: {name}")
        return int(default)
    return int(raw)


def get_env_float(
    name: str,
    default: float | object = _MISSING,
    *,
    required: bool = False,
) -> float:
    raw = _get_env_value(name, default=_MISSING, required=required)
    if raw is None:
        if default is _MISSING:
            raise KeyError(f"Required float environment variable is not set: {name}")
        return float(default)
    return float(raw)


def get_env_path(
    name: str,
    *,
    default: str | os.PathLike[str] | object = _MISSING,
    required: bool = False,
    repo_root: str | os.PathLike[str] | None = None,
) -> Path | None:
    raw = _get_env_value(name, default=_MISSING, required=required)
    if raw is None:
        if default is _MISSING:
            return None
        resolved_default = resolve_repo_path(default, repo_root=repo_root)
        if resolved_default is None:
            raise ValueError(f"Failed to resolve default path for {name}")
        return resolved_default
    resolved = resolve_repo_path(raw, repo_root=repo_root)
    if resolved is None:
        if required:
            raise ValueError(f"Required path environment variable is empty: {name}")
        return None
    return resolved


def get_models_dir() -> Path:
    resolved = get_env_path(MODELS_DIR_ENV_KEY, required=True)
    assert resolved is not None
    return resolved


def models_path(*parts: str) -> Path:
    return get_models_dir().joinpath(*parts).resolve()
