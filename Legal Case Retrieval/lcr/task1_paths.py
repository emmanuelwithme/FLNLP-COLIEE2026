from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from repo_config import (
    get_env,
    get_env_bool,
    get_env_float,
    get_env_int,
    get_env_path,
    get_repo_root,
    load_dotenv_if_present,
    resolve_repo_path,
)

ENV_YEAR_KEY = "COLIEE_TASK1_YEAR"
ENV_ROOT_KEY = "COLIEE_TASK1_ROOT"
ENV_DIR_KEY = "COLIEE_TASK1_DIR"
ENV_MODEL_NAME_KEY = "TASK1_RETRIEVAL_MODEL_NAME"
ENV_MODEL_ROOT_KEY = "TASK1_MODEL_ROOT_DIR"
ENV_BASE_ENCODER_DIR_KEY = "TASK1_BASE_ENCODER_DIR"


get_env_flag = get_env_bool


def get_task1_year() -> str:
    value = get_env(ENV_YEAR_KEY, required=True)
    assert value is not None
    return value


def get_task1_root() -> str:
    explicit_root = get_env_path(ENV_ROOT_KEY)
    if explicit_root is not None:
        return str(explicit_root)
    return str(Path(get_task1_dir()).parent.resolve())


def get_task1_dir() -> str:
    explicit_dir = get_env_path(ENV_DIR_KEY)
    if explicit_dir is not None:
        return str(explicit_dir)

    task1_root = get_env_path(ENV_ROOT_KEY, required=True)
    assert task1_root is not None
    return str((task1_root / get_task1_year()).resolve())


def build_default_task1_model_root_dir(
    *,
    year: str | None = None,
    scope_filter: bool = True,
    quick_test: bool = False,
) -> Path:
    raise RuntimeError(
        "TASK1 model root is now configured directly via TASK1_MODEL_ROOT_DIR in shell/.env. "
        "Automatic model-root synthesis has been removed."
    )


def get_task1_model_name() -> str:
    value = get_env(ENV_MODEL_NAME_KEY, required=True)
    assert value is not None
    return value


def get_task1_model_root_dir(*, scope_filter: bool = True, quick_test: bool = False) -> str:
    resolved = get_env_path(ENV_MODEL_ROOT_KEY, required=True)
    assert resolved is not None
    return str(resolved)


def get_task1_base_encoder_dir() -> str:
    resolved = get_env_path(ENV_BASE_ENCODER_DIR_KEY, required=True)
    assert resolved is not None
    return str(resolved)


def task1_join(*parts: str) -> str:
    load_dotenv_if_present(get_repo_root())
    return str(Path(get_task1_dir()).joinpath(*parts))
