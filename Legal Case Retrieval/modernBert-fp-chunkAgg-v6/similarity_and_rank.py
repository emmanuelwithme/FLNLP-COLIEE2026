from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from env_utils import load_chunkagg_v6_dotenv

_LOADED_DOTENV_PATH = load_chunkagg_v6_dotenv(__file__)

from lcr.data import EmbeddingsData, load_query_ids
from lcr.similarity import compute_similarity_and_save
from lcr.task1_paths import get_env_path, get_task1_dir

TASK1_DIR = get_task1_dir()
MODEL_NAME = os.environ["TASK1_CHUNKAGG_V6_MODEL_NAME"].strip()


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _quick_embedding_path(path: Path) -> Path:
    if not _get_env_bool("TASK1_CHUNKAGG_V6_INFER_QUICK_TEST", False):
        return path
    return path.with_name(f"{path.stem}_quick{path.suffix}")


def main() -> None:
    configured_embeddings_path = get_env_path(
        "TASK1_CHUNKAGG_V6_EMBEDDINGS_PATH", required=True
    )
    assert configured_embeddings_path is not None
    embeddings_path = _quick_embedding_path(configured_embeddings_path)
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"找不到 v6 embeddings: {embeddings_path}; 請先執行 inference.py"
        )
    embeddings = EmbeddingsData.load(embeddings_path)
    quick_test = _get_env_bool("TASK1_CHUNKAGG_V6_INFER_QUICK_TEST", False)
    if not quick_test and len(embeddings) != 7703:
        raise RuntimeError(
            f"v6 ranking data contract drift: embeddings={len(embeddings)}, expected=7703"
        )

    scope_path = get_env_path(
        "TASK1_CHUNKAGG_V6_SCOPE_PATH",
        default=(
            f"{TASK1_DIR}/lht_process/scope_compare/"
            "query_candidate_scope_raw_plus0_empty_exclude.json"
        ),
        required=True,
    )
    assert scope_path is not None
    if not scope_path.exists():
        raise FileNotFoundError(f"找不到 TASK1_CHUNKAGG_V6_SCOPE_PATH: {scope_path}")

    results_dir = get_env_path("TASK1_CHUNKAGG_V6_RESULTS_DIR", required=True)
    assert results_dir is not None
    results_dir.mkdir(parents=True, exist_ok=True)
    qid_paths = {
        "train": Path(f"{TASK1_DIR}/train_qid.tsv"),
        "valid": Path(f"{TASK1_DIR}/valid_qid.tsv"),
    }
    print(f"🔹 Query/candidate embeddings: {embeddings_path} ({len(embeddings)} documents)")
    print(f"🔹 Scope: {scope_path}")
    print("🔹 Ranking contract: eligible processed only, scope first, self excluded, top-5")

    for split, qid_path in qid_paths.items():
        qids = load_query_ids(qid_path)
        for metric in ("dot", "cos"):
            output_path = results_dir / f"output_{MODEL_NAME}_{metric}_{split}.tsv"
            missing = compute_similarity_and_save(
                qids,
                embeddings,
                embeddings,
                output_path,
                metric=metric,
                run_tag=f"{MODEL_NAME}_{metric}",
                topk=5,
                query_candidate_scope_path=scope_path,
                fallback_to_all_candidates_if_scope_missing=False,
                exclude_self=True,
            )
            if missing:
                print(
                    f"⚠️ {split}/{metric}: {len(missing)} queries lack eligible embeddings; "
                    "they remain empty predictions during global evaluation"
                )
            print(f"✅ {split}/{metric}: {output_path}")


if __name__ == "__main__":
    main()
