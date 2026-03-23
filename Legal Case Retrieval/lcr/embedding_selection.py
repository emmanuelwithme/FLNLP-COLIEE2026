from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Task1EmbeddingSelection:
    role: str
    path: str
    source: str
    origin: str


def _first_defined_env(names: Sequence[str]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value, name
    return None, None


def select_task1_embedding_path(
    *,
    role: str,
    processed_path: str,
    processed_new_path: str,
    default_source: str = "processed",
    source_env_names: Sequence[str] = (),
    path_env_names: Sequence[str] = (),
) -> Task1EmbeddingSelection:
    override_path, override_env = _first_defined_env(path_env_names)
    if override_path is not None:
        return Task1EmbeddingSelection(
            role=role,
            path=override_path,
            source="custom_path",
            origin=override_env or "custom_path",
        )

    requested_source, source_env = _first_defined_env(source_env_names)
    source = (requested_source or default_source).strip().lower()
    candidates = {
        "processed": processed_path,
        "processed_new": processed_new_path,
    }
    if source not in candidates:
        allowed = ", ".join(sorted(candidates))
        raise ValueError(
            f"Unsupported {role} embedding source `{source}`. "
            f"Expected one of: {allowed}."
        )

    return Task1EmbeddingSelection(
        role=role,
        path=candidates[source],
        source=source,
        origin=source_env or "default",
    )


def log_task1_embedding_choices(
    *,
    processed_path: str,
    processed_new_path: str,
    query_selection: Task1EmbeddingSelection,
    candidate_selection: Task1EmbeddingSelection,
) -> None:
    print("🔹 可用 embeddings:")
    print(f"   processed     -> {processed_path}")
    print(f"   processed_new -> {processed_new_path}")

    for selection in (query_selection, candidate_selection):
        if selection.source == "custom_path":
            print(
                f"🔹 {selection.role.capitalize()} embeddings 使用自訂路徑 "
                f"({selection.origin}): {selection.path}"
            )
        elif selection.origin == "default":
            print(
                f"🔹 {selection.role.capitalize()} embeddings 預設使用 "
                f"{selection.source}: {selection.path}"
            )
        else:
            print(
                f"🔹 {selection.role.capitalize()} embeddings 由 {selection.origin} "
                f"切換為 {selection.source}: {selection.path}"
            )
