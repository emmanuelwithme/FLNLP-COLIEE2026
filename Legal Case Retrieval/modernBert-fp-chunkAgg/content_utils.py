from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def non_whitespace_character_count(text: str) -> int:
    return sum(1 for character in text if not character.isspace())


def has_minimum_document_content(text: str, minimum_characters: int) -> bool:
    if minimum_characters < 0:
        raise ValueError("minimum_characters must be >= 0")
    return non_whitespace_character_count(text) >= minimum_characters


def scan_eligible_document_ids(
    directory: str | Path,
    *,
    minimum_characters: int,
) -> tuple[set[str], dict[str, int]]:
    """Return eligible IDs and rejected non-whitespace lengths for a text corpus."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"document directory does not exist: {directory}")

    eligible_ids: set[str] = set()
    rejected_lengths: dict[str, int] = {}
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        content_length = non_whitespace_character_count(text)
        if content_length >= minimum_characters:
            eligible_ids.add(path.stem)
        else:
            rejected_lengths[path.stem] = content_length
    return eligible_ids, rejected_lengths


def filter_id_text_pairs(
    ids: Sequence[str],
    texts: Sequence[str],
    *,
    minimum_characters: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    if len(ids) != len(texts):
        raise ValueError(f"ids/texts length mismatch: {len(ids)} != {len(texts)}")

    kept_ids: list[str] = []
    kept_texts: list[str] = []
    rejected_lengths: dict[str, int] = {}
    for doc_id, text in zip(ids, texts):
        content_length = non_whitespace_character_count(text)
        if content_length >= minimum_characters:
            kept_ids.append(str(doc_id))
            kept_texts.append(text)
        else:
            rejected_lengths[str(doc_id)] = content_length
    return kept_ids, kept_texts, rejected_lengths


def summarize_rejected_documents(rejected_lengths: dict[str, int]) -> str:
    items: Iterable[tuple[str, int]] = sorted(
        rejected_lengths.items(),
        key=lambda item: (item[1], item[0]),
    )
    return ", ".join(f"{doc_id}({length})" for doc_id, length in items)
