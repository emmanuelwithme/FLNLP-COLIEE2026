"""
Shared utilities for the Legal Case Retrieval project.

This package consolidates common functionality that was previously duplicated
across the `modernBert`, `SAILER`, and `utils` directories.  The goal is to
keep inference, similarity calculation, evaluation, and result tracking
composable and reusable.
"""

from .data import (
    EmbeddingsData,
    load_query_candidate_scope,
    load_query_ids,
    normalize_case_id,
    normalize_query_candidate_scope,
    read_text_directory,
    resolve_query_candidate_scope,
)
from .device import get_device
from .embeddings import (
    generate_embeddings,
    generate_embeddings_for_directory,
    process_directory_to_embeddings,
)
from .metrics import (
    classification_report,
    random_guess_baseline,
    random_guess_predictions,
    rel_file_to_dict,
    trec_file_to_dict,
)
from .results import record_result
from .similarity import (
    compute_similarity_and_save,
    rank_candidates,
    rank_candidates_with_scores,
    score_queries,
)
from .retrieval import SimilarityArtifacts, generate_similarity_artifacts

__all__ = [
    "EmbeddingsData",
    "normalize_case_id",
    "normalize_query_candidate_scope",
    "load_query_candidate_scope",
    "resolve_query_candidate_scope",
    "load_query_ids",
    "read_text_directory",
    "get_device",
    "generate_embeddings",
    "generate_embeddings_for_directory",
    "process_directory_to_embeddings",
    "classification_report",
    "random_guess_baseline",
    "random_guess_predictions",
    "rel_file_to_dict",
    "trec_file_to_dict",
    "record_result",
    "compute_similarity_and_save",
    "score_queries",
    "rank_candidates",
    "rank_candidates_with_scores",
    "SimilarityArtifacts",
    "generate_similarity_artifacts",
]
