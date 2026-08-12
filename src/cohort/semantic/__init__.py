"""Semantic layer: document meaning, cohort discovery, cohort naming."""

from cohort.semantic.cluster import ClusterResult, discover_cohorts
from cohort.semantic.embed import EmbeddingResult, build_backend
from cohort.semantic.naming import name_cohorts

__all__ = [
    "ClusterResult",
    "EmbeddingResult",
    "build_backend",
    "discover_cohorts",
    "name_cohorts",
]
