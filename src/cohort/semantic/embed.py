"""Pluggable document embedding.

Two backends, same interface:

``tfidf-svd``
    Word 1-2 grams -> TF-IDF -> truncated SVD -> L2 normalise. Deterministic,
    CPU-only, no model download, ~2 seconds for 15k documents. This is the
    default so that a clean checkout reproduces the published metrics exactly,
    with no network access and no GPU.

``sentence-transformers``
    ``BAAI/bge-m3`` by default. Better semantics and — the reason it exists here —
    it is *cross-lingual*, so a German Rahmenvertrag lands in the same cohort as
    an English master service agreement. The TF-IDF backend cannot do that, and
    the evaluation report states so rather than hiding it.

The choice is a config flag, not a code change, because the rest of the pipeline
only ever sees an (n_docs, dim) float32 matrix with unit-norm rows.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from cohort.config import SemanticConfig


@dataclass
class EmbeddingResult:
    vectors: np.ndarray  # (n, dim) float32, unit-norm rows
    backend: str
    dim: int
    elapsed_s: float
    docs_per_s: float
    #: Vocabulary of the fitted vectoriser, when the backend has one. The cohort
    #: namer reuses it instead of re-tokenising the corpus.
    vectorizer: TfidfVectorizer | None = None


class Backend(Protocol):
    name: str

    def encode(self, texts: list[str]) -> tuple[np.ndarray, TfidfVectorizer | None]: ...


class TfidfSvdBackend:
    """Deterministic offline backend."""

    name = "tfidf-svd"

    def __init__(self, cfg: SemanticConfig) -> None:
        self.cfg = cfg

    def encode(self, texts: list[str]) -> tuple[np.ndarray, TfidfVectorizer | None]:
        vec = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.55,  # drop the shared corporate boilerplate
            max_features=self.cfg.max_features,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        X = vec.fit_transform(texts)

        # SVD to a dense, fixed-width representation. n_components must stay
        # below the vocabulary rank; clamp rather than let sklearn raise.
        k = min(self.cfg.dim, X.shape[1] - 1, X.shape[0] - 1)
        svd = TruncatedSVD(n_components=k, random_state=0, algorithm="randomized", n_iter=7)
        Z = svd.fit_transform(X)

        # Unit-norm rows so Euclidean distance is a monotone function of cosine
        # distance, which is what lets HDBSCAN run with the default metric.
        Z = normalize(Z.astype(np.float32), norm="l2")
        return Z, vec


class SentenceTransformerBackend:
    """Optional transformer backend. Requires the `transformers` extra."""

    name = "sentence-transformers"

    def __init__(self, cfg: SemanticConfig) -> None:
        self.cfg = cfg

    def encode(self, texts: list[str]) -> tuple[np.ndarray, None]:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "backend='sentence-transformers' needs the transformers extra:\n"
                "    pip install -e '.[transformers]'"
            ) from exc

        model = SentenceTransformer(self.cfg.model_name)
        Z = model.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return Z.astype(np.float32), None


def build_backend(cfg: SemanticConfig) -> Backend:
    if cfg.backend == "sentence-transformers":
        return SentenceTransformerBackend(cfg)
    return TfidfSvdBackend(cfg)


def embed_documents(texts: list[str], cfg: SemanticConfig) -> EmbeddingResult:
    backend = build_backend(cfg)
    t0 = time.perf_counter()
    vectors, vec = backend.encode(texts)
    elapsed = time.perf_counter() - t0
    return EmbeddingResult(
        vectors=vectors,
        backend=backend.name,
        dim=int(vectors.shape[1]),
        elapsed_s=elapsed,
        docs_per_s=len(texts) / elapsed if elapsed > 0 else float("inf"),
        vectorizer=vec,
    )
