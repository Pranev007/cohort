"""Cohort discovery — recovering document categories from meaning alone.

No category list is supplied anywhere in this module. HDBSCAN finds however many
dense regions exist, which is the property that makes the approach rules-free:
a customer with a document type nobody anticipated still gets a cohort for it.

Two design notes worth defending in review:

* Clustering runs on a further-reduced view (``cluster_dim``, default 40) rather
  than the stored 256-d vectors. Density-based methods lose contrast as
  dimensionality grows — pairwise distances concentrate — so HDBSCAN on raw
  high-dimensional embeddings tends to return one giant cluster plus noise.
* HDBSCAN labels genuine outliers ``-1``. Those documents are not discarded:
  they are re-attached to their nearest cohort when cosine similarity clears a
  threshold, and otherwise scored against a global baseline in the
  ``UNASSIGNED`` cohort. Silently dropping them would quietly exclude exactly the
  unusual documents a security tool exists to look at.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from cohort.config import SemanticConfig

UNASSIGNED = -1


@dataclass
class ClusterResult:
    labels: np.ndarray  # (n,) int, -1 == UNASSIGNED
    centroids: dict[int, np.ndarray] = field(default_factory=dict)
    n_cohorts: int = 0
    raw_noise_rate: float = 0.0  # before reassignment
    unassigned_rate: float = 0.0  # after reassignment
    silhouette: float = float("nan")
    #: The similarity bar actually used to re-absorb outliers (diagnostic).
    reassign_bar: float = float("nan")
    elapsed_s: float = 0.0
    sizes: dict[int, int] = field(default_factory=dict)


def _reduce_for_clustering(vectors: np.ndarray, cfg: SemanticConfig) -> np.ndarray:
    target = min(cfg.cluster_dim, vectors.shape[1], max(2, vectors.shape[0] - 1))
    if target >= vectors.shape[1]:
        return vectors

    if cfg.reducer == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=target, metric="cosine", random_state=0, n_neighbors=15, min_dist=0.0
            )
            return normalize(reducer.fit_transform(vectors).astype(np.float32), norm="l2")
        except ImportError:
            pass  # fall through to SVD; documented in the run report

    svd = TruncatedSVD(n_components=target, random_state=0)
    return normalize(svd.fit_transform(vectors).astype(np.float32), norm="l2")


def _centroids(vectors: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for lab in np.unique(labels):
        if lab == UNASSIGNED:
            continue
        member = vectors[labels == lab]
        c = member.mean(axis=0)
        n = np.linalg.norm(c)
        out[int(lab)] = (c / n if n > 0 else c).astype(np.float32)
    return out


def _reassign_bar(
    vectors: np.ndarray,
    labels: np.ndarray,
    cents: dict[int, np.ndarray],
    cfg: SemanticConfig,
) -> float:
    """How similar an outlier must be to a cohort before it is absorbed.

    Calibrated to the embedding's own geometry: take every confidently-assigned
    document's cosine to its own centroid, and use a low percentile of that
    distribution as the bar. A noise point that is as close to a cohort as the
    cohort's least typical genuine member is plausibly a member.

    A fixed cosine cannot do this job. Template-generated documents cluster far
    more tightly than real prose, so a threshold tuned on one lands in completely
    the wrong place on the other.
    """
    if cfg.noise_reassign_threshold is not None:
        return cfg.noise_reassign_threshold

    sims: list[float] = []
    for lab, centroid in cents.items():
        member = vectors[labels == lab]
        if member.size:
            sims.append(np.quantile(member @ centroid, cfg.noise_reassign_percentile))
    if not sims:
        return 1.0
    return float(np.mean(sims))


def discover_cohorts(vectors: np.ndarray, cfg: SemanticConfig) -> ClusterResult:
    t0 = time.perf_counter()
    Z = _reduce_for_clustering(vectors, cfg)

    hdb = HDBSCAN(
        min_cluster_size=cfg.effective_min_cluster_size(vectors.shape[0]),
        min_samples=cfg.min_samples,
        cluster_selection_epsilon=cfg.cluster_selection_epsilon,
        metric="euclidean",  # rows are unit-norm, so this tracks cosine
        n_jobs=-1,
    )
    labels = hdb.fit_predict(Z).astype(int)
    raw_noise = float(np.mean(labels == UNASSIGNED))

    # Re-attach outliers to their nearest cohort when they are close enough.
    cents = _centroids(vectors, labels)
    reassign_bar = float("nan")
    if cents:
        keys = sorted(cents)
        C = np.stack([cents[k] for k in keys])  # (k, dim), unit-norm
        noise_idx = np.flatnonzero(labels == UNASSIGNED)
        if noise_idx.size:
            reassign_bar = _reassign_bar(vectors, labels, cents, cfg)
            sims = vectors[noise_idx] @ C.T  # cosine, rows unit-norm
            best = sims.argmax(axis=1)
            best_sim = sims.max(axis=1)
            take = best_sim >= reassign_bar
            labels[noise_idx[take]] = np.array(keys)[best[take]]

    cents = _centroids(vectors, labels)
    sizes = {int(k): int((labels == k).sum()) for k in np.unique(labels)}

    sil = float("nan")
    assigned = labels != UNASSIGNED
    if cents and assigned.sum() > 10 and len(cents) > 1:
        # Silhouette is O(n^2); sample for anything corpus-sized.
        idx = np.flatnonzero(assigned)
        if idx.size > 4000:
            idx = np.random.default_rng(0).choice(idx, size=4000, replace=False)
        # Raises when a sampled subset happens to contain a single cluster.
        with contextlib.suppress(ValueError):
            sil = float(silhouette_score(vectors[idx], labels[idx], metric="cosine"))

    return ClusterResult(
        labels=labels,
        centroids=cents,
        n_cohorts=len(cents),
        raw_noise_rate=raw_noise,
        unassigned_rate=float(np.mean(labels == UNASSIGNED)),
        silhouette=sil,
        reassign_bar=reassign_bar,
        elapsed_s=time.perf_counter() - t0,
        sizes=sizes,
    )
