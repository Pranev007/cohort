"""MinHash + LSH near-duplicate detection.

Why a security tool cares about duplicates: permissions do not travel with
content. Someone downloads the board pack, edits a slide, and re-uploads it to a
folder with different inheritance. The original stays locked down and the copy is
the exposure. Cohort detects the copy, links it to its family, and lets risk be
read across the family rather than per file.

Implemented directly on numpy rather than pulling in `datasketch`, for the same
reason the default embedding backend is TF-IDF: the core pipeline should install
and run with nothing but numpy/scipy/sklearn/pandas.

Cost is O(n * num_perm) to sign plus roughly linear candidate generation from the
banding, instead of the O(n^2) pairwise comparison the naive version implies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from cohort.config import LineageConfig

_MERSENNE = (1 << 61) - 1
_MAXHASH = (1 << 32) - 1
_TOKEN = re.compile(r"\w+")


@dataclass
class LineageResult:
    #: doc index -> family id. Singletons get their own family.
    family_of: np.ndarray
    #: family id -> member indices
    families: dict[int, list[int]] = field(default_factory=dict)
    #: per-document count of *other* members in its family
    dup_count: np.ndarray = field(default_factory=lambda: np.array([]))
    n_families_gt1: int = 0
    largest_family: int = 0


def _shingles(text: str, k: int) -> set[int]:
    """Hash the document's word k-grams to 32-bit ints."""
    tokens = _TOKEN.findall(text.lower())
    if len(tokens) < k:
        return {hash(" ".join(tokens)) & _MAXHASH} if tokens else set()
    return {hash(" ".join(tokens[i : i + k])) & _MAXHASH for i in range(len(tokens) - k + 1)}


def _signatures(texts: list[str], cfg: LineageConfig, seed: int = 7) -> np.ndarray:
    """(n_docs, num_perm) uint64 MinHash signature matrix."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _MERSENNE, size=cfg.num_perm, dtype=np.uint64)
    b = rng.integers(0, _MERSENNE, size=cfg.num_perm, dtype=np.uint64)

    sig = np.full((len(texts), cfg.num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    for i, text in enumerate(texts):
        sh = _shingles(text, cfg.shingle_size)
        if not sh:
            continue
        h = np.fromiter(sh, dtype=np.uint64, count=len(sh))
        # Universal hashing: (a*h + b) mod (2^61 - 1), one column per permutation.
        perm = (np.outer(h, a) + b) % np.uint64(_MERSENNE)
        sig[i] = perm.min(axis=0)
    return sig


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[max(rx, ry)] = min(rx, ry)


def build_lineage(texts: list[str], cfg: LineageConfig) -> LineageResult:
    n = len(texts)
    if n == 0:
        return LineageResult(family_of=np.array([], dtype=int))

    sig = _signatures(texts, cfg)
    rows_per_band = max(1, cfg.num_perm // cfg.n_bands)

    # LSH banding: documents sharing an identical band are candidate pairs. Two
    # documents agreeing on every row of any one band is overwhelmingly likely to
    # mean high Jaccard similarity, which is what keeps this sub-quadratic.
    uf = _UnionFind(n)
    for band in range(cfg.n_bands):
        lo = band * rows_per_band
        hi = min(lo + rows_per_band, cfg.num_perm)
        if lo >= hi:
            break
        buckets: dict[bytes, list[int]] = {}
        chunk = sig[:, lo:hi]
        for i in range(n):
            key = chunk[i].tobytes()
            buckets.setdefault(key, []).append(i)

        for members in buckets.values():
            if len(members) < 2:
                continue
            anchor = members[0]
            # Verify against the full signature before merging: banding produces
            # candidates, not conclusions.
            for other in members[1:]:
                est = float(np.mean(sig[anchor] == sig[other]))
                if est >= cfg.threshold:
                    uf.union(anchor, other)

    family_of = np.array([uf.find(i) for i in range(n)], dtype=int)
    families: dict[int, list[int]] = {}
    for i, fam in enumerate(family_of):
        families.setdefault(int(fam), []).append(i)

    dup_count = np.array([len(families[int(f)]) - 1 for f in family_of], dtype=float)
    multi = [m for m in families.values() if len(m) > 1]

    return LineageResult(
        family_of=family_of,
        families=families,
        dup_count=dup_count,
        n_families_gt1=len(multi),
        largest_family=max((len(m) for m in multi), default=0),
    )
