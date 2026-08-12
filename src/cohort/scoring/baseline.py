r"""Per-cohort posture baselines and the additive surprisal score.

The whole method in one line: **a document is risky to the extent that its
security posture is improbable among documents that mean the same thing.**

For a document :math:`d` in cohort :math:`c`,

.. math::

    S(d) = \sum_{j} s_j(d), \qquad
    s_j(d) = \max\!\big(0,\; -\log \tilde{p}_j(d) - \bar{s}_j(c)\big)

where :math:`\tilde{p}_j` is the probability of a value at least as extreme as
this one among peers, and :math:`\bar{s}_j(c)` is the expected surprisal for an
ordinary member of the cohort. Subtracting that expectation is what makes a
typical document score ~0 rather than accumulating a noise floor across fifteen
features.

**Categorical features** use Dirichlet-smoothed probabilities, with the cohort's
entropy as the expectation term — the exact analytic value of :math:`\bar{s}_j`.

**Continuous features** use the *empirical tail probability* within the cohort,
not a fitted Gaussian. This is the single most consequential modelling choice in
the file and it was made after measurement, not before: a location-scale model
put 61% of the score on ``n_principals`` and ``accessor_dept_entropy`` while both
ranked anomalies no better than chance. The reason is that those features are
multi-modal by construction — a document either carries a broad group grant or it
does not, so the distribution has two humps and its median sits in the valley
between them. Perfectly ordinary documents in the upper mode looked extreme.
An ECDF has no such assumption: it handles multi-modality, zero-inflation
(``n_external_domains`` is 0 for most documents) and discreteness alike.

Both families shrink toward the corpus-wide baseline by empirical Bayes with
weight :math:`w_c = n_c/(n_c + \kappa)`, which is what stops a nine-document
cohort from declaring its own accidents to be normal.

The sum is the score and each term is its own exact attribution. There is no
post-hoc explainer approximating a black box — the explanation *is* the model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cohort.config import ScoringConfig
from cohort.schema import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_DIRECTION,
)

_EPS = 1e-12


@dataclass(slots=True)
class CategoricalStat:
    probs: dict[str, float]  # shrunk, smoothed P(value | cohort)
    entropy: float  # expected surprisal under those probs
    n_values: int  # vocabulary size, for unseen-value handling


@dataclass(slots=True)
class ContinuousStat:
    #: Sorted cohort values — the empirical distribution the document is judged against.
    sorted_values: np.ndarray
    direction: str  # high | low | both
    informative: bool  # False when the feature is constant corpus-wide
    #: Expected surprisal for an ordinary member of this cohort, subtracted at
    #: scoring time. The continuous analogue of subtracting the entropy.
    offset: float = 0.0


@dataclass
class CohortBaseline:
    """What normal looks like for one cohort."""

    cohort_id: int
    n: int
    weight: float  # empirical-Bayes w_c
    categorical: dict[str, CategoricalStat] = field(default_factory=dict)
    continuous: dict[str, ContinuousStat] = field(default_factory=dict)


def _as_key(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return str(value)


def _tail_prob(sorted_vals: np.ndarray, x: np.ndarray, direction: str) -> np.ndarray:
    """P(peer value at least as extreme as x), Laplace-smoothed.

    Add-one smoothing on both numerator and denominator keeps the result strictly
    inside (0, 1], so ``-log p`` is always finite even for a value beyond every
    observation in the cohort.
    """
    n = sorted_vals.size
    if n == 0:
        return np.ones_like(x, dtype=float)

    if direction == "high":
        ge = n - np.searchsorted(sorted_vals, x, side="left")
        return (ge + 1.0) / (n + 1.0)
    if direction == "low":
        le = np.searchsorted(sorted_vals, x, side="right")
        return (le + 1.0) / (n + 1.0)

    ge = n - np.searchsorted(sorted_vals, x, side="left")
    le = np.searchsorted(sorted_vals, x, side="right")
    two_sided = 2.0 * (np.minimum(ge, le) + 1.0) / (n + 1.0)
    return np.minimum(1.0, two_sided)


@dataclass
class BaselineModel:
    """Baselines for every cohort, plus the global fallback they shrink toward."""

    cohorts: dict[int, CohortBaseline]
    global_baseline: CohortBaseline
    cfg: ScoringConfig

    def for_cohort(self, cohort_id: int) -> CohortBaseline:
        return self.cohorts.get(cohort_id, self.global_baseline)

    # -- scoring -----------------------------------------------------------
    def _categorical_surprisal(
        self, values: pd.Series, base: CohortBaseline, feature: str
    ) -> np.ndarray:
        stat = base.categorical[feature]
        fallback = self.cfg.dirichlet_alpha / (
            base.n + self.cfg.dirichlet_alpha * (stat.n_values + 1)
        )
        p = values.map(_as_key).map(stat.probs).fillna(fallback).to_numpy(dtype=float)
        s = -np.log(np.maximum(p, _EPS)) - stat.entropy
        return np.clip(s, 0.0, self.cfg.max_feature_surprisal)

    def _continuous_surprisal(
        self, values: pd.Series, base: CohortBaseline, feature: str
    ) -> np.ndarray:
        stat = base.continuous[feature]
        x = values.to_numpy(dtype=float)
        if not stat.informative:
            return np.zeros_like(x)

        g = self.global_baseline.continuous[feature]
        p_local = _tail_prob(stat.sorted_values, x, stat.direction)
        p_global = _tail_prob(g.sorted_values, x, stat.direction)
        w = base.weight
        p = w * p_local + (1.0 - w) * p_global

        s = -np.log(np.maximum(p, _EPS)) - stat.offset
        return np.clip(s, 0.0, self.cfg.max_feature_surprisal)

    def surprisal_frame(self, df: pd.DataFrame, cohort_ids: np.ndarray) -> pd.DataFrame:
        """Per-feature excess surprisal, in nats. One column per posture feature.

        Computed cohort-block at a time so every operation is vectorised; a
        per-row Python loop over 15 features costs roughly 20x more.
        """
        cols = CATEGORICAL_FEATURES + CONTINUOUS_FEATURES
        out = pd.DataFrame(0.0, index=df.index, columns=cols)

        for cid in np.unique(cohort_ids):
            mask = cohort_ids == int(cid)
            base = self.for_cohort(int(cid))
            sub = df.loc[mask]
            for f in CATEGORICAL_FEATURES:
                out.loc[mask, f] = self._categorical_surprisal(sub[f], base, f)
            for f in CONTINUOUS_FEATURES:
                out.loc[mask, f] = self._continuous_surprisal(sub[f], base, f)

        return out


def _fit_categorical(
    values: pd.Series, alpha: float, vocab: list[str]
) -> tuple[dict[str, float], float]:
    counts = values.map(_as_key).value_counts()
    n = float(counts.sum())
    denom = n + alpha * len(vocab)
    probs = {val: (float(counts.get(val, 0)) + alpha) / denom for val in vocab}
    entropy = -sum(p * math.log(max(p, _EPS)) for p in probs.values())
    return probs, entropy


def _fit_offset(
    x: np.ndarray,
    local_sorted: np.ndarray,
    global_sorted: np.ndarray,
    weight: float,
    direction: str,
    trim: float = 0.95,
) -> float:
    """Expected surprisal for an ordinary member of this cohort.

    Estimated empirically over the cohort's own members. The top 5% is trimmed
    first: those members include the anomalies being hunted, and leaving them in
    would inflate the offset and partially subtract away the signal.
    """
    if x.size == 0:
        return 0.0
    p = weight * _tail_prob(local_sorted, x, direction) + (1.0 - weight) * _tail_prob(
        global_sorted, x, direction
    )
    s = -np.log(np.maximum(p, _EPS))
    cut = float(np.quantile(s, trim))
    kept = s[s <= cut]
    return float(kept.mean()) if kept.size else 0.0


def fit_baselines(
    df: pd.DataFrame,
    cohort_ids: np.ndarray,
    cfg: ScoringConfig,
) -> BaselineModel:
    """Fit one baseline per cohort, shrunk toward the corpus-wide baseline.

    `df` must contain every column in POSTURE_FEATURES. It must **not** contain a
    label; nothing here is supervised.
    """
    vocab = {f: sorted({_as_key(v) for v in df[f]}) for f in CATEGORICAL_FEATURES}

    # ---- global baseline (the shrinkage target) ----------------------------
    g_cat: dict[str, CategoricalStat] = {}
    for f in CATEGORICAL_FEATURES:
        probs, ent = _fit_categorical(df[f], cfg.dirichlet_alpha, vocab[f])
        g_cat[f] = CategoricalStat(probs, ent, len(vocab[f]))

    g_cont: dict[str, ContinuousStat] = {}
    for f in CONTINUOUS_FEATURES:
        x = df[f].to_numpy(dtype=float)
        sorted_vals = np.sort(x)
        # A feature that never varies carries no information about anything.
        informative = bool(sorted_vals.size and sorted_vals[0] != sorted_vals[-1])
        stat = ContinuousStat(
            sorted_values=sorted_vals,
            direction=FEATURE_DIRECTION.get(f, "both"),
            informative=informative,
        )
        if informative:
            stat.offset = _fit_offset(x, sorted_vals, sorted_vals, 1.0, stat.direction)
        g_cont[f] = stat

    global_baseline = CohortBaseline(
        cohort_id=-999, n=len(df), weight=1.0, categorical=g_cat, continuous=g_cont
    )

    # ---- per-cohort baselines ---------------------------------------------
    cohorts: dict[int, CohortBaseline] = {}
    for cid in np.unique(cohort_ids):
        cid = int(cid)
        mask = cohort_ids == cid
        sub = df.loc[mask]
        n = int(mask.sum())
        w = n / (n + cfg.shrinkage_kappa)

        cat: dict[str, CategoricalStat] = {}
        for f in CATEGORICAL_FEATURES:
            local, _ = _fit_categorical(sub[f], cfg.dirichlet_alpha, vocab[f])
            g = g_cat[f].probs
            blended = {val: w * local[val] + (1.0 - w) * g[val] for val in vocab[f]}
            total = sum(blended.values()) or 1.0
            blended = {k: v / total for k, v in blended.items()}
            ent = -sum(p * math.log(max(p, _EPS)) for p in blended.values())
            cat[f] = CategoricalStat(blended, ent, len(vocab[f]))

        cont: dict[str, ContinuousStat] = {}
        for f in CONTINUOUS_FEATURES:
            x = sub[f].to_numpy(dtype=float)
            sorted_vals = np.sort(x)
            g_stat = g_cont[f]
            stat = ContinuousStat(
                sorted_values=sorted_vals,
                direction=g_stat.direction,
                informative=g_stat.informative,
            )
            if stat.informative:
                stat.offset = _fit_offset(x, sorted_vals, g_stat.sorted_values, w, stat.direction)
            cont[f] = stat

        cohorts[cid] = CohortBaseline(cid, n, w, cat, cont)

    return BaselineModel(cohorts=cohorts, global_baseline=global_baseline, cfg=cfg)
