"""Scorer invariants.

The most important test in the file is `test_score_equals_sum_of_attributions`.
The entire explainability claim rests on the score being decomposable, so if that
identity ever breaks, every narrative the tool produces becomes a plausible
fiction. It is asserted to floating-point tolerance rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cohort.config import ScoringConfig
from cohort.schema import CATEGORICAL_FEATURES, POSTURE_FEATURES
from cohort.scoring.baseline import _tail_prob, fit_baselines
from cohort.scoring.conformal import ConformalCalibrator
from cohort.scoring.engine import RiskEngine


# ---------------------------------------------------------------- tail prob
@given(
    x=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
    direction=st.sampled_from(["high", "low", "both"]),
)
def test_tail_prob_is_a_probability(x, direction):
    vals = np.sort(np.array([0.0, 1.0, 2.0, 5.0, 10.0, 100.0]))
    p = _tail_prob(vals, np.array([x]), direction)
    assert np.all(p > 0.0), "must stay strictly positive so -log p is finite"
    assert np.all(p <= 1.0)


def test_tail_prob_is_monotone_on_the_risky_side():
    vals = np.sort(np.random.default_rng(0).normal(size=500))
    xs = np.linspace(-3, 3, 40)
    p = _tail_prob(vals, xs, "high")
    assert np.all(np.diff(p) <= 1e-12), "higher values must be at least as extreme"


def test_tail_prob_handles_zero_inflated_features():
    """The case that broke the original MAD-based implementation."""
    vals = np.sort(np.array([0.0] * 95 + [1.0, 2.0, 3.0, 4.0, 5.0]))
    p_zero = _tail_prob(vals, np.array([0.0]), "high")[0]
    p_five = _tail_prob(vals, np.array([5.0]), "high")[0]
    assert p_zero > 0.9, "the common value must be unsurprising"
    assert p_five < 0.05, "the rare value must be surprising"


# ----------------------------------------------------------------- baselines
def test_shrinkage_pulls_small_cohorts_toward_global(scored):
    df, labels, _ = scored
    posture = df[POSTURE_FEATURES]

    strong = fit_baselines(posture, labels, ScoringConfig(shrinkage_kappa=0.0))
    shrunk = fit_baselines(posture, labels, ScoringConfig(shrinkage_kappa=1e6))

    cid = int(pd.Series(labels).value_counts().idxmin())
    feature = CATEGORICAL_FEATURES[0]
    g = shrunk.global_baseline.categorical[feature].probs

    local = strong.cohorts[cid].categorical[feature].probs
    pulled = shrunk.cohorts[cid].categorical[feature].probs

    d_local = sum(abs(local[k] - g[k]) for k in g)
    d_pulled = sum(abs(pulled[k] - g[k]) for k in g)
    assert d_pulled < d_local, "heavy shrinkage must move the cohort toward global"


def test_unseen_categorical_value_is_finite(scored):
    df, labels, _ = scored
    model = fit_baselines(df[POSTURE_FEATURES], labels, ScoringConfig())

    row = df[POSTURE_FEATURES].iloc[[0]].copy()
    row.loc[row.index[0], "repo_type"] = "some_repo_nobody_has_ever_seen"
    s = model.surprisal_frame(row, np.array([labels[0]]))
    assert np.isfinite(s.to_numpy()).all()
    assert (s.to_numpy() >= 0).all()


def test_surprisal_is_non_negative(scored):
    df, labels, _ = scored
    model = fit_baselines(df[POSTURE_FEATURES], labels, ScoringConfig())
    S = model.surprisal_frame(df[POSTURE_FEATURES], labels)
    assert (S.to_numpy() >= -1e-9).all()
    assert np.isfinite(S.to_numpy()).all()


# -------------------------------------------------------------------- engine
def test_score_equals_sum_of_attributions(scored):
    """The explainability contract, asserted numerically."""
    df, labels, _ = scored
    cfg = ScoringConfig(aggregation="sum", iforest_weight=0.0, robust_passes=0)
    res = RiskEngine(cfg).fit_score(df, labels, df["doc_id"].to_numpy())
    np.testing.assert_allclose(
        res.risk, res.attributions.to_numpy().sum(axis=1), rtol=1e-9, atol=1e-9
    )


def test_topk_score_equals_sum_of_top_k_attributions(scored):
    df, labels, _ = scored
    cfg = ScoringConfig(aggregation="topk", top_k=2, iforest_weight=0.0, robust_passes=0)
    res = RiskEngine(cfg).fit_score(df, labels, df["doc_id"].to_numpy())
    M = res.attributions.to_numpy()
    expected = np.sort(M, axis=1)[:, -2:].sum(axis=1)
    np.testing.assert_allclose(res.risk, expected, rtol=1e-9, atol=1e-9)


def test_missing_feature_raises(scored):
    df, labels, _ = scored
    broken = df.drop(columns=["link_scope"])
    with pytest.raises(ValueError, match="posture features missing"):
        RiskEngine(ScoringConfig()).fit(broken, labels)


def test_score_before_fit_raises(scored):
    df, labels, _ = scored
    with pytest.raises(RuntimeError, match="fit"):
        RiskEngine(ScoringConfig()).score(df, labels, df["doc_id"].to_numpy())


def test_scoring_is_deterministic(scored):
    df, labels, _ = scored
    a = RiskEngine(ScoringConfig()).fit_score(df, labels, df["doc_id"].to_numpy(), rng_seed=3)
    b = RiskEngine(ScoringConfig()).fit_score(df, labels, df["doc_id"].to_numpy(), rng_seed=3)
    np.testing.assert_allclose(a.risk, b.risk)


# ---------------------------------------------------------------- conformal
def test_conformal_flag_rate_tracks_alpha():
    """On clean exchangeable data the realised flag rate should sit near alpha."""
    rng = np.random.default_rng(0)
    scores = rng.normal(size=8000)
    cohorts = np.zeros(8000, dtype=int)

    cfg = ScoringConfig(conformal_alpha=0.05, min_cohort_for_local_conformal=10**9)
    cal = ConformalCalibrator(cfg).fit(scores[:4000], cohorts[:4000])
    p = cal.p_values(scores[4000:], cohorts[4000:])
    rate = float((p <= 0.05).mean())
    assert 0.03 < rate < 0.07, f"flag rate {rate:.3f} far from nominal 0.05"


def test_conformal_p_values_are_in_unit_interval():
    rng = np.random.default_rng(1)
    scores = rng.gamma(2.0, size=2000)
    cohorts = rng.integers(0, 4, size=2000)
    cal = ConformalCalibrator(ScoringConfig()).fit(scores, cohorts)
    p = cal.p_values(scores, cohorts)
    assert np.all(p > 0) and np.all(p <= 1.0)


@settings(deadline=None, max_examples=25)
@given(alpha=st.floats(min_value=0.005, max_value=0.5))
def test_higher_alpha_flags_at_least_as_much(alpha):
    rng = np.random.default_rng(2)
    scores = rng.normal(size=3000)
    cohorts = np.zeros(3000, dtype=int)
    cal = ConformalCalibrator(ScoringConfig(min_cohort_for_local_conformal=10**9)).fit(
        scores, cohorts
    )
    p = cal.p_values(scores, cohorts)
    assert (p <= alpha).sum() >= (p <= alpha / 2).sum()
