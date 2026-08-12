"""Lineage, attribution, and counterfactual remediation."""

from __future__ import annotations

import numpy as np
import pytest

from cohort.config import ExplainConfig, LineageConfig, ScoringConfig
from cohort.explain.attribution import top_attributions
from cohort.explain.counterfactual import plan_remediation
from cohort.explain.narrate import narrate
from cohort.lineage import build_lineage
from cohort.schema import MUTABLE_FEATURES, POSTURE_FEATURES
from cohort.scoring.engine import RiskEngine


# ------------------------------------------------------------------ lineage
def test_identical_documents_share_a_family():
    text = "The supplier shall provide services described in each statement of work executed hereunder."
    res = build_lineage(
        [text, text, "Completely different content about kubernetes ingress rules."],
        LineageConfig(),
    )
    assert res.family_of[0] == res.family_of[1]
    assert res.family_of[2] != res.family_of[0]


def test_lightly_edited_copy_is_detected():
    original = " ".join(
        f"Clause {i}: the receiving party shall protect confidential information disclosed."
        for i in range(20)
    )
    edited = original.replace("Clause 3:", "Clause 3 (revised):") + " Updated after review."
    res = build_lineage([original, edited], LineageConfig())
    assert res.family_of[0] == res.family_of[1]


def test_unrelated_documents_stay_separate():
    docs = [
        "Quarterly revenue grew driven by favourable product mix and lower unit costs.",
        "The deployment manifest pins the container image to a digest from the release branch.",
        "Offer of employment: your annual base salary will be reviewed each performance cycle.",
    ]
    res = build_lineage(docs, LineageConfig())
    assert len({int(f) for f in res.family_of}) == 3


def test_dup_count_excludes_self():
    text = "identical shingled content for the minhash banding test across many words here now"
    res = build_lineage([text] * 4, LineageConfig())
    assert list(res.dup_count) == [3.0, 3.0, 3.0, 3.0]


def test_empty_corpus_is_handled():
    res = build_lineage([], LineageConfig())
    assert res.family_of.size == 0


# -------------------------------------------------------------- attribution
def test_attributions_are_ranked_and_bounded(scored):
    df, labels, _ = scored
    engine = RiskEngine(ScoringConfig()).fit(df, labels)
    res = engine.score(df, labels, df["doc_id"].to_numpy())

    i = int(np.argmax(res.risk))
    attrs = top_attributions(
        df.iloc[i], res.attributions.iloc[i], engine.baselines, int(labels[i]), k=3
    )
    assert attrs, "the top-scoring document must have at least one driver"
    assert [a.surprisal for a in attrs] == sorted((a.surprisal for a in attrs), reverse=True)
    for a in attrs:
        assert 0.0 <= a.peer_share <= 1.0
        assert a.peer_n > 0


def test_narrative_quotes_the_peer_count(scored):
    df, labels, _ = scored
    engine = RiskEngine(ScoringConfig()).fit(df, labels)
    res = engine.score(df, labels, df["doc_id"].to_numpy())
    i = int(np.argmax(res.risk))
    attrs = top_attributions(df.iloc[i], res.attributions.iloc[i], engine.baselines, int(labels[i]))
    text = narrate(
        "test cohort", attrs, float(res.risk[i]), float(res.p_values[i]), ExplainConfig()
    )
    assert "peer" in text.lower()
    assert str(attrs[0].peer_n) in text
    assert "nats" in text


def test_narrative_handles_no_drivers():
    text = narrate("empty cohort", [], 0.0, 1.0, ExplainConfig())
    assert "No individual posture feature" in text


# ------------------------------------------------------------ counterfactual
def test_remediation_reduces_measured_risk(scored):
    df, labels, _ = scored
    engine = RiskEngine(ScoringConfig()).fit(df, labels)
    res = engine.score(df, labels, df["doc_id"].to_numpy())
    target = engine.calibrator.threshold()

    def score_fn(frame, cids):
        return engine.risk_of(frame, cids)

    improved = 0
    for i in np.argsort(-res.risk)[:25]:
        plan = plan_remediation(
            df.iloc[i],
            engine.baselines,
            int(labels[i]),
            score_fn,
            target,
            ExplainConfig(),
            surprisal_row=res.attributions.iloc[i],
        )
        if plan.edits:
            assert plan.residual_score < plan.original_score + 1e-9
            improved += 1
    assert improved > 0, "no remediation plan was produced for any top finding"


def test_remediation_only_touches_mutable_features(scored):
    df, labels, _ = scored
    engine = RiskEngine(ScoringConfig()).fit(df, labels)
    res = engine.score(df, labels, df["doc_id"].to_numpy())

    def score_fn(frame, cids):
        return engine.risk_of(frame, cids)

    for i in np.argsort(-res.risk)[:15]:
        plan = plan_remediation(
            df.iloc[i],
            engine.baselines,
            int(labels[i]),
            score_fn,
            engine.calibrator.threshold(),
            ExplainConfig(),
        )
        for e in plan.edits:
            assert e.feature in MUTABLE_FEATURES, f"proposed editing immutable {e.feature}"


def test_residual_score_is_verified_not_predicted(scored):
    """Re-scoring the edited row must reproduce the plan's residual exactly."""
    df, labels, _ = scored
    engine = RiskEngine(ScoringConfig()).fit(df, labels)
    res = engine.score(df, labels, df["doc_id"].to_numpy())

    def score_fn(frame, cids):
        return engine.risk_of(frame, cids)

    for i in np.argsort(-res.risk)[:10]:
        row = df.iloc[i]
        plan = plan_remediation(
            row,
            engine.baselines,
            int(labels[i]),
            score_fn,
            engine.calibrator.threshold(),
            ExplainConfig(),
        )
        if not plan.edits:
            continue
        edited = row[POSTURE_FEATURES].copy()
        for e in plan.edits:
            edited[e.feature] = e.to_value
            if e.feature == "has_external_principal" and not e.to_value:
                edited["n_external_domains"] = 0.0
        actual = float(score_fn(edited.to_frame().T, np.array([int(labels[i])]))[0])
        assert abs(actual - plan.residual_score) < 1e-6
        return
    pytest.skip("no plan with edits among the top findings")
