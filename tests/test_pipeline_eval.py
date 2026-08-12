"""End-to-end pipeline and evaluation.

The regression guard at the bottom is the one that matters in CI: it asserts that
peer-group baselining still beats a single global baseline by a wide margin. That
is the project's central claim, and a refactor that quietly breaks it should fail
the build rather than ship.
"""

from __future__ import annotations

import numpy as np
import pytest

from cohort.config import ScoringConfig
from cohort.evaluate.metrics import anomaly_metrics, cluster_metrics
from cohort.evaluate.report import run_evaluation
from cohort.pipeline import run_pipeline, write_artifacts
from cohort.scoring.engine import RiskEngine


def test_metrics_on_a_perfect_and_a_random_ranker():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    perfect = np.array([9, 8, 1, 1, 1, 1, 1, 1, 1, 1], dtype=float)
    m = anomaly_metrics(y, perfect, ks=(2,))
    assert m.pr_auc == pytest.approx(1.0)
    assert m.precision_at[2] == pytest.approx(1.0)

    inverted = -perfect
    assert anomaly_metrics(y, inverted, ks=(2,)).precision_at[2] == pytest.approx(0.0)


def test_cluster_metrics_ignore_the_unassigned_bucket():
    truth = np.array(["a", "a", "b", "b", "c"])
    labels = np.array([0, 0, 1, 1, -1])
    cm = cluster_metrics(truth, labels, unassigned_rate=0.2, silhouette=0.5)
    assert cm.n_cohorts == 2
    assert cm.ari == pytest.approx(1.0)


def test_per_type_metrics_exclude_other_anomaly_types():
    """Another anomaly type must not count as a false positive for this one."""
    y = np.array([1, 1, 0, 0, 0], dtype=bool)
    types = np.array(["A", "B", "", "", ""])
    scores = np.array([5.0, 4.0, 1.0, 1.0, 1.0])
    m = anomaly_metrics(y, scores, types, ks=(1,))
    # Type A is ranked top among {A + clean}; type B's presence must not hurt it.
    assert m.per_type["A"]["pr_auc"] == pytest.approx(1.0)


@pytest.mark.slow
def test_pipeline_runs_end_to_end(small_config, corpus_paths, tmp_path):
    small_config.paths.root = tmp_path / "artifacts"
    result = run_pipeline(small_config, corpus_paths["corpus"], max_findings=50)

    assert len(result.corpus) == small_config.synthorg.n_documents
    assert result.cohorts.n_cohorts >= 2
    assert len(result.findings) == 50
    assert result.findings["risk_score"].is_monotonic_decreasing
    assert result.findings["narrative"].str.len().min() > 20

    written = write_artifacts(result, small_config)
    for path in written.values():
        assert path.exists() and path.stat().st_size > 0

    for stage in ("embed", "cluster", "lineage", "fit", "score", "explain", "total"):
        assert stage in result.timings


@pytest.mark.slow
def test_evaluation_produces_a_report(small_config, corpus_paths):
    report = run_evaluation(
        small_config,
        corpus_paths["corpus"],
        corpus_paths["ground_truth"],
        with_ablations=False,
        with_sensitivity=False,
    )
    md = report.to_markdown()
    assert "PR-AUC" in md and "Per anomaly type" in md
    assert 0.0 <= report.headline.pr_auc <= 1.0
    assert report.headline.n_positive > 0


@pytest.mark.slow
def test_peer_grouping_beats_global_baseline(medium_scored):
    """The project's central claim, guarded in CI.

    Peer-relative scoring must substantially outperform a single corpus-wide
    baseline. If this margin collapses, the premise has broken and no amount of
    passing unit tests makes the tool worth running.

    Measured margin scales with corpus size — ~1.9x at 4k documents, ~2.9x at
    15k — so the threshold here is set below the 4k figure, not the headline one.
    """
    df, labels, truth = medium_scored
    y = truth["is_anomaly"].to_numpy().astype(bool)
    ids = df["doc_id"].to_numpy()
    cfg = ScoringConfig()

    peer = RiskEngine(cfg).fit_score(df, labels, ids).risk
    glob = RiskEngine(cfg).fit_score(df, np.zeros(len(df), dtype=int), ids).risk

    peer_ap = anomaly_metrics(y, peer).pr_auc
    global_ap = anomaly_metrics(y, glob).pr_auc

    assert peer_ap > global_ap * 1.5, (
        f"peer baselining ({peer_ap:.3f}) must clearly beat global ({global_ap:.3f})"
    )


@pytest.mark.slow
def test_semantic_grouping_beats_random_grouping(medium_scored):
    """Separates 'grouping helps' from 'grouping by meaning helps'."""
    df, labels, truth = medium_scored
    y = truth["is_anomaly"].to_numpy().astype(bool)
    ids = df["doc_id"].to_numpy()
    cfg = ScoringConfig()

    shuffled = labels.copy()
    np.random.default_rng(0).shuffle(shuffled)

    real_ap = anomaly_metrics(y, RiskEngine(cfg).fit_score(df, labels, ids).risk).pr_auc
    rand_ap = anomaly_metrics(y, RiskEngine(cfg).fit_score(df, shuffled, ids).risk).pr_auc
    assert real_ap > rand_ap * 1.4
