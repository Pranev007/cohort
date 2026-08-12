"""Generator invariants.

The benchmark is only worth anything if these hold. A corpus that leaks its
labels into the features, or whose "anomalies" are ordinary for their category,
produces numbers that mean nothing.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from cohort.config import SynthOrgConfig
from cohort.schema import POSTURE_FEATURES, AnomalyType
from cohort.synthorg import generate_corpus
from cohort.synthorg.anomalies import eligible_anomalies
from cohort.synthorg.categories import CATEGORIES
from cohort.synthorg.org import build_organisation


def _digest(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()


def test_generation_is_deterministic(tmp_path):
    cfg = SynthOrgConfig(n_employees=120, n_documents=400, seed=5)
    a = pd.read_parquet(generate_corpus(cfg, tmp_path / "a")["corpus"])
    b = pd.read_parquet(generate_corpus(cfg, tmp_path / "b")["corpus"])
    assert _digest(a) == _digest(b), "same seed must reproduce the corpus exactly"


def test_different_seeds_differ(tmp_path):
    a = pd.read_parquet(
        generate_corpus(SynthOrgConfig(n_employees=120, n_documents=400, seed=1), tmp_path / "a")[
            "corpus"
        ]
    )
    b = pd.read_parquet(
        generate_corpus(SynthOrgConfig(n_employees=120, n_documents=400, seed=2), tmp_path / "b")[
            "corpus"
        ]
    )
    assert _digest(a) != _digest(b)


def test_corpus_does_not_leak_labels(corpus):
    """The pipeline must not be able to see the answer."""
    forbidden = {"is_anomaly", "anomaly_type", "true_category"}
    assert forbidden.isdisjoint(corpus.columns)


def test_all_posture_features_present(corpus):
    missing = [f for f in POSTURE_FEATURES if f not in corpus.columns]
    assert not missing, f"generator did not emit: {missing}"


def test_anomaly_rate_close_to_configured(truth, small_config):
    observed = truth["is_anomaly"].mean()
    target = small_config.synthorg.anomaly_rate
    # Duplicates are injected at 1.5x, so the realised rate sits slightly above.
    assert target * 0.6 < observed < target * 2.0


def test_every_anomaly_type_is_represented(truth):
    present = set(truth.loc[truth.is_anomaly, "anomaly_type"])
    assert present == set(AnomalyType.all()), f"missing types: {set(AnomalyType.all()) - present}"


@pytest.mark.parametrize("cat", CATEGORIES, ids=lambda c: c.key)
def test_eligibility_never_injects_normal_behaviour(cat):
    """An 'anomaly' must actually be abnormal for the category it lands in.

    This is the guard on the benchmark's honesty. An earlier eligibility rule
    admitted mislabelling into a category that is one-third 'internal' natively,
    which made a perfectly ordinary state count as a missed detection.
    """
    eligible = eligible_anomalies(cat)
    pol = cat.policy

    if AnomalyType.EXTERNAL_LINK in eligible:
        assert pol.link_scope.get("anyone_with_link", 0.0) < 0.02
    if AnomalyType.MISLABELED_DOWN in eligible:
        downgrade = pol.label_tier.get("public", 0.0) + pol.label_tier.get("internal", 0.0)
        assert downgrade < 0.10
    if AnomalyType.THIRD_PARTY_ACCESS in eligible:
        assert pol.external_prob < 0.15
    if AnomalyType.OVERSHARED in eligible:
        assert sum(p for sel, p in pol.group_grants if sel == "all_employees") < 0.15


def test_group_expansion_is_transitive_and_cycle_safe():
    rng = np.random.default_rng(0)
    org = build_organisation(rng, n_employees=200, n_partners=2)

    everyone = org.expand_group(org.all_company_group)
    engineering = org.expand_group(org.dept_groups["Engineering"])
    assert engineering <= everyone, "nested department must be contained in All Employees"
    assert len(everyone) > len(engineering)

    # Introduce a cycle and confirm expansion still terminates.
    org.groups[org.all_company_group].child_group_ids.append(org.all_company_group)
    org._expand_cache.clear()
    assert org.expand_group(org.all_company_group)


def test_overshared_documents_reach_more_principals(corpus, truth):
    overshared = truth["anomaly_type"] == AnomalyType.OVERSHARED
    if overshared.sum() == 0:
        pytest.skip("no overshared anomalies in this small corpus")
    clean = ~truth["is_anomaly"].to_numpy()
    assert corpus.loc[overshared.to_numpy(), "n_principals"].mean() > (
        corpus.loc[clean, "n_principals"].mean()
    )
