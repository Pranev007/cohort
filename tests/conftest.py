"""Shared fixtures. Corpora here are deliberately small so the suite stays fast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cohort.config import CohortConfig, SynthOrgConfig
from cohort.lineage import build_lineage
from cohort.semantic.cluster import discover_cohorts
from cohort.semantic.embed import embed_documents
from cohort.synthorg import generate_corpus


@pytest.fixture(scope="session")
def small_config() -> CohortConfig:
    cfg = CohortConfig()
    cfg.synthorg = SynthOrgConfig(n_employees=250, n_documents=1200, seed=99)
    cfg.semantic.min_cluster_size = 40
    cfg.semantic.min_cluster_size_frac = None
    return cfg


@pytest.fixture(scope="session")
def corpus_paths(tmp_path_factory, small_config: CohortConfig) -> dict:
    out = tmp_path_factory.mktemp("corpus")
    return generate_corpus(small_config.synthorg, out)


@pytest.fixture(scope="session")
def corpus(corpus_paths) -> pd.DataFrame:
    return pd.read_parquet(corpus_paths["corpus"])


@pytest.fixture(scope="session")
def truth(corpus_paths, corpus) -> pd.DataFrame:
    return pd.read_parquet(corpus_paths["ground_truth"]).set_index("doc_id").loc[corpus.doc_id]


def _prepare(df: pd.DataFrame, cfg: CohortConfig):
    texts = (df["title"] + ". " + df["body"]).tolist()
    vectors = embed_documents(texts, cfg.semantic).vectors
    clusters = discover_cohorts(vectors, cfg.semantic)
    out = df.copy()
    out["dup_count"] = build_lineage(out["body"].tolist(), cfg.lineage).dup_count
    return out, clusters.labels, np.asarray(vectors)


@pytest.fixture(scope="session")
def scored(corpus, small_config):
    """Corpus with cohorts assigned and dup_count populated."""
    return _prepare(corpus, small_config)


@pytest.fixture(scope="session")
def medium_config() -> CohortConfig:
    cfg = CohortConfig()
    cfg.synthorg = SynthOrgConfig(n_employees=400, n_documents=4000, seed=99)
    return cfg


@pytest.fixture(scope="session")
def medium_scored(tmp_path_factory, medium_config: CohortConfig):
    """A corpus large enough to actually test the peer-vs-global claim.

    The advantage of peer baselining scales with how many peers each cohort has:
    measured at roughly 1.1x on 1.2k documents, 1.9x on 4k and 2.9x on 15k. On the
    1.2k-document fixture used elsewhere the effect is real but within noise, so
    asserting it there would produce a flaky test rather than a meaningful guard.
    """
    out = tmp_path_factory.mktemp("corpus_medium")
    paths = generate_corpus(medium_config.synthorg, out)
    df = pd.read_parquet(paths["corpus"])
    truth = pd.read_parquet(paths["ground_truth"]).set_index("doc_id").loc[df.doc_id]
    prepared, labels, _vectors = _prepare(df, medium_config)
    return prepared, labels, truth
