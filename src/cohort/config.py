"""Configuration objects for the Cohort pipeline.

Every knob that changes a result lives here, so a run is reproducible from
`configs/*.yaml` + a seed. Nothing in the pipeline reads a magic constant
defined at its call site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

EmbeddingBackend = Literal["tfidf-svd", "sentence-transformers"]
ReducerKind = Literal["svd", "umap", "none"]


class SynthOrgConfig(BaseModel):
    """Synthetic enterprise generator."""

    seed: int = 1337
    n_employees: int = 2000
    n_documents: int = 15000

    #: Fraction of documents that receive an injected posture anomaly.
    anomaly_rate: float = 0.02
    #: Fraction of contract-family documents emitted in German/French.
    multilingual_fraction: float = 0.08
    #: Fraction of documents that are near-duplicates of an existing document.
    duplicate_rate: float = 0.06
    #: Fraction of documents drawn from no clean category (realistic corpus noise).
    noise_rate: float = 0.04

    n_external_partners: int = 4
    n_departments: int = 9


class SemanticConfig(BaseModel):
    """Embedding + category discovery."""

    backend: EmbeddingBackend = "tfidf-svd"
    #: Only consulted when backend == "sentence-transformers".
    model_name: str = "BAAI/bge-m3"

    #: TF-IDF vocabulary ceiling for the offline backend.
    max_features: int = 60_000
    #: Dimensionality after reduction; also the vector width stored per document.
    dim: int = 256
    reducer: ReducerKind = "svd"
    #: Density-based clustering degrades badly in high dimensions, so cohort
    #: discovery runs on a further-reduced view while the stored vectors keep
    #: their full width for centroid similarity and lineage.
    cluster_dim: int = 40

    #: HDBSCAN. min_cluster_size is the single most result-sensitive knob here:
    #: too small and one document type shatters into a dozen cohorts, each with
    #: too few members to baseline against; too large and distinct types get
    #: merged and their differing norms average out.
    #:
    #: Expressed as a fraction of corpus size by default so the same config works
    #: on 5k and 500k documents. `min_cluster_size` is the floor and the value
    #: used when the fraction is disabled.
    min_cluster_size: int = 25
    min_cluster_size_frac: float | None = 0.015
    min_samples: int | None = 8
    cluster_selection_epsilon: float = 0.0

    def effective_min_cluster_size(self, n_docs: int) -> int:
        if self.min_cluster_size_frac is None:
            return self.min_cluster_size
        return max(self.min_cluster_size, int(self.min_cluster_size_frac * n_docs))

    #: Documents HDBSCAN marks as noise (-1) are re-attached to their nearest
    #: cohort when they are close enough to it; otherwise they land in the
    #: UNASSIGNED cohort and are scored against a global model.
    #:
    #: "Close enough" is calibrated against the data rather than fixed, because an
    #: absolute cosine bar does not transfer between embedding geometries. On the
    #: synthetic corpus a noise document sits ~0.6 cosine from the nearest
    #: centroid; on real newsgroup text the median is 0.307, so the original
    #: hard-coded 0.55 rescued 1.9% of outliers and left 85% of a real corpus
    #: unassigned. The percentile form asks a scale-free question instead: is this
    #: document as close to the cohort as the cohort's own least typical members?
    noise_reassign_percentile: float = 0.10
    #: Absolute cosine floor. Set to override the percentile rule entirely.
    noise_reassign_threshold: float | None = None

    #: Number of keyphrases used to auto-name a discovered cohort.
    name_top_k: int = 6


class ScoringConfig(BaseModel):
    """Peer-baseline (Risk Distance style) anomaly scorer."""

    #: Dirichlet concentration for categorical surprisal smoothing.
    dirichlet_alpha: float = 0.5

    #: Empirical-Bayes shrinkage strength toward the global baseline. A cohort
    #: with n members gets weight n/(n+kappa) on its own estimate.
    shrinkage_kappa: float = 25.0

    #: How per-feature surprisals combine into one score.
    #:
    #: "sum"  — total surprisal across all features. Principled under an
    #:          independence assumption, but it dilutes: fifteen features means a
    #:          document extreme on two of them competes with one mildly odd on
    #:          all fifteen, and the latter is usually just noise.
    #: "topk" — sum of the k largest surprisals. Treats an anomaly as something
    #:          extreme on *a few* dimensions, which is what posture anomalies
    #:          actually are. Keeps attribution exact: the top k features are
    #:          both the score and the explanation.
    aggregation: Literal["sum", "topk"] = "topk"
    top_k: int = 2

    #: Blend weight for the IsolationForest interaction term. The remaining
    #: (1 - w) is the exactly-attributable additive score.
    #:
    #: Defaults to OFF, which is a measured decision rather than a simplification.
    #: The term was built, swept, and did not pay for itself: on the benchmark it
    #: moved PR-AUC 0.619 -> 0.611 at lambda=0.25 while making a quarter of every
    #: score unattributable to any feature. The eval harness re-runs the sweep on
    #: every evaluation so the decision stays evidence-backed rather than
    #: inherited. See docs/adr/0003-topk-aggregation.md.
    iforest_weight: float = 0.0
    iforest_n_estimators: int = 300

    #: Robust refitting passes. Baselines are learned from the same corpus being
    #: judged, so the anomalies contaminate the distribution they are measured
    #: against — with a 2% anomaly rate concentrated on one feature value, that
    #: value stops looking rare. Each pass drops the highest-scoring
    #: `trim_fraction` of documents and refits on the remainder, sharpening the
    #: baseline toward what compliant documents actually look like.
    #: 0 disables (single fit).
    robust_passes: int = 1
    trim_fraction: float = 0.05

    #: Split-conformal target false-positive rate.
    conformal_alpha: float = 0.05
    #: Fraction of the corpus held out to calibrate conformal p-values.
    calibration_fraction: float = 0.30

    #: Per-feature surprisal is clipped here (in nats) to stop one impossible
    #: categorical value from saturating the total score. Robustness of the
    #: baseline itself comes from using empirical tail probabilities rather than
    #: a fitted location-scale model — see scoring/baseline.py.
    max_feature_surprisal: float = 12.0

    #: Conformal p-values are computed within cohort when the cohort has at least
    #: this many calibration points, otherwise against the global calibration set.
    min_cohort_for_local_conformal: int = 50


class LineageConfig(BaseModel):
    """Near-duplicate / derivative detection."""

    shingle_size: int = 5
    num_perm: int = 128
    threshold: float = 0.72
    n_bands: int = 32


class ExplainConfig(BaseModel):
    #: Number of top attributions quoted in a finding narrative.
    top_k_attributions: int = 3
    #: Maximum permission edits a counterfactual remediation may propose.
    max_counterfactual_edits: int = 4
    #: Use an LLM to polish narratives. Off by default: the templates are
    #: deterministic and the pipeline must run with no API key.
    use_llm: bool = False
    llm_model: str = "claude-sonnet-5"


class Paths(BaseModel):
    root: Path = Path("artifacts")

    @property
    def corpus(self) -> Path:
        return self.root / "corpus"

    @property
    def vectors(self) -> Path:
        return self.root / "vectors"

    @property
    def findings(self) -> Path:
        return self.root / "findings"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> None:
        for p in (self.root, self.corpus, self.vectors, self.findings, self.reports):
            p.mkdir(parents=True, exist_ok=True)


class CohortConfig(BaseModel):
    synthorg: SynthOrgConfig = Field(default_factory=SynthOrgConfig)
    semantic: SemanticConfig = Field(default_factory=SemanticConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    lineage: LineageConfig = Field(default_factory=LineageConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    paths: Paths = Field(default_factory=Paths)

    @classmethod
    def load(cls, path: str | Path | None = None) -> CohortConfig:
        if path is None:
            return cls()
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
