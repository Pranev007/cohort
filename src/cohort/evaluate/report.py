"""The evaluation harness.

Produces the numbers in the README, and the same function backs the CI job that
comments a metrics table on every pull request.

The ablations are the point. A single PR-AUC proves nothing on its own — it could
come from peer grouping, or from the features, or from nothing at all. So the
harness runs the controls that could falsify the central claim:

* **global** — one baseline for the whole corpus, no peer grouping. If this
  matches the headline, the entire premise of the project is wrong.
* **random cohorts** — the same cohort *sizes*, membership shuffled. This is the
  stronger control: it separates "grouping helps" from "grouping *by meaning*
  helps". Without it, someone can reasonably argue the gain is a bucketing
  artefact.
* **no shrinkage / global-only shrinkage** — isolates the empirical-Bayes term.
* **aggregation and lambda sweeps** — the two tuned knobs, re-measured every run
  so the defaults stay evidence-backed rather than inherited.
* **cohort-quality sensitivity** — degrades clustering deliberately and plots
  detection against it, answering the obvious question: what happens when the
  cohorts are not this clean?
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cohort.config import CohortConfig, ScoringConfig
from cohort.evaluate.metrics import (
    AnomalyMetrics,
    ClusterMetrics,
    anomaly_metrics,
    cluster_metrics,
    posture_coupling,
)
from cohort.lineage import build_lineage
from cohort.schema import CATEGORICAL_FEATURES, CONTINUOUS_FEATURES
from cohort.scoring.engine import RiskEngine
from cohort.semantic.cluster import discover_cohorts
from cohort.semantic.embed import embed_documents


@dataclass
class EvalReport:
    headline: AnomalyMetrics
    clustering: ClusterMetrics
    ablations: dict[str, dict] = field(default_factory=dict)
    sensitivity: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "headline": self.headline.as_dict(),
            "clustering": self.clustering.as_dict(),
            "ablations": self.ablations,
            "cohort_quality_sensitivity": self.sensitivity,
            "stats": self.stats,
            "timings_s": {k: round(v, 2) for k, v in self.timings.items()},
        }

    def to_markdown(self) -> str:
        h = self.headline
        c = self.clustering
        s = self.stats
        L: list[str] = []

        L.append("## Detection results\n")
        L.append(
            f"Corpus: **{h.n_total:,} documents**, **{h.n_positive}** injected "
            f"anomalies ({h.n_positive / max(h.n_total, 1):.2%} base rate), "
            f"{c.n_cohorts} discovered cohorts.\n"
        )
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| **PR-AUC** (average precision) | **{h.pr_auc:.3f}** |")
        L.append(f"| ROC-AUC | {h.roc_auc:.3f} |")
        for k in sorted(h.precision_at):
            L.append(f"| Precision@{k} | {h.precision_at[k]:.3f} |")
        for k in sorted(h.recall_at):
            L.append(f"| Recall@{k} | {h.recall_at[k]:.3f} |")
        L.append(
            f"| Flag rate at conformal α={s.get('conformal_alpha', 0.05)} | {h.flag_rate:.2%} |"
        )
        L.append(f"| Precision among flagged | {h.flagged_precision:.3f} |")
        L.append(f"| Recall among flagged | {h.flagged_recall:.3f} |")
        L.append("")

        if h.per_type:
            L.append("### Per anomaly type\n")
            L.append(
                "| Anomaly type | n | PR-AUC | Recall@100 | Recall@300 | Recall at threshold |"
            )
            L.append("|---|---:|---:|---:|---:|---:|")
            for t, d in sorted(h.per_type.items(), key=lambda kv: -kv[1]["pr_auc"]):
                L.append(
                    f"| `{t}` | {int(d['n'])} | {d['pr_auc']:.3f} | "
                    f"{d.get('recall_at_100', float('nan')):.3f} | "
                    f"{d.get('recall_at_300', float('nan')):.3f} | "
                    f"{d.get('recall_at_threshold', float('nan')):.3f} |"
                )
            L.append("")

        L.append("### Cohort discovery (unsupervised, vs generator categories)\n")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| Cohorts discovered | {c.n_cohorts} |")
        L.append(f"| Adjusted Rand Index | {c.ari:.3f} |")
        L.append(f"| Homogeneity | {c.homogeneity:.3f} |")
        L.append(f"| Completeness | {c.completeness:.3f} |")
        L.append(f"| Silhouette (cosine) | {c.silhouette:.3f} |")
        L.append(f"| Unassigned | {c.unassigned_rate:.2%} |")
        L.append("")

        if self.ablations:
            L.append("### Ablations\n")
            L.append("| Configuration | PR-AUC | ROC-AUC | P@50 | P@100 | vs headline |")
            L.append("|---|---:|---:|---:|---:|---:|")
            base = h.pr_auc
            for name, entry in self.ablations.items():
                am = entry["metrics"]
                delta = am["pr_auc"] - base
                L.append(
                    f"| {name} | {am['pr_auc']:.3f} | {am['roc_auc']:.3f} | "
                    f"{am['precision_at']['50']:.3f} | {am['precision_at']['100']:.3f} | "
                    f"{delta:+.3f} |"
                )
            L.append("")

        if self.sensitivity:
            L.append("### Sensitivity to cohort quality\n")
            L.append(
                "Clustering is degraded deliberately by lowering `min_cluster_size`, "
                "which over-segments categories into many small cohorts.\n"
            )
            L.append("| min_cluster_size | cohorts | cluster ARI | completeness | PR-AUC |")
            L.append("|---:|---:|---:|---:|---:|")
            for row in self.sensitivity:
                L.append(
                    f"| {row['min_cluster_size']} | {row['n_cohorts']} | "
                    f"{row['ari']:.3f} | {row['completeness']:.3f} | {row['pr_auc']:.3f} |"
                )
            L.append("")

        if self.stats:
            L.append("### Throughput\n")
            L.append("| Stage | Value |")
            L.append("|---|---|")
            for stat_key in (
                "embedding_backend",
                "embed_docs_per_s",
                "score_docs_per_s",
                "end_to_end_docs_per_s",
            ):
                if stat_key in self.stats:
                    L.append(f"| {stat_key.replace('_', ' ')} | {self.stats[stat_key]} |")
            L.append("")

        return "\n".join(L)


def _score_with(
    corpus: pd.DataFrame, cohort_ids: np.ndarray, scfg: ScoringConfig, seed: int
) -> np.ndarray:
    return (
        RiskEngine(scfg)
        .fit_score(corpus, cohort_ids, corpus["doc_id"].to_numpy(), rng_seed=seed)
        .risk
    )


def run_evaluation(
    cfg: CohortConfig,
    corpus_path: Path,
    truth_path: Path,
    with_ablations: bool = True,
    with_sensitivity: bool = True,
) -> EvalReport:
    timings: dict[str, float] = {}
    seed = cfg.synthorg.seed

    corpus = pd.read_parquet(corpus_path)
    truth = pd.read_parquet(truth_path).set_index("doc_id").loc[corpus["doc_id"]]
    y = truth["is_anomaly"].to_numpy().astype(bool)
    types = truth["anomaly_type"].fillna("").to_numpy()

    texts = (corpus["title"].astype(str) + ". " + corpus["body"].astype(str)).tolist()

    t0 = time.perf_counter()
    emb = embed_documents(texts, cfg.semantic)
    timings["embed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    clusters = discover_cohorts(emb.vectors, cfg.semantic)
    timings["cluster"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lineage = build_lineage(corpus["body"].astype(str).tolist(), cfg.lineage)
    corpus = corpus.copy()
    corpus["dup_count"] = lineage.dup_count
    timings["lineage"] = time.perf_counter() - t0

    # ---- headline --------------------------------------------------------
    t0 = time.perf_counter()
    engine = RiskEngine(cfg.scoring).fit(corpus, clusters.labels, rng_seed=seed)
    res = engine.score(corpus, clusters.labels, corpus["doc_id"].to_numpy())
    timings["score"] = time.perf_counter() - t0

    headline = anomaly_metrics(y, res.risk, types, res.flagged)
    clustering = cluster_metrics(
        truth["true_category"].to_numpy(),
        clusters.labels,
        clusters.unassigned_rate,
        clusters.silhouette,
    )

    # How much of the posture variation cohort membership explains — the
    # precondition peer baselining depends on. Reported here so the synthetic
    # figure can be compared directly against the same statistic on a real corpus.
    coupling = posture_coupling(corpus, clusters.labels, CATEGORICAL_FEATURES, CONTINUOUS_FEATURES)

    report = EvalReport(headline=headline, clustering=clustering, timings=timings)
    report.stats = {
        "posture_cohort_coupling": round(coupling["mean_coupling"], 4),
        "coupling_by_feature": {
            k: round(v, 3)
            for k, v in sorted(coupling.items(), key=lambda kv: -kv[1])
            if k != "mean_coupling"
        },
        "conformal_alpha": cfg.scoring.conformal_alpha,
        "aggregation": cfg.scoring.aggregation,
        "top_k": cfg.scoring.top_k,
        "iforest_weight": cfg.scoring.iforest_weight,
        "shrinkage_kappa": cfg.scoring.shrinkage_kappa,
        "embedding_backend": emb.backend,
        "embed_docs_per_s": round(emb.docs_per_s, 1),
        "score_docs_per_s": round(len(corpus) / max(timings["score"], 1e-9), 1),
        "end_to_end_docs_per_s": round(len(corpus) / max(sum(timings.values()), 1e-9), 1),
        "seed": seed,
    }

    if with_ablations:
        t0 = time.perf_counter()
        report.ablations = _run_ablations(corpus, clusters.labels, cfg, y, types, seed)
        timings["ablations"] = time.perf_counter() - t0

    if with_sensitivity:
        t0 = time.perf_counter()
        report.sensitivity = _run_sensitivity(
            corpus, emb.vectors, cfg, y, truth["true_category"].to_numpy(), seed
        )
        timings["sensitivity"] = time.perf_counter() - t0

    return report


def _run_ablations(
    corpus: pd.DataFrame,
    labels: np.ndarray,
    cfg: CohortConfig,
    y: np.ndarray,
    types: np.ndarray,
    seed: int,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    rng = np.random.default_rng(seed)

    def record(name: str, scores: np.ndarray, note: str) -> None:
        m = anomaly_metrics(y, scores, types)
        out[name] = {"note": note, "metrics": m.as_dict()}

    # -- the central control: no peer grouping at all ----------------------
    record(
        "**global baseline** (no peer grouping)",
        _score_with(corpus, np.zeros(len(corpus), dtype=int), cfg.scoring, seed),
        "One baseline for the entire corpus. Falsifies the premise if it matches.",
    )

    # -- same bucket sizes, meaningless membership -------------------------
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    record(
        "**random cohorts** (same sizes, shuffled)",
        _score_with(corpus, shuffled, cfg.scoring, seed),
        "Separates 'grouping helps' from 'grouping by meaning helps'.",
    )

    # -- shrinkage ---------------------------------------------------------
    for kappa, label in [(0.0, "no shrinkage (κ=0)"), (500.0, "heavy shrinkage (κ=500)")]:
        record(
            label,
            _score_with(
                corpus, labels, cfg.scoring.model_copy(update={"shrinkage_kappa": kappa}), seed
            ),
            "Empirical-Bayes strength.",
        )

    # -- aggregation -------------------------------------------------------
    record(
        "aggregation = sum (all 15 features)",
        _score_with(corpus, labels, cfg.scoring.model_copy(update={"aggregation": "sum"}), seed),
        "Unweighted total surprisal.",
    )
    for k in (1, 3, 4):
        record(
            f"aggregation = top-{k}",
            _score_with(corpus, labels, cfg.scoring.model_copy(update={"top_k": k}), seed),
            "Top-k surprisal sum.",
        )

    # -- robust refitting --------------------------------------------------
    for passes, label in [(0, "no robust refit (single fit)"), (2, "robust refit x2")]:
        record(
            label,
            _score_with(
                corpus, labels, cfg.scoring.model_copy(update={"robust_passes": passes}), seed
            ),
            "Trimmed iterative refitting against baseline contamination.",
        )

    # -- interaction term --------------------------------------------------
    for lam in (0.15, 0.25, 0.40):
        record(
            f"+ IsolationForest interaction (λ={lam})",
            _score_with(
                corpus, labels, cfg.scoring.model_copy(update={"iforest_weight": lam}), seed
            ),
            "Blended interaction term over the surprisal matrix.",
        )

    return out


def _run_sensitivity(
    corpus: pd.DataFrame,
    vectors: np.ndarray,
    cfg: CohortConfig,
    y: np.ndarray,
    true_categories: np.ndarray,
    seed: int,
) -> list[dict]:
    """How much detection degrades as cohort quality degrades.

    Cohorts on this corpus come out near-perfect, which is a property of a
    template-generated corpus and will not hold on real documents. Rather than
    leave that as an unexamined caveat, force the clusterer to over-segment and
    measure what it costs.
    """
    rows: list[dict] = []
    for mcs in (40, 80, 150, 250):
        scfg = cfg.semantic.model_copy(
            update={"min_cluster_size": mcs, "min_cluster_size_frac": None}
        )
        cl = discover_cohorts(vectors, scfg)
        if cl.n_cohorts < 2:
            continue
        cm = cluster_metrics(true_categories, cl.labels, cl.unassigned_rate, cl.silhouette)
        scores = _score_with(corpus, cl.labels, cfg.scoring, seed)
        m = anomaly_metrics(y, scores)
        rows.append(
            {
                "min_cluster_size": mcs,
                "n_cohorts": cl.n_cohorts,
                "ari": cm.ari,
                "completeness": cm.completeness,
                "pr_auc": m.pr_auc,
            }
        )
    return rows


def write_report(report: EvalReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / "evaluation.json"
    m = out_dir / "evaluation.md"
    j.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    m.write_text(report.to_markdown(), encoding="utf-8")
    return {"json": j, "markdown": m}
