"""Real-corpus experiments.

Two experiments, deliberately kept separate from the synthetic benchmark rather
than blended into it:

**A — cohort discovery on 20 Newsgroups.** Real prose with 20 gold labels. Tests
the half of the pipeline the synthetic corpus cannot: whether documents that were
not generated from templates still group by meaning.

**B — peer-baseline detection on Enron.** Real content, real recipient graphs,
real folder structure, real timestamps. Anomalies are injected into the real
posture distribution, drawn from its own upper tail, so a flagged document is
abnormal only relative to its peers. Runs the same peer / global / random-cohort
ablation as the synthetic harness, which is the comparison that matters: does the
central claim survive contact with real data?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cohort.config import CohortConfig
from cohort.evaluate.metrics import anomaly_metrics, cluster_metrics, posture_coupling
from cohort.lineage import build_lineage
from cohort.real.inject import inject_into_real
from cohort.schema import POSTURE_FEATURES
from cohort.scoring.engine import RiskEngine
from cohort.semantic.cluster import discover_cohorts
from cohort.semantic.embed import embed_documents

#: Features a mail archive cannot supply. Filled with a constant so the scorer
#: runs unmodified: a constant categorical has zero entropy and a constant
#: continuous feature is flagged uninformative, so both contribute exactly 0 nats.
#: Nothing is faked — the features are simply inert, and the report says so.
ABSENT_ON_ENRON = {
    "link_scope": "unknown",  # email has no sharing-link concept
    "label_tier": "unknown",  # no sensitivity labelling in the archive
    "acl_origin": "unknown",  # no permission inheritance model
    "n_groups": 0.0,  # no group grants, only explicit recipients
    "owner_dept_is_modal": True,  # no org chart to compare against
}


@dataclass
class RealReport:
    name: str
    n_documents: int
    notes: dict = field(default_factory=dict)
    clustering: dict = field(default_factory=dict)
    detection: dict = field(default_factory=dict)
    ablations: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "corpus": self.name,
            "n_documents": self.n_documents,
            "notes": self.notes,
            "clustering": self.clustering,
            "detection": self.detection,
            "ablations": self.ablations,
        }


def run_newsgroups(cfg: CohortConfig, parquet: Path, max_docs: int | None = None) -> RealReport:
    """Experiment A — cohort discovery against 20 gold newsgroup labels."""
    from cohort.real.newsgroups import load_corpus

    corpus, truth = load_corpus(parquet, max_docs=max_docs)
    vectors = embed_documents(corpus["body"].tolist(), cfg.semantic).vectors
    clusters = discover_cohorts(vectors, cfg.semantic)

    cm = cluster_metrics(
        truth["true_category"].to_numpy(),
        clusters.labels,
        clusters.unassigned_rate,
        clusters.silhouette,
    )
    return RealReport(
        name="20 Newsgroups",
        n_documents=len(corpus),
        notes={
            "gold_categories": int(truth["true_category"].nunique()),
            "reducer": cfg.semantic.reducer,
            "cluster_dim": cfg.semantic.cluster_dim,
            "min_cluster_size": cfg.semantic.effective_min_cluster_size(len(corpus)),
            "reassign_bar": round(float(clusters.reassign_bar), 4),
            "raw_hdbscan_noise": round(clusters.raw_noise_rate, 4),
        },
        clustering=cm.as_dict(),
    )


def run_enron(
    cfg: CohortConfig,
    archive: Path,
    limit: int = 40_000,
    anomaly_rate: float = 0.02,
) -> RealReport:
    """Experiment B — peer-baseline detection over real email posture."""
    from cohort.real.enron import ENRON_CATEGORICAL, ENRON_CONTINUOUS, ENRON_FEATURES, parse_corpus

    docs, stats = parse_corpus(archive, limit=limit)
    if docs.empty:
        raise RuntimeError("no usable messages parsed from the archive")

    texts = (docs["title"].astype(str) + ". " + docs["body"].astype(str)).tolist()
    vectors = embed_documents(texts, cfg.semantic).vectors
    clusters = discover_cohorts(vectors, cfg.semantic)

    docs = docs.copy()
    docs["dup_count"] = build_lineage(docs["body"].astype(str).tolist(), cfg.lineage).dup_count

    injected = inject_into_real(docs, clusters.labels, rate=anomaly_rate, seed=cfg.synthorg.seed)
    frame = injected.corpus
    for feature, constant in ABSENT_ON_ENRON.items():
        frame[feature] = constant

    y = injected.truth["is_anomaly"].to_numpy().astype(bool)
    types = injected.truth["anomaly_type"].to_numpy()
    ids = frame["doc_id"].to_numpy()

    def score(labels: np.ndarray) -> np.ndarray:
        return (
            RiskEngine(cfg.scoring)
            .fit_score(
                frame[[*POSTURE_FEATURES, "doc_id"]], labels, ids, rng_seed=cfg.synthorg.seed
            )
            .risk
        )

    peer = RiskEngine(cfg.scoring).fit_score(
        frame[[*POSTURE_FEATURES, "doc_id"]], clusters.labels, ids, rng_seed=cfg.synthorg.seed
    )
    headline = anomaly_metrics(y, peer.risk, types, peer.flagged)

    # Does semantic cohort membership actually predict posture here? This is the
    # precondition peer baselining depends on, and it is the number that explains
    # the gap between the synthetic and real results rather than hand-waving at it.
    coupling = posture_coupling(docs, clusters.labels, ENRON_CATEGORICAL, ENRON_CONTINUOUS)

    shuffled = clusters.labels.copy()
    np.random.default_rng(cfg.synthorg.seed).shuffle(shuffled)

    ablations = {
        "global baseline (no peer grouping)": anomaly_metrics(
            y, score(np.zeros(len(frame), dtype=int)), types
        ).as_dict(),
        "random cohorts (same sizes)": anomaly_metrics(y, score(shuffled), types).as_dict(),
    }

    return RealReport(
        name="Enron",
        n_documents=len(frame),
        notes={
            "messages_scanned": stats.scanned,
            "messages_kept": stats.kept,
            "active_features": len(ENRON_FEATURES),
            "total_features": len(POSTURE_FEATURES),
            "inert_features": sorted(ABSENT_ON_ENRON),
            "anomaly_counts": injected.counts,
            "injected_from_real_tails": {k: round(v, 2) for k, v in injected.tail_values.items()},
            "reducer": cfg.semantic.reducer,
            "reassign_bar": round(float(clusters.reassign_bar), 4),
            "raw_hdbscan_noise": round(clusters.raw_noise_rate, 4),
            "posture_cohort_coupling": round(coupling["mean_coupling"], 4),
            "coupling_by_feature": {
                k: round(v, 3)
                for k, v in sorted(coupling.items(), key=lambda kv: -kv[1])
                if k != "mean_coupling"
            },
        },
        clustering={
            "n_cohorts": clusters.n_cohorts,
            "unassigned_rate": round(clusters.unassigned_rate, 4),
            "silhouette": round(clusters.silhouette, 4),
        },
        detection=headline.as_dict(),
        ablations=ablations,
    )


def to_markdown(reports: list[RealReport]) -> str:
    L: list[str] = ["# Real-corpus results\n"]
    for r in reports:
        L.append(f"## {r.name} — {r.n_documents:,} documents\n")

        if r.notes:
            L.append("| Setting | Value |")
            L.append("|---|---|")
            for k, v in r.notes.items():
                L.append(f"| {k.replace('_', ' ')} | {v} |")
            L.append("")

        if r.clustering:
            L.append("### Cohort discovery\n")
            L.append("| Metric | Value |")
            L.append("|---|---|")
            for k, v in r.clustering.items():
                L.append(f"| {k.replace('_', ' ')} | {v} |")
            L.append("")

        if r.detection:
            d = r.detection
            L.append("### Detection\n")
            L.append("| Metric | Value |")
            L.append("|---|---|")
            L.append(f"| PR-AUC | {d['pr_auc']:.3f} |")
            L.append(f"| ROC-AUC | {d['roc_auc']:.3f} |")
            for k in sorted(d["precision_at"], key=int):
                L.append(f"| Precision@{k} | {d['precision_at'][k]:.3f} |")
            L.append(f"| Flag rate | {d['flag_rate']:.2%} |")
            L.append("")
            if d.get("per_type"):
                L.append("| Anomaly type | n | PR-AUC | Recall at threshold |")
                L.append("|---|---:|---:|---:|")
                for t, v in sorted(d["per_type"].items(), key=lambda kv: -kv[1]["pr_auc"]):
                    L.append(
                        f"| `{t}` | {int(v['n'])} | {v['pr_auc']:.3f} | "
                        f"{v.get('recall_at_threshold', float('nan')):.3f} |"
                    )
                L.append("")

        if r.ablations:
            L.append("### Ablations\n")
            L.append("| Configuration | PR-AUC | vs peer cohorts |")
            L.append("|---|---:|---:|")
            base = r.detection["pr_auc"]
            L.append(f"| **peer cohorts (semantic)** | **{base:.3f}** | — |")
            for name, m in r.ablations.items():
                L.append(f"| {name} | {m['pr_auc']:.3f} | {m['pr_auc'] - base:+.3f} |")
            L.append("")
    return "\n".join(L)


def write_reports(reports: list[RealReport], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / "real_corpora.json"
    m = out_dir / "real_corpora.md"
    j.write_text(json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8")
    m.write_text(to_markdown(reports), encoding="utf-8")
    return {"json": j, "markdown": m}
