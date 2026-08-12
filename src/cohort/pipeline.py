"""End-to-end scan: corpus in, ranked findings out.

    extract -> embed -> discover cohorts -> name -> lineage -> baseline -> score
            -> calibrate -> attribute -> narrate -> plan remediation

Every stage is timed and the timings ship with the run report, because throughput
is a first-class result here: a DSPM engine that cannot process a corpus faster
than the corpus grows is not a product.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from cohort.config import CohortConfig
from cohort.explain.attribution import top_attributions
from cohort.explain.counterfactual import plan_remediation
from cohort.explain.narrate import narrate
from cohort.lineage import build_lineage
from cohort.schema import POSTURE_FEATURES
from cohort.scoring.engine import RiskEngine, ScoreResult
from cohort.semantic.cluster import ClusterResult, discover_cohorts
from cohort.semantic.embed import embed_documents
from cohort.semantic.naming import CohortName, name_cohorts


@dataclass
class PipelineResult:
    corpus: pd.DataFrame
    cohorts: ClusterResult
    names: dict[int, CohortName]
    scores: ScoreResult
    findings: pd.DataFrame
    engine: RiskEngine
    timings: dict[str, float] = field(default_factory=dict)
    stats: dict[str, float | int | str] = field(default_factory=dict)

    def summary(self) -> str:
        t = self.timings
        return (
            f"{len(self.corpus):,} documents | {self.cohorts.n_cohorts} cohorts "
            f"({self.cohorts.unassigned_rate:.1%} unassigned) | "
            f"{int(self.scores.flagged.sum()):,} flagged "
            f"({self.scores.flagged.mean():.2%}) | "
            f"{t.get('total', 0):.1f}s total"
        )


def run_pipeline(
    cfg: CohortConfig,
    corpus_path: Path,
    max_findings: int = 500,
    with_remediation: bool = True,
) -> PipelineResult:
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    corpus = pd.read_parquet(corpus_path)
    texts = (corpus["title"].astype(str) + ". " + corpus["body"].astype(str)).tolist()

    # -- semantic layer ----------------------------------------------------
    t0 = time.perf_counter()
    emb = embed_documents(texts, cfg.semantic)
    timings["embed"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cohorts = discover_cohorts(emb.vectors, cfg.semantic)
    timings["cluster"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    names = name_cohorts(texts, cohorts.labels, cfg.semantic, cfg.explain)
    timings["name"] = time.perf_counter() - t0

    # -- lineage: fills dup_count, which is a scored posture feature --------
    t0 = time.perf_counter()
    lineage = build_lineage(corpus["body"].astype(str).tolist(), cfg.lineage)
    corpus = corpus.copy()
    corpus["dup_count"] = lineage.dup_count
    corpus["lineage_family"] = lineage.family_of
    timings["lineage"] = time.perf_counter() - t0

    # -- baseline + score --------------------------------------------------
    t0 = time.perf_counter()
    engine = RiskEngine(cfg.scoring).fit(corpus, cohorts.labels, rng_seed=cfg.synthorg.seed)
    timings["fit"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    scores = engine.score(corpus, cohorts.labels, corpus["doc_id"].to_numpy())
    timings["score"] = time.perf_counter() - t0

    # -- explain -----------------------------------------------------------
    t0 = time.perf_counter()
    findings = _build_findings(
        corpus, cohorts, names, scores, engine, cfg, max_findings, with_remediation
    )
    timings["explain"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_start

    stats: dict[str, float | int | str] = {
        "n_documents": len(corpus),
        "embedding_backend": emb.backend,
        "embedding_dim": emb.dim,
        "embed_docs_per_s": round(emb.docs_per_s, 1),
        "n_cohorts": cohorts.n_cohorts,
        "unassigned_rate": round(cohorts.unassigned_rate, 4),
        "silhouette": round(cohorts.silhouette, 4),
        "lineage_families": lineage.n_families_gt1,
        "largest_lineage_family": lineage.largest_family,
        "flag_rate": round(float(scores.flagged.mean()), 4),
        "n_flagged": int(scores.flagged.sum()),
        "docs_per_s_end_to_end": round(len(corpus) / max(timings["total"], 1e-9), 1),
    }

    return PipelineResult(
        corpus=corpus,
        cohorts=cohorts,
        names=names,
        scores=scores,
        findings=findings,
        engine=engine,
        timings=timings,
        stats=stats,
    )


def _build_findings(
    corpus: pd.DataFrame,
    cohorts: ClusterResult,
    names: dict[int, CohortName],
    scores: ScoreResult,
    engine: RiskEngine,
    cfg: CohortConfig,
    max_findings: int,
    with_remediation: bool,
) -> pd.DataFrame:
    order = np.argsort(-scores.risk)[:max_findings]
    target = engine.fitted_calibrator.threshold()

    def score_fn(frame: pd.DataFrame, cids: np.ndarray) -> np.ndarray:
        return engine.risk_of(frame, cids)

    rows = []
    for i in order:
        cid = int(cohorts.labels[i])
        cname = names.get(cid)
        label = cname.label if cname else f"cohort {cid}"

        attrs = top_attributions(
            corpus.iloc[i],
            scores.attributions.iloc[i],
            engine.fitted_baselines,
            cid,
            k=cfg.explain.top_k_attributions,
        )
        text = narrate(label, attrs, float(scores.risk[i]), float(scores.p_values[i]), cfg.explain)

        plan_desc, residual, resolved, n_edits = "", float(scores.risk[i]), False, 0
        if with_remediation and scores.flagged[i]:
            plan = plan_remediation(
                corpus.iloc[i],
                engine.fitted_baselines,
                cid,
                score_fn,
                target,
                cfg.explain,
                surprisal_row=scores.attributions.iloc[i],
            )
            plan_desc, residual = plan.describe(), plan.residual_score
            resolved, n_edits = plan.resolved, len(plan.edits)

        rows.append(
            {
                "doc_id": corpus.iloc[i]["doc_id"],
                "title": corpus.iloc[i]["title"],
                "cohort_id": cid,
                "cohort_name": label,
                "risk_score": float(scores.risk[i]),
                "conformal_p": float(scores.p_values[i]),
                "is_flagged": bool(scores.flagged[i]),
                "top_features": ", ".join(f"{a.feature}({a.surprisal:.2f})" for a in attrs),
                "narrative": text,
                "remediation": plan_desc,
                "residual_score": residual,
                "remediation_resolves": resolved,
                "n_edits": n_edits,
            }
        )
    return pd.DataFrame(rows)


def write_artifacts(result: PipelineResult, cfg: CohortConfig) -> dict[str, Path]:
    cfg.paths.ensure()
    out: dict[str, Path] = {}

    p = cfg.paths.findings / "findings.parquet"
    result.findings.to_parquet(p, index=False)
    out["findings"] = p

    p = cfg.paths.findings / "scores.parquet"
    result.scores.to_frame().to_parquet(p, index=False)
    out["scores"] = p

    p = cfg.paths.reports / "run_stats.json"
    p.write_text(
        json.dumps({"stats": result.stats, "timings_s": result.timings}, indent=2),
        encoding="utf-8",
    )
    out["run_stats"] = p

    p = cfg.paths.reports / "cohorts.json"
    p.write_text(
        json.dumps(
            [
                {
                    "cohort_id": n.cohort_id,
                    "label": n.label,
                    "size": n.size,
                    "keyphrases": n.keyphrases,
                    "source": n.source,
                }
                for n in sorted(result.names.values(), key=lambda x: -x.size)
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    out["cohorts"] = p
    return out


def posture_frame(corpus: pd.DataFrame) -> pd.DataFrame:
    return corpus[POSTURE_FEATURES]
