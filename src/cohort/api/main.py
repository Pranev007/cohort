"""Findings API.

Serves the scan output, and — the part that makes this a service rather than a
report viewer — exposes `POST /score`, which evaluates an arbitrary posture
payload against the fitted peer baselines. That is the endpoint a remediation
workflow actually needs: *before* I widen access on this document, tell me what
it does to the risk, and tell me why.

Requires the `api` extra:  pip install -e '.[api]'
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from cohort.config import CohortConfig
from cohort.explain.attribution import top_attributions
from cohort.explain.narrate import narrate
from cohort.lineage import build_lineage
from cohort.schema import POSTURE_FEATURES
from cohort.scoring.engine import RiskEngine
from cohort.semantic.cluster import discover_cohorts
from cohort.semantic.embed import embed_documents
from cohort.semantic.naming import name_cohorts

app = FastAPI(
    title="Cohort",
    description="Peer-baseline anomaly detection for unstructured enterprise data.",
    version="1.0.0",
)

STATE: dict[str, Any] = {}


class ScoreRequest(BaseModel):
    """A posture payload to evaluate against a cohort's baseline."""

    cohort_id: int = Field(..., description="Cohort to judge against. Use /cohorts to list.")
    link_scope: str = "none"
    repo_type: str = "sharepoint"
    label_tier: str = "internal"
    acl_origin: str = "inherited"
    has_external_principal: bool = False
    owner_dept_is_modal: bool = True
    n_principals: float = 10
    n_groups: float = 1
    n_external_domains: float = 0
    accessor_dept_entropy: float = 0.3
    path_depth: float = 4
    age_days: float = 120
    staleness_days: float = 60
    pii_density: float = 1.0
    dup_count: float = 0


class ScoreResponse(BaseModel):
    risk_score: float
    conformal_p: float
    is_flagged: bool
    narrative: str
    attributions: list[dict]


@app.on_event("startup")
def _startup() -> None:
    """Load artifacts and refit baselines so /score can serve live queries."""
    cfg = CohortConfig()
    STATE["config"] = cfg

    corpus_path = cfg.paths.corpus / "corpus.parquet"
    if not corpus_path.exists():
        STATE["ready"] = False
        return

    corpus = pd.read_parquet(corpus_path)
    texts = (corpus["title"].astype(str) + ". " + corpus["body"].astype(str)).tolist()
    vectors = embed_documents(texts, cfg.semantic).vectors
    clusters = discover_cohorts(vectors, cfg.semantic)

    corpus = corpus.copy()
    corpus["dup_count"] = build_lineage(corpus["body"].astype(str).tolist(), cfg.lineage).dup_count

    engine = RiskEngine(cfg.scoring).fit(corpus, clusters.labels, rng_seed=cfg.synthorg.seed)

    STATE.update(
        ready=True,
        engine=engine,
        labels=clusters.labels,
        names=name_cohorts(texts, clusters.labels, cfg.semantic, cfg.explain),
        corpus=corpus,
    )

    findings_path = cfg.paths.findings / "findings.parquet"
    STATE["findings"] = pd.read_parquet(findings_path) if findings_path.exists() else pd.DataFrame()

    stats_path = cfg.paths.reports / "run_stats.json"
    STATE["stats"] = (
        json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    )


def _require_ready() -> None:
    if not STATE.get("ready"):
        raise HTTPException(
            status_code=503,
            detail="No corpus loaded. Run `cohort generate && cohort scan` first.",
        )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "ready": bool(STATE.get("ready")), "version": app.version}


@app.get("/stats")
def stats() -> dict:
    _require_ready()
    return STATE.get("stats", {})


@app.get("/cohorts")
def cohorts() -> list[dict]:
    _require_ready()
    return [
        {
            "cohort_id": n.cohort_id,
            "label": n.label,
            "size": n.size,
            "keyphrases": n.keyphrases,
            "source": n.source,
        }
        for n in sorted(STATE["names"].values(), key=lambda x: -x.size)
    ]


@app.get("/findings")
def findings(
    limit: int = Query(50, ge=1, le=1000),
    min_risk: float = Query(0.0, ge=0.0),
    cohort_id: int | None = None,
    flagged_only: bool = False,
) -> list[dict]:
    _require_ready()
    df = STATE["findings"]
    if df.empty:
        return []
    if cohort_id is not None:
        df = df[df.cohort_id == cohort_id]
    if flagged_only:
        df = df[df.is_flagged]
    df = df[df.risk_score >= min_risk]
    return json.loads(df.head(limit).to_json(orient="records"))


@app.get("/findings/{doc_id}")
def finding(doc_id: str) -> dict:
    _require_ready()
    df = STATE["findings"]
    row = df[df.doc_id == doc_id] if not df.empty else df
    if row.empty:
        raise HTTPException(status_code=404, detail=f"{doc_id} not among the ranked findings")
    return json.loads(row.iloc[0].to_json())


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    """Score a hypothetical posture against a cohort baseline.

    Lets a change-management workflow ask "what would this permission change do?"
    before making it, rather than discovering the answer in the next scan.
    """
    _require_ready()
    engine: RiskEngine = STATE["engine"]

    known = {int(c) for c in np.unique(STATE["labels"])}
    if req.cohort_id not in known:
        raise HTTPException(
            status_code=400,
            detail=f"unknown cohort {req.cohort_id}; known cohorts: {sorted(known)}",
        )

    payload = req.model_dump()
    payload.pop("cohort_id")
    row = pd.DataFrame([payload])[POSTURE_FEATURES]
    cids = np.array([req.cohort_id])

    risk = float(engine.risk_of(row, cids)[0])
    p = float(engine.fitted_calibrator.p_values(np.array([risk]), cids)[0])

    surprisal = engine.fitted_baselines.surprisal_frame(row, cids)
    attrs = top_attributions(
        row.iloc[0], surprisal.iloc[0], engine.fitted_baselines, req.cohort_id, k=3
    )
    name = STATE["names"].get(req.cohort_id)

    return ScoreResponse(
        risk_score=risk,
        conformal_p=p,
        is_flagged=p <= STATE["config"].scoring.conformal_alpha,
        narrative=narrate(
            name.label if name else f"cohort {req.cohort_id}",
            attrs,
            risk,
            p,
            STATE["config"].explain,
        ),
        attributions=[
            {
                "feature": a.feature,
                "nats": round(a.surprisal, 4),
                "observed": a.observed,
                "peer_share": round(a.peer_share, 4),
                "peer_typical": a.peer_typical,
                "peer_n": a.peer_n,
            }
            for a in attrs
        ],
    )
