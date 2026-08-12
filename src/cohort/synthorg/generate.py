"""Corpus generation orchestrator.

Emits three artifacts:

* ``corpus.parquet``       — text + posture features, i.e. everything the pipeline is allowed to see
* ``ground_truth.parquet`` — doc_id, is_anomaly, anomaly_type, true_category (evaluation only)
* ``org.json``             — people and groups, used by the explainer to name principals

The split matters: no stage of the pipeline reads ``ground_truth.parquet`` except
``cohort.evaluate``. The scorer never learns a category label or an anomaly flag.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from cohort.config import SynthOrgConfig
from cohort.schema import POSTURE_FEATURES, AnomalyType
from cohort.synthorg import content as C
from cohort.synthorg.acl import posture_features, sample_access
from cohort.synthorg.anomalies import eligible_anomalies, inject
from cohort.synthorg.categories import CATEGORIES, normalised_base_rates
from cohort.synthorg.org import build_organisation


def _anomaly_weights() -> dict[str, float]:
    """Inverse-coverage weights so injected types come out roughly balanced.

    Eligibility is not uniform: dormant-external-access is abnormal almost
    everywhere, whereas mislabelling only makes sense for categories whose peers
    are actually confidential. Sampling uniformly from each category's eligible
    set would therefore over-produce the broadly-eligible types and leave some
    types with a dozen examples, which is too few to report per-type recall on.
    """
    rates = normalised_base_rates()
    coverage: dict[str, float] = dict.fromkeys(AnomalyType.all(), 0.0)
    for cat, r in zip(CATEGORIES, rates):
        for a in eligible_anomalies(cat):
            coverage[a] += float(r)
    return {k: (1.0 / v if v > 0 else 0.0) for k, v in coverage.items()}


def _choose_anomaly(cat, rng: np.random.Generator, weights: dict[str, float]) -> str:
    options = eligible_anomalies(cat)
    if not options:
        return ""
    w = np.array([weights[o] for o in options], dtype=float)
    if w.sum() <= 0:
        return ""
    return str(rng.choice(options, p=w / w.sum()))


def _build_owner_pools(org) -> tuple[dict[str, list[str]], list[str]]:
    """Owner candidates per category, materialised once rather than per document."""
    internal = [p.person_id for p in org.internal_people()]
    by_dept: dict[str, list[str]] = {}
    for p in org.internal_people():
        by_dept.setdefault(p.dept, []).append(p.person_id)

    pools: dict[str, list[str]] = {}
    for cat in CATEGORIES:
        pool = [pid for d in cat.owning_depts for pid in by_dept.get(d, [])]
        pools[cat.key] = pool or internal
    return pools, internal


def _pick_owner(cat, pools: dict[str, list[str]], rng: np.random.Generator) -> str:
    return str(rng.choice(pools[cat.key]))


def generate_corpus(cfg: SynthOrgConfig, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    org = build_organisation(rng, cfg.n_employees, cfg.n_external_partners)
    rates = normalised_base_rates()
    anomaly_weights = _anomaly_weights()
    owner_pools, internal_pool = _build_owner_pools(org)

    rows: list[dict] = []
    truth: list[dict] = []

    n_primary = int(cfg.n_documents * (1 - cfg.duplicate_rate))
    n_dupes = cfg.n_documents - n_primary

    for i in range(n_primary):
        doc_id = f"doc-{i:06d}"

        if rng.random() < cfg.noise_rate:
            title, body = C.render_noise_document(rng)
            # Unstructured scratch material still lives somewhere and still has
            # permissions, so give it a plausible engineering-ish posture.
            cat = CATEGORIES[int(np.argmax(rates))]
            owner = _pick_owner(cat, owner_pools, rng)
            state = sample_access(cat, org, rng, owner, internal_pool)
            feats = posture_features(state, org)
            rows.append(_row(doc_id, title, body, "en", owner, state, feats))
            truth.append(
                {
                    "doc_id": doc_id,
                    "is_anomaly": False,
                    "anomaly_type": "",
                    "true_category": "__noise__",
                }
            )
            continue

        cat = CATEGORIES[int(rng.choice(len(CATEGORIES), p=rates))]

        language = "en"
        if cat.multilingual and rng.random() < cfg.multilingual_fraction:
            language = str(rng.choice(["de", "fr"]))

        title, body = C.render_document(cat, rng, language)
        owner = _pick_owner(cat, owner_pools, rng)
        state = sample_access(cat, org, rng, owner, internal_pool)

        anomaly_type = ""
        if rng.random() < cfg.anomaly_rate:
            anomaly_type = _choose_anomaly(cat, rng, anomaly_weights)
            if anomaly_type:
                state = inject(state, cat, org, rng, AnomalyType(anomaly_type))

        feats = posture_features(state, org)
        rows.append(_row(doc_id, title, body, language, owner, state, feats))
        truth.append(
            {
                "doc_id": doc_id,
                "is_anomaly": bool(anomaly_type),
                "anomaly_type": anomaly_type,
                "true_category": cat.key,
            }
        )

    # ---- derivative copies -------------------------------------------------
    # A copy of an existing document, lightly edited, filed somewhere else with
    # independently sampled permissions. Eligible for anomaly injection like any
    # other document, which is how "sensitive file copied and then overshared"
    # enters the corpus.
    clean_idx = [j for j, t in enumerate(truth) if t["true_category"] != "__noise__"]
    for k in range(n_dupes):
        src = int(rng.choice(clean_idx))
        src_row, src_truth = rows[src], truth[src]
        cat = next(c for c in CATEGORIES if c.key == src_truth["true_category"])
        doc_id = f"doc-{n_primary + k:06d}"

        body = C.make_near_duplicate(src_row["body"], rng)
        owner = _pick_owner(cat, owner_pools, rng)
        state = sample_access(cat, org, rng, owner, internal_pool)

        anomaly_type = ""
        if rng.random() < cfg.anomaly_rate * 1.5:  # copies drift more than originals
            anomaly_type = _choose_anomaly(cat, rng, anomaly_weights)
            if anomaly_type:
                state = inject(state, cat, org, rng, AnomalyType(anomaly_type))

        feats = posture_features(state, org)
        rows.append(_row(doc_id, src_row["title"], body, src_row["language"], owner, state, feats))
        truth.append(
            {
                "doc_id": doc_id,
                "is_anomaly": bool(anomaly_type),
                "anomaly_type": anomaly_type,
                "true_category": cat.key,
            }
        )

    corpus = pd.DataFrame(rows)
    gt = pd.DataFrame(truth)

    corpus_path = out_dir / "corpus.parquet"
    truth_path = out_dir / "ground_truth.parquet"
    org_path = out_dir / "org.json"

    corpus.to_parquet(corpus_path, index=False)
    gt.to_parquet(truth_path, index=False)
    org_path.write_text(
        json.dumps(
            {
                "people": {k: asdict(v) for k, v in org.people.items()},
                "groups": {k: asdict(v) for k, v in org.groups.items()},
            },
            indent=0,
        ),
        encoding="utf-8",
    )

    return {"corpus": corpus_path, "ground_truth": truth_path, "org": org_path}


def _row(doc_id, title, body, language, owner, state, feats) -> dict:
    row = {
        "doc_id": doc_id,
        "title": title,
        "body": body,
        "language": language,
        "owner_id": owner,
        "group_ids": ",".join(state.group_ids),
        "direct_ids": ",".join(state.direct_ids),
    }
    for f in POSTURE_FEATURES:
        row[f] = feats[f]
    return row
