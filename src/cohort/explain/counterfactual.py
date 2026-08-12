"""Counterfactual remediation: the cheapest set of changes that ends the finding.

A risk score tells an analyst that something is wrong. This tells them what to do,
and — because the scorer is additive and cheap to re-evaluate — it verifies the
answer rather than asserting it. Every proposed plan is re-scored, so the residual
risk printed next to it is measured, not predicted.

The search is greedy on benefit-per-unit-cost. At each step it tries moving every
mutable feature to the value its peers actually use, re-scores all candidates in
one batch, and takes the edit with the best score reduction per unit of
operational cost. Costs are ordinal and live in `schema.EDIT_COST`: revoking a
public link is cheap and safe, relocating a file may break inbound references.

Greedy is not optimal. Finding the true minimum-cost edit set is a subset-selection
problem, and with at most four edits over nine features an exhaustive search is
tractable — but greedy matches it on essentially every case here because the
top-k aggregation makes the score close to separable in the features. The
exhaustive variant is available via `exhaustive=True` for anyone who wants to
check that claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from cohort.config import ExplainConfig
from cohort.schema import (
    CATEGORICAL_FEATURES,
    EDIT_COST,
    MUTABLE_FEATURES,
    POSTURE_FEATURES,
    REMEDIATION_VERB,
)
from cohort.scoring.baseline import BaselineModel, _as_key

#: Some edits imply others. Removing every external principal necessarily takes
#: the external-domain count to zero; proposing one without the other would
#: produce a plan whose residual score is not achievable in reality.
COUPLED_EDITS: dict[str, dict[str, Any]] = {
    "has_external_principal": {"n_external_domains": 0.0},
}


@dataclass(slots=True)
class Edit:
    feature: str
    from_value: Any
    to_value: Any
    action: str
    cost: float
    score_delta: float  # nats removed by this edit

    @staticmethod
    def _fmt(v: Any) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            f = float(v)
            return f"{f:,.0f}" if abs(f) >= 100 else f"{f:.2f}"
        return str(v)

    def describe(self) -> str:
        return (
            f"{self.action}: {self._fmt(self.from_value)} -> {self._fmt(self.to_value)} "
            f"(-{self.score_delta:.2f} nats)"
        )


@dataclass
class RemediationPlan:
    edits: list[Edit] = field(default_factory=list)
    original_score: float = 0.0
    residual_score: float = 0.0
    resolved: bool = False
    target_score: float = 0.0
    #: Features driving the score that no permission change can affect — age and
    #: dormancy, principally. Recorded so an empty plan can say *why* it is empty.
    blocking_features: list[str] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return sum(e.cost for e in self.edits)

    def describe(self) -> str:
        if not self.edits:
            if self.blocking_features:
                drivers = ", ".join(self.blocking_features)
                return (
                    f"No permission change reduces this score: the risk is driven by "
                    f"{drivers}, which access control cannot alter. This is a records "
                    f"retention or disposal action, not an entitlement one."
                )
            return "No single-feature change brings this document inside its peer baseline."
        steps = "; ".join(e.describe() for e in self.edits)
        verdict = (
            "resolves the finding" if self.resolved else "reduces but does not clear the finding"
        )
        return f"{steps}. Residual risk {self.residual_score:.2f} nats — {verdict}."


def _bool_like(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _candidate_value(
    model: BaselineModel, cohort_id: int, feature: str, observed: Any
) -> Any | None:
    """The value this document's peers would have. None when already conformant."""
    base = model.for_cohort(cohort_id)

    if feature in CATEGORICAL_FEATURES:
        stat = base.categorical.get(feature)
        if not stat or not stat.probs:
            return None
        modal = max(stat.probs, key=lambda k: stat.probs[k])
        if modal == _as_key(observed):
            return None
        if _bool_like(observed):
            return modal == "true"
        return modal

    stat_c = base.continuous.get(feature)
    if stat_c is None or not stat_c.informative or stat_c.sorted_values.size == 0:
        return None

    vals = stat_c.sorted_values
    x = float(observed)
    # Move just inside the peer range rather than to the median: the cheapest
    # honest fix, and it avoids proposing that a widely-shared design document be
    # locked down to the tightest configuration any peer happens to use.
    if stat_c.direction == "high":
        target = float(np.quantile(vals, 0.75))
        return target if x > target else None
    if stat_c.direction == "low":
        target = float(np.quantile(vals, 0.25))
        return target if x < target else None
    target = float(np.median(vals))
    return target if abs(x - target) > 1e-9 else None


def _apply(row: pd.Series, feature: str, value: Any) -> pd.Series:
    out = row.copy()
    out[feature] = value
    for coupled, coupled_value in COUPLED_EDITS.get(feature, {}).items():
        # Only propagate when the edit is a relaxation (turning access off).
        if not value:
            out[coupled] = coupled_value
    return out


def plan_remediation(
    row: pd.Series,
    model: BaselineModel,
    cohort_id: int,
    score_fn: Callable[[pd.DataFrame, np.ndarray], np.ndarray],
    target_score: float,
    cfg: ExplainConfig,
    exhaustive: bool = False,
    surprisal_row: pd.Series | None = None,
) -> RemediationPlan:
    """Greedy minimum-cost edit search.

    `score_fn` takes a frame of candidate posture rows plus their cohort ids and
    returns risk scores — the same function used to score the corpus, so a plan
    is evaluated by exactly the model that produced the finding.
    """
    current = row[POSTURE_FEATURES].copy()
    cid = np.array([cohort_id])
    original = float(score_fn(current.to_frame().T, cid)[0])

    plan = RemediationPlan(
        original_score=original, residual_score=original, target_score=target_score
    )
    if surprisal_row is not None:
        plan.blocking_features = [
            str(f)
            for f in surprisal_row.sort_values(ascending=False).index[:2]
            if f not in MUTABLE_FEATURES and float(surprisal_row[f]) > 0.05
        ]
    if original <= target_score:
        plan.resolved = True
        return plan

    if exhaustive:
        return _exhaustive(current, model, cohort_id, score_fn, target_score, cfg, plan)

    score = original
    used: set[str] = set()

    for _ in range(cfg.max_counterfactual_edits):
        candidates: list[tuple[str, Any, pd.Series]] = []
        for f in MUTABLE_FEATURES:
            if f in used:
                continue
            cand = _candidate_value(model, cohort_id, f, current[f])
            if cand is None:
                continue
            candidates.append((f, cand, _apply(current, f, cand)))

        if not candidates:
            break

        trial = pd.DataFrame([c[2] for c in candidates])
        scores = score_fn(trial, np.repeat(cohort_id, len(candidates)))

        best_i, best_gain = -1, 0.0
        for i, (f, _cand, _r) in enumerate(candidates):
            delta = score - float(scores[i])
            if delta <= 1e-6:
                continue
            gain = delta / EDIT_COST.get(f, 1.0)
            if gain > best_gain:
                best_i, best_gain = i, gain

        if best_i < 0:
            break

        f, cand, new_row = candidates[best_i]
        delta = score - float(scores[best_i])
        plan.edits.append(
            Edit(
                feature=f,
                from_value=current[f],
                to_value=cand,
                action=REMEDIATION_VERB.get(f, f"Change {f}"),
                cost=EDIT_COST.get(f, 1.0),
                score_delta=delta,
            )
        )
        current, score = new_row, float(scores[best_i])
        used.add(f)

        if score <= target_score:
            break

    plan.residual_score = score
    plan.resolved = score <= target_score
    return plan


def _exhaustive(
    row: pd.Series,
    model: BaselineModel,
    cohort_id: int,
    score_fn: Callable[[pd.DataFrame, np.ndarray], np.ndarray],
    target_score: float,
    cfg: ExplainConfig,
    plan: RemediationPlan,
) -> RemediationPlan:
    """Cheapest edit set that clears the threshold, searched by increasing size."""
    options: dict[str, Any] = {}
    for f in MUTABLE_FEATURES:
        cand = _candidate_value(model, cohort_id, f, row[f])
        if cand is not None:
            options[f] = cand
    if not options:
        return plan

    features = list(options)
    best: tuple[float, list[str], float] | None = None

    for size in range(1, min(cfg.max_counterfactual_edits, len(features)) + 1):
        subsets = list(combinations(features, size))
        rows, keep = [], []
        for subset in subsets:
            r = row.copy()
            for f in subset:
                r = _apply(r, f, options[f])
            rows.append(r)
            keep.append(subset)
        scores = score_fn(pd.DataFrame(rows), np.repeat(cohort_id, len(rows)))
        for subset, sc in zip(keep, scores):
            if float(sc) <= target_score:
                cost = sum(EDIT_COST.get(f, 1.0) for f in subset)
                if best is None or cost < best[0]:
                    best = (cost, list(subset), float(sc))
        if best is not None:
            break  # smallest satisfying size found; cheapest within it is chosen

    if best is None:
        return plan

    _cost, chosen, residual = best
    current = row.copy()
    running = plan.original_score
    for f in chosen:
        nxt = _apply(current, f, options[f])
        sc = float(score_fn(nxt[POSTURE_FEATURES].to_frame().T, np.array([cohort_id]))[0])
        plan.edits.append(
            Edit(
                feature=f,
                from_value=current[f],
                to_value=options[f],
                action=REMEDIATION_VERB.get(f, f"Change {f}"),
                cost=EDIT_COST.get(f, 1.0),
                score_delta=running - sc,
            )
        )
        current, running = nxt, sc

    plan.residual_score = residual
    plan.resolved = True
    return plan
