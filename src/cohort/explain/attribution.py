"""Exact attributions, plus the peer context needed to phrase them.

Nothing here approximates anything. The score is a sum of per-feature surprisals,
so an attribution is simply the term that was added — no SHAP, no LIME, no
surrogate model, no sampling variance. When the tool says "this contributed 4.1
of the 6.3 nats", that is arithmetic rather than an estimate.

What the raw term lacks is context. "link_scope contributed 4.1 nats" is true and
useless. So each attribution is paired with the peer evidence that produced it:
how many peers share this value, what they do instead, and where the document
sits in the peer distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from cohort.schema import CATEGORICAL_FEATURES
from cohort.scoring.baseline import BaselineModel, _as_key, _tail_prob


@dataclass(slots=True)
class Attribution:
    feature: str
    surprisal: float  # nats contributed, exact
    observed: Any
    kind: str  # "categorical" | "continuous"
    #: Categorical: share of peers holding the observed value.
    #: Continuous: share of peers at or beyond the observed value on the risky side.
    peer_share: float
    #: Categorical: the value most peers hold. Continuous: the peer median.
    peer_typical: Any
    peer_n: int
    #: Which tail was scored, so the narrative can phrase the comparison correctly.
    direction: str = "high"

    @property
    def peer_count(self) -> int:
        return round(self.peer_share * self.peer_n)


def _categorical_context(
    model: BaselineModel, cohort_id: int, feature: str, observed: Any
) -> tuple[float, Any, int]:
    base = model.for_cohort(cohort_id)
    stat = base.categorical[feature]
    key = _as_key(observed)
    share = float(stat.probs.get(key, 0.0))
    typical = max(stat.probs, key=lambda k: stat.probs[k]) if stat.probs else "unknown"
    return share, typical, base.n


def _continuous_context(
    model: BaselineModel, cohort_id: int, feature: str, observed: float
) -> tuple[float, float, int]:
    """Peer share on the side that actually drove the score.

    Reuses the scorer's own tail-probability function rather than recomputing a
    one-sided share. Getting this wrong produced narratives like "it sits at path
    depth 2 — 100% of peers reach this", because `path_depth` is two-sided and an
    unusually *shallow* path was being described with the upper-tail share.
    """
    base = model.for_cohort(cohort_id)
    stat = base.continuous[feature]
    vals = stat.sorted_values
    n = int(vals.size)
    if n == 0:
        return 1.0, float(observed), 0

    share = float(_tail_prob(vals, np.array([float(observed)]), stat.direction)[0])
    return share, float(np.median(vals)), n


def top_attributions(
    row: pd.Series,
    surprisal_row: pd.Series,
    model: BaselineModel,
    cohort_id: int,
    k: int = 3,
    min_nats: float = 0.05,
) -> list[Attribution]:
    """The k features that actually drove this document's score."""
    ranked = surprisal_row.sort_values(ascending=False)
    out: list[Attribution] = []

    for feature, nats in ranked.items():
        if len(out) >= k or float(nats) < min_nats:
            break
        feature = str(feature)
        observed = row[feature]

        if feature in CATEGORICAL_FEATURES:
            share, typical, peer_n = _categorical_context(model, cohort_id, feature, observed)
            kind, direction = "categorical", "high"
        else:
            share, typical, peer_n = _continuous_context(model, cohort_id, feature, float(observed))
            kind = "continuous"
            direction = model.for_cohort(cohort_id).continuous[feature].direction

        out.append(
            Attribution(
                feature=feature,
                surprisal=float(nats),
                observed=observed,
                kind=kind,
                peer_share=share,
                peer_typical=typical,
                peer_n=peer_n,
                direction=direction,
            )
        )
    return out
