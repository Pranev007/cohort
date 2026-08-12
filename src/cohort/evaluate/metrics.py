"""Metrics.

**PR-AUC, not ROC-AUC.** The corpus is 2% anomalous. At that base rate ROC-AUC is
close to meaningless: a scorer can post 0.93 while its top 100 findings are 70%
false positives, because the enormous true-negative pool flatters the false
positive rate. Average precision answers the question an analyst actually has —
of the documents you put in front of me, how many were worth opening. Both are
reported, ROC second, so the gap between them stays visible.

**Precision@k** is reported for k in {50, 100, 300} because a security team works
a queue. The metric that matters is the hit rate of the first screenful.

**Per-anomaly-type recall** matters more than the aggregate. A detector that finds
every overshared file and no mislabelled one has a blind spot that an aggregate
number hides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    homogeneity_completeness_v_measure,
    roc_auc_score,
)


@dataclass
class AnomalyMetrics:
    pr_auc: float
    roc_auc: float
    precision_at: dict[int, float] = field(default_factory=dict)
    recall_at: dict[int, float] = field(default_factory=dict)
    per_type: dict[str, dict[str, float]] = field(default_factory=dict)
    n_positive: int = 0
    n_total: int = 0
    #: Realised flag rate and precision at the conformal threshold.
    flag_rate: float = 0.0
    flagged_precision: float = 0.0
    flagged_recall: float = 0.0

    def as_dict(self) -> dict:
        return {
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            # String keys so the dict survives a JSON round-trip unchanged;
            # json.dump would coerce int keys anyway and the report reads both
            # the in-memory and the reloaded form.
            "precision_at": {str(k): round(v, 4) for k, v in self.precision_at.items()},
            "recall_at": {str(k): round(v, 4) for k, v in self.recall_at.items()},
            "per_type": {
                k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in self.per_type.items()
            },
            "n_positive": self.n_positive,
            "n_total": self.n_total,
            "flag_rate": round(self.flag_rate, 4),
            "flagged_precision": round(self.flagged_precision, 4),
            "flagged_recall": round(self.flagged_recall, 4),
        }


@dataclass
class ClusterMetrics:
    n_cohorts: int
    ari: float
    homogeneity: float
    completeness: float
    v_measure: float
    unassigned_rate: float
    silhouette: float

    def as_dict(self) -> dict:
        return {
            "n_cohorts": self.n_cohorts,
            "ari": round(self.ari, 4),
            "homogeneity": round(self.homogeneity, 4),
            "completeness": round(self.completeness, 4),
            "v_measure": round(self.v_measure, 4),
            "unassigned_rate": round(self.unassigned_rate, 4),
            "silhouette": round(self.silhouette, 4),
        }


def anomaly_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    anomaly_types: np.ndarray | None = None,
    flagged: np.ndarray | None = None,
    ks: tuple[int, ...] = (50, 100, 300),
) -> AnomalyMetrics:
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())

    m = AnomalyMetrics(
        pr_auc=float(average_precision_score(y_true, scores)) if n_pos else float("nan"),
        roc_auc=float(roc_auc_score(y_true, scores)) if 0 < n_pos < len(y_true) else float("nan"),
        n_positive=n_pos,
        n_total=len(y_true),
    )

    order = np.argsort(-scores)
    for k in ks:
        kk = min(k, len(order))
        hits = int(y_true[order[:kk]].sum())
        m.precision_at[k] = hits / kk if kk else 0.0
        m.recall_at[k] = hits / n_pos if n_pos else 0.0

    if anomaly_types is not None:
        types = np.asarray(anomaly_types).astype(str)
        for t in sorted({t for t in types if t and t != "nan"}):
            mask_t = types == t
            n_t = int(mask_t.sum())
            if n_t == 0:
                continue
            # One-vs-rest: this type against all clean documents. Other anomaly
            # types are excluded rather than counted as negatives — scoring them
            # as false positives would penalise the detector for being right.
            keep = mask_t | ~y_true
            pr = float(average_precision_score(mask_t[keep], scores[keep]))

            entry = {"n": float(n_t), "pr_auc": pr}
            for k in ks:
                kk = min(k, len(order))
                entry[f"recall_at_{k}"] = float(mask_t[order[:kk]].sum()) / n_t
            if flagged is not None:
                entry["recall_at_threshold"] = float((mask_t & flagged).sum()) / n_t
            m.per_type[t] = entry

    if flagged is not None:
        flagged = np.asarray(flagged).astype(bool)
        m.flag_rate = float(flagged.mean())
        m.flagged_precision = float(y_true[flagged].mean()) if flagged.any() else 0.0
        m.flagged_recall = float((y_true & flagged).sum()) / n_pos if n_pos else 0.0

    return m


def posture_coupling(
    frame: pd.DataFrame,
    cohort_ids: np.ndarray,
    categorical: list[str],
    continuous: list[str],
) -> dict[str, float]:
    r"""How much of the posture variation is explained by cohort membership.

    This is the precondition the whole method rests on, and it is measurable
    rather than assumed. Peer baselining can only beat a single global baseline
    when documents that *mean* the same thing are also *handled* the same way —
    when semantic category predicts security posture.

    In a document repository that coupling is strong by construction: contracts
    go to the legal team, offer letters to HR, and each has its own sharing norm.
    In an email archive it is weak, because how many people you copy has far more
    to do with the moment than with the topic.

    Continuous features are scored by :math:`\eta^2`, the share of total variance
    lying between cohorts rather than within them. Categorical features use
    normalised mutual information between cohort and value. Both are in [0, 1]
    and are averaged into `mean_coupling`.
    """
    from sklearn.metrics import normalized_mutual_info_score

    cohort_ids = np.asarray(cohort_ids)
    out: dict[str, float] = {}

    for f in continuous:
        x = frame[f].to_numpy(dtype=float)
        grand = x.mean()
        ss_total = float(((x - grand) ** 2).sum())
        if ss_total <= 1e-12:
            out[f] = 0.0
            continue
        ss_between = 0.0
        for c in np.unique(cohort_ids):
            xc = x[cohort_ids == c]
            if xc.size:
                ss_between += xc.size * (xc.mean() - grand) ** 2
        out[f] = float(min(1.0, ss_between / ss_total))

    for f in categorical:
        values = frame[f].astype(str).to_numpy()
        out[f] = float(normalized_mutual_info_score(cohort_ids, values))

    out["mean_coupling"] = float(np.mean([v for k, v in out.items()])) if out else 0.0
    return out


def cluster_metrics(
    true_categories: np.ndarray,
    labels: np.ndarray,
    unassigned_rate: float,
    silhouette: float,
) -> ClusterMetrics:
    """Cohort quality against the generator's categories.

    Scored on assigned documents only. Including the UNASSIGNED bucket would
    conflate two different things — how well the clusterer groups what it groups,
    and how much it declines to group — and the second is reported separately as
    `unassigned_rate`.
    """
    labels = np.asarray(labels)
    truth = np.asarray(true_categories).astype(str)
    mask = labels != -1

    if mask.sum() < 2 or len(set(labels[mask])) < 2:
        return ClusterMetrics(
            n_cohorts=len(set(labels[mask])),
            ari=float("nan"),
            homogeneity=float("nan"),
            completeness=float("nan"),
            v_measure=float("nan"),
            unassigned_rate=unassigned_rate,
            silhouette=silhouette,
        )

    h, c, v = homogeneity_completeness_v_measure(truth[mask], labels[mask])
    return ClusterMetrics(
        n_cohorts=len(set(labels[mask])),
        ari=float(adjusted_rand_score(truth[mask], labels[mask])),
        homogeneity=float(h),
        completeness=float(c),
        v_measure=float(v),
        unassigned_rate=unassigned_rate,
        silhouette=silhouette,
    )
