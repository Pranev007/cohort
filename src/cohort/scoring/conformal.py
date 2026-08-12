"""Split-conformal calibration of the risk score.

A raw anomaly score is not a decision. "4.7 nats" tells an analyst nothing about
how many false positives a threshold will generate, and alert volume is the thing
that decides whether a security product survives contact with a SOC.

Split conformal turns the score into a p-value. Hold out a calibration slice,
then for a new document

.. math::

    p(d) = \\frac{1 + |\\{c \\in \\text{cal} : S(c) \\ge S(d)\\}|}{n_{\\text{cal}} + 1}

and flag when :math:`p \\le \\alpha`. Under exchangeability this bounds the
false-positive rate at :math:`\\alpha`.

**The caveat, stated plainly.** Exchangeability requires the calibration set to be
drawn from the null. Ours is not: it is unlabelled production-like data
containing roughly the same 2% anomaly rate as everything else. The guarantee is
therefore approximate and mildly *conservative* in the direction that matters —
contamination pushes the calibration quantile up, so the realised flag rate comes
in at or below the nominal alpha. The evaluation reports the realised rate next
to the nominal one so the gap is visible rather than assumed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cohort.config import ScoringConfig


@dataclass
class ConformalCalibrator:
    cfg: ScoringConfig
    _global_cal: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)
    _cohort_cal: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def fit(self, scores: np.ndarray, cohort_ids: np.ndarray) -> ConformalCalibrator:
        self._global_cal = np.sort(np.asarray(scores, dtype=float))
        self._cohort_cal = {}
        for cid in np.unique(cohort_ids):
            s = np.sort(np.asarray(scores[cohort_ids == cid], dtype=float))
            if s.size >= self.cfg.min_cohort_for_local_conformal:
                self._cohort_cal[int(cid)] = s
        return self

    @staticmethod
    def _p(sorted_cal: np.ndarray, values: np.ndarray) -> np.ndarray:
        n = sorted_cal.size
        if n == 0:
            return np.ones_like(values, dtype=float)
        # count of calibration scores >= value
        ge = n - np.searchsorted(sorted_cal, values, side="left")
        return (1.0 + ge) / (n + 1.0)

    def p_values(self, scores: np.ndarray, cohort_ids: np.ndarray) -> np.ndarray:
        """Per-document conformal p-values.

        Computed within cohort where the cohort has enough calibration points,
        otherwise against the global calibration set. Within-cohort is the more
        honest choice — score distributions differ markedly between, say, board
        minutes and marketing briefs — but a cohort with twelve calibration
        points can only ever produce twelve distinct p-values.
        """
        scores = np.asarray(scores, dtype=float)
        out = self._p(self._global_cal, scores)
        for cid, cal in self._cohort_cal.items():
            mask = cohort_ids == cid
            if mask.any():
                out[mask] = self._p(cal, scores[mask])
        return out

    def threshold(self, alpha: float | None = None) -> float:
        """Global score threshold corresponding to the nominal alpha."""
        a = self.cfg.conformal_alpha if alpha is None else alpha
        if self._global_cal.size == 0:
            return float("inf")
        return float(np.quantile(self._global_cal, 1.0 - a))
