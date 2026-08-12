"""The scoring engine: additive surprisal, an interaction term, and calibration.

The interaction term deserves a note, because where it is fitted is the whole
trick. An IsolationForest over *raw* posture features would relearn what the
baselines already know, and would do it globally — blind to peer group. Instead
it is fitted over the **surprisal matrix**: each column is already "how unusual
is this feature for this document's peers", so the forest can only contribute
what the additive model structurally cannot — signal that lives in *combinations*
of mild deviations.

Dormant external access is the motivating case. A contract untouched for two
years is unremarkable. Outside counsel holding a grant is unremarkable. Both at
once is the finding.

The blend is

.. math:: R(d) = (1-\\lambda)\\,S_{\\text{additive}}(d) + \\lambda\\,S_{\\text{interaction}}(d)

and :math:`\\lambda` is a config knob rather than a conviction. The evaluation
harness sweeps it; see `docs/adr/0003-topk-aggregation.md` for the measured
outcome and why the default is what it is.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from cohort.config import ScoringConfig
from cohort.schema import POSTURE_FEATURES
from cohort.scoring.baseline import BaselineModel, fit_baselines
from cohort.scoring.conformal import ConformalCalibrator

_EPS = 1e-12


@dataclass
class ScoreResult:
    doc_ids: np.ndarray
    cohort_ids: np.ndarray
    risk: np.ndarray
    additive: np.ndarray
    interaction: np.ndarray
    p_values: np.ndarray
    flagged: np.ndarray
    attributions: pd.DataFrame  # (n, n_features) per-feature surprisal, nats
    elapsed_s: float = 0.0

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "doc_id": self.doc_ids,
                "cohort_id": self.cohort_ids,
                "risk_score": self.risk,
                "additive_score": self.additive,
                "interaction_score": self.interaction,
                "conformal_p": self.p_values,
                "is_flagged": self.flagged,
            }
        )


class RiskEngine:
    """Fit peer baselines, then score documents against them."""

    def __init__(self, cfg: ScoringConfig) -> None:
        self.cfg = cfg
        self.baselines: BaselineModel | None = None
        self.iforest: IsolationForest | None = None
        self.calibrator: ConformalCalibrator | None = None
        self._if_cal: np.ndarray = np.array([])

    # -- post-fit accessors -------------------------------------------------
    # `baselines` and `calibrator` are None until fit() runs. These turn what
    # would be a late AttributeError deep inside the explainer into a clear
    # error at the call site, and let callers avoid Optional handling.
    @property
    def fitted_baselines(self) -> BaselineModel:
        if self.baselines is None:
            raise RuntimeError("RiskEngine.fit() must be called first")
        return self.baselines

    @property
    def fitted_calibrator(self) -> ConformalCalibrator:
        if self.calibrator is None:
            raise RuntimeError("RiskEngine.fit() must be called first")
        return self.calibrator

    # -- fitting -----------------------------------------------------------
    def fit(self, df: pd.DataFrame, cohort_ids: np.ndarray, rng_seed: int = 0) -> RiskEngine:
        missing = [f for f in POSTURE_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"posture features missing from frame: {missing}")

        posture = df[POSTURE_FEATURES]
        self.baselines = fit_baselines(posture, cohort_ids, self.cfg)
        S = self.baselines.surprisal_frame(posture, cohort_ids)

        # -- robust refitting ------------------------------------------------
        # The baseline is learned from the corpus it judges, so anomalies pollute
        # it. Drop the highest-scoring tail and refit on what remains; repeat.
        # Trimming is applied *within cohort* so a cohort whose posture is simply
        # tighter than average does not lose a disproportionate share of its
        # members and end up with an artificially narrow baseline.
        for _ in range(max(0, self.cfg.robust_passes)):
            scores = self._risk_from_surprisal(S)
            keep = np.ones(len(posture), dtype=bool)
            for cid in np.unique(cohort_ids):
                idx = np.flatnonzero(cohort_ids == cid)
                if idx.size < 20:
                    continue
                n_drop = int(np.floor(self.cfg.trim_fraction * idx.size))
                if n_drop <= 0:
                    continue
                worst = idx[np.argsort(-scores[idx])[:n_drop]]
                keep[worst] = False
            if not keep.any() or keep.all():
                break
            self.baselines = fit_baselines(posture.loc[keep], cohort_ids[keep], self.cfg)
            S = self.baselines.surprisal_frame(posture, cohort_ids)

        if self.cfg.iforest_weight > 0:
            self.iforest = IsolationForest(
                n_estimators=self.cfg.iforest_n_estimators,
                max_samples="auto",
                contamination="auto",
                random_state=rng_seed,
                n_jobs=-1,
            ).fit(S.to_numpy(dtype=float))
            # Empirical distribution of the raw forest score, used to map it
            # onto the same nats-ish scale as the additive term.
            self._if_cal = np.sort(-self.iforest.score_samples(S.to_numpy(dtype=float)))

        # Calibration split: a random slice is held out so conformal p-values are
        # not computed against scores the same documents helped produce.
        rng = np.random.default_rng(rng_seed)
        n = len(df)
        k = max(1, int(self.cfg.calibration_fraction * n))
        cal_idx = rng.choice(n, size=k, replace=False)

        cal_scores = self._risk_from_surprisal(S.iloc[cal_idx])
        self.calibrator = ConformalCalibrator(self.cfg).fit(cal_scores, cohort_ids[cal_idx])
        return self

    # -- scoring -----------------------------------------------------------
    def _interaction(self, S: pd.DataFrame) -> np.ndarray:
        if self.iforest is None or self.cfg.iforest_weight <= 0:
            return np.zeros(len(S), dtype=float)
        raw = -self.iforest.score_samples(S.to_numpy(dtype=float))
        if self._if_cal.size == 0:
            return np.zeros(len(S), dtype=float)
        # Empirical CDF against the fitted distribution, then -log(1-u) so the
        # term lives on the same unbounded-positive scale as surprisal.
        u = np.searchsorted(self._if_cal, raw, side="left") / (self._if_cal.size + 1.0)
        s = -np.log(np.maximum(1.0 - u, _EPS))
        return np.minimum(s, self.cfg.max_feature_surprisal)

    def _aggregate(self, S: pd.DataFrame) -> np.ndarray:
        """Collapse the per-feature surprisal matrix into one additive score."""
        M = S.to_numpy(dtype=float)
        if self.cfg.aggregation == "sum":
            return M.sum(axis=1)
        k = max(1, min(self.cfg.top_k, M.shape[1]))
        # Partial sort is enough: we need the k largest, not their order.
        part = np.partition(M, -k, axis=1)[:, -k:]
        return part.sum(axis=1)

    def _risk_from_surprisal(self, S: pd.DataFrame) -> np.ndarray:
        additive = self._aggregate(S)
        if self.cfg.iforest_weight <= 0:
            return additive
        inter = self._interaction(S)
        w = self.cfg.iforest_weight
        return (1.0 - w) * additive + w * inter

    def risk_of(self, df: pd.DataFrame, cohort_ids: np.ndarray) -> np.ndarray:
        """Risk score for arbitrary posture rows against already-fitted baselines.

        This is what makes counterfactual remediation trustworthy: a proposed fix
        is evaluated by exactly the model that raised the finding, so the residual
        risk reported beside a plan is measured rather than predicted.
        """
        if self.baselines is None:
            raise RuntimeError("RiskEngine.fit() must be called before risk_of()")
        S = self.baselines.surprisal_frame(df[POSTURE_FEATURES], cohort_ids)
        return self._risk_from_surprisal(S)

    def score(self, df: pd.DataFrame, cohort_ids: np.ndarray, doc_ids: np.ndarray) -> ScoreResult:
        if self.baselines is None or self.calibrator is None:
            raise RuntimeError("RiskEngine.fit() must be called before score()")

        t0 = time.perf_counter()
        S = self.baselines.surprisal_frame(df[POSTURE_FEATURES], cohort_ids)
        additive = self._aggregate(S)
        interaction = self._interaction(S)

        w = self.cfg.iforest_weight
        risk = (1.0 - w) * additive + w * interaction if w > 0 else additive

        p = self.calibrator.p_values(risk, cohort_ids)
        flagged = p <= self.cfg.conformal_alpha

        return ScoreResult(
            doc_ids=np.asarray(doc_ids),
            cohort_ids=np.asarray(cohort_ids),
            risk=risk,
            additive=additive,
            interaction=interaction,
            p_values=p,
            flagged=flagged,
            attributions=S,
            elapsed_s=time.perf_counter() - t0,
        )

    def fit_score(
        self, df: pd.DataFrame, cohort_ids: np.ndarray, doc_ids: np.ndarray, rng_seed: int = 0
    ) -> ScoreResult:
        """Fit and score the same corpus.

        This is the operational mode: a scan has no labelled training set and no
        held-out future. Baselines are learned from the corpus being judged,
        which is legitimate because the method is unsupervised — but it does mean
        a cohort consisting *entirely* of overshared documents would learn that
        oversharing is normal. That failure mode is real and is documented in
        MODEL_CARD.md rather than engineered around.
        """
        return self.fit(df, cohort_ids, rng_seed).score(df, cohort_ids, doc_ids)
