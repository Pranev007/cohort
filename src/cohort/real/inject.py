"""Injecting labelled anomalies into a *real* posture distribution.

The synthetic benchmark generates both the normal behaviour and the anomalies.
This module keeps the real behaviour and injects only the anomalies, which is a
strictly harder and more informative test: the baselines the scorer learns are
now real, heavy-tailed, and messy rather than drawn from a policy the generator
also wrote.

**The design decision that makes this a fair test.** An injected value is drawn
from the *upper tail of the corpus-wide real distribution*, not invented. If a
document is made "overshared", its recipient count is set to a value that
genuinely occurs elsewhere in Enron — because plenty of real messages go to
eighty people. The document is therefore anomalous **only relative to its
peers**, never globally.

That is the whole claim under test. A global detector sees a value it has seen
many times before and has no reason to flag it; a peer-relative detector sees a
value that eighty-person distribution lists make normal for all-hands mail and
bizarre for a two-person legal thread. Injecting globally-extreme values instead
would produce a benchmark that any detector passes, and would prove nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from cohort.schema import AnomalyType

#: Anomaly types expressible with the features a mail archive actually has.
#: `external_link_on_sensitive` needs a sharing-link model and `mislabeled_down`
#: needs sensitivity labels; email has neither, so they are not injected rather
#: than being simulated with a field that does not exist.
ENRON_ANOMALIES = [
    AnomalyType.OVERSHARED,
    AnomalyType.THIRD_PARTY_ACCESS,
    AnomalyType.STALE_EXTERNAL,
    AnomalyType.WRONG_LOCATION,
]


@dataclass
class InjectionResult:
    corpus: pd.DataFrame
    truth: pd.DataFrame
    counts: dict[str, int]
    #: The real-tail values the injector drew from, for the write-up.
    tail_values: dict[str, float]


def _upper_tail(series: pd.Series, lo: float = 0.95, hi: float = 0.999) -> tuple[float, float]:
    return float(series.quantile(lo)), float(series.quantile(hi))


def inject_into_real(
    corpus: pd.DataFrame,
    cohort_ids: np.ndarray,
    rate: float = 0.02,
    seed: int = 1337,
) -> InjectionResult:
    """Return (corpus with anomalies, ground truth) over a real posture frame."""
    rng = np.random.default_rng(seed)
    df = corpus.copy().reset_index(drop=True)
    n = len(df)

    # Real corpora hand back integer columns (recipient counts, domain counts).
    # Writing a float into one is a silent dtype coercion today and an error in
    # a future pandas, so widen them once rather than rounding injected values
    # and quietly changing the distribution being injected.
    for column in (
        "n_principals",
        "n_external_domains",
        "accessor_dept_entropy",
        "staleness_days",
        "age_days",
        "path_depth",
    ):
        if column in df.columns:
            df[column] = df[column].astype(float)

    # Real upper tails — the pool injected values are drawn from.
    p_lo, p_hi = _upper_tail(df["n_principals"])
    e_lo, e_hi = _upper_tail(df["n_external_domains"].replace(0, np.nan).dropna(), 0.80, 0.99)
    s_lo, s_hi = _upper_tail(df["staleness_days"], 0.90, 0.999)
    h_lo, h_hi = _upper_tail(df["accessor_dept_entropy"], 0.90, 0.999)
    tail_values = {
        "n_principals_p95": p_lo,
        "n_principals_p999": p_hi,
        "n_external_domains_p80": e_lo,
        "staleness_days_p90": s_lo,
        "accessor_dept_entropy_p90": h_lo,
    }

    # Folders that exist in the corpus but are rare, for the relocation anomaly.
    folder_counts = df["repo_type"].value_counts()
    rare_folders = [f for f, c in folder_counts.items() if c < 0.02 * n] or list(
        folder_counts.index[-3:]
    )

    is_anom = np.zeros(n, dtype=bool)
    anom_type = np.array([""] * n, dtype=object)
    targets = rng.choice(n, size=int(rate * n), replace=False)

    for i in targets:
        kind = ENRON_ANOMALIES[int(rng.integers(len(ENRON_ANOMALIES)))]

        if kind == AnomalyType.OVERSHARED:
            df.at[i, "n_principals"] = float(rng.uniform(p_lo, p_hi))
            df.at[i, "accessor_dept_entropy"] = float(rng.uniform(h_lo, h_hi))

        elif kind == AnomalyType.THIRD_PARTY_ACCESS:
            df.at[i, "n_external_domains"] = float(max(1.0, rng.uniform(e_lo, e_hi)))
            df.at[i, "has_external_principal"] = True
            df.at[i, "accessor_dept_entropy"] = float(rng.uniform(h_lo, h_hi))

        elif kind == AnomalyType.STALE_EXTERNAL:
            stale = float(rng.uniform(s_lo, s_hi))
            df.at[i, "staleness_days"] = stale
            df.at[i, "age_days"] = max(float(df.at[i, "age_days"]), stale)
            df.at[i, "has_external_principal"] = True
            df.at[i, "n_external_domains"] = float(max(1.0, df.at[i, "n_external_domains"]))

        elif kind == AnomalyType.WRONG_LOCATION:
            # Relocate into a folder that is real but rare for this document's
            # cohort — the mail equivalent of a contract in a personal drive.
            cohort = cohort_ids[i]
            here = df.loc[cohort_ids == cohort, "repo_type"].value_counts()
            options = [f for f in rare_folders if here.get(f, 0) == 0] or rare_folders
            df.at[i, "repo_type"] = str(rng.choice(options))
            df.at[i, "path_depth"] = float(max(1, df.at[i, "path_depth"] - rng.integers(0, 2)))

        is_anom[i] = True
        anom_type[i] = kind.value

    truth = pd.DataFrame(
        {
            "doc_id": df["doc_id"],
            "is_anomaly": is_anom,
            "anomaly_type": anom_type,
            "true_category": [f"cohort_{c}" for c in cohort_ids],
        }
    )
    counts = {k: int((anom_type == k).sum()) for k in {a.value for a in ENRON_ANOMALIES}}
    return InjectionResult(corpus=df, truth=truth, counts=counts, tail_values=tail_values)
