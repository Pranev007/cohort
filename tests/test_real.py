"""Real-corpus adapters and injection.

These run without the corpora present — they exercise the parsing and injection
logic on constructed inputs, so CI does not need a 443 MB download.

The test that matters most is `test_injected_values_stay_inside_the_real_range`.
It guards the experimental design of the Enron run: injected anomalies must be
values that genuinely occur in the corpus, so a flagged document is abnormal
*relative to its peers* and not simply globally extreme. If that invariant slips,
the real-data result stops testing peer baselining and starts testing whether the
scorer can spot a number nobody has ever seen — which any detector passes.
"""

from __future__ import annotations

import email
import email.header

import numpy as np
import pandas as pd
import pytest

from cohort.real.enron import _clean_body, _entropy, _header, _pii_density, _recipients
from cohort.real.inject import ENRON_ANOMALIES, inject_into_real
from cohort.real.newsgroups import _clean


# ------------------------------------------------------------------ parsing
def test_clean_body_drops_quoted_replies_and_forwards():
    raw = (
        "Here is my actual point about the gas contract.\n"
        "> you wrote something I am quoting back\n"
        "-----Original Message-----\n"
        "From: someone\nEverything after this is not mine.\n"
    )
    out = _clean_body(raw)
    assert "actual point" in out
    assert "quoting back" not in out
    assert "not mine" not in out


def test_header_handles_rfc2047_header_objects():
    """Real mail returns Header objects, not str, and a regex over one raises."""
    msg = email.message.Message()
    msg["Subject"] = email.header.Header("Q4 r\xe9sultats", "utf-8")
    value = _header(msg, "Subject")
    assert isinstance(value, str)
    assert _header(msg, "Nonexistent-Header") == ""


def test_recipients_deduplicates_across_all_address_fields():
    msg = email.message.Message()
    msg["To"] = "a@enron.com, b@enron.com"
    msg["Cc"] = "b@enron.com"
    msg["X-cc"] = "c@partner.com"
    assert _recipients(msg) == ["a@enron.com", "b@enron.com", "c@partner.com"]


def test_pii_density_scales_with_entity_count():
    filler = " ".join(["word"] * 200)
    assert _pii_density(filler) == 0.0
    assert _pii_density(filler + " 123-45-6789 person@example.com $4,000.00") > 0


def test_entropy_is_zero_for_a_single_domain():
    assert _entropy([10]) == pytest.approx(0.0)
    assert _entropy([5, 5]) > _entropy([9, 1])


def test_newsgroups_cleaner_strips_usenet_headers():
    raw = (
        "From: someone@uni.edu\nSubject: Re: encryption\nLines: 12\n"
        "NNTP-Posting-Host: foo.bar\n\n"
        "> quoted text from the parent post\n"
        "My own argument about key escrow.\n-- \nsig block\n"
    )
    out = _clean(raw)
    assert "key escrow" in out
    assert "NNTP" not in out and "quoted text" not in out and "sig block" not in out


# ---------------------------------------------------------------- injection
@pytest.fixture
def real_like_frame() -> pd.DataFrame:
    """A frame with Enron-shaped distributions: zero-inflated and heavy-tailed."""
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame(
        {
            "doc_id": [f"d-{i:05d}" for i in range(n)],
            "n_principals": np.maximum(1, rng.lognormal(0.5, 1.4, n).round()),
            "n_external_domains": np.where(rng.random(n) < 0.7, 0, rng.integers(1, 20, n)),
            "accessor_dept_entropy": np.where(rng.random(n) < 0.6, 0.0, rng.random(n) * 3),
            "staleness_days": rng.integers(3000, 9000, n).astype(float),
            "age_days": rng.integers(3000, 9000, n).astype(float),
            "path_depth": rng.integers(1, 5, n).astype(float),
            "pii_density": rng.exponential(5, n),
            "dup_count": np.zeros(n),
            "repo_type": rng.choice(
                ["inbox", "sent", "deleted_items", "rare_a", "rare_b"],
                size=n,
                p=[0.4, 0.3, 0.28, 0.01, 0.01],
            ),
            "has_external_principal": rng.random(n) < 0.3,
        }
    )


def test_injection_produces_the_requested_rate(real_like_frame):
    cohorts = np.zeros(len(real_like_frame), dtype=int)
    res = inject_into_real(real_like_frame, cohorts, rate=0.02, seed=7)
    assert res.truth["is_anomaly"].sum() == int(0.02 * len(real_like_frame))
    assert set(res.counts) == {a.value for a in ENRON_ANOMALIES}


def test_injected_values_stay_inside_the_real_range(real_like_frame):
    """The experimental-design guard.

    Every injected value must already occur in the real corpus. An injected
    document is then anomalous only against its own cohort — which is the claim
    the Enron run exists to test.
    """
    cohorts = np.random.default_rng(1).integers(0, 6, len(real_like_frame))
    res = inject_into_real(real_like_frame, cohorts, rate=0.05, seed=3)

    anom = res.truth["is_anomaly"].to_numpy()
    for feature in (
        "n_principals",
        "n_external_domains",
        "accessor_dept_entropy",
        "staleness_days",
    ):
        real_max = float(real_like_frame[feature].max())
        injected_max = float(res.corpus.loc[anom, feature].max())
        assert injected_max <= real_max + 1e-9, (
            f"{feature}: injected {injected_max} exceeds the real maximum {real_max}; "
            "the anomaly would be globally extreme rather than peer-relative"
        )

    # Relocation must use a folder that actually exists in the corpus.
    assert set(res.corpus["repo_type"]) <= set(real_like_frame["repo_type"])


def test_injection_only_touches_targeted_rows(real_like_frame):
    cohorts = np.zeros(len(real_like_frame), dtype=int)
    res = inject_into_real(real_like_frame, cohorts, rate=0.02, seed=11)
    clean = ~res.truth["is_anomaly"].to_numpy()
    pd.testing.assert_frame_equal(
        res.corpus.loc[clean].reset_index(drop=True),
        real_like_frame.loc[clean].reset_index(drop=True),
        check_dtype=False,
    )


def test_injection_is_deterministic(real_like_frame):
    cohorts = np.zeros(len(real_like_frame), dtype=int)
    a = inject_into_real(real_like_frame, cohorts, rate=0.02, seed=5)
    b = inject_into_real(real_like_frame, cohorts, rate=0.02, seed=5)
    pd.testing.assert_frame_equal(a.corpus, b.corpus)
    assert a.counts == b.counts


# --------------------------------------------------------------- html report
def test_html_report_is_self_contained_and_escaped():
    """The report must open anywhere, with no network and no injection risk."""
    import re

    from cohort.report_html import render_report

    findings = pd.DataFrame(
        [
            {
                "doc_id": "doc-000001",
                "title": "<script>alert('xss')</script> & co",
                "cohort_name": "Legal / Contracts",
                "risk_score": 12.5,
                "conformal_p": 0.003,
                "is_flagged": True,
                "top_features": "link_scope(7.1), repo_type(3.2)",
                "narrative": 'Shared with "everyone" & nobody else does.',
                "remediation": "Restrict link scope: anyone_with_link -> none.",
            }
        ]
    )
    html = render_report(
        findings,
        {"n_documents": 15000, "n_cohorts": 14, "n_flagged": 614, "flag_rate": 0.0409},
        [
            {"cohort_id": 0, "label": "Legal / Contracts", "size": 1316, "keyphrases": ["msa"]},
            {"cohort_id": -1, "label": "Unassigned", "size": 184, "keyphrases": []},
        ],
    )

    assert not re.search(r'(?:src|href)=["\']https?://', html), "must make no external requests"
    assert "<script>alert" not in html, "finding text must be HTML-escaped"
    assert "&lt;script&gt;" in html
    assert "prefers-color-scheme:dark" in html
    # A cohort with no keyphrases must not render an empty pill.
    assert "<span class='chip'></span>" not in html
    assert html.lstrip().startswith("<!doctype html>")


def test_html_report_handles_no_findings():
    from cohort.report_html import render_report

    empty = pd.DataFrame(
        columns=[
            "doc_id",
            "title",
            "cohort_name",
            "risk_score",
            "conformal_p",
            "is_flagged",
            "top_features",
            "narrative",
            "remediation",
        ]
    )
    html = render_report(empty, {"n_documents": 0}, [])
    assert "No findings match" in html
