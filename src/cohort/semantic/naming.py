"""Naming discovered cohorts.

Cohorts arrive as integers. An analyst cannot act on "cohort 7 has an anomaly",
so each one gets a label derived from what actually distinguishes its documents
from the rest of the corpus.

The default method is class-based TF-IDF: treat each cohort as a single long
document, score terms by frequency within the cohort against frequency across
cohorts, and take the top phrases. It is deterministic, costs one pass over the
corpus, and needs no API key.

The optional LLM path turns those keyphrases into a fluent label. Note what it is
*not* given: the documents. It sees ten keyphrases and returns a noun phrase.
That is one call per cohort — roughly sixty calls for a corpus of any size — which
is the whole argument for using an LLM here and nowhere else in the hot path.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from cohort.config import ExplainConfig, SemanticConfig
from cohort.semantic.cluster import UNASSIGNED

_STOP_EXTRA = {
    "document",
    "company",
    "northwind",
    "grid",
    "shall",
    "party",
    "parties",
    "agreement",
    "date",
    "prepared",
    "owner",
    "reviewed",
    "internal",
    "team",
}


@dataclass(slots=True)
class CohortName:
    cohort_id: int
    label: str
    keyphrases: list[str]
    size: int
    source: str = "c-tfidf"


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "cohort"


def _ctfidf_terms(
    texts: list[str], labels: np.ndarray, cfg: SemanticConfig
) -> dict[int, list[str]]:
    """Class-based TF-IDF: which terms make each cohort different from the rest."""
    vec = CountVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.5,
        stop_words="english",
        strip_accents="unicode",
    )
    X = vec.fit_transform(texts)  # (n_docs, n_terms)
    vocab = np.array(vec.get_feature_names_out())

    cohort_ids = sorted({int(c) for c in np.unique(labels) if c != UNASSIGNED})
    if not cohort_ids:
        return {}

    # Sum term counts within each cohort -> (n_cohorts, n_terms)
    rows = []
    for cid in cohort_ids:
        rows.append(np.asarray(X[labels == cid].sum(axis=0)).ravel())
    C = np.vstack(rows).astype(np.float64)

    tf = C / np.maximum(C.sum(axis=1, keepdims=True), 1.0)
    avg_len = C.sum() / max(len(cohort_ids), 1)
    freq_across = np.maximum(C.sum(axis=0), 1.0)
    idf = np.log(1.0 + avg_len / freq_across)
    score = tf * idf

    out: dict[int, list[str]] = {}
    for i, cid in enumerate(cohort_ids):
        order = np.argsort(-score[i])
        phrases: list[str] = []
        for j in order:
            term = str(vocab[j])
            if any(w in _STOP_EXTRA for w in term.split()):
                continue
            # Skip a bigram whose words are already covered by chosen phrases.
            if any(term in p or p in term for p in phrases):
                continue
            phrases.append(term)
            if len(phrases) >= cfg.name_top_k:
                break
        out[cid] = phrases
    return out


def _label_from_phrases(phrases: list[str]) -> str:
    if not phrases:
        return "Unnamed cohort"
    lead = phrases[:3]
    return " / ".join(p.title() for p in lead)


def name_cohorts(
    texts: list[str],
    labels: np.ndarray,
    cfg: SemanticConfig,
    explain_cfg: ExplainConfig | None = None,
) -> dict[int, CohortName]:
    terms = _ctfidf_terms(texts, labels, cfg)
    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels)}

    names: dict[int, CohortName] = {}
    for cid, phrases in terms.items():
        names[cid] = CohortName(
            cohort_id=cid,
            label=_label_from_phrases(phrases),
            keyphrases=phrases,
            size=sizes.get(cid, 0),
        )

    if UNASSIGNED in sizes:
        names[UNASSIGNED] = CohortName(
            cohort_id=UNASSIGNED,
            label="Unassigned (no dense peer group)",
            keyphrases=[],
            size=sizes[UNASSIGNED],
            source="fallback",
        )

    if explain_cfg and explain_cfg.use_llm:
        names = _llm_relabel(names, explain_cfg)
    return names


def _llm_relabel(names: dict[int, CohortName], cfg: ExplainConfig) -> dict[int, CohortName]:
    """One short call per cohort. Silently keeps the deterministic label on failure.

    Deliberately fail-open: a missing API key or a rate limit must not break a
    scan. Naming is presentation, not detection.
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return names
    try:
        from anthropic import Anthropic
    except ImportError:
        return names

    client = Anthropic()
    for cid, cn in names.items():
        if cid == UNASSIGNED or not cn.keyphrases:
            continue
        try:
            msg = client.messages.create(
                model=cfg.llm_model,
                max_tokens=32,
                system=(
                    "You name document categories for a data security tool. "
                    "Given keyphrases, reply with a 2-5 word noun phrase naming the "
                    "document type. No preamble, no punctuation, no quotes."
                ),
                messages=[{"role": "user", "content": ", ".join(cn.keyphrases)}],
            )
            text = msg.content[0].text.strip()
            if text:
                names[cid] = CohortName(cid, text, cn.keyphrases, cn.size, source="llm")
        except Exception:
            continue
    return names
