"""20 Newsgroups adapter — real text with real category labels.

This corpus exists in the project for one purpose: to answer honestly the
question the synthetic benchmark cannot. On generated data, cohort discovery
scores ARI 1.000, which is a property of documents built from per-category
sentence templates and says nothing about real writing.

20 Newsgroups is the standard adversarial case for exactly this. It is real
human text, it has 20 gold labels, and several of those labels are genuinely
hard to separate — `comp.sys.ibm.pc.hardware` versus `comp.sys.mac.hardware`,
`talk.religion.misc` versus `soc.religion.christian`, `talk.politics.misc`
versus `talk.politics.guns`. Any clustering method that reports a high score
here is either very good or measuring something else.

It carries no permission metadata, so it validates the *semantic* half of the
pipeline only. The Enron adapter covers the posture half.
"""

from __future__ import annotations

import io
import re
import urllib.request
from pathlib import Path

import pandas as pd

PARQUET_URL = (
    "https://huggingface.co/api/datasets/SetFit/20_newsgroups/parquet/default/train/0.parquet"
)

# Usenet headers and quoted replies. Left in place these dominate the vocabulary
# and let a clusterer separate groups on mail-server routing rather than topic,
# which would be a flattering result measuring the wrong thing.
_HEADER = re.compile(
    r"^(From|Subject|Organization|Lines|NNTP-Posting-Host|Distribution|Reply-To|"
    r"Keywords|Article-I\.D\.|References|Sender|X-Newsreader|In-Reply-To|Path|"
    r"Newsgroups|Message-ID|Date|Summary|Expires|Followup-To):.*$",
    re.MULTILINE | re.IGNORECASE,
)
_QUOTED = re.compile(r"^\s*(>|\|).*$", re.MULTILINE)
_EMAIL = re.compile(r"[\w\.\-\+]+@[\w\.\-]+\.\w+")
_SIGNATURE = re.compile(r"\n-- \n.*", re.DOTALL)


def download(dest: Path, url: str = PARQUET_URL) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "cohort/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    dest.write_bytes(data)
    return dest


def _clean(text: str) -> str:
    text = _HEADER.sub("", text)
    text = _SIGNATURE.sub("", text)
    text = _QUOTED.sub("", text)
    text = _EMAIL.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_corpus(
    path: Path,
    min_chars: int = 200,
    max_docs: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (corpus, ground_truth) frames in Cohort's schema.

    Ground truth carries the newsgroup label as `true_category` and no anomaly
    labels — this corpus tests cohort discovery, not detection.
    """
    raw = pd.read_parquet(io.BytesIO(path.read_bytes()))
    text_col = "text" if "text" in raw.columns else raw.columns[0]
    label_col = "label_text" if "label_text" in raw.columns else "label"

    raw = raw.dropna(subset=[text_col, label_col])
    bodies = raw[text_col].astype(str).map(_clean)
    keep = bodies.str.len() >= min_chars

    raw = raw.loc[keep].reset_index(drop=True)
    bodies = bodies.loc[keep].reset_index(drop=True)
    if max_docs:
        raw = raw.head(max_docs)
        bodies = bodies.head(max_docs)

    corpus = pd.DataFrame(
        {
            "doc_id": [f"news-{i:06d}" for i in range(len(raw))],
            "title": bodies.str.slice(0, 70),
            "body": bodies,
            "language": "en",
        }
    )
    truth = pd.DataFrame(
        {
            "doc_id": corpus["doc_id"],
            "is_anomaly": False,
            "anomaly_type": "",
            "true_category": raw[label_col].astype(str).to_numpy(),
        }
    )
    return corpus, truth
