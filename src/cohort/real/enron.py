"""Enron corpus adapter — real documents, real sharing behaviour.

Why this corpus. Every posture feature Cohort scores is a property of *who can
reach a document and where it lives*, and public document corpora almost never
carry that. Enron does, implicitly: a message has real recipients (an access
list), real external domains, a real filing location inside a user's maildir, and
a real timestamp. That makes it the closest thing to a real DSPM corpus that can
be downloaded without a customer contract.

What is real here, and what is not:

===========================  ==========================================================
n_principals                 REAL — distinct To/Cc/Bcc recipients
n_external_domains           REAL — recipient domains outside enron.com
has_external_principal       REAL
accessor_dept_entropy        REAL — Shannon entropy over recipient domains
repo_type                    REAL — the maildir folder (inbox, sent, deleted_items, ...)
path_depth                   REAL — folder nesting depth
age_days / staleness_days    REAL — derived from the Date header
dup_count                    REAL — computed by Cohort's own lineage stage
pii_density                  DERIVED — entity extraction over the body
link_scope                   ABSENT — email has no sharing-link concept
label_tier                   ABSENT — Enron has no sensitivity labels
acl_origin, n_groups         ABSENT — no inheritance or group model in email
owner_dept_is_modal          ABSENT — no org chart
===========================  ==========================================================

The five absent features are left inert rather than faked, so the Enron run
scores on 10 of 15. A constant categorical has zero entropy and a constant
continuous feature is flagged uninformative, so each contributes exactly 0 nats
and the scorer runs unmodified. The reduced feature count is reported alongside
the results rather than papered over.

Streamed, not extracted: the archive is 443 MB of gzip containing ~517k tiny
files, and unpacking that on a laptop filesystem takes far longer than reading it.
"""

from __future__ import annotations

import email
import email.header
import email.utils
import math
import re
import tarfile
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ENRON_URL = "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz"
INTERNAL_DOMAIN = "enron.com"

#: Features Enron can actually support. The rest of POSTURE_FEATURES is dropped.
ENRON_CATEGORICAL = ["repo_type", "has_external_principal"]
ENRON_CONTINUOUS = [
    "n_principals",
    "n_external_domains",
    "accessor_dept_entropy",
    "path_depth",
    "age_days",
    "staleness_days",
    "pii_density",
    "dup_count",
]
ENRON_FEATURES = ENRON_CATEGORICAL + ENRON_CONTINUOUS

_ADDR = re.compile(r"[\w\.\-\+']+@[\w\.\-]+\.\w+")
_PHONE = re.compile(r"\b(?:\+?\d{1,2}[\s\-\.])?\(?\d{3}\)?[\s\-\.]\d{3}[\s\-\.]\d{4}\b")
_MONEY = re.compile(r"\$\s?\d[\d,]*(?:\.\d{2})?")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ACCOUNT = re.compile(r"\b\d{9,16}\b")

# Boilerplate that appears on a large share of Enron mail and would otherwise
# dominate the leading singular vectors.
_FORWARD_MARKER = re.compile(
    r"(-{2,}\s*Original Message\s*-{2,}|Forwarded by .{0,80}on \d)", re.IGNORECASE
)


@dataclass
class EnronStats:
    scanned: int = 0
    kept: int = 0
    skipped_no_body: int = 0
    skipped_no_recipients: int = 0
    skipped_no_date: int = 0
    bytes_read: int = 0
    #: Set when the archive ended early (partial download / corrupt tail).
    truncated_at: str = ""


ARCHIVE_BYTES = 443_254_787


def download(
    dest: Path,
    url: str = ENRON_URL,
    chunk: int = 1 << 20,
    expected_bytes: int = ARCHIVE_BYTES,
    max_attempts: int = 8,
) -> Path:
    """Fetch the archive, resuming a partial file rather than restarting.

    443 MB over an unreliable link will not always arrive in one piece — this one
    was reset by the server at 88% during development. Without range resume the
    only recovery is to throw away 388 MB and start again, which is both
    infuriating and, on a metered connection, expensive. Each attempt asks for
    the remaining byte range and appends.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    for _ in range(max_attempts):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= expected_bytes:
            return dest

        headers = {"User-Agent": "cohort/1.0"}
        if have:
            headers["Range"] = f"bytes={have}-"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r:
                # A server that ignores Range replies 200 with the whole file;
                # appending then would corrupt what is already on disk.
                mode = "ab" if (have and r.status == 206) else "wb"
                with open(dest, mode) as f:
                    while True:
                        buf = r.read(chunk)
                        if not buf:
                            break
                        f.write(buf)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            continue  # retry from wherever the file now ends

        if dest.exists() and dest.stat().st_size >= expected_bytes:
            return dest

    if not dest.exists():
        raise RuntimeError(f"could not download {url}")
    return dest  # partial; parse_corpus stops cleanly at the truncation point


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def _clean_body(raw: str) -> str:
    """Strip quoted-reply chains and signature noise.

    Enron mail is heavily forwarded; without this, half the corpus is other
    people's text and clustering groups threads rather than topics.
    """
    cut = _FORWARD_MARKER.search(raw)
    if cut:
        raw = raw[: cut.start()]
    lines = [ln for ln in raw.splitlines() if not ln.lstrip().startswith(">")]
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _pii_density(body: str) -> float:
    """Entities per 1,000 tokens.

    Pattern extraction, not classification: the pipeline never uses a rule to
    decide what a document *is*, only to count identifiers inside it.
    """
    tokens = max(len(body.split()), 1)
    hits = (
        len(_ADDR.findall(body))
        + len(_PHONE.findall(body))
        + len(_MONEY.findall(body))
        + 3 * len(_SSN.findall(body))
        + len(_ACCOUNT.findall(body))
    )
    return 1000.0 * hits / tokens


def _header(msg: email.message.Message, field: str) -> str:
    """Header value as plain text.

    Real mail does not cooperate: RFC 2047 encoded-words make `Message.get`
    return an `email.header.Header` rather than a `str`, and a regex over that
    raises. Coerce everything to text and never let one malformed header take
    down a scan of half a million messages.
    """
    value = msg.get(field)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return str(email.header.make_header(email.header.decode_header(str(value))))
    except Exception:
        return str(value)


def _recipients(msg: email.message.Message) -> list[str]:
    out: list[str] = []
    for field in ("To", "Cc", "Bcc", "X-To", "X-cc", "X-bcc"):
        value = _header(msg, field)
        if value:
            out.extend(_ADDR.findall(value))
    return sorted({a.lower() for a in out})


def _iter_members(tar: tarfile.TarFile, stats: EnronStats):
    """Yield members, stopping at the first read failure instead of raising."""
    while True:
        try:
            member = tar.next()
        except (tarfile.ReadError, EOFError, OSError) as exc:
            stats.truncated_at = type(exc).__name__
            return
        if member is None:
            return
        yield member


def parse_corpus(
    archive: Path,
    limit: int = 40_000,
    stride: int = 12,
    min_body_chars: int = 220,
) -> tuple[pd.DataFrame, EnronStats]:
    """Stream the archive and build a document frame.

    `stride` samples every Nth qualifying message so the sample spans the whole
    archive — all ~150 custodians — instead of only the users whose names sort
    first. The archive is read once, sequentially, and never unpacked.
    """
    stats = EnronStats()
    rows: list[dict] = []
    now = datetime.now(UTC)
    qualifying = 0

    with tarfile.open(archive, mode="r|gz") as tar:
        # A truncated or partially-downloaded archive raises on the member that
        # straddles the cut. Everything read before that point is still valid, so
        # stop cleanly and report what was parsed rather than losing the lot.
        for member in _iter_members(tar, stats):
            if not member.isfile():
                continue
            stats.scanned += 1

            parts = Path(member.name).parts
            # maildir/<custodian>/<folder...>/<n>
            if len(parts) < 4 or parts[0] != "maildir":
                continue

            fh = tar.extractfile(member)
            if fh is None:
                continue
            raw = fh.read()
            stats.bytes_read += len(raw)

            try:
                msg = email.message_from_bytes(raw)
            except Exception:
                continue

            payload = msg.get_payload()
            if not isinstance(payload, str):
                continue
            body = _clean_body(payload)
            if len(body) < min_body_chars:
                stats.skipped_no_body += 1
                continue

            recips = _recipients(msg)
            if not recips:
                stats.skipped_no_recipients += 1
                continue

            date_hdr = _header(msg, "Date")
            try:
                dt = email.utils.parsedate_to_datetime(date_hdr) if date_hdr else None
            except (TypeError, ValueError):
                dt = None
            if dt is None:
                stats.skipped_no_date += 1
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)

            qualifying += 1
            if qualifying % stride != 0:
                continue

            custodian = parts[1]
            folder = parts[2]
            domains = Counter(a.split("@")[-1] for a in recips)
            external = {d: c for d, c in domains.items() if d != INTERNAL_DOMAIN}
            age = max(0.0, (now - dt).days)

            rows.append(
                {
                    "doc_id": f"enron-{len(rows):06d}",
                    "title": (_header(msg, "Subject") or "(no subject)").strip()[:160],
                    "body": body[:12_000],
                    "language": "en",
                    "owner_id": custodian,
                    "repo_type": folder,
                    "has_external_principal": bool(external),
                    "n_principals": float(len(recips)),
                    "n_external_domains": float(len(external)),
                    "accessor_dept_entropy": _entropy(list(domains.values())),
                    "path_depth": float(len(parts) - 2),
                    "age_days": age,
                    # No modification history in a mail archive: the message is
                    # frozen at send time, so dormancy equals age. Kept as its own
                    # column because the scorer treats them as separate features.
                    "staleness_days": age,
                    "pii_density": _pii_density(body),
                    "dup_count": 0.0,
                }
            )
            stats.kept += 1
            if stats.kept >= limit:
                break

    return pd.DataFrame(rows), stats
