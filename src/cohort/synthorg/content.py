"""Document body rendering.

Each occurrence of a placeholder is filled independently, so a template with two
`{term}` slots yields two different domain terms. That keeps the surface text
varied enough that TF-IDF does not simply memorise one template per category.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from cohort.synthorg import names as N
from cohort.synthorg.categories import BOILERPLATE, CLOSERS, Category

_SLOT = re.compile(r"\{(\w+)\}")


@dataclass(slots=True)
class RenderContext:
    company: str
    counterparty: str
    person: str
    project: str
    city: str
    service: str
    dept: str
    title: str


def _new_context(rng: np.random.Generator) -> RenderContext:
    dept = str(rng.choice(N.DEPARTMENTS))
    return RenderContext(
        company=N.COMPANY,
        counterparty=str(rng.choice(N.COUNTERPARTIES)),
        person=N.person_name(rng),
        project=str(rng.choice(N.PROJECTS)),
        city=str(rng.choice(N.CITIES)),
        service=str(rng.choice(N.SERVICES)),
        dept=dept,
        title=str(rng.choice(N.TITLES[dept])),
    )


def _fill(template: str, rng: np.random.Generator, ctx: RenderContext, lexicon: list[str]) -> str:
    def sub(m: re.Match[str]) -> str:
        slot = m.group(1)
        if slot == "term":
            return str(rng.choice(lexicon))
        if slot == "money":
            return N.money(rng)
        if slot == "date":
            return N.date_str(rng)
        if slot == "n":
            return str(int(rng.integers(1, 800)))
        if slot == "pct":
            return f"{rng.integers(1, 60)}.{rng.integers(0, 10)}%"
        if slot == "ref":
            return f"{rng.integers(10000, 99999)}"
        if slot == "person":
            return N.person_name(rng)
        if slot == "counterparty":
            return str(rng.choice(N.COUNTERPARTIES))
        return str(getattr(ctx, slot, slot))

    out = _SLOT.sub(sub, template)
    return out[:1].upper() + out[1:] if out else out


def render_document(
    cat: Category,
    rng: np.random.Generator,
    language: str = "en",
) -> tuple[str, str]:
    """Return (title, body) for one document of the given category."""
    ctx = _new_context(rng)

    if language == "de" and cat.de_templates:
        pool = cat.de_templates
    elif language == "fr" and cat.fr_templates:
        pool = cat.fr_templates
    else:
        pool = cat.templates

    n_sent = int(rng.integers(6, min(len(pool), 11) + 1))
    idx = rng.choice(len(pool), size=min(n_sent, len(pool)), replace=False)
    # Keep near-template order so documents read like documents, not word salad.
    idx = np.sort(idx)
    sentences = [_fill(pool[i], rng, ctx, cat.lexicon) for i in idx]

    # Shared corporate furniture, present on most but not all documents. Only on
    # English documents: mixing an English confidentiality footer into a German
    # contract would hand the clustering stage a language-invariant crib that a
    # real multilingual corpus would not contain.
    if language == "en":
        if rng.random() < 0.55:
            sentences.insert(0, _fill(str(rng.choice(BOILERPLATE)), rng, ctx, cat.lexicon))
        if rng.random() < 0.45:
            sentences.append(_fill(str(rng.choice(BOILERPLATE)), rng, ctx, cat.lexicon))
        if rng.random() < 0.70:
            sentences.append(_fill(str(rng.choice(CLOSERS)), rng, ctx, cat.lexicon))

    title = _fill(str(rng.choice(cat.title_patterns)), rng, ctx, cat.lexicon)
    body = " ".join(sentences)
    return title, body


def render_noise_document(rng: np.random.Generator) -> tuple[str, str]:
    """A document belonging to no clean category.

    Real corpora are full of these: meeting scratch notes, forwarded threads,
    half-finished drafts. They exist so the clustering stage has to cope with
    genuine outliers rather than a perfectly partitioned corpus.
    """
    ctx = _new_context(rng)
    fragments = [
        "Notes from the sync — {person} to follow up on the open items before {date}.",
        "Forwarded thread: see below for context on the {project} discussion.",
        "Draft, do not circulate. Numbers still to be confirmed with {dept}.",
        "Action items: 1) confirm scope 2) chase {counterparty} 3) update the tracker.",
        "Placeholder page created during the {project} kickoff, content pending.",
        "Attendees: {person}, {person}. Apologies: {person}.",
        "Rough working file — superseded by the version in the shared folder.",
        "Agenda: status round, risks, decisions needed, any other business.",
        "Copied from the {city} workshop whiteboard, needs tidying.",
        "Ref {ref}. Owner to be assigned.",
    ]
    n = int(rng.integers(3, 7))
    idx = rng.choice(len(fragments), size=n, replace=False)
    sentences = [_fill(fragments[i], rng, ctx, ["item", "note", "topic"]) for i in idx]
    return _fill("Untitled notes {ref}", rng, ctx, ["note"]), " ".join(sentences)


def make_near_duplicate(body: str, rng: np.random.Generator) -> str:
    """Produce a derivative copy: same document, lightly edited.

    Mirrors what actually happens in an enterprise — someone downloads a file,
    tweaks a line, and re-uploads it somewhere with different permissions.
    """
    sentences = [s for s in body.split(". ") if s]
    if len(sentences) > 3 and rng.random() < 0.6:
        drop = int(rng.integers(0, len(sentences)))
        sentences.pop(drop)
    if rng.random() < 0.5:
        sentences.insert(0, "REVISED COPY — supersedes the previously circulated version")
    if rng.random() < 0.4:
        sentences.append("Updated following review comments")
    return ". ".join(sentences)
