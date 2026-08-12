"""Natural-language findings.

Deterministic templates, not a language model. Three reasons, in order of how
much they matter:

1. A security finding is evidence. "0 of 312 peer contracts use link sharing" is
   a claim about the data that must be exactly true, and a template that reads
   the numbers straight out of the attribution cannot drift from them.
2. It runs with no API key, no network, and no per-finding cost.
3. It is reproducible, so the same scan produces the same report twice.

The optional LLM pass (`ExplainConfig.use_llm`) rewrites the assembled sentence
for fluency only, and is given the numbers rather than the corpus. Off by default.
"""

from __future__ import annotations

import os

from cohort.config import ExplainConfig
from cohort.explain.attribution import Attribution

_FEATURE_PHRASE: dict[str, str] = {
    "link_scope": "its sharing-link scope is {observed}",
    "repo_type": "it is stored in {observed}",
    "label_tier": "it carries the {observed} sensitivity label",
    "acl_origin": "its permissions are {observed} rather than inherited",
    "has_external_principal": "it grants access to external principals",
    "owner_dept_is_modal": "its owner sits outside the department that normally holds this material",
    "n_principals": "{observed:.0f} principals can reach it",
    "n_groups": "it carries {observed:.0f} group grants",
    "n_external_domains": "it is exposed to {observed:.0f} external domain(s)",
    "accessor_dept_entropy": "its audience spans an unusually wide set of departments",
    "path_depth": "it sits at path depth {observed:.0f}",
    "age_days": "it is {observed:.0f} days old",
    "staleness_days": "it has been untouched for {observed:.0f} days",
    "pii_density": "it carries an unusually high density of personal data",
    "dup_count": "it has {observed:.0f} near-duplicate copies elsewhere",
}


def _phrase(a: Attribution) -> str:
    template = _FEATURE_PHRASE.get(a.feature, f"{a.feature} is {{observed}}")
    if a.feature == "has_external_principal" and not a.observed:
        return "it grants no external access"
    try:
        return template.format(observed=a.observed)
    except (ValueError, TypeError):
        return template.replace("{observed:.0f}", str(a.observed)).replace(
            "{observed}", str(a.observed)
        )


def _num(x: float) -> str:
    """Format a peer statistic without rounding small quantities away.

    `accessor_dept_entropy` has a median around 0.26; printing it as "0" made the
    evidence line read as though peers had no departmental spread at all.
    """
    ax = abs(x)
    if ax >= 100:
        return f"{x:,.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def _evidence(a: Attribution) -> str:
    """The peer comparison that justifies the claim."""
    if a.peer_n == 0:
        return "no peer baseline available"

    if a.kind == "categorical":
        count = a.peer_count
        if count == 0:
            return f"no other document in this cohort of {a.peer_n} does"
        return f"only {count} of {a.peer_n} peers do, and most use '{a.peer_typical}'"

    pct = a.peer_share * 100
    median = _num(float(a.peer_typical))
    verb = "are this far from the norm" if a.direction == "both" else "reach this"
    if pct < 1:
        return f"fewer than 1% of {a.peer_n} peers {verb}, median is {median}"
    return f"{pct:.0f}% of {a.peer_n} peers {verb}, median is {median}"


def narrate(
    cohort_name: str,
    attributions: list[Attribution],
    risk_score: float,
    conformal_p: float,
    cfg: ExplainConfig | None = None,
) -> str:
    if not attributions:
        return (
            f"No individual posture feature stands out against the '{cohort_name}' "
            f"baseline (risk {risk_score:.1f} nats, p={conformal_p:.3f})."
        )

    lead = attributions[0]
    total = sum(a.surprisal for a in attributions)
    share = lead.surprisal / total if total > 0 else 0.0

    parts = [
        f"Compared with {lead.peer_n} peer documents in '{cohort_name}', "
        f"{_phrase(lead)} — {_evidence(lead)}."
    ]
    for a in attributions[1:]:
        parts.append(f"In addition, {_phrase(a)} ({_evidence(a)}).")

    parts.append(
        f"Risk {risk_score:.1f} nats (conformal p={conformal_p:.4f}); "
        f"{share:.0%} of that comes from {lead.feature}."
    )
    text = " ".join(parts)

    if cfg and cfg.use_llm:
        text = _llm_polish(text, cfg) or text
    return text


def _llm_polish(text: str, cfg: ExplainConfig) -> str | None:
    """Fluency pass only. Fails open — a narration failure must not fail a scan."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        from anthropic import Anthropic

        msg = Anthropic().messages.create(
            model=cfg.llm_model,
            max_tokens=220,
            system=(
                "Rewrite the security finding as two clear sentences for a SOC analyst. "
                "Preserve every number and every claim exactly. Do not add facts, "
                "speculation, or recommendations. No preamble."
            ),
            messages=[{"role": "user", "content": text}],
        )
        return msg.content[0].text.strip() or None
    except Exception:
        return None
