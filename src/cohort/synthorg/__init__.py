"""synthorg — a deterministic synthetic enterprise with labelled posture anomalies.

The point of this package is evaluation. Real DSPM corpora have no ground truth:
nobody can tell you which of a customer's ten million files are genuinely
overshared. So we build an organisation whose *normal* is defined by explicit
per-category access policy, inject a known rate of known anomaly types, and emit
the labels alongside the corpus. That turns "does peer-baseline scoring work?"
into a measurable question.
"""

from cohort.synthorg.generate import generate_corpus

__all__ = ["generate_corpus"]
