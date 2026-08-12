# ADR 0004 — Conformal calibration under a contaminated null

**Status:** accepted · **Date:** 2026-08

## Context

A raw anomaly score is not a decision. "4.7 nats" tells an analyst nothing about how
many false positives a threshold will generate, and alert volume is what decides
whether a security product survives contact with a SOC. Thresholds picked by eye do
not transfer between corpora, because the score distribution depends on how many
cohorts there are and how tight each one is.

## Decision

**Split-conformal p-values.** Hold out a calibration slice (30%), then

$$p(d) = \frac{1 + |\{c \in \text{cal} : S(c) \ge S(d)\}|}{n_{\text{cal}} + 1}$$

and flag when $p \le \alpha$. Under exchangeability this bounds the false-positive
rate at $\alpha$, and the bound is distribution-free — no assumption about the shape
of the score distribution, which is exactly the property a score built from summed
surprisals needs.

Computed **within cohort** when a cohort has ≥50 calibration points, otherwise
globally. Within-cohort is more honest — board minutes and marketing briefs have
genuinely different score distributions — but a cohort with twelve calibration points
can only express twelve distinct p-values, so the fallback matters.

## The caveat, stated rather than assumed away

Exchangeability requires the calibration set to be drawn from the null. **Ours is
not.** It is unlabelled production-like data carrying the same ~2% anomaly rate as
everything else. The guarantee is therefore approximate.

The direction of the error is the forgiving one: contamination pushes the calibration
quantile *up*, so the realised flag rate comes in at or below nominal. Measured:
**4.01% realised at α = 0.05**. The evaluation prints both numbers side by side so
the gap stays visible.

A test (`test_conformal_flag_rate_tracks_alpha`) verifies the mechanism on genuinely
clean exchangeable data, where the realised rate lands within 0.03–0.07 of nominal.
That separates "the implementation is correct" from "the assumption is imperfect",
which are different claims and deserve different evidence.

## Ranking by p-value was tried and is worse

Raw surprisals are compared across cohorts with different baseline entropies, which
is arguably unfair: a tightly-distributed cohort produces larger surprisals for the
same degree of abnormality. Conformal p-values are computed within cohort, so ranking
by them should correct that.

Measured, it does not:

| Ranking | PR-AUC | P@50 |
|---|---:|---:|
| **Raw surprisal** | **0.597** | **0.98** |
| Within-cohort conformal p | 0.544 | 0.90 |

Within-cohort p-values are discrete — bounded by cohort size — and the resolution
lost to ties costs more than the cross-cohort comparability gains. Ranking uses the
raw score; the p-value is used for the flag decision, where it belongs.

## Consequences

- `conformal_alpha` is the single operational knob, and it means something an
  operator can reason about: expected alert volume.
- Alert budget becomes predictable across corpora without retuning.
- The guarantee is approximate under contamination, and the README says so.
- Robust refitting (`robust_passes`) reduces contamination in the *baseline* but not
  in the calibration set; removing suspected anomalies from calibration would break
  exchangeability in the other direction and was not done.
