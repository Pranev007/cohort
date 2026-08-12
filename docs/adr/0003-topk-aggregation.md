# ADR 0003 — Top-k aggregation, and the interaction term that didn't earn its place

**Status:** accepted · **Date:** 2026-08

## Context

Fifteen per-feature surprisals have to become one score. The default is to add them
up, which is what independence implies. But summing dilutes: a document extreme on
two dimensions competes with one mildly odd across all fifteen, and the second is
usually just a document.

Measured on normal documents, `pii_density` contributed 11.5% of the total score
while having a solo PR-AUC of 0.023 — pure noise, structurally guaranteed a seat at
the table by the sum.

The tempting fix is to learn feature weights. That is not available here: the method
is unsupervised, and hard-coding weights derived from the labelled synthetic
benchmark would be fitting the detector to its own generator.

## Decision

**Sum only the k largest surprisals, k = 2.**

An anomaly is a document extreme on *a few* dimensions, not one mildly odd on all of
them. This needs no labels, no weights, and — critically — preserves exact
attribution: the top-k features are simultaneously the score and the explanation.

| Aggregation | PR-AUC | P@50 |
|---|---:|---:|
| sum (all 15) | 0.434 | 0.760 |
| top-1 | 0.443 | 0.620 |
| **top-2** | **0.623** | **1.000** |
| top-3 | 0.562 | 0.860 |
| top-4 | 0.508 | 0.820 |

### Guarding against tuning on the benchmark

k = 2 was chosen on labelled data, which is legitimate hyperparameter selection and
illegitimate if left unvalidated. Two checks:

1. **Two unseen corpus seeds.** k = 2 wins on both — 0.534 (seed 4242) and 0.561
   (seed 777), beating k = 1, 3, 4 and sum in each case.
2. **The curve is flat, not a spike.** k = 2 and k = 3 differ by 0.055; this is not
   a knife-edge that happens to sit on one benchmark.

The a-priori objection stands and is worth stating: injected anomalies here perturb
one or two features, so k = 2 is somewhat matched to the generator by construction.
On a corpus where anomalies are diffuse across many features, `sum` would be the
better choice. It is a config flag (`scoring.aggregation`) for exactly that reason.

## The interaction term, and why it is off

An IsolationForest was added over the **surprisal matrix** — not raw features, so it
could only contribute what the additive model structurally cannot: signal living in
*combinations* of mild deviations. The motivating case is real. A contract untouched
for two years is unremarkable; outside counsel holding a grant is unremarkable; both
at once is the finding.

It was built, swept, and did not pay for itself:

| λ (blend weight) | PR-AUC |
|---:|---:|
| **0.00 (off)** | **0.623** |
| 0.15 | 0.614 |
| 0.25 | 0.608 |
| 0.40 | 0.594 |

It costs PR-AUC monotonically, and it makes a λ-share of every score unattributable
to any feature — which is the one property the whole explainability story rests on.

**Default λ = 0.** The code stays, the sweep re-runs on every evaluation, and if a
future feature set makes interactions matter the harness will say so. A negative
result that is measured and kept visible is worth more than a plausible component
shipped on intuition.

## Consequences

- Single-feature anomalies are handicapped. `mislabeled_down` has exactly one strong
  signal, so its score is one real term plus one noise term, and it ranks at 111–555
  of 15,000 rather than at the top (PR-AUC 0.044). It is still caught by the
  calibrated threshold at 47% recall. This is the measured price of +0.18 PR-AUC
  overall, and it is stated in the README rather than buried.
- Attribution stays exact. `test_topk_score_equals_sum_of_top_k_attributions`
  asserts the identity numerically.
- Complexity stays low: one `np.partition` call, no training, no weights to maintain.
