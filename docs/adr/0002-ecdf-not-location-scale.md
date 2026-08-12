# ADR 0002 — Empirical tail probabilities, not a location-scale model

**Status:** accepted · **Date:** 2026-08 · **Supersedes:** the median/MAD scorer

## Context

Continuous posture features need a surprisal. The obvious choice is a robust
location-scale model: centre on the cohort median, scale by MAD × 1.4826, score
`0.5·z²` on the risky tail. Robust to outliers, cheap, textbook.

It was implemented and it performed badly. Overall PR-AUC came in at **0.130**.

Diagnosis showed two distinct failures.

**Degenerate scale on zero-inflated counts.** `n_external_domains` is 0 for most
documents, so both its median *and* its MAD are 0. The implementation treated a zero
scale as "this feature carries no information" and scored it at 0 forever. Three
features — `n_external_domains`, `n_groups`, `dup_count` — contributed *exactly*
nothing. That is precisely backwards: a nonzero external-domain count is the finding.

A ladder of fallback estimators (MAD → IQR → inter-decile → standard deviation)
fixed the degeneracy and lifted PR-AUC to 0.130 → 0.447. But it did not fix the
second problem.

**Multi-modality.** `n_principals` and `accessor_dept_entropy` together contributed
**61% of the score on normal documents** while ranking anomalies no better than
chance (solo PR-AUC 0.025 and 0.027). The reason is structural: a document either
carries a broad group grant or it does not, so the within-cohort distribution has
two humps and the median sits in the valley between them. Perfectly ordinary
documents in the upper mode looked extreme, and the noise they generated drowned
out the features that actually carried signal.

No amount of robustness in the *scale* estimator fixes a wrong *shape* assumption.

## Decision

Score continuous features by the **empirical tail probability** within the cohort:

```
p = (#{peers at least as extreme as x} + 1) / (n + 1)
s = max(0, -log p - offset)
```

- Distribution-free. Multi-modality, zero-inflation and discreteness all just work.
- Laplace smoothing keeps `p` strictly inside (0, 1], so `-log p` is finite even for
  a value beyond every observation in the cohort.
- Direction-aware: only the risky tail counts. A contract shared with *fewer* people
  than its peers is not a finding, and a symmetric score would flag it just as loudly.
- The `offset` is the expected surprisal for an ordinary cohort member, estimated
  empirically over the cohort's own values with the top 5% trimmed. Trimming matters:
  those members include the anomalies being hunted, and leaving them in would inflate
  the offset and partially subtract away the signal.
- Cohort and global tail probabilities are blended by the empirical-Bayes weight, so
  shrinkage happens in probability space rather than parameter space.

## Consequences

| Configuration | PR-AUC |
|---|---:|
| Median/MAD location-scale | 0.130 |
| \+ robust scale ladder | 0.447 |
| **Empirical tail probability** | **0.597** |
| \+ top-2 aggregation (ADR 0003) | 0.623 |

Vectorising the ECDF path cohort-block at a time also made scoring ~4× faster than
the per-row loop it replaced (~14,000 docs/s).

**Costs.** Each cohort stores its sorted values, so the model is O(n) in memory
rather than O(features). At corpus scale that is a few megabytes and irrelevant; at
billions of documents it would need to become a quantile sketch (t-digest or
KLL), which is a known, bounded piece of work rather than a redesign.

Resolution is bounded by cohort size: a 40-document cohort can express at most 41
distinct tail probabilities. Shrinkage toward the global ECDF covers this, which is
the same mechanism that handles small cohorts generally.

## Alternatives considered

- **Kernel density estimation.** Handles multi-modality, but bandwidth selection
  becomes a per-feature per-cohort hyperparameter and the estimate is unstable in the
  tails — which is the only region that matters here.
- **Per-feature parametric families** (Poisson for counts, Beta for entropy). More
  principled per feature, but it makes adding a feature a modelling exercise rather
  than a schema edit, and the tails are exactly where a misspecified family hurts most.
- **Gaussian copula over all features jointly.** Captures dependence, but destroys
  exact per-feature attribution — which ADR 0003 shows was not worth paying for.
