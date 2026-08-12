# ADR 0005 — Synthetic ground truth, and how to keep it honest

**Status:** accepted · **Date:** 2026-08

## Context

Real DSPM corpora have no labels. Nobody can tell you which of a customer's ten
million files are genuinely overshared — that is the entire reason the product
exists. So "does peer-baseline scoring work?" is unanswerable on real data without
an expensive manual audit.

Public document corpora (Enron, CUAD, RVL-CDIP, EDGAR) supply realistic *content* but
no permission metadata at all, and permissions are the thing being scored.

## Decision

**Generate a synthetic enterprise whose "normal" is defined by explicit per-category
access policy, inject a known rate of known anomaly types, and emit the labels
alongside the corpus.**

This converts an unanswerable question into a measurable one, at the cost of
measuring on data you made up. Four safeguards keep that cost bounded.

**1. Eligibility is derived from policy, not asserted.** An anomaly type is only
injected into a category where it is genuinely abnormal *for that category*.
Anyone-with-link is normal for marketing briefs, so it is never injected there.

This is the failure mode that matters most, and it bit during development. An earlier
rule injected `mislabeled_down` wherever `confidential + restricted > 0.60`, which
admitted incident postmortems — a category that is **33% "internal" natively**.
Relabelling one of those "internal" is not an anomaly; it is the second most common
state. The detector was being penalised for correctly ignoring non-anomalies. The
rule now keys on the *destination* label being rare (`public + internal < 0.10`), and
a parametrised test asserts the invariant for every category.

**2. The features never see the label.** Injected anomalies pass through exactly the
same `posture_features()` code as clean documents. Only `cohort.evaluate` reads
`ground_truth.parquet`; no other module imports it.

**3. Realism where realism is cheap.** Nested AD-style groups requiring transitive
expansion; external partners with legitimate access, so the scorer cannot simply
learn "external = bad"; a low-sensitivity category (marketing) where broad sharing is
normal, so it cannot learn "many principals = bad"; 4% category-less scratch
documents; 6% near-duplicates with independently sampled permissions; German and
French contracts.

**4. Byte-reproducibility.** Same seed, same corpus, asserted in CI by generating
twice and comparing SHA-256 digests. A benchmark that is not reproducible cannot
support a regression gate.

## What this cannot tell you

**Cluster quality does not transfer.** ARI is 1.000 because documents come from
per-category sentence templates and are more lexically separable than real documents
ever are. This is stated prominently in the README rather than presented as a result.
Because that number is doing no work, the harness deliberately degrades clustering
(min_cluster_size 250 → 40, ARI 1.000 → 0.730) and reports detection against it:
PR-AUC falls 0.623 → 0.474, still 2.3× the global baseline.

**Absolute PR-AUC does not transfer.** It depends on the injected anomaly mix and the
2% rate. The *ablation contrasts* are what generalise, because every arm sees the
identical corpus.

**Detectability is partly designed in.** Anomalies perturb one or two features, which
is what makes top-2 aggregation look good (ADR 0003). Real oversharing is messier.

## Consequences

- Every claim in the README is falsifiable by `make baseline` on a clean checkout.
- The peer-vs-global and peer-vs-random contrasts become measurable, which is the
  only reason the central claim is more than an assertion.
- Two CI tests guard the claim directly, so a refactor that breaks the premise fails
  the build.
- **The obvious unfinished work is validating on real corpora.** The ingestion path
  for PDF/DOCX/EML plus OCR is scaffolded behind the `ingest` extra, but no published
  number here comes from a real document. Enron in particular would supply genuine
  organisational sharing behaviour — real senders, real recipient graphs — against
  which the *cohort discovery* half could be checked properly, even though the
  anomaly half would still lack labels.
