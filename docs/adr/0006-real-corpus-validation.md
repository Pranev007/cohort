# ADR 0006 — Validating on real corpora, and what it cost

**Status:** accepted · **Date:** 2026-08

## Context

Every number in v1.0 came from generated data. Two claims in particular were
untested and load-bearing:

1. Cohort discovery scored ARI 1.000 — on documents built from per-category
   sentence templates, which are far more lexically separable than real writing.
   The README said so, but saying so is not measuring it.
2. Peer baselining beat a global baseline 2.9x — on posture distributions the
   same generator also wrote, using a policy that deliberately tied document
   category to sharing norm.

Both needed a corpus nobody in this project designed.

## Decision

Two public corpora, each testing a different half, kept in `cohort.real` and
reported separately from the synthetic benchmark rather than blended into it.

**20 Newsgroups** — real prose, 20 gold labels, several genuinely hard to
separate (`comp.sys.ibm.pc.hardware` vs `comp.sys.mac.hardware`). Tests cohort
discovery. Usenet headers, quoted replies and signature blocks are stripped
first; left in, a clusterer separates groups on mail-server routing and posts a
flattering score for the wrong reason.

**Enron** — ~500k real messages with real recipient lists, real external domains,
real folder structure and real timestamps. That combination is what makes it
usable here: public *document* corpora carry content but no access metadata, and
access metadata is the thing being scored. Streamed from the 443 MB archive
rather than unpacked, sampled every 12th qualifying message so the sample spans
all custodians.

Enron supplies 10 of the 15 posture features. The other five — `link_scope`,
`label_tier`, `acl_origin`, `n_groups`, `owner_dept_is_modal` — have no analogue
in email and are left **inert** rather than simulated: a constant categorical has
zero entropy and a constant continuous feature is marked uninformative, so each
contributes exactly 0 nats and the scorer runs unmodified.

### The experimental design that makes the Enron run fair

Anomalies are injected into the **real** posture distribution, and every injected
value is drawn from that distribution's own upper tail. A document made
"overshared" gets a recipient count that genuinely occurs elsewhere in Enron,
because plenty of real messages go to eighty people.

The document is therefore anomalous **only relative to its peers**, never
globally. That is precisely the claim under test. Injecting globally-extreme
values would produce a benchmark any detector passes and would prove nothing.
`test_injected_values_stay_inside_the_real_range` guards the invariant.

## What it found

### Cohort discovery collapses on real text under the v1.0 defaults

| Configuration | cohorts | unassigned | ARI | homogeneity |
|---|---:|---:|---:|---:|
| SVD + fixed 0.55 bar (v1.0 default) | 3 | 85.0% | 0.129 | 0.268 |
| SVD + adaptive bar | 3 | 45.4% | 0.051 | 0.164 |
| UMAP + fixed 0.55 bar | 36 | 31.2% | **0.429** | 0.602 |
| **UMAP + adaptive bar** (`configs/real.yaml`) | 36 | **17.9%** | 0.377 | 0.562 |

Against ARI 1.000 on synthetic. Two distinct faults, both real:

**HDBSCAN finds no density structure in TF-IDF space.** 86.6% of real documents
came back as noise. The class signal exists — within-class cosine 0.180 versus
between-class 0.117 — but it is far too weak for a density method to latch onto.
UMAP builds the structure HDBSCAN needs and takes the same corpus from ARI 0.129
to 0.429, which is why `configs/real.yaml` exists and why `umap-learn` is now a
`real` extra.

**A fixed cosine threshold does not transfer between embedding geometries.** The
0.55 reassignment bar was tuned on synthetic vectors. On real text the median
noise-to-centroid cosine is 0.307, so it rescued 1.9% of outliers and left 85% of
the corpus unassigned — and an unassigned document falls back to the global
baseline, which is the configuration we measured as 2.9x worse. The bar is now
calibrated per run against the cohorts' own internal similarity (`_reassign_bar`).

Note the trade-off in the table: the adaptive bar buys coverage and costs purity
(17.9% unassigned at ARI 0.377, versus 31.2% at ARI 0.429). ARI is not the
objective — detection is — so neither setting dominates and both remain
available. On the synthetic benchmark the change is exactly neutral: identical
PR-AUC 0.623, because the documents it would absorb are the German and French
contracts, which are far from every centroid under either rule.

### On Enron, peer baselining provides no benefit at all

22,966 messages sampled across the full 517,401-message archive, 459 injected
anomalies, 117 discovered cohorts.

| Configuration | PR-AUC |
|---|---:|
| **Peer cohorts (semantic)** | **0.236** |
| Global baseline (no peer grouping) | **0.236** |
| Random cohorts (same sizes) | 0.189 |

Peer and global are **identical to three decimal places**. On the synthetic
benchmark the same comparison is 0.623 versus 0.213. The central claim of the
project simply does not hold on this corpus.

(Random cohorts scoring *worse* than global is the expected sanity check rather
than an anomaly: splitting into ~200-document buckets adds estimation noise while
contributing no signal, so it is strictly worse than pooling everything.)

### Why — measured, not argued

Peer baselining can only beat a global baseline when documents that *mean* the
same thing are also *handled* the same way. `posture_coupling` measures exactly
that precondition: η² between cohorts for continuous features, normalised mutual
information for categorical ones.

Comparing like with like — the same ten features Enron supports:

| Feature | Synthetic | Enron |
|---|---:|---:|
| `n_principals` | 0.306 | **0.055** |
| `accessor_dept_entropy` | 0.318 | **0.093** |
| `n_external_domains` | 0.150 | **0.048** |
| `staleness_days` | 0.176 | 0.097 |
| `repo_type` | 0.222 | 0.162 |
| `path_depth` | 0.467 | 0.031 |
| **Mean (10 features)** | **0.312** | **0.170** |

The features the injected anomalies actually perturb have **near-zero coupling on
Enron** — 0.048 to 0.093. Cohort membership carries almost no information about
recipient counts or external exposure, so a peer distribution is indistinguishable
from the global one and peer-relative scoring has nothing to be relative *to*.
Peer equalling global is not a bug; it is the arithmetic consequence.

This is intuitive in hindsight. In a document repository the coupling is
structural: contracts go to the legal team, offer letters to HR, and each type
carries its own sharing norm. In an email archive, how many people you copy has
far more to do with the moment than with the topic.

**This is the deployment guidance the synthetic benchmark could not produce.**
Cohort is a document-repository tool. Pointed at a mail archive it runs, produces
plausible findings, and delivers nothing a global baseline would not — and the
repository now reports the statistic that predicts which situation you are in
*before* you trust the output. A coupling near 0.05 on the features you care
about means peer baselining will not pay for its complexity.

## Consequences

- `cohort real-eval` runs both experiments; results are written to
  `artifacts/reports/real_corpora.{md,json}` and never merged into the synthetic
  baseline or the CI regression gate.
- Two genuine bugs fixed: a non-transferable similarity threshold, and an
  `email.header.Header` returned where a `str` was assumed, which raised on real
  RFC 2047 headers and would have taken down a scan.
- A truncated archive now degrades to a partial parse instead of raising.
- `posture_coupling` is a first-class metric, because it predicts whether the
  method will help *before* anyone measures detection.
- The headline README numbers stay synthetic and stay labelled as such. The real
  numbers are lower and are reported next to them rather than in place of them.
