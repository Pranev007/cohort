# Model card — Cohort peer-baseline risk scorer

**Version** 1.0.0 · **Type** unsupervised anomaly detection over document semantics
and access metadata · **Reference run** seed 1337, 15,000 documents

## What it does

Groups documents into cohorts by meaning, learns the distribution of fifteen security
posture features within each cohort, and scores every document by how improbable its
posture is among its peers. Output is a risk score in nats, a conformal p-value, an
exact per-feature attribution, a natural-language finding, and a verified
minimum-cost remediation plan.

Nothing in the pipeline is supervised. No category list, no labelled examples, no
rules, no regex.

## Intended use

- **Triage.** Rank a corpus so an analyst opens the twenty files most likely to be
  genuinely mis-permissioned. Precision@50 is 1.000 on the reference benchmark.
- **Posture review.** Surface systematic drift — a whole cohort whose sharing norms
  have loosened.
- **Change preview.** `POST /score` evaluates a hypothetical permission change before
  it is made.
- **Research and teaching.** A complete, reproducible implementation of peer-baseline
  risk analysis with an evaluation harness attached.

## Out of scope

- **Not an authorisation system.** It observes posture; it does not enforce anything.
- **Not a compliance attestation.** A low score is not evidence of GDPR, HIPAA or
  PCI-DSS conformance. Absence of evidence is not evidence of absence.
- **Not for automated destructive action.** Findings should drive review or a
  reversible change. Auto-revoking access on an unsupervised score with a 38%
  precision-among-flagged rate would break legitimate work.
- **Not for judging individuals.** Features include document ownership. Using this to
  evaluate employee behaviour would be a misuse: the model has no concept of intent,
  and a person who inherits a badly-permissioned folder is not doing anything wrong.
- **Not for mail archives.** Measured on Enron: no advantage over a global
  baseline. Intended for document repositories, where document type and handling
  norm are structurally coupled.

## Performance (reference benchmark)

| Metric | Value |
|---|---|
| PR-AUC | 0.623 |
| ROC-AUC | 0.969 |
| Precision@50 / @100 / @300 | 1.000 / 0.800 / 0.620 |
| Flag rate at α=0.05 | 4.01% (nominal 5%) |
| Precision among flagged | 0.392 |
| Recall among flagged | 0.752 |
| Cluster ARI vs. true categories | 1.000 |

Cross-seed PR-AUC: 0.623 / 0.534 / 0.561.

**Precision among flagged is 0.392.** Roughly three in five flagged documents are not
injected anomalies. Some are genuinely unusual-but-benign, which is inherent to
unsupervised detection. Anyone reading the 1.000 Precision@50 should read this number
next to it: the top of the ranking is clean, the tail of the flag set is not.

## Validated on real corpora — and it does not always work

| Corpus | Result |
|---|---|
| **Synthetic** (15k docs) | peer 0.623 vs global 0.213 — **2.9x** |
| **Enron** (23k real messages) | peer 0.236 vs global 0.236 — **no benefit** |
| **20 Newsgroups** (8.8k real posts) | cohort ARI 0.429 vs 1.000 on synthetic |

**On Enron, peer baselining is worth nothing.** The cause is measured rather than
guessed: `posture_coupling` — the share of posture variation explained by cohort
membership — is 0.312 on synthetic and 0.170 on Enron, and on the features the
anomalies actually touch it collapses to 0.048-0.093. Cohort membership carries
almost no information about recipient counts or external exposure in email, so
the peer distribution equals the global one.

**Run `posture_coupling` on your corpus before deploying.** Coupling near 0.05 on
the features you care about means this method will not beat a single global
baseline and is not worth its complexity. See
[ADR 0006](docs/adr/0006-real-corpus-validation.md).

## Known failure modes

**A wholly-compromised cohort learns the wrong normal.** Baselines come from the
corpus being judged. If every document of a type is overshared, oversharing becomes
that cohort's baseline and nothing is flagged. Unsupervised detection cannot escape
this. Mitigation is external: compare baselines against a reference tenant or against
the same cohort's own history.

**Anomalies contaminate their own baseline.** At a 2% anomaly rate concentrated on one
feature value, that value looks ~10× more common than it is. Trimmed iterative
refitting (`robust_passes=1`) recovers part of the loss.

**Single-feature anomalies rank low.** Top-2 aggregation gives them one real signal
plus one noise term. `mislabeled_down` scores PR-AUC 0.045 while still reaching 47%
recall at the calibrated threshold. Set `aggregation="sum"` if diffuse anomalies
matter more than sharp ones in your corpus.

**Weak category-to-posture coupling makes the method pointless.** The dominant
failure mode, and the one that showed up on real data. Measured, not theoretical —
see the section above.

**Small corpora gain little.** The advantage over a global baseline is ~1.1× at 1,200
documents, ~1.9× at 4,000, ~2.9× at 15,000. Below roughly 100 documents per cohort,
peer baselining is not worth its complexity.

**Lexical embeddings do not cross languages.** With the default backend, German and
French contracts form their own cohorts or go unassigned (1.23% of the reference
corpus). Use the `sentence-transformers` backend for multilingual corpora.

**Conformal guarantee is approximate.** The calibration set is contaminated, so
coverage is nominal-ish and mildly conservative rather than exact. See
[ADR 0004](docs/adr/0004-conformal-calibration.md).

**Cohort naming is descriptive, not authoritative.** c-TF-IDF keyphrases produce
labels like "Customer / Liability / Services" for what a human would call "vendor
contracts". Useful for orientation; not a classification.

## Fairness and privacy

The scored features are properties of *documents and their permissions*, not of
people. Two carry indirect signal about individuals — `owner_dept_is_modal` and
`pii_density` — and neither should be read as a judgement about the owner.

The synthetic corpus contains no real personal data. Names are drawn from a small
combinatorial pool and any resemblance to real people is coincidental. Applied to a
real corpus, the tool processes document contents and access metadata and inherits
whatever data-protection obligations attach to them; the embedding step reads full
document text.

## Compute and cost

CPU only. 15,000 documents scan end to end in ~23 seconds on one core (~660 docs/s);
scoring alone runs at ~14,000 docs/s. No GPU, no model download, no API key on the
default path. Optional LLM cohort naming costs one short call per cohort — roughly
fourteen calls for a corpus of any size — and is off by default.

## Reproducing

```bash
pip install -e . && make baseline
```

Byte-reproducible from `seed`; CI asserts it by generating twice and comparing
SHA-256 digests.

## Attribution

Peer-baseline risk analysis is a concept published by Concentric AI as Risk Distance™.
This is an independent reimplementation built from their public descriptions of the
idea — no proprietary code, data, or internal detail was used or is claimed. Not
affiliated with or endorsed by Concentric AI. The trademark belongs to them; the
implementation, benchmark, and results here are mine.
