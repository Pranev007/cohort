# ADR 0001 — Offline-first embedding backend

**Status:** accepted · **Date:** 2026-08

## Context

Cohort discovery needs document embeddings. The obvious choice for a 2026 project is
a transformer — `BAAI/bge-m3` is multilingual, strong, and one `pip install` away.

The problem is what that does to the published numbers. A reader who clones the repo
to check a PR-AUC of 0.623 would need a model download, a compatible torch build,
several gigabytes of disk, and network access. Anyone behind a proxy, on an air-gapped
machine, or running CI with no model cache gets a different experience from the one
the README describes — and a benchmark nobody can re-run is a screenshot.

## Decision

**Two backends behind one interface. Offline TF-IDF → SVD is the default.**

```
tfidf-svd                 word 1–2 grams → TF-IDF → truncated SVD(256) → L2 normalise
sentence-transformers     BAAI/bge-m3, optional, behind the `transformers` extra
```

The rest of the pipeline only ever sees an `(n, dim)` float32 matrix with unit-norm
rows, so the choice is a config flag rather than a code path.

Supporting choices:
- `max_df=0.55` drops the shared corporate boilerplate that appears across every
  category, which would otherwise dominate the leading singular vectors.
- Unit-norm rows make Euclidean distance monotone in cosine distance, which is what
  lets HDBSCAN run with its default metric.
- Clustering runs on a further-reduced 40-d view; density-based methods lose contrast
  as dimensionality grows, and HDBSCAN on 256-d embeddings returns one giant cluster
  plus noise.

The same principle governs the rest of the project: OCR, PII extraction, LLM cohort
naming and the remediation agent are all optional extras, never on the default path.

## Consequences

**Good.** `pip install -e . && cohort demo` reproduces every published number with
numpy, scipy, scikit-learn and pandas — no network, no GPU, no API key. CI runs the
full benchmark in under two minutes on a free runner. Corpus generation is
byte-reproducible, which is what makes the regression gate in `eval.yml` meaningful.
Embedding throughput is 882 docs/s on one CPU core.

**Bad, and visible in the results.** TF-IDF is lexical, so it cannot group a German
*Rahmenvertrag* with an English master service agreement. The 1.23% of documents left
unassigned by clustering are *exactly* the German and French contracts.

That is reported rather than hidden, because it is the honest demonstration of the
trade-off: the transformer backend exists precisely for the multilingual case, and
switching to it is `semantic.backend: sentence-transformers`. Any enterprise corpus
worth scanning spans languages, so a real deployment would take the transformer
path. The offline default is a reproducibility decision for a portfolio benchmark,
not a claim that lexical embeddings are sufficient.

## Alternatives considered

- **Transformer by default, TF-IDF as fallback.** Inverts the problem: the numbers in
  the README would come from a path most readers cannot reproduce cheaply.
- **Vendor a small quantised model in the repo.** Fixes reproducibility, breaks
  `pip install`, and puts model weights in git history.
- **Hashed character n-grams.** Partially cross-lingual for related languages, but
  markedly worse cluster quality and harder to explain than word TF-IDF, whose
  vocabulary the cohort namer reuses directly for c-TF-IDF keyphrases.
