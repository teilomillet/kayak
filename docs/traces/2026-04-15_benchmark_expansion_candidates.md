# 2026-04-15: benchmark expansion candidates for modular retrieval evaluation

## Goal

Identify paper-backed retrieval benchmarks that are worth adding next, without
optimizing Kayak around one benchmark family or one query-shape regime.

The selection rule for this note is explicit:

- prefer primary-source papers
- prefer benchmarks that stress meaningfully different failure modes
- favor benchmarks that can slot into the existing dataset-loading surfaces
  instead of forcing benchmark-specific search logic

## Verified local constraints

I checked the local code before recommending additions:

- the repo already has an `ir_datasets` ingestion path in
  [python/kayak_bridge/beir_subset.py](../../python/kayak_bridge/beir_subset.py)
- the repo already has Hugging Face `datasets` ingestion paths in
  [python/kayak_bridge/limit_subset.py](../../python/kayak_bridge/limit_subset.py)
  and
  [python/kayak_bridge/browsecomp_plus_decrypt.py](../../python/kayak_bridge/browsecomp_plus_decrypt.py)
- the new query-bucket frontier runner added in this loop keeps benchmark
  slicing modular by operating on `StoredJudgedTask` queries instead of
  changing planner or kernel behavior

This means benchmark expansion is primarily a dataset-wrapping and reporting
task, not a systems-architecture blocker.

## Recommended benchmark matrix

### 1. Keep BEIR as the broad zero-shot anchor

Source:

- BEIR: A Heterogenous Benchmark for Zero-shot Evaluation of Information
  Retrieval Models
  https://arxiv.org/abs/2104.08663

Why it still matters:

- BEIR spans 18 public datasets across diverse retrieval tasks and domains.
- The benchmark paper explicitly found that late-interaction models were among
  the strongest zero-shot baselines, while dense and sparse retrievers remained
  more efficient but weaker on average out of distribution.
- Kayak already uses BEIR-derived slices (`SciFact`, `FiQA`), so BEIR is the
  lowest-risk path for broadening coverage without changing benchmark plumbing.

Role in the matrix:

- baseline anchor for heterogeneous zero-shot retrieval
- regression check that new optimizations do not only help deep-research or
  synthetic slices

### 2. Add LongEmbed / LEMB for long-context retrieval stress

Source:

- LONGEMBED: Extending Embedding Models for Long Context Retrieval
  https://aclanthology.org/2024.emnlp-main.47/

Why it is a distinct signal:

- LongEmbed introduces LEMB specifically to measure long-context retrieval,
  where standard embedding models degrade as document length grows.
- This is directly useful for Kayak because it stresses document-vector budgets,
  truncation behavior, and long-evidence recall rather than only short-query
  semantic matching.
- The `NarrativeQARetrieval` slice is operationally attractive because its
  small corpus keeps repeated local runs tractable while still exposing genuine
  long-document behavior.

Role in the matrix:

- long-document retrieval benchmark
- document-vector-budget and truncation stress test

Implementation note:

- I verified the `mteb/LEMBNarrativeQARetrieval` packaging path locally and it
  fits the existing Hugging Face subset-builder pattern used in this repo.

### 3. Add BRIGHT for reasoning-intensive retrieval

Source:

- BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive
  Retrieval
  https://arxiv.org/abs/2407.12883

Why it is a distinct signal:

- BRIGHT was introduced specifically because many prior retrieval benchmarks
  were dominated by surface-form or semantic matching.
- The paper reports 1,398 real-world queries across domains such as economics,
  psychology, robotics, software engineering, and earth sciences.
- The abstract reports a strong leaderboard model dropping from `59.0 nDCG@10`
  on MTEB retrieval to `18.0 nDCG@10` on BRIGHT, which makes it a useful hard
  retrieval target instead of another easy zero-shot slice.

Role in the matrix:

- reasoning-heavy retrieval benchmark
- better fit than BEIR alone for testing whether late-interaction advantages
  survive on hard semantic alignment

Implementation note:

- I verified the `xlangai/BRIGHT` Hugging Face packaging path locally and it
  fits the existing subset-builder pattern used in this repo.

### 4. Add BrowseComp-Plus as the controlled deep-research benchmark

Source:

- BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of
  Deep-Research Agent
  https://arxiv.org/abs/2508.06600

Why it is a distinct signal:

- BrowseComp-Plus replaces live-web black-box evaluation with a fixed curated
  corpus, human-verified supporting documents, and mined hard negatives.
- That is directly aligned with Kayak’s systems goal because it isolates
  retriever quality from opaque API effects.
- The benchmark is explicitly meant for disentangling retriever performance,
  citation behavior, and context engineering in deep-research systems.

Role in the matrix:

- controlled deep-research benchmark
- benchmark for long evidence documents and hard negatives
- best current public fit for the user’s BrowseComp-adjacent direction

Implementation note:

- the repo already has BrowseComp-Plus support, so this is the benchmark to
  keep using while broadening the surrounding matrix
- I also verified a Hugging Face dataset entry exists

### 5. Optional stretch benchmark: R2MED for domain-specific reasoning

Source:

- R2MED: A Benchmark for Reasoning-Driven Medical Retrieval
  https://arxiv.org/abs/2505.14558

Why it is a distinct signal:

- R2MED targets a failure mode that generic retrieval benchmarks usually miss:
  relevant evidence matching an inferred diagnosis rather than the surface form
  of the symptoms.
- The paper reports 876 queries across three medical retrieval tasks and
  notes that even the best evaluated retriever only reached `31.4 nDCG@10`.
- This is a good stress test for reasoning-driven retrieval beyond consumer
  web search or general factual QA.

Role in the matrix:

- specialized reasoning benchmark
- useful if Kayak wants to claim robustness on hard domain retrieval rather
  than only general-purpose web or science slices

Implementation note:

- I verified Hugging Face R2MED entries exist, and the subset-builder path is
  compatible with the existing Hugging Face ingestion flow in this repo

## What not to treat as enough on its own

### `limit_small`

Current status:

- keep it as a local regression slice
- do not treat it as the only external-facing benchmark argument

Reason:

- it is valuable as a compositional sanity check already present in the repo
- but in this pass I did not verify a canonical academic packaging path strong
  enough to make it a first-wave benchmark expansion recommendation

### One benchmark family or one query width

Reason:

- BEIR alone misses reasoning-heavy and multilingual failure modes
- BrowseComp-Plus alone risks over-indexing on long evidence retrieval
- BRIGHT alone does not cover multilingual behavior or common zero-shot IR
- any single aggregate score can hide that a planner or candidate generator is
  only good for short queries or only good for wide candidate windows

That is why the query-bucket frontier work in this same loop is important:

- keep dataset choice modular
- keep query vector count explicit
- let users change bucket definitions or dataset sets without changing kernels

## Recommended rollout order

1. Keep the current BEIR-derived slices and BrowseComp-Plus slices.
2. Add BRIGHT next.
   Reason: it adds the clearest reasoning-heavy retrieval target from a recent
   benchmark paper.
3. Add LongEmbed / LEMB after BRIGHT.
   Reason: it adds an orthogonal long-document stress axis without requiring
   benchmark-specific search logic.
4. Keep BrowseComp-Plus as the deep-research control benchmark.
5. Add R2MED only if domain-specific reasoning is a product or research goal.

## Why this matters right now

The new modular query-bucket frontier run from this same loop wrote:

- `.cache/kayak/public_query_bucket_frontier.json`

and the measured result was that all five current public slices only populated
the same query-vector bucket:

- `qv_17_32`

So the current benchmark set is not yet exercising multiple query-shape
regimes, even though query vector count is a first-class cost axis in Kayak.
That is the strongest local reason to add BRIGHT and LongEmbed next rather than
continuing to optimize only within the current public suite.

## Decision

The sound near-term benchmark matrix for Kayak is:

- BEIR-derived slices for broad zero-shot comparability
- LongEmbed / LEMB for long-document retrieval stress
- BRIGHT for reasoning-intensive retrieval
- BrowseComp-Plus for controlled deep-research retrieval
- R2MED as an optional specialized reasoning benchmark

That combination is justified because each benchmark covers a different failure
mode, and the repo already has the ingestion surfaces needed to add them
without hard-coding benchmark-specific planner logic.
