# Benchmark Ladder

This note defines the benchmark ladder that Kayak should maintain instead of
growing ad hoc benchmark scripts.

The goal is not to maximize the number of datasets.

The goal is to keep one explicit evaluation ladder where each rung answers a
different systems question and has a concrete exit criterion.

## Why A Ladder

Reason:
- the repository already has several useful benchmark surfaces
- without an explicit ladder it is too easy to mix:
  - smoke tests
  - hard-recall comparisons
  - stronger local ceilings
  - future code or multimodal ambitions

Decision:
- every new benchmark family should justify which rung it belongs to
- every rung should state what evidence is required before it counts as
  implemented

## The Rungs

### 1. Trivial Sanity

Current families:
- `SciFact`
- `FIQA`
- `LIMIT-small`

Question this rung answers:
- does the exact late-interaction loop still work end to end on real judged
  data?

Exit criteria:
- the dataset loads through the repo’s existing benchmark path
- exact late interaction runs end to end
- judged quality and stage-aware JSON are emitted
- the family is treated as a regression/sanity slice, not as the strongest
  hard-recall proof

### 2. Harder Public Text

Current families:
- `BrowseComp-Plus` evidence slice
- `BrowseComp-Plus` gold slice

Question this rung answers:
- how much exact-reference candidate recall do current stage-1 plans lose on a
  materially harder public text retrieval slice?

Exit criteria:
- the dataset is mirrored into the explicit collection/snapshot path
- stage-aware output reports candidate recall against exact full scan
- vector counts, byte counts, and candidate budgets are explicit

### 3. Scalable Synthetic Hard Recall

Current families:
- `synthetic_hard_recall`
- `contradiction_hard_recall`
- `long_document_hard_recall`

Question this rung answers:
- how do stage-1 failures move once the corpus shape or document shape becomes
  harder than the current public slices?

Exit criteria:
- query vector count and nominal document vector count are explicit
- the family scales beyond the tiny public slices
- at least one non-exact stage-1 plan measurably loses exact-reference recall
  at bounded `candidate_k`
- exact late-interaction reranking can still recover when the candidate window
  fully covers the oracle

Important distinction:
- `synthetic_hard_recall` stresses conjunction-style ambiguity
- `contradiction_hard_recall` stresses same-topic opposite-polarity confusion
- `long_document_hard_recall` stresses long noisy prefixes with late exact
  evidence

### 4. Stronger Local Ceiling

Current path:
- `exact_clause_text_ceiling`

Question this rung answers:
- how much quality exists above exact vector-only late interaction when Kayak
  uses one explicitly more expensive local path?

Exit criteria:
- the expensive path is labeled by the actual execution path used
- candidate generation, stage-2 reference, and stage-3 verifier semantics stay
  explicit in the output
- latency is reported alongside quality

Naming rule:
- a clause-text reranker is a `local_stronger_ceiling`
- it is not a cross-encoder ceiling
- it is not a long-context LLM ceiling

### 5. Corpus-Scale PLAID Serving

Current status:
- scaffolded, not yet a decision-quality full-corpus benchmark

Current scaffold:
- `python/scripts/profile_task_plaid_i8_candidate_generation.py`

Target families:
- MS MARCO passage full corpus
- one long-context corpus: LoCoV1, LEMB/NarrativeQA, or local arXiv chunks

Question this rung answers:
- does Kayak's advantage come from candidate generation, pruning, compact
  storage, and residual/exact materialization rather than from isolated MaxSim
  kernels?

Exit criteria:
- corpus size and vector counts are explicit: document count, query count,
  total document vectors, query-vector distribution, document-vector
  distribution
- candidate generation is measured separately from exact rerank
- residual or i8 token payload materialization is measured or estimated as its
  own field
- external baselines include FastPlaid where the comparison is batch/offline
  and NextPlaid where the comparison is serving/index lifecycle
- full-corpus runs use the quiet benchmark wrapper, and any sampled smoke run
  is labeled as a pipeline validation only

Reason:
- the current GPU/FastPlaid rows are useful primitive tests, but they are too
  small to prove corpus-scale PLAID serving behavior. Long documents and large
  corpora are where posting fanout, candidate pruning, and residual
  materialization become the system bottleneck.

### 6. Future Code Or Multimodal Lane

Current status:
- not implemented

Question this rung will answer:
- can the same stage-aware reporting surface support code or beyond-text
  retrieval without hiding encoder or representation assumptions?

Exit criteria:
- one real loader or fixture exists through the normal benchmark path
- encoder identity remains explicit
- the resulting artifact uses the same stage-aware JSON surface as the text
  families

## Current Commands

Sanity and public hard-recall paths:

```bash
pixi run bench_hard_recall_real_subset
```

Synthetic hard-recall paths:

```bash
pixi run bench_synthetic_hard_recall_stage_aware
pixi run bench_contradiction_hard_recall_stage_aware
pixi run bench_long_document_hard_recall_stage_aware
```

Local stronger ceiling:

```bash
pixi run bench_browsecomp_plus_gold_ceiling_comparison
```

Corpus-scale scaffold on an already encoded task JSON:

```bash
PYTHONPATH=python python python/scripts/profile_task_plaid_i8_candidate_generation.py \
  --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json \
  --query-limit 4 \
  --candidate-k 256 \
  --emit-quiet-mean
```

## What This Note Does Not Claim

This note does not claim:
- that the current ladder is final
- that every rung already has the best possible dataset
- that a local stronger ceiling is a substitute for a verified cross-attention
  or long-context ceiling

It only sets the current contract:
- keep the ladder explicit
- keep vector counts explicit
- keep stronger ceilings labeled by the actual path used
