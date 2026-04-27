# 2026-04-27: NextPlaid And Large-Corpus PLAID Profile Plan

## Claim

The next Kayak optimization target should be corpus-scale candidate generation
and materialization, not another isolated MaxSim kernel.

Reason: PLAID-family serving spends its decisive work before exact MaxSim:
centroid routing, IVF/posting traversal, approximate candidate scoring,
candidate pruning, and then residual or token payload materialization for a
small rerank set. If Kayak wins only on small synthetic rerank windows, that is
not enough evidence for a system-level claim.

## Local Evidence

Verified in this repo:

- `docs/traces/2026-04-27_gpu_i8_next_optimization_priority.md` records CPU
  candidate generation at about `57%` to `82%` of the scoped GPU rerank
  envelope on the then-current policy rows.
- `docs/traces/2026-04-27_gpu_i8_candidate_generation_negative_results.md`
  records several rejected local CPU candidate-loop edits. Those failures
  point to a larger primitive or benchmark boundary, not more local loop work.
- `docs/traces/2026-04-27_gpu_i8_exact_rerank_probe.md` records two rejected
  exact-rerank kernel probes. Both matched CPU scores, but they were roughly
  two orders of magnitude slower than the existing partial-score GPU path on
  the smoke shape.
- `python/scripts/profile_task_plaid_i8_candidate_generation.py` now provides
  a real-task profiler for Kayak's internal PLAID i8 candidate-generation
  substeps on encoded task JSON.

Interpretation: the current synthetic GPU/FastPlaid matrices are valid
primitive checks. They are not full evidence that Kayak beats PLAID-family
systems on serving-shaped long-document or large-corpus workloads.

## NextPlaid Check

Checked public NextPlaid sources on 2026-04-27:

- LightOn's NextPlaid announcement:
  https://lighton.ai/lighton-blogs/introducing-lighton-nextplaid
- NextPlaid GitHub repository:
  https://github.com/lightonai/next-plaid
- NextPlaid docs:
  https://lightonai.github.io/next-plaid/
- NextPlaid Rust crate docs:
  https://docs.rs/next-plaid

Verified from those sources:

- NextPlaid is framed as the production serving/API layer, while FastPlaid is
  framed as the bulk/offline GPU indexing path.
- The search path is standard PLAID-shaped: score query vectors against
  centroids, probe selected centroids/IVF postings, approximate-score
  candidates with compact assignments, then reconstruct/materialize only a
  reduced set for exact MaxSim.
- Residual storage and materialization are first-class. Public material
  describes 4-bit residual compression and memory-mapped residuals, with
  materialization during exact reranking.
- NextPlaid adds serving concerns that our current synthetic FastPlaid matrix
  does not test: memory-mapped index loading, incremental updates, deletion,
  SQL metadata pre-filtering, and adaptive IVF probing under filters.
- The public NextPlaid benchmark table is API-level and includes encoding time
  with parameters such as `top_k=100`, `n_ivf_probe=8`, and
  `n_full_scores=4096`. That is not the same scope as our internal prepared
  primitive timings.

Decision: compare Kayak against FastPlaid for batch/offline PLAID baselines,
and compare against NextPlaid for serving/lifecycle baselines. Do not collapse
those into one baseline.

## Target Corpora

### MS MARCO Full

Source checked:
https://microsoft.github.io/msmarco/Datasets.html

The official passage-ranking corpus lists `8,841,823` passages, `1,010,916`
queries, dev qrels, train qrels, and top-1000 reranking files.

Role:

- large-corpus posting fanout and pruning stress
- standard neural IR comparability
- short-passage corpus, so it isolates corpus scale more than long-document
  vector budget

Required fields:

- corpus record count
- query count used
- total document vector count
- query-vector distribution
- document-vector distribution
- centroid count and selected centroids per query vector
- posting visits, touched documents, candidate count, candidate recall
- approximate candidate time, exact/rerank materialization time, end-to-end time

### Long-Context Corpus

Primary target: LoCoV1 if the loader is practical.

Source checked:
https://huggingface.co/papers/2402.07440

The paper page describes LoCoV1 as a 12-task benchmark for long-context
retrieval where chunking is not possible or ineffective, with documents up to
`32K` tokens.

Fallback target: LEMB/NarrativeQA or local arXiv chunks.

Reason:

- MS MARCO full stresses corpus scale, but not necessarily long-document
  residual materialization.
- LoCo/LEMB/arXiv chunks stress document-vector count, truncation, and
  candidate rerank payload size.

## Measurement Contract

Every corpus-scale row must record:

- `document_count`
- `query_count`
- `query_vector_count` distribution
- `document_vector_count` distribution
- `total_document_vector_count`
- `centroid_count`
- `centroids_per_query_vector`
- `candidate_k`
- centroid scoring time
- centroid selection time
- posting traversal or accumulation time
- final candidate top-k time
- candidate recall versus exact, when exact is feasible
- i8 token payload bytes or residual materialization bytes
- exact rerank time
- full end-to-end time
- external baseline name and scope: FastPlaid batch/offline, NextPlaid serving

Reason: vector count and materialization volume are the axes that make toy
benchmark conclusions unreliable.

## Scaffold Added

Added:

- `python/kayak_bridge/plaid_task_candidate_profile.py`
- `python/scripts/profile_task_plaid_i8_candidate_generation.py`
- `python/tests/test_plaid_task_candidate_profile.py`
- `python/kayak_bridge/msmarco_passage_task.py`
- `python/scripts/build_msmarco_passage_task_json.py`
- `python/scripts/build_lemb_narrativeqa_task_json.py`
- `python/tests/test_msmarco_passage_task.py`

What it measures:

- real encoded task JSON, not synthetic tensors only
- PLAID i8 candidate-generation substeps through the same Mojo prepared i8
  index path used by the GPU work
- posting visits and touched documents from the existing Mojo profile
- candidate document-vector count
- Kayak i8 token payload byte estimate for candidate rerank
- NextPlaid-style 4-bit residual byte estimate for comparison planning
- optional candidate recall versus exact full scan

What it does not claim:

- it is not a full MS MARCO or LoCo result
- it is not a NextPlaid benchmark adapter
- it is not a public Kayak backend

Corpus materialization scaffolds:

- MS MARCO passage local builder consumes official `collection.tsv`,
  `queries.tsv`, and qrels files. It does not download the corpus. With a
  document limit, it can still force selected positive documents into the
  subset so candidate recall is measurable on smoke runs. The script guards
  accidental no-limit JSON builds because full MS MARCO should move to a
  streaming snapshot path.
- LEMB/NarrativeQA builder exposes the existing long-document Hugging Face
  loader as an encoded task JSON command.

Validation run:

- helper and MS MARCO local parser tests:
  `pixi run env PYTHONPATH=python python -m unittest python/tests/test_plaid_task_candidate_profile.py python/tests/test_msmarco_passage_task.py`
  passed `7 / 7`
- syntax:
  `pixi run env PYTHONPATH=python python -m py_compile python/kayak_bridge/plaid_task_candidate_profile.py python/kayak_bridge/msmarco_passage_task.py python/scripts/profile_task_plaid_i8_candidate_generation.py python/scripts/build_msmarco_passage_task_json.py python/scripts/build_lemb_narrativeqa_task_json.py`
  passed
- in-memory Mojo smoke:
  one synthetic encoded task with `8` documents, `4` document vectors per
  document, `1` query, `3` query vectors, dim128, `candidate_k=4`, and
  `measurement_iterations=1`

Observed smoke result:

- status: `ok`
- profiled query count: `1`
- candidate recall versus exact: `0.5`

Example smoke command:

```bash
PYTHONPATH=python python python/scripts/build_msmarco_passage_task_json.py \
  --collection /data/msmarco/collection.tsv \
  --queries /data/msmarco/queries.dev.tsv \
  --qrels /data/msmarco/qrels.dev.tsv \
  --document-limit 10000 \
  --query-limit 16 \
  --output .cache/kayak/msmarco_passage_smoke/python_task.json

PYTHONPATH=python python python/scripts/profile_task_plaid_i8_candidate_generation.py \
  --task .cache/kayak/msmarco_passage_smoke/python_task.json \
  --query-limit 4 \
  --candidate-k 256 \
  --emit-quiet-mean
```

## Next Work

1. Run the new profiler on existing encoded long-document tasks first.
   Reason: this validates the measurement surface without paying full MS MARCO
   indexing cost.
2. Add or reuse a full-corpus MS MARCO materialization path.
   Reason: the current public benchmark ladder has real public slices, but not
   the full 8.8M-passage corpus.
3. Add a long-context corpus path for LoCoV1 or arXiv chunks.
   Reason: this is the document-vector-count stress axis missing from MS MARCO.
4. Add a NextPlaid baseline adapter only after the corpus runner can emit the
   same fields for Kayak.
   Reason: without a shared scope, comparing API-level NextPlaid numbers to
   internal Kayak primitive numbers would be misleading.
