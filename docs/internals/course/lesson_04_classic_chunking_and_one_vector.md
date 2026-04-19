# Lesson 4 Draft: Classic Chunking And One-Vector Retrieval

## Problem

Many teams do one of these two things by default:

- compress the whole document to one vector
- chunk the document, then compress each chunk to one vector

This lesson makes that comparison explicit inside the Kayak course material.

The point is not to mock those baselines.
The point is to show:

- what they are good at
- what they erase
- how to recognize when you are in one of those regimes

One important boundary should stay explicit internally:

- the current `chunk16` and `chunk32` comparisons in this lesson are
  vector-level teaching baselines
- they are **not** intended to represent current production chunk sizes such as
  `512+` token chunks

For the recurring user confusion around:

- "but if the whole document is chunked, isn't the whole document already covered?"

pair this lesson with:

- [chunk_coverage_vs_joint_scoring.md](chunk_coverage_vs_joint_scoring.md)

## Natural Opening

This lesson should begin from the learner's current system, not from Kayak vocabulary.

Open with something like:

- "we chunked it and quality moved"
- "we compressed every document to one vector because that is what the stack made easy"
- "we do not know whether chunking helped or just changed the failure"

Then introduce the controlled comparison.

## Important Translation

Inside the repo, the current single-vector-per-document proxy is called
`document_proxy`.

For course purposes, that should be taught as:

- query token vectors mean-pooled to one vector
- document token vectors mean-pooled to one vector
- one score per document from those pooled representations

That is the nearest controlled local stand-in for "one vector per document"
without changing the underlying encoder.

## Why This Comparison Is Sound

The internal comparison here keeps the encoder fixed.

We do **not** switch to a different embedding model.
Instead, we derive simpler baselines from the exact same token vectors:

- exact late interaction
- one vector per whole document
- one vector per fixed-size chunk

Reason:
- that isolates the effect of representation reduction and chunking
- it avoids confusing "different encoder" with "different retrieval geometry"

This is an epistemic constraint for us.
The learner does not need to hear that sentence first.
They need to feel first that:

- we changed one thing at a time
- so the comparison is interpretable

But there is also a realism boundary:

- the judged-slice task JSONs in this folder already contain encoded
  late-interaction vector sequences
- on the currently cached slices, document vector counts top out around `180`
- so `chunk16` and `chunk32` are chunking the stored late-interaction vectors,
  not recreating a modern raw-text `512` or `800` token ingestion pipeline

That means this lesson currently answers:

- "what happens when I reduce the number of retrieval units inside the stored
  late-interaction representation?"

not:

- "what is the best production token chunk size for a modern RAG stack?"

And it also does **not** by itself answer:

- "when does chunk coverage become real joint document scoring?"

That distinction is explained directly in:

- [chunk_coverage_vs_joint_scoring.md](chunk_coverage_vs_joint_scoring.md)

## Verified Local Bridges

| Claim | Status | Evidence |
| --- | --- | --- |
| The repo's `document_proxy` candidate path is the one-vector-per-document baseline under the same token vectors. | Verified locally | [../../../python/tests/test_course_chunking_baselines_smoke.py](../../../python/tests/test_course_chunking_baselines_smoke.py) |
| Chunking can help when one local evidence region is buried inside a long noisy document. | Verified locally | [../../../python/tests/test_course_chunking_baselines_smoke.py](../../../python/tests/test_course_chunking_baselines_smoke.py) |
| Chunking can hurt when the needed evidence spans chunk boundaries. | Verified locally | [../../../python/tests/test_course_chunking_baselines_smoke.py](../../../python/tests/test_course_chunking_baselines_smoke.py) |

Notebook:
- [notebooks/classic_chunking_and_one_vector.ipynb](notebooks/classic_chunking_and_one_vector.ipynb)

## Cross-Slice Local Result

On `2026-04-19`, I ran a controlled local comparison on cached judged slices
already present in `.cache/kayak/`.

Baselines:

- `exact`: full late interaction
- `onevec`: one vector per whole document, one vector per query
- `chunk16`: fixed-size chunking into `16` token vectors per chunk, then one
  vector per chunk
- `chunk32`: fixed-size chunking into `32` token vectors per chunk, then one
  vector per chunk

### Summary

| Slice | Primary metric | Exact | One vector | Chunk 16 | Chunk 32 | Exact recall@k | One vector recall@k | Chunk 16 recall@k | Chunk 32 recall@k |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `limit_small_real_subset` | `nDCG@10` | `0.9674` | `0.4987` | `0.1841` | `0.1606` | `1.0000` | `0.6719` | `0.2969` | `0.3125` |
| `bright_stackoverflow_real_subset` | `nDCG@10` | `0.2914` | `0.1074` | `0.2074` | `0.0986` | `0.5139` | `0.1389` | `0.2153` | `0.0903` |
| `legal_rag_bench_real_subset` | `MRR@10` | `0.1667` | `0.0125` | `0.0000` | `0.0417` | `0.2500` | `0.1250` | `0.0000` | `0.1250` |
| `r2med_biology_real_subset` | `nDCG@10` | `0.8646` | `0.7955` | `0.8177` | `0.7985` | `0.8109` | `0.7585` | `0.7966` | `0.7901` |

## What This Shows

### 1. One vector per document is a real baseline, but it throws away a lot

This is visible especially on:

- `LIMIT-small`
- `BRIGHT`
- `Legal RAG Bench`

### 2. Chunking is not uniformly worse or uniformly better

Chunking helped relative to one vector per whole document on:

- `bright_stackoverflow_real_subset`
- `r2med_biology_real_subset`

Chunking hurt badly on:

- `limit_small_real_subset`
- `legal_rag_bench_real_subset`

So the course should not teach:

- "chunking always helps"
- "chunking always hurts"

It should teach:

- chunking is a heuristic for local evidence isolation
- it becomes harmful when the evidence spans chunks or when the chunk-level
  representation is still too lossy

That is the real-life teaching move:

- not "chunking is wrong"
- not "chunking is the answer"
- but "chunking changes which kind of miss you get"

### 3. Exact late interaction remained best on all four cached slices

That is the most stable course-grade conclusion from this local comparison.

## Realistic Chunking Boundary

If we want the course to compare against what teams really deploy today, the
next benchmark should use raw text and token chunking settings in the
few-hundred-token regime, not just `16` and `32` stored vectors.

Reason:

- current production retrieval stacks often use chunk sizes much larger than
  the pedagogical vector-level chunks in this lesson
- for example, OpenAI's Retrieval docs currently state a default chunk size of
  `800` tokens with `400` overlap, and show a custom example at
  `1200` / `200`

Source:
- https://developers.openai.com/api/docs/guides/retrieval

So the epistemically sound interpretation is:

- this lesson is a controlled geometry lesson
- it is not yet the repo's production-chunking lesson

The production-grade follow-up should compare, on raw text:

- full document
- `256` token chunks
- `384` token chunks
- `512` token chunks

with overlap made explicit.

That follow-up now has a concrete benchmark path:

- [raw_text_chunk_sweep.md](raw_text_chunk_sweep.md)
- [openai_like_dense_chunk_baseline.md](openai_like_dense_chunk_baseline.md)

Current status:

- implemented and smoke-run on `bright_stackoverflow_real_subset`
- tokenizer-limited to `<=512` source tokens, but the current ColBERT document
  encoder also has `doc_maxlen = 180`, so the raw-text sweep now reports cap
  pressure explicitly instead of treating those chunk sizes as fully faithful

## Diagnosis Rule

If your current system is:

- one vector per whole document

then your first check should be:

- does exact late interaction on the same vectors recover judged quality?

If your current system is:

- fixed-size chunks with one vector per chunk

then your first check should be:

- does chunking help because evidence is local?
- or does chunking hurt because the evidence spans chunks?

Kayak helps because it makes both comparisons explicit.

This section should feel like:

- "what should I try next Monday on my actual pipeline?"

not:

- "what abstract position should I now hold about chunking?"

## Where This Does Not Yet Generalize

This lesson does not prove:

- that fixed chunk size `16` or `32` are representative of the whole field
- that the repo has already evaluated realistic production token chunk sizes
- that mean pooling is the exact retrieval geometry every dense system uses
- that these four cached slices are enough to characterize all production RAG
  workloads

What it does prove:

- the internal course can now compare exact late interaction, one-vector
  retrieval, and chunked one-vector retrieval on both deterministic toys and
  real judged slices
