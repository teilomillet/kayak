# OpenAI-Like Dense Chunk Baseline

This note defines the internal benchmark path for comparing Kayak against a
production-shaped dense chunk retrieval recipe closer to what many teams deploy
today.

It is intentionally separate from the controlled ColBERT geometry lessons.

## What This Owns

This note owns:

- the benchmark path for a one-vector dense chunk baseline starting from raw
  text
- the OpenAI-specific adapter for embeddings plus tiktoken chunking
- the comparison contract for "OpenAI-like" retrieval versus Kayak exact late
  interaction

It does **not** own:

- a claim that OpenAI hosted File Search is exactly reproduced here
- a claim that the current local implementation has already been live-run on the
  benchmark slices
- a claim that single-vector dense retrieval and OpenAI hosted retrieval are the
  same system

## Why This Exists

The current internal chunking lesson is a controlled encoder-fixed comparison.
That is still useful, but it does not answer the practical question:

- "how does Kayak compare to the kind of dense chunk retrieval many teams
  actually use?"

The OpenAI retrieval guides checked on `2026-04-19` document that:

- vector stores automatically chunk, embed, and index uploaded files
- default chunking is `800` tokens with `400` overlap
- `max_chunk_size_tokens` can be between `100` and `4096`
- File Search uses semantic and keyword search

Sources:

- https://developers.openai.com/api/docs/guides/retrieval
- https://developers.openai.com/api/docs/guides/tools-file-search

That makes an OpenAI-like dense chunk lane a justified comparison surface.

## Comparison Contract

The current benchmark path does the following:

- start from the raw document and query text already stored in the judged task
  JSON
- split document text into source-token chunks with a tiktoken tokenizer
- embed each chunk as one dense vector through the OpenAI embeddings API
- embed each query as one dense vector through the same API
- L2-normalize vectors so Kayak dot-product search acts as cosine-style dense
  retrieval
- rank chunk vectors globally
- map chunk hits back to parent documents with explicit max-chunk-style dedup
- evaluate parent-document rankings against the judged relevant documents

This is not "OpenAI hosted File Search reproduced exactly."
It is:

- OpenAI-like source chunking defaults
- OpenAI embeddings for one-vector chunks
- explicit cosine-style dense retrieval
- explicit parent aggregation

That distinction must stay visible.

## Verified Local Facts

Verified locally on `2026-04-19`:

- the public encoder registry currently exposes built-in `"colbert"` and
  `"callable"` encoder kinds
- the `"callable"` seam is the right way to plug in a different one-vector text
  embedder without pretending it is late interaction
- the cached judged task JSONs are built with ColBERT token vectors by default,
  so any OpenAI-like comparison must re-encode from raw text to avoid leaking
  the ColBERT encoder into the baseline

Evidence:

- [../../../python/kayak/encoders/registry.py](../../../python/kayak/encoders/registry.py)
- [../../../python/kayak/encoders/callable.py](../../../python/kayak/encoders/callable.py)
- [../../../python/kayak_bridge/retrieval_task_builder.py](../../../python/kayak_bridge/retrieval_task_builder.py)

## Implementation Status

Implemented locally:

- generic dense chunk benchmark core:
  [../../../python/kayak_bridge/single_vector_chunk_baseline.py](../../../python/kayak_bridge/single_vector_chunk_baseline.py)
- shared raw-text chunk primitives:
  [../../../python/kayak_bridge/source_text_chunking.py](../../../python/kayak_bridge/source_text_chunking.py)
- OpenAI embeddings adapter plus explicit file cache:
  [../../../python/kayak_bridge/openai_dense_baseline.py](../../../python/kayak_bridge/openai_dense_baseline.py)
- benchmark CLI:
  [../../../python/scripts/bench_openai_like_dense_chunk_baseline.py](../../../python/scripts/bench_openai_like_dense_chunk_baseline.py)

Mechanically verified locally:

- [../../../python/tests/test_single_vector_chunk_baseline.py](../../../python/tests/test_single_vector_chunk_baseline.py)
- [../../../python/tests/test_openai_dense_baseline.py](../../../python/tests/test_openai_dense_baseline.py)

Not yet verified locally in this environment:

- a live benchmark run against the OpenAI embeddings API

Reason:

- `OPENAI_API_KEY` was not present in the environment on `2026-04-19`
- `openai` and `tiktoken` were also not installed in the active `.venv` during
  implementation, so I verified the script surface and the injected-client tests
  instead of fabricating live numbers

## How To Run

Prerequisites:

- set `OPENAI_API_KEY`
- run with `openai` and `tiktoken` available

Example:

```bash
PYTHONPATH=python \
uv run --python 3.11 --with openai --with tiktoken \
python python/scripts/bench_openai_like_dense_chunk_baseline.py \
  --task .cache/kayak/bright_stackoverflow_real_subset/python_task.json \
  --output /tmp/bright_openai_like_dense_chunk_baseline.json \
  --embedding-model text-embedding-3-small \
  --chunk-spec 800:400 \
  --chunk-spec 1200:200 \
  --measurement-iterations 1
```

Notes:

- the CLI defaults to `800:400` and `1200:200`
- vectors are normalized by default so the ranking surface is cosine-like
- the adapter uses an explicit file cache under `.cache/openai_embeddings`
  unless `--cache-dir` overrides it or caching is disabled in code

## What This Can Prove

Once live-run, this path can answer:

- how far Kayak exact late interaction is above or below a dense single-vector
  chunk baseline on the same judged slice
- whether the usual dense chunk defaults are enough on hard-recall or
  reasoning-heavy slices
- whether parent-document recall is mainly lost in chunk retrieval, chunk
  aggregation, or dense embedding quality

## What This Still Cannot Prove

Even after a live run, this path still would not prove:

- the exact behavior of OpenAI hosted File Search end-to-end
- the effect of OpenAI's internal keyword-semantic fusion details
- the effect of any hosted ranker beyond the explicit dense chunk retrieval
  implemented here

So the correct interpretation is:

- this is a strong OpenAI-like dense chunk baseline
- it is not a full black-box reproduction of the hosted retrieval product
