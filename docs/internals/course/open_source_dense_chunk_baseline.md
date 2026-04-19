# Open-Source Dense Chunk Baseline

This note records the first live local benchmark run for the dense one-vector
chunk baseline using an open-source embedding model that can run entirely in the
current environment.

It is intentionally separate from the OpenAI-like lane.

## What This Owns

This note owns:

- the local Hugging Face dense embedding adapter
- the first live judged-slice measurement for a dense chunk baseline that does
  not depend on an external API
- the interpretation boundary for what this result does and does not prove

It does **not** own:

- a claim that `all-MiniLM-L6-v2` is the best open-source dense retrieval model
- a claim that this reproduces OpenAI hosted retrieval
- a claim that late interaction is weaker than dense retrieval in general

## Why This Exists

The OpenAI-like dense lane was implemented first, but this environment did not
have a live API key.

So the next justified step was:

- keep the same explicit dense chunk benchmark contract
- swap in a locally runnable open-source embedder
- measure a real judged slice instead of speculating

That is what this note records.

## Verified Local Setup

Verified locally on `2026-04-20`:

- `.venv` already had `torch` and `transformers`
- `.venv` did not have `sentence-transformers`
- the local adapter therefore uses `AutoTokenizer` plus `AutoModel` with
  explicit mean pooling, implemented in
  [../../../python/kayak_bridge/hf_dense_baseline.py](../../../python/kayak_bridge/hf_dense_baseline.py)
- the dense chunk benchmark contract is still the generic path in
  [../../../python/kayak_bridge/single_vector_chunk_baseline.py](../../../python/kayak_bridge/single_vector_chunk_baseline.py)
- the benchmark CLI is
  [../../../python/scripts/bench_hf_dense_chunk_baseline.py](../../../python/scripts/bench_hf_dense_chunk_baseline.py)

Mechanically verified locally:

- [../../../python/tests/test_single_vector_chunk_baseline.py](../../../python/tests/test_single_vector_chunk_baseline.py)
- [../../../python/tests/test_hf_dense_baseline.py](../../../python/tests/test_hf_dense_baseline.py)

## Measured Run

Command used:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python ./.venv/bin/python \
python/scripts/bench_hf_dense_chunk_baseline.py \
  --task .cache/kayak/bright_stackoverflow_real_subset/python_task.json \
  --output /tmp/bright_hf_dense_chunk_baseline_sweep.json \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --chunk-spec 256:64 \
  --chunk-spec 384:128 \
  --chunk-spec 512:128 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --embedding-batch-size 32 \
  --backend numpy_reference
```

Task facts:

- dataset: `xlangai/BRIGHT/stackoverflow`
- slice: `bright_stackoverflow_real_subset`
- judged queries: `8`
- documents: `150`
- metric: `nDCG@10`

Exact baseline from the same task JSON:

- exact model: `colbert-ir/colbertv2.0`
- exact full-document score: `0.2914247870199321`
- exact full-document mean recall@10: `0.5138888888888888`

Dense chunk results with `sentence-transformers/all-MiniLM-L6-v2`:

| Source chunking | nDCG@10 | Recall@10 | MRR | Mean chunks / doc |
| --- | ---: | ---: | ---: | ---: |
| `256 / 64` | `0.7955019806906065` | `0.8819444444444444` | `0.8375` | `3.8267` |
| `384 / 128` | `0.8034979363518409` | `0.8819444444444444` | `0.8375` | `2.8267` |
| `512 / 128` | `0.7672431856058413` | `0.8194444444444444` | `0.8333333333333334` | `2.12` |

## What Is Actually Verified

Verified:

- this open-source dense chunk baseline is live-runnable locally
- on this judged BRIGHT slice, it scores far above the exact full-document
  ColBERT baseline stored in the same task JSON
- the result is stable across three nearby chunk settings, not just one lucky
  chunk size

That means the course must not teach a simplistic story such as:

- "late interaction always beats what users already do"

The measured comparison here is narrower and more honest:

- full-document exact ColBERT can lose badly to a chunked dense baseline on at
  least one real slice

## What Is Still Uncertain

This result alone does **not** isolate why the dense chunk baseline wins.

The leading hypotheses are:

- chunking exposes more retrievable surface area than the full-document ColBERT
  path on this slice
- the compared systems have different encoder families, not just different
  retrieval geometry
- the full-document ColBERT path may be pressured by its own document-length
  budget

Those are hypotheses, not verified explanations.

The next evidence that would separate them is:

- compare against exact late interaction on the same chunked retrieval units
- compare against a stronger or longer-window dense encoder
- inspect whether judged-relevant evidence appears late in the raw documents

## Tokenizer Warning Fix

The first live run emitted a Hugging Face max-length warning during source
tokenization.

Direct reproduction showed that the warning came from tokenizing the full raw
document before chunking, not from embedding a too-long chunk.

That warning path is now fixed in
[../../../python/kayak_bridge/source_text_chunking.py](../../../python/kayak_bridge/source_text_chunking.py):

- source tokenization now prefers the callable tokenizer path with
  `truncation=False` and `verbose=False`
- this keeps the pre-chunk token stream untruncated while avoiding a misleading
  warning

The behavior is covered by:

- [../../../python/tests/test_raw_text_chunk_sweep.py](../../../python/tests/test_raw_text_chunk_sweep.py)

## Why This Matters For The Course

This is a useful teaching note because it prevents the course from converging to
the wrong slogan.

The safer teaching claim is:

- retrieval unit choice, chunk coverage, and encoder budget can dominate the
  outcome
- Kayak should be taught as a way to diagnose and fix those failures, not as a
  blanket guarantee that late interaction wins every direct comparison

## Rerun

Unit coverage:

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest \
  python.tests.test_raw_text_chunk_sweep \
  python.tests.test_single_vector_chunk_baseline \
  python.tests.test_hf_dense_baseline -v
```

Benchmark rerun:

```bash
PATH="$PWD/.venv/bin:$PATH" PYTHONPATH=python ./.venv/bin/python \
python/scripts/bench_hf_dense_chunk_baseline.py \
  --task .cache/kayak/bright_stackoverflow_real_subset/python_task.json \
  --output /tmp/bright_hf_dense_chunk_baseline_sweep.json \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --chunk-spec 256:64 \
  --chunk-spec 384:128 \
  --chunk-spec 512:128 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --embedding-batch-size 32 \
  --backend numpy_reference
```
