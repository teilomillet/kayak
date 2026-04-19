# Raw-Text Chunk Sweep Note

This note records the new benchmark path for comparing Kayak against more
realistic chunking behavior.

It exists because the earlier `chunk16` / `chunk32` comparisons are useful
geometry lessons, but they are not faithful stand-ins for the source-token
chunking regimes many teams actually deploy.

## What This Owns

This note owns:

- the raw-text chunk-sweep benchmark path
- the current ColBERT-capped default sweep
- the first measured smoke result on a real judged slice

It does **not** own:

- a general production claim about the best chunk size
- a field-wide survey of chunking defaults
- a benchmark on encoders that can faithfully represent `800+` source tokens in
  one pass

## Why This Exists

The repo's judged task JSONs already contain:

- raw document text
- query text
- judged relevant documents
- stored late-interaction vectors

That means we can do something better than the vector-level teaching baseline:

- start from raw text
- split documents into explicit source-token chunks
- re-encode those chunks with the same ColBERT encoder
- compare the resulting one-vector-per-chunk baseline against the judged task

For the architecture distinction between:

- covering the whole document with chunks
- jointly scoring the whole document

see:

- [chunk_coverage_vs_joint_scoring.md](chunk_coverage_vs_joint_scoring.md)

For the separate external dense-comparison lane closer to common one-vector
production chunk retrieval, see:

- [openai_like_dense_chunk_baseline.md](openai_like_dense_chunk_baseline.md)

## Important Constraints

The current encoder path is still ColBERT on a BERT-family tokenizer.

Verified locally in this repo today:

- the tokenizer-side maximum is `512`
- the actual ColBERT document budget is `doc_maxlen = 180`
- so `256`, `384`, and `512` source-token chunks are **not** faithfully
  preserved as full ColBERT document representations; many hit the `180`-vector
  cap during encoding
- `800` / `400` and `1200` / `200` are important external reference points, but
  they are **not** faithfully representable under the current encoder without a
  different encoder path or an explicit overflow policy

That is why the current default sweep is:

- `256 / 64`
- `384 / 128`
- `512 / 128`

Those settings are still useful because they match chunk sizes users actually
ask about, but they must now be interpreted together with:

- `encoder_doc_maxlen`
- `at_doc_maxlen_chunk_fraction`

and not as "fully preserved chunk text under the current ColBERT path."

They also do **not** justify comparisons to:

- `800 / 400`
- `1200 / 200`

## Benchmark Path

Module:

- [../../../python/kayak_bridge/raw_text_chunk_sweep.py](../../../python/kayak_bridge/raw_text_chunk_sweep.py)

CLI:

- [../../../python/scripts/bench_raw_text_chunk_sweep.py](../../../python/scripts/bench_raw_text_chunk_sweep.py)

Focused unit tests:

- [../../../python/tests/test_raw_text_chunk_sweep.py](../../../python/tests/test_raw_text_chunk_sweep.py)

## How Chunks Are Embedded Today

The current raw-text chunk benchmark does the following:

- split raw document text into source-token chunks with the Hugging Face
  tokenizer for `task["model_name"]`
- decode those source-token windows back to text
- re-encode each chunk with ColBERT document encoding via
  `checkpoint.docFromText(...)`
- trim zero-padded rows from the resulting chunk token matrix
- mean-pool each chunk matrix down to one vector
- mean-pool each query matrix down to one vector
- run global retrieval over chunk vectors
- map winning chunk ids back to parent documents by deduplication

So the current raw-text chunk sweep is not "chunked late interaction." It is:

- raw-text chunking
- ColBERT re-encoding per chunk
- then one-vector compression per chunk

That distinction matters for interpreting the results.

## First Measured Smoke Run

On `2026-04-19`, I ran:

```bash
PATH="$PWD/.venv/bin:$PATH" \
PYTHONPATH=python ./.venv/bin/python python/scripts/bench_raw_text_chunk_sweep.py \
  --task .cache/kayak/bright_stackoverflow_real_subset/python_task.json \
  --output /tmp/bright_raw_text_chunk_sweep_smoke.json \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --encode-batch-size 8 \
  --backend numpy_reference
```

Primary metric on `bright_stackoverflow_real_subset`:

- exact late interaction: `0.2914`
- one vector per whole document: `0.1074`

Raw-text chunked one-vector sweep:

| Source chunk tokens | Overlap | Primary value | Mean recall@k | Generated chunks | Mean chunks / doc | Mean stored vectors / chunk | Encoder doc maxlen | At doc maxlen fraction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `256` | `64` | `0.1583` | `0.2153` | `574` | `3.8267` | `166.63` | `180` | `0.8171` |
| `384` | `128` | `0.1874` | `0.2014` | `424` | `2.8267` | `169.73` | `180` | `0.8679` |
| `512` | `128` | `0.1491` | `0.2014` | `318` | `2.1200` | `166.77` | `180` | `0.8365` |

## Focused Diagnostic: Chunking Or Compression?

On `2026-04-19`, on the same BRIGHT slice, I ran one focused comparison at
`384 / 128` to separate:

- chunking itself
- one-vector compression after chunk encoding

Measured results:

| Retrieval geometry | Primary value | Mean recall@k |
| --- | ---: | ---: |
| full-document exact late interaction | `0.2914` | `0.5139` |
| raw-text chunking + exact late interaction on chunk token matrices, then parent dedup | `0.4931` | `0.6389` |
| raw-text chunking + mean-pooled chunk vectors, full queries | `0.1874` | `0.2014` |
| raw-text chunking + mean-pooled chunk vectors, mean-pooled queries | `0.1874` | `0.2014` |

Interpretation:

- on this slice, chunking itself is not the main loss driver
- the large drop comes from compressing each encoded chunk to one vector
- on this slice, mean-pooling the query did not further change judged metrics
  once the documents were already compressed to one vector

This is still only one slice and one chunk setting, but it directly answers the
question "is the loss coming from chunking, or from how the chunks are
embedded?"

## What This Shows

On this slice:

- realistic raw-text chunking helps over one vector per whole document
- raw-text chunking with full late interaction over chunk matrices can beat the
  full-document exact baseline
- the current chunked one-vector baseline stays clearly below late interaction
  over full chunk matrices
- most of the measured degradation in the current raw-text chunk baseline comes
  from one-vector compression after chunk encoding
- chunk-size choice is non-monotonic even inside the current compared range
- a large share of chunks sit at the ColBERT `doc_maxlen` cap, so chunk-size
  comparisons are also affected by encoder budget pressure
- `384 / 128` is best of the three tested settings here, but only on this one
  slice

That is already a useful course-grade lesson.

It supports:

- "yes, chunk size matters"
- "no, there is not one universal chunk size story"

## What This Does Not Yet Show

This note does not prove:

- that `384 / 128` is a generally good setting
- that BRIGHT is representative of every production RAG workload
- that the current encoder path can faithfully simulate production chunking once
  many chunks hit `doc_maxlen = 180`
- that the current encoder path can faithfully simulate OpenAI-style
  `800 / 400` chunking

## Next Recommended Runs

The next epistemically sound expansion would be:

1. rerun the same sweep on:
   - `lemb_narrativeqa_real_subset`
   - `browsecomp_plus_real_subset`
   - `legal_rag_bench_real_subset`
2. compare whether the best chunk size under the current capped ColBERT path is
   stable across those
   slices
3. only after that discuss whether we need a different encoder path for
   `800+` token chunk comparisons
