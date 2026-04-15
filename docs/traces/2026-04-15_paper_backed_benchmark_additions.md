# 2026-04-15: paper-backed benchmark additions

## Goal

Add recent, paper-backed retrieval benchmarks without breaking the public
benchmark loop or pretending that every benchmark has the same operational
cost.

## Decision rule

- add benchmarks only after verifying a real local packaging path
- measure first-materialization cost instead of inferring it from corpus size
- keep the public default set limited to slices that stay practical for repeat
  local runs
- keep heavier but still useful slices available as opt-in entrypoints

## Primary sources

- BRIGHT: A Realistic and Challenging Benchmark for Reasoning-Intensive
  Retrieval
  https://arxiv.org/abs/2407.12883
- LONGEMBED: Extending Embedding Models for Long Context Retrieval
  https://aclanthology.org/2024.emnlp-main.47/
- R2MED: A Benchmark for Reasoning-Driven Medical Retrieval
  https://arxiv.org/abs/2505.14558

## Verified local packaging paths

- `xlangai/BRIGHT` on Hugging Face
- `mteb/LEMBNarrativeQARetrieval` on Hugging Face
- `R2MED/Biology` on Hugging Face

I validated all three locally through the subset-builder path used by
`python/kayak_bridge/*_subset.py`.

## Measured builder costs

Measured with:

```bash
PYTHONPATH=python pixi run python - <<'PY'
import time
from kayak_bridge.bright_subset import build_bright_colbert_subset
from kayak_bridge.lemb_narrativeqa_subset import build_lemb_narrativeqa_colbert_subset
from kayak_bridge.r2med_biology_subset import build_r2med_biology_colbert_subset

for name, fn in [
    ("bright", build_bright_colbert_subset),
    ("lemb", build_lemb_narrativeqa_colbert_subset),
    ("r2med", build_r2med_biology_colbert_subset),
]:
    start = time.perf_counter()
    task = fn()
    print(
        name,
        round(time.perf_counter() - start, 3),
        len(task["queries"]),
        len(task["documents"]),
        task["nominal_document_vector_count"],
    )
PY
```

Observed results:

- `bright`: `24.786s`, `8` queries, `150` documents, `154` nominal document
  vectors
- `lemb`: `118.644s`, `8` queries, `355` documents, `180` nominal document
  vectors
- `r2med`: `22.818s`, `8` queries, `178` documents, `130` nominal document
  vectors

Interpretation:

- BRIGHT and R2MED have similar first-materialization costs.
- LEMB is substantially heavier because the corpus documents are much longer,
  even though the corpus itself is small.
- The added cost for LEMB is justified by the new long-document stress axis,
  but it should be a conscious inclusion rather than an accidental one.

## End-to-end benchmark runs

Validated through the Mojo entrypoints with the Pixi-backed Python wrapper:

- `pixi run bench_bright_stackoverflow_raw`
- `pixi run bench_lemb_narrativeqa_raw`
- `pixi run bench_r2med_biology_raw`

Observed benchmark means:

- BRIGHT StackOverflow:
  `0.007812338578088578s`
- LEMB NarrativeQA:
  `0.010910489119170984s`
- R2MED Biology:
  `0.005766707122507123s`

The LEMB run and the R2MED run both emitted a Python
`resource_tracker` semaphore-cleanup warning at process shutdown. The benchmark
outputs were still written successfully, so this is an operational warning, not
currently a benchmark blocker.

## Public benchmark inclusion decision

### Kept in the default public set

- `bright_stackoverflow_real_subset`
- `lemb_narrativeqa_real_subset`

Reason:

- BRIGHT adds a recent reasoning-intensive retrieval benchmark.
- LEMB adds a recent long-document retrieval benchmark that stresses document
  vector budgets and truncation.
- Both integrate cleanly into the existing public benchmark registry and were
  validated end to end.

### Added as opt-in, not default

- `r2med_biology_real_subset`

Reason:

- It is useful and verified, but it is a domain-specific benchmark rather than
  a broadly representative public default slice.
- Keeping it opt-in preserves modularity without forcing every default public
  run to pay for a biomedical benchmark.

### Removed from the default path

- `miracl_es_real_subset`

Reason:

- The attempted MIRACL path was operationally too heavy for a sound default
  public slice in this repo.
- The extraction path kept degenerating toward broad corpus materialization,
  which failed the same tractability bar applied to the newly added datasets.

## Integration checks

These completed successfully after the registry changes:

- `pixi run bench_public_query_bucket_frontier_raw`
- `pixi run bench_public_small_window_frontier_raw`
- `pixi run bench_public_wide_window_native_frontier_raw`
- `pixi run bench_public_wide_window_candidate_generation_raw`

This verifies that the public benchmark surfaces still load and execute with
the new dataset set.

## Query-bucket implication

The public query-bucket frontier still populated only one bucket:

- `qv_17_32`

Reason:

- the current ColBERT query encoder continues to emit `32` query vectors on the
  sampled BRIGHT, LEMB, and R2MED slices
- these benchmark additions diversify semantics and document length, but they
  do not yet broaden the measured query-width distribution under the current
  encoder
