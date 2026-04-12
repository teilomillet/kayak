# Centroid Sidecar Summaries

Date: `2026-04-12`

This step tightened the centroid-postings sidecar layout rather than changing
retrieval semantics.

## What Changed

The `centroid_postings` artifact now persists segment-local centroid summary
arrays:
- `centroid_document_counts.tsv`
- `centroid_token_counts.tsv`

The in-memory `CentroidPostingIndex` now carries those summaries directly:
- `centroid_document_counts`
- `centroid_token_counts`
- `total_centroid_token_count`

Why this is justified:
- the imputed scorer already needed centroid token totals
- before this change it recomputed them from postings on every query
- these values are deterministic functions of the postings and belong in the
  search-native sidecar

## Compatibility Boundary

This step keeps the old layout loadable.

Verified behavior:
- old centroid sidecars without the new summary files still load
- partial summary layouts are rejected
- `ensure_stored_centroid_posting_index()` upgrades old sidecars into the new
  layout instead of silently reusing them forever

## Storage Impact

Regenerated artifact:
- `.cache/kayak/public_real_slice_collection_storage.json`

Observed `bytes_per_vector` after adding the summary files:

- `beir/scifact/test`: `523.204689`
  previous trace value: `523.13`
  delta: `+0.074689`
- `beir/fiqa/test`: `524.163796`
  previous trace value: `524.09`
  delta: `+0.073796`
- `orionweller/LIMIT-small`: `524.523143`
  previous trace value: `524.45`
  delta: `+0.073143`
- `Tevatron/browsecomp-plus/evidence-slice`: `520.397499`
  previous trace value: `520.35`
  delta: `+0.047499`
- `Tevatron/browsecomp-plus/gold-slice`: `520.396738`
  previous trace value: `520.35`
  delta: `+0.046738`

Inference:
- the new summaries have a real but small storage cost
- this is acceptable as long as they remove repeated hot-path recomputation and
  make future native search layout work easier

## Public Candidate-Stage Timing

Regenerated artifact:
- `.cache/kayak/public_candidate_window_sweep.json`

The candidate-window artifact now records
`mean_candidate_generation_seconds`.

Representative rows at `candidate_k = 40`:

- `beir/scifact/test`
  - `document_proxy`: recall `1.0`, candidate stage `0.0000065s`
  - `centroid_postings`: recall `0.95`, candidate stage `0.0000923s`
  - `centroid_postings_imputed`: recall `0.9`, candidate stage `0.000449s`
  - `exact_full_scan`: recall `1.0`, candidate stage `0.0007987s`
- `orionweller/LIMIT-small`
  - `document_proxy`: recall `0.965625`, candidate stage `0.0000054s`
  - `centroid_postings`: recall `0.903125`, candidate stage `0.0000877s`
  - `centroid_postings_imputed`: recall `0.86875`, candidate stage `0.0003398s`
  - `exact_full_scan`: recall `1.0`, candidate stage `0.0006515s`
- `Tevatron/browsecomp-plus/evidence-slice`
  - `document_proxy`: recall `1.0`, candidate stage `0.0000098s`
  - `centroid_postings`: recall `0.8`, candidate stage `0.0001053s`
  - `centroid_postings_imputed`: recall `0.75`, candidate stage `0.000471s`
  - `exact_full_scan`: recall `1.0`, candidate stage `0.0016248s`

## Benchmark Measurement Guardrail

The first implementation used the default `std.benchmark.run()` settings and
made the full candidate-window sweep too slow to be a routine artifact.

This was corrected by bounding the timing path to one explicit pass over the
query set:
- `num_warmup_iters = 0`
- `max_iters = len(task.queries)`
- `min_runtime_secs = 0.0`
- `max_batch_size = 1`

Primary source used for that correction:
- official Modular stdlib benchmark docs:
  https://docs.modular.com/mojo/stdlib/benchmark/benchmark/Report/

## Decision

The sound conclusion is:
- keep the centroid summary files in the default sidecar layout
- keep the candidate-stage timing field in the public candidate-window artifact
- treat this as a storage-and-observability step, not a retrieval-quality step
- move the next heavier work toward more expressive native layouts, not more
  heuristic recomputation on the same postings format
