# 2026-04-27: GPU I8 Selected-Centroid Boundary

## Question

What should the first GPU posting-accumulation probe take as input?

## Decision

Expose selected centroid positions and proxy scores from the existing CPU i8
candidate-generation path before writing a GPU posting kernel.

Reason: selected centroid scoring/selection and posting traversal are separate
substeps in the CPU profiler. A GPU posting-accumulation probe should start
from the same selected centroids as the CPU reference so any correctness or
timing difference is attributable to posting traversal, not to centroid
selection.

## Change

Added an internal selected-centroid export:

- Mojo binding:
  `plaid_i8_selected_centroids_prepared_batch_address`
- Python wrapper:
  `KayakPlaidApproxIndex.i8_selected_centroids_batch(...)`
- Data contract:
  `KayakPlaidI8SelectedCentroids`

The exported rows are flattened in query-vector-major order. For each query:

- `positions_by_query[q]` has
  `query_vector_count * centroids_per_query_vector` centroid ids
- `scores_by_query[q]` has the matching proxy score for each selected centroid

## Validation

Command:

```bash
pixi run env PYTHONPATH=python python -m unittest python/tests/test_fastplaid_speed_track.py
```

Result:

- `13` tests passed
- selected-centroid export reports explicit query count, query vector count,
  centroid budget, total selected count, and ndarray shapes

## Next Step

Use this boundary in a benchmark-only GPU posting-traversal probe.

Initial correctness target:

- CPU reference: selected centroid ids and scores from this export
- GPU input: selected centroid ids/scores plus resident centroid posting arrays
- GPU output: document score or touched-document state for the same selected
  centroids

Reason: this keeps centroid selection fixed while testing whether the posting
accumulation stage itself is worth moving to GPU.
