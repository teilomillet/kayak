# 2026-04-25: FastPlaid Speed Track

## Claim

Kayak should compete with FastPlaid on speed, not only on transparency.

FastPlaid is the closest open-source system-level reference for Kayak's narrowed
late-interaction search direction:

- LightOn positions it as a Rust late-interaction retrieval engine optimized
  for production and GPUs
- the repository exposes PLAID-style K-means/PQ indexing, CPU/GPU device
  controls, and mutable `update(...)` indexing

Sources:

- <https://lighton.ai/lighton-blogs/fastplaid>
- <https://github.com/lightonai/fast-plaid>

## Implementation

Added:

- `python/scripts/bench_fastplaid_speed_track.py`
- `python/tests/test_fastplaid_speed_track.py`
- `python/kayak_bridge/plaid_approx.py`
- `kayak/search/plaid_approx_dim128.mojo`

Changed:

- `kayak/search/__init__.mojo`
- `python/kayak_bridge/_mojo_exact_cpu_bindings.mojo`
- `pyproject.toml`
- `docs/product_direction.md`
- `docs/search_layer_optimization_scorecard.md`

The harness:

- generates one deterministic synthetic multi-vector shape
- keeps document vectors, query vectors, vector dimension, and `top_k` explicit
- always runs Kayak exact as the reference
- optionally installs/runs FastPlaid through `uv --with fast-plaid==1.4.6.2110`
- reports FastPlaid recall against Kayak exact top-k
- reports query latency, QPS, build time, update time when requested, and index
  bytes
- includes an opt-in `kayak_plaid` lane that uses sampled centroid postings and
  exact MaxSim rerank over the selected candidate window
- runs the Kayak approximation hot path in Mojo: centroid assignment, candidate
  scoring, and rerank are all implemented in `_mojo_exact_cpu_bindings.mojo`

Reason:

- FastPlaid's PLAID/PQ path is approximate, so speed without recall against
  exact MaxSim is incomplete
- Kayak's current defensible advantage is exact-reference accountability; the
  benchmark must preserve that advantage while exposing the speed gap
- the `kayak_plaid` lane is still an explicit benchmark/product-direction probe,
  not a replacement for Kayak exact or a public production contract

## Validation

Focused tests:

```bash
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_fastplaid_speed_track -v
pixi run test_hybrid_flat_dim128
```

Observed:

- `6/6` tests passed
- `8/8` Mojo hybrid-flat tests passed, including the PLAID approximation
  full-candidate-window exact-rerank check

Kayak-only smoke task:

```bash
pixi run bench_fastplaid_speed_track_kayak_smoke_raw
```

Observed:

- emitted `.cache/kayak/fastplaid_speed_track_kayak_smoke/summary.json`
- Kayak `mojo_exact_cpu` status was `ok`

Full Python API suite:

```bash
pixi run test_python_api
```

Observed:

- `275` tests passed
- `3` tests skipped

Repository and docs checks:

```bash
pixi run test_repo_semantic_inventory
git diff --check
git -C public diff --check
uv run --no-project --python 3.11 \
  --with 'mkdocs>=1.6,<2.0' \
  --with 'mkdocs-material>=9.5,<10.0' \
  mkdocs build --strict
```

Observed:

- semantic inventory passed
- root and public diff whitespace checks passed
- strict public docs build passed

FastPlaid import check:

```bash
uv run --python 3.11 --with fast-plaid==1.4.6.2110 python - <<'PY'
from fast_plaid import search
import torch
print(search.FastPlaid)
print(torch.__version__)
PY
```

Observed:

- `fast_plaid.search.fast_plaid.FastPlaid` imported
- torch version was `2.11.0`

The first FastPlaid create attempt failed when passing a 3D document tensor:

```text
ValueError: too many values to unpack (expected 2)
```

The harness was changed to pass a list of per-document tensors. Reason: the
FastPlaid README documents list input, and that path works with the installed
wheel.

## Measured CPU Smoke

Command:

```bash
pixi run bench_fastplaid_speed_track_raw
```

Artifact:

- `.cache/kayak/fastplaid_speed_track_scorecard/summary.json`

Shape:

- documents: `256`
- document vectors/document: `32`
- total document vectors: `8192`
- queries: `8`
- query vectors/query: `16`
- total query vectors: `128`
- vector dim: `128`
- top_k: `10`
- FastPlaid version: `1.4.6.2110`
- FastPlaid device: `cpu`
- FastPlaid nbits: `4`
- Kayak PLAID candidate_k: `160`
- Kayak PLAID centroids/query vector: `32`
- Kayak PLAID centroid count: `128`

Results:

- Kayak exact query batch mean: `0.3453409026672792 s`
- Kayak exact QPS: `23.165515402928246`
- Kayak exact build time: `0.0016379580010834616 s`
- Kayak exact index bytes: `4196360`
- Kayak PLAID-style Mojo probe query batch mean: `0.008783846998994704 s`
- Kayak PLAID-style Mojo probe QPS: `910.7626761845448`
- Kayak PLAID-style Mojo probe build time: `0.04264354199767695 s`
- Kayak PLAID-style Mojo probe index bytes: `4255712`
- Kayak PLAID-style Mojo probe recall@10 vs Kayak exact:
  `0.7500000000000001`
- FastPlaid query batch mean: `0.08756494433206778 s`
- FastPlaid QPS: `91.36076155843867`
- FastPlaid build time: `0.531085582999367 s`
- FastPlaid index bytes: `5716494`
- FastPlaid recall@10 vs Kayak exact: `0.6`

Pairwise:

- Kayak PLAID-style query batch latency ratio vs Kayak exact:
  `0.02543529286902211`
- Kayak PLAID-style QPS ratio vs Kayak exact: `39.3154505886547`
- Kayak PLAID-style index bytes ratio vs Kayak exact:
  `1.0141436864330038`
- FastPlaid query batch latency ratio vs Kayak exact:
  `0.2535608833351338`
- FastPlaid QPS ratio vs Kayak exact: `3.9438259831201585`
- FastPlaid index bytes ratio vs Kayak exact: `1.3622506172015747`

Interpretation:

- Kayak's opt-in Mojo PLAID-style probe is faster than FastPlaid on this
  synthetic CPU query path
- Kayak's opt-in Mojo PLAID-style probe also has higher recall@10 vs Kayak
  exact on this run: `0.75` vs FastPlaid's `0.6`
- the result validates the product direction: keep exact as the reference, then
  expose approximate speed as an explicit parameter
- the reported Kayak approximation bytes include exact token values because the
  prepared object owns the vectors required for exact candidate rerank

## FastPlaid-Token-Shape Smoke

Command shape:

```bash
UV_CACHE_DIR=.cache/uv PYTHONPATH=python uv run --python 3.11 \
  --with fast-plaid==1.4.6.2110 \
  python python/scripts/bench_fastplaid_speed_track.py \
  --engines kayak_exact,kayak_plaid,fastplaid \
  --document-count 128 \
  --document-vector-count 300 \
  --query-count 4 \
  --query-vector-count 50 \
  --vector-dim 128 \
  --top-k 10 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --kayak-plaid-candidate-k 96 \
  --kayak-plaid-centroids-per-query-vector 32 \
  --index-root .cache/kayak/fastplaid_speed_track_turf_smoke/indexes \
  --overwrite-index-root \
  --output .cache/kayak/fastplaid_speed_track_turf_smoke/summary.json \
  --require-fastplaid
```

Observed:

- artifact: `.cache/kayak/fastplaid_speed_track_turf_smoke/summary.json`
- shape: `128` documents, `300` document vectors/document, `4` queries, `50`
  query vectors/query, vector dim `128`, `top_k=10`
- Kayak PLAID-style Mojo probe query batch mean:
  `0.05640716599737061 s`
- Kayak PLAID-style Mojo probe QPS: `70.91297584754493`
- Kayak PLAID-style Mojo probe recall@10 vs Kayak exact: `0.675`
- FastPlaid query batch mean: `0.07875895899996976 s`
- FastPlaid QPS: `50.787872907278214`
- FastPlaid recall@10 vs Kayak exact: `0.5499999999999999`

Interpretation:

- this is a raw single-iteration falsification smoke, not a quiet-host
  benchmark claim
- the Mojo approximation remained faster on a token shape closer to the
  FastPlaid README style (`300` document vectors and `50` query vectors)

## Mutable-Index Smoke

Command shape:

```bash
UV_CACHE_DIR=.cache/uv PYTHONPATH=python uv run --python 3.11 \
  --with fast-plaid==1.4.6.2110 \
  python python/scripts/bench_fastplaid_speed_track.py \
  --document-count 80 \
  --document-vector-count 16 \
  --query-count 4 \
  --query-vector-count 8 \
  --vector-dim 128 \
  --top-k 10 \
  --update-document-count 16 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --index-root .cache/kayak/fastplaid_speed_track_fastplaid_update_smoke/indexes \
  --overwrite-index-root \
  --output .cache/kayak/fastplaid_speed_track_fastplaid_update_smoke/summary.json \
  --require-fastplaid
```

Observed:

- FastPlaid update time: `0.5240594580027391 s`
- FastPlaid QPS ratio vs Kayak exact after update: `3.3684699169176335`
- FastPlaid recall@10 vs Kayak exact after update: `0.75`

## Caveats

- These are synthetic CPU results only.
- The quiet-wrapper run needed `--force` because the local host did not stay
  below the competing-CPU threshold.
- The benchmark has not yet been run on GPU.
- The benchmark has not yet been run on real encoded task JSON.
- No claim is made that the index-byte relationship generalizes beyond this
  small shape.
- The Kayak speed win is from an opt-in Mojo benchmark lane, not yet from a
  public product API.

## Next Work

The next speed work should target the gap surfaced here:

- add larger synthetic scale points that match FastPlaid README-style shapes
  such as `300` document tokens and `50` query tokens
- add a task-JSON FastPlaid comparison using real encoded documents and judged
  metrics
- add GPU lanes when hardware is available
- optimize Kayak's Mojo exact batch path and candidate path with this track as
  the external speed reference
