# 2026-04-25: Search-Layer Scorecard And Mojo Payload Cache

## Claim

After narrowing Kayak to the late-interaction search layer, the next work should
optimize exact repeated-query throughput before more speculative stage-1 work.

The first concrete implementation claim was:

- repeated exact search over the same `hybrid_flat_dim128` index should not
  rebuild the same Mojo index payload on every call

## Implementation

Added:

- `python/kayak_bridge/mojo_payload_cache.py`

Changed:

- `python/kayak_bridge/backend_dispatch.py`
- `python/kayak_bridge/batch_dispatch.py`
- `python/scripts/bench_batch_maxsim.py`
- `python/kayak_engine/prepared_exact_process_runtime.py`
- `MANIFEST.in`
- `setup.py`
- `python/tests/test_sdist_manifest.py`

The cache is:

- bounded to `8` index payloads
- scoped by `LateIndex` object identity
- guarded by an `RLock`
- internal to the Mojo bridge, not a public API

Reason:

- `LateIndex` instances are immutable at the Python SDK layer
- repeated-query workloads reuse the same index object
- the packed path already has a prepared-index cache, while the hybrid-flat
  path was still rebuilding list payloads for the Mojo bridge

Verification also exposed a prepared-runtime stats race: the process worker
published individual success envelopes before batch metrics, so callers could
observe successful futures before `completed_request_count` advanced. The
worker now publishes the batch metrics envelope before the per-request result
envelopes. Reason: execution metrics describe the completed batch and should be
visible before futures unblock.

## Benchmark Evidence

Benchmark command shape:

```bash
bash scripts/run_bench_quiet.sh --timeout-seconds 5 --force -- \
  env PYTHONPATH=python pixi run python python/scripts/bench_batch_maxsim.py \
    --mode shared_batch \
    --document-count 2000 \
    --document-vector-count 32 \
    --batch-size 4 \
    --query-vector-count 6 \
    --repeats 3 \
    --warmup-runs 1 \
    [--clear-index-payload-cache-per-run]
```

Measured shape:

- query layout: `flat_dim128`
- index layout: `hybrid_flat_dim128`
- batch size: `4`
- query vectors/query: `6`
- documents: `2000`
- document vectors/document: `32`
- total index vectors: `64000`

Forced uncached payload rebuild:

- log dir: `.cache/kayak/bench_quiet/20260425T093215Z`
- median run mean: `1.9359409443325906 s`

Cached payload reuse:

- log dir: `.cache/kayak/bench_quiet/20260425T093320Z`
- median run mean: `1.7000958473339172 s`

Derived:

- speedup: `1.139x`
- time reduction: `12.18%`

Important caveat:

- both quiet-wrapper runs used `--force` because the host did not remain quiet
  under the default threshold
- this supports a scoped repeated-query improvement, not a universal latency
  claim

## Negative Or Ambiguous Evidence

The default small benchmark shape did not show a clean quiet-wrapper win:

- uncached median run mean: `0.3006730979997883 s`
- cached median run mean: `0.3089791686505123 s`

Interpretation:

- the cache is not justified by small-index timing
- the win appears only once the repeated index payload is large enough to matter
- future work should isolate query payload conversion, index payload conversion,
  Mojo scoring, and result materialization before adding another optimization

## Validation

Commands:

```bash
PYTHONPATH=python pixi run python -m unittest python.tests.test_mojo_payload_cache -v
PYTHONPATH=python pixi run python -m unittest python.tests.test_batch_api -v
PYTHONPATH=python pixi run python -m unittest python.tests.test_sdist_manifest -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_public_api_contract \
  python.tests.test_mojo_payload_cache \
  python.tests.test_batch_api -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_prepared_exact_search_scheduler.PreparedExactSearchRuntimeTests.test_runtime_coalesces_concurrent_submitters -v
PYTHONPATH=python pixi run python -m unittest \
  python.tests.test_public_typing_surface \
  python.tests.test_encoder_api -v
pixi run test_python_api
pixi run test_repo_semantic_inventory
git diff --check
git -C public diff --check
uv run --no-project --python 3.11 \
  --with 'mkdocs>=1.6,<2.0' \
  --with 'mkdocs-material>=9.5,<10.0' \
  mkdocs build --strict
```

Observed:

- new cache tests: `4/4` passed
- batch API tests: `9/9` passed
- sdist manifest tests: `3/3` passed
- combined public/cache/batch tests: `17/17` passed
- targeted prepared-runtime stats test passed
- targeted public typing plus encoder registry tests: `19/19` passed
- full Python API suite: `269` tests passed, `3` skipped
- semantic inventory: `3/3` passed
- whitespace checks passed for root and public docs
- strict public MkDocs build passed

Additional scorecard checks:

```bash
pixi run bench_python_store_search_raw
pixi run bench_real_subset_planner_benchmark_smoke_raw
PYTHONPATH=python pixi run python python/scripts/bench_prepared_exact_runtime.py ...
```

These are recorded in
[docs/search_layer_optimization_scorecard.md](../search_layer_optimization_scorecard.md).
