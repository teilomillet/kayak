# 2026-04-14 LanceDB Multivector BrowseComp Baseline

## Goal

Create the first runnable LanceDB comparison harness that:
- consumes the same encoded BrowseComp task JSON already used in-repo
- runs on the same machine as Kayak
- reports judged metrics and storage context in one artifact

This is intentionally narrower than a full external-comparison contract.

It answers only the first question:
- can Kayak and LanceDB consume the same ColBERT-shaped multivector task and
  return the same judged ranking quality on the current small BrowseComp
  slices?

## Why This Shape

Reason:
- the repo already has machine-readable public-slice benchmark artifacts for
  Kayak exact search
- the new external baseline should reuse the same encoded corpus and queries
  before introducing adapter or re-encoding variables
- LanceDB multivector search is a native late-interaction comparison surface

Important boundary:
- LanceDB multivector search currently supports cosine similarity
- Kayak exact MaxSim uses dot products
- the cached ColBERT vectors are unit-normalized when non-zero, so cosine and
  dot product are compatible on the non-zero rows
- the cached task also includes exact zero vectors, which are degenerate under
  cosine and therefore must be filtered explicitly

That filtering is part of the reported artifact:
- original nominal vector counts stay visible
- stored vector counts after zero filtering are recorded separately

## Implementation

Added:
- [python/kayak_bridge/judged_metrics.py](../../python/kayak_bridge/judged_metrics.py)
  for Python-side judged metric semantics aligned with
  [kayak/eval/metrics.mojo](../../kayak/eval/metrics.mojo)
- [python/kayak_bridge/lancedb_benchmark.py](../../python/kayak_bridge/lancedb_benchmark.py)
  for the LanceDB multivector benchmark runner
- [python/kayak_bridge/comparison_scorecard.py](../../python/kayak_bridge/comparison_scorecard.py)
  for frozen comparison scorecards over benchmark artifacts
- [python/kayak_bridge/kayak_task_benchmark.py](../../python/kayak_bridge/kayak_task_benchmark.py)
  for Python-driven Kayak exact benchmarking on encoded task JSON
- [python/kayak_bridge/lancedb_storage_comparison.py](../../python/kayak_bridge/lancedb_storage_comparison.py)
  for same-storage comparisons where both branches read the same LanceDB table
- [python/kayak_bridge/task_scale.py](../../python/kayak_bridge/task_scale.py)
  for deterministic corpus inflation during scale sweeps
- [python/kayak_bridge/scale_comparison.py](../../python/kayak_bridge/scale_comparison.py)
  for same-task scale sweeps between Kayak exact and LanceDB scan
- [python/kayak_bridge/storage_scale_comparison.py](../../python/kayak_bridge/storage_scale_comparison.py)
  for same-task storage sweeps between Kayak packed storage and LanceDB storage
- [python/kayak_bridge/prepared_index_cache.py](../../python/kayak_bridge/prepared_index_cache.py)
  for bounded caching of native prepared packed-index objects on the Python
  exact path
- [python/kayak_bridge/prepared_index_storage_artifact.py](../../python/kayak_bridge/prepared_index_storage_artifact.py)
  for the transient flat-dim128 artifact used only on the Python exact cold
  path
- [python/scripts/bench_lancedb_multivector.py](../../python/scripts/bench_lancedb_multivector.py)
  for a CLI entrypoint over encoded task JSON
- [python/scripts/bench_lancedb_storage_compare.py](../../python/scripts/bench_lancedb_storage_compare.py)
  for same-LanceDB-storage search comparisons
- [python/scripts/bench_lancedb_scale_sweep.py](../../python/scripts/bench_lancedb_scale_sweep.py)
  for gold-slice scale sweeps under the Python driver
- [python/scripts/bench_storage_engine_scale_compare.py](../../python/scripts/bench_storage_engine_scale_compare.py)
  for gold and evidence storage sweeps on identical filtered vectors
- [benchmarks/task_json_storage_encoding.mojo](../../benchmarks/task_json_storage_encoding.mojo)
  for native packed-index store/load/search benchmarking from task JSON
- [python/scripts/build_comparison_scorecard.py](../../python/scripts/build_comparison_scorecard.py)
  for compact multi-artifact scorecards
- `pixi` tasks in [pyproject.toml](../../pyproject.toml) for BrowseComp
  evidence and gold slices via `uv run --with lancedb`
- indexed `IVF_PQ` variants via `uv run --with lancedb --with faiss-cpu`
- scorecard tasks for gold and evidence slices
- rebuild-variance tasks for the gold and evidence indexed paths
- frozen indexed-summary tasks that use `mean_across_5_rebuilds`
- storage-controlled gold and evidence tasks where LanceDB owns storage in both
  branches
- gold and evidence scale-sweep tasks for seeing whether the current delta
  persists as corpus size grows synthetically
- gold and evidence storage-scale tasks for testing whether Kayak packed
  storage stays smaller or cheaper to build than LanceDB on the same filtered
  vectors
- the Python exact benchmark helper now uses the existing batched public API so
  the Mojo module and index payload are reused across the query set
- the Mojo exact binding now exposes a native prepared packed-index Python
  object so the packed index is decoded once per index reuse instead of once
  per query batch
- the dim128 packed cold path now writes only `doc_ids.tsv`, `doc_offsets.bin`,
  and `token_values.bin`, then loads them into a `HybridFlatDim128Index`-backed
  prepared object instead of rebuilding nested Python float lists
- [kayak/index/packed_index.mojo](../../kayak/index/packed_index.mojo) now
  exposes a lightweight `Writable` summary because Mojo Python type binding
  requires the bound native object graph to be printable
- [kayak/index/hybrid_flat_dim128.mojo](../../kayak/index/hybrid_flat_dim128.mojo)
  now also exposes a lightweight `Writable` summary for that same binding
  reason on the flat prepared path

## Verification

Tested:

```bash
PYTHONPATH=python pixi run python -m unittest python/tests/test_judged_metrics.py
PYTHONPATH=python uv run --python 3.11 --with lancedb python python/scripts/bench_lancedb_multivector.py --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json --output .cache/kayak/lancedb_browsecomp_plus_gold_benchmark.json --db-root .cache/kayak/lancedb_browsecomp_plus_gold --table-name browsecomp_plus_gold --warmup-iterations 1 --measurement-iterations 3
PYTHONPATH=python uv run --python 3.11 --with lancedb python python/scripts/bench_lancedb_multivector.py --task .cache/kayak/browsecomp_plus_real_subset/python_task_evidence.json --output .cache/kayak/lancedb_browsecomp_plus_evidence_benchmark.json --db-root .cache/kayak/lancedb_browsecomp_plus_evidence --table-name browsecomp_plus_evidence --warmup-iterations 1 --measurement-iterations 3
PYTHONPATH=python uv run --python 3.11 --with lancedb --with faiss-cpu python python/scripts/bench_lancedb_multivector.py --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json --output /tmp/lancedb_indexed_gold.json --db-root /tmp/lancedb_indexed_gold_db --table-name browsecomp_plus_gold --warmup-iterations 1 --measurement-iterations 3 --build-index
pixi run bench_lancedb_browsecomp_plus_gold_raw
pixi run bench_lancedb_browsecomp_plus_gold_indexed_variance_raw
pixi run bench_lancedb_browsecomp_plus_evidence_indexed_variance_raw
pixi run bench_lancedb_browsecomp_plus_gold_indexed_frozen_raw
pixi run bench_lancedb_browsecomp_plus_evidence_indexed_frozen_raw
pixi run bench_lancedb_browsecomp_plus_gold_scorecard_raw
pixi run bench_lancedb_browsecomp_plus_evidence_scorecard_raw
pixi run bench_lancedb_browsecomp_plus_gold_storage_compare_raw
pixi run bench_lancedb_browsecomp_plus_evidence_storage_compare_raw
pixi run bench_lancedb_browsecomp_plus_gold_scale_sweep_raw
pixi run bench_lancedb_browsecomp_plus_evidence_scale_sweep_raw
PYTHONPATH=python pixi run python -m unittest python/tests/test_prepared_index_cache.py python/tests/test_storage_scale_comparison.py python/tests/test_task_scale.py python/tests/test_kayak_task_benchmark.py python/tests/test_batch_api.py
pixi run bench_storage_engine_browsecomp_plus_gold_scale_raw
pixi run bench_storage_engine_browsecomp_plus_evidence_scale_raw
```

Observed:
- the LanceDB gold benchmark artifact was written successfully
- the LanceDB evidence benchmark artifact was written successfully
- the `pixi` gold task also ran successfully end to end
- the indexed LanceDB gold run also completed successfully with local `faiss-cpu`
- the indexed gold and evidence rebuild-variance artifacts were also written
  successfully
- the indexed gold and evidence frozen-summary artifacts were also written
  successfully
- the gold and evidence comparison scorecards were also written successfully
- the gold same-storage comparison artifact was written successfully
- the evidence same-storage comparison artifact was written successfully
- the gold scale-sweep artifact was written successfully
- the evidence scale-sweep artifact was written successfully
- the gold storage-scale artifact was written successfully
- the evidence storage-scale artifact was written successfully

## Current Results

Using the existing cached task artifacts:

### BrowseComp-Plus Gold

Kayak exact artifact:
- `.cache/kayak/browsecomp_plus_gold_benchmark.json`

LanceDB artifact:
- `.cache/kayak/lancedb_browsecomp_plus_gold_benchmark.json`

Indexed LanceDB artifact:
- `/tmp/lancedb_indexed_gold.json`

Observed equality on judged quality up to floating-point noise:
- Kayak `mean_ndcg@10 = 0.2851267779084149`
- LanceDB `mean_ndcg@10 = 0.28512677790387286`
- Kayak `mean_recall@10 = 0.30833333333333335`
- LanceDB `mean_recall@10 = 0.30833333333333335`
- Kayak `MRR@10 = 0.4375`
- LanceDB `MRR@10 = 0.4375`

Latency on the current small slice is much higher in the current LanceDB path:
- Kayak exact `mean_search_seconds ≈ 0.000768`
- LanceDB multivector `mean_search_seconds ≈ 0.02555`

Indexed LanceDB on the same slice is a distinct operating point:
- LanceDB `IVF_PQ` `mean_ndcg@10 = 0.33944250317087515`
- LanceDB `IVF_PQ` `mean_recall@10 = 0.5583333333333333`
- LanceDB `IVF_PQ` `success_rate@10 = 1.0`
- LanceDB `IVF_PQ` `mean_search_seconds ≈ 0.02195`

Interpretation:
- on this tiny slice, `IVF_PQ` is only modestly faster than the scan path
- it also changes the ranking materially
- therefore it must be tracked as a separate baseline configuration rather than
  treated as a transparent acceleration of the same result

Indexed rebuild variance on gold is also material:
- artifact:
  - `.cache/kayak/lancedb_browsecomp_plus_gold_indexed_variance.json`
- over `5` rebuilds with the same task and settings:
  - frozen policy:
    - `mean_across_5_rebuilds`
  - frozen indexed artifact:
    - `.cache/kayak/lancedb_browsecomp_plus_gold_indexed_frozen_benchmark.json`
  - scorecard:
    - `.cache/kayak/lancedb_browsecomp_plus_gold_scorecard.json`
  - frozen indexed `mean_ndcg@10 = 0.24778915953852823`
  - frozen indexed `mean_search_seconds ≈ 0.02602`
  - source `mean_ndcg@10` ranged from `0.18007673124358647` to
    `0.32046983940500645`
  - source `mean_search_seconds` ranged from `0.02257967025313216` to
    `0.029648833316362772`

Interpretation:
- the current indexed LanceDB point is not stable enough to summarize with one
  run only
- this repo now freezes the indexed gold point as the arithmetic mean across a
  fixed `5` rebuild budget, while retaining min/max in the variance artifact
- that is stricter than best-of-`N` and less brittle than freezing one lucky
  rebuild

Storage reported by the LanceDB artifact:
- `storage_byte_size = 7,801,876`
- `bytes_per_document ≈ 86,687.51`
- `bytes_per_vector ≈ 610.57`
- original nominal document vectors/document: `175`
- stored document vectors/document after zero filtering: `142`
- zero document vectors filtered: `2,978`

### Same LanceDB Storage, Different Search Engines

Artifact:
- `.cache/kayak/lancedb_browsecomp_plus_gold_storage_compare.json`

Engine boundary verified from code:
- [benchmarks/browsecomp_plus_gold_real_subset_search.mojo](../../benchmarks/browsecomp_plus_gold_real_subset_search.mojo)
  uses `ExactCpuBackend()` plus `search_exact(...)`
- [kayak/search/exact_search.mojo](../../kayak/search/exact_search.mojo)
  scores Kayak's own packed index
- [python/kayak_bridge/lancedb_benchmark.py](../../python/kayak_bridge/lancedb_benchmark.py)
  is the separate LanceDB path

Measured on the same LanceDB-stored gold corpus:
- LanceDB scan `mean_ndcg@10 = 0.28512677790387286`
- Kayak exact from LanceDB `mean_ndcg@10 = 0.28512677790387286`
- LanceDB scan `mean_search_seconds ≈ 0.02567`
- Kayak exact from LanceDB `mean_search_seconds ≈ 0.00121`
- Kayak materialization from LanceDB rows to a packed Kayak index took
  `≈ 0.20881s` once per load

Interpretation:
- the storage substrate alone does not explain the earlier latency gap
- when both branches are driven from Python and both read the same LanceDB
  rows, judged quality stays equal and Kayak exact from LanceDB is now faster
  on the base gold slice
- the current prepared-index path moves that same-storage base-slice ratio to
  `≈ 21.15`
- this does **not** contradict the native Mojo benchmark above
- it shows that driver and integration overhead are a first-class part of the
  comparison surface

### Gold Scale Sweep

Artifact:
- `.cache/kayak/lancedb_browsecomp_plus_gold_scale_sweep.json`

Policy:
- the gold corpus is inflated by repeating only non-protected distractor
  documents with unique `doc_id`s
- protected means:
  - any judged relevant document
  - any document that already appears in the base exact top-`k`
- this keeps the sweep additive while reducing avoidable quality drift from the
  duplicated distractors

Measured under the current Python driver:
- `90` docs:
  - Kayak exact `≈ 0.00153s`
  - LanceDB scan `≈ 0.02736s`
  - LanceDB/Kayak latency ratio `≈ 17.84`
- `180` docs:
  - Kayak exact `≈ 0.00248s`
  - LanceDB scan `≈ 0.03236s`
  - ratio `≈ 13.04`
- `360` docs:
  - Kayak exact `≈ 0.00454s`
  - LanceDB scan `≈ 0.04041s`
  - ratio `≈ 8.91`
- `720` docs:
  - Kayak exact `≈ 0.00863s`
  - LanceDB scan `≈ 0.05896s`
  - ratio `≈ 6.84`

Observed quality across those points stayed equal up to floating-point noise:
- Kayak exact `mean_ndcg@10 = 0.28512677790387286`
- LanceDB scan `mean_ndcg@10 = 0.28512677790387286`

Interpretation:
- under the current Python-driven harness, Kayak exact now wins at every
  tested gold scale point from `90` to `720` documents
- the remaining trend is not a reversal in winner but a narrowing advantage:
  LanceDB/Kayak drops from `≈ 17.84x` at `90` docs to `≈ 6.84x` at `720` docs
- this sweep should **not** be treated as a native Kayak-versus-LanceDB engine
  claim, because the Kayak side is paying Python-to-Mojo call overhead on
  every query
- the native Mojo benchmark remains the sound reference for Kayak's current
  standalone exact path

### Gold Storage Engine Scale Sweep

Artifacts:
- `.cache/kayak/storage_engine_browsecomp_plus_gold_scale.json`
- `.cache/kayak/task_json_storage_encoding_output.json`

Scope:
- both branches use the same filtered task, so the stored document vectors are
  `12,778` at `90` docs instead of the original `15,756`
- Kayak is measured with the native `packed_index` persisted format using
  `binary_le`, which keeps vector precision aligned with the Float32 LanceDB
  baseline
- scale uses the same non-protected-distractor duplication policy as the
  search sweep

Measured:
- `90` docs / `12,778` stored vectors:
  - Kayak packed storage `6,543,574` bytes
  - LanceDB storage `7,801,876` bytes
  - LanceDB/Kayak byte ratio `≈ 1.19`
  - Kayak build `≈ 0.00532s`
  - LanceDB build `≈ 0.15261s`
  - build ratio `≈ 28.67`
- `180` docs / `25,606` stored vectors:
  - Kayak `13,114,381` bytes
  - LanceDB `14,373,698` bytes
  - ratio `≈ 1.10`
  - Kayak build `≈ 0.00891s`
  - LanceDB build `≈ 0.15281s`
  - build ratio `≈ 17.15`
- `360` docs / `51,262` stored vectors:
  - Kayak `26,255,993` bytes
  - LanceDB `27,517,058` bytes
  - ratio `≈ 1.05`
  - Kayak build `≈ 0.01590s`
  - LanceDB build `≈ 0.34493s`
  - build ratio `≈ 21.69`
- `720` docs / `102,574` stored vectors:
  - Kayak `52,539,236` bytes
  - LanceDB `53,803,984` bytes
  - ratio `≈ 1.02`
  - Kayak build `≈ 0.03068s`
  - LanceDB build `≈ 0.60914s`
  - build ratio `≈ 19.86`

Observed search on the reloaded native-storage path:
- `90` docs:
  - Kayak exact `≈ 0.00308s`
  - LanceDB scan `≈ 0.03923s`
  - ratio `≈ 12.72`
- `720` docs:
  - Kayak exact `≈ 0.00945s`
  - LanceDB scan `≈ 0.05838s`
  - ratio `≈ 6.18`

Interpretation:
- on the gold slice, Kayak packed storage is smaller than LanceDB at every
  tested point when both store the same filtered vectors
- the byte gap narrows with scale, but it does not reverse through `720` docs
- build cost is not close on this host: LanceDB build time stayed roughly
  `17x` to `29x` higher than Kayak packed storage build time across the sweep
- this is the first fair storage comparison in this trace because the earlier
  cached Kayak directory kept zero vectors while the LanceDB baseline filtered
  them out

### BrowseComp-Plus Evidence

Observed equality on judged quality up to floating-point noise:
- Kayak `mean_ndcg@10 = 0.26234761965070796`
- LanceDB `mean_ndcg@10 = 0.26234761964459435`

Latency remains much higher in the current LanceDB path:
- Kayak exact `mean_search_seconds ≈ 0.000764`
- LanceDB multivector `mean_search_seconds ≈ 0.02592`

Indexed evidence is now frozen under the same policy:
- variance artifact:
  - `.cache/kayak/lancedb_browsecomp_plus_evidence_indexed_variance.json`
- frozen indexed artifact:
  - `.cache/kayak/lancedb_browsecomp_plus_evidence_indexed_frozen_benchmark.json`
- scorecard:
  - `.cache/kayak/lancedb_browsecomp_plus_evidence_scorecard.json`
- frozen indexed `mean_ndcg@10 = 0.30560470617929153`
- frozen indexed `mean_search_seconds ≈ 0.02586`
- source `mean_ndcg@10` ranged from `0.2796118143166798` to
  `0.3271793262768601`

### Evidence Same LanceDB Storage, Different Search Engines

Artifact:
- `.cache/kayak/lancedb_browsecomp_plus_evidence_storage_compare.json`

Measured on the same LanceDB-stored evidence corpus:
- LanceDB scan `mean_ndcg@10 = 0.26234761964459435`
- Kayak exact from LanceDB `mean_ndcg@10 = 0.26234761964459435`
- LanceDB scan `mean_search_seconds ≈ 0.02654`
- Kayak exact from LanceDB `mean_search_seconds ≈ 0.00212`
- Kayak materialization from LanceDB rows to a packed Kayak index took
  `≈ 0.21454s` once per load

Interpretation:
- the same-storage result generalizes across both BrowseComp slices
- on both slices, Kayak exact from LanceDB is now materially faster than
  LanceDB scan on the base slice while preserving identical judged quality
- the ratio is `≈ 21.15x` on gold and `≈ 12.52x` on evidence
- that consistency strengthens the conclusion that the main gap here is driver
  overhead, not an idiosyncrasy in one slice

### Evidence Scale Sweep

Artifact:
- `.cache/kayak/lancedb_browsecomp_plus_evidence_scale_sweep.json`

Measured under the current Python driver:
- `90` docs:
  - Kayak exact `≈ 0.00151s`
  - LanceDB scan `≈ 0.02666s`
  - LanceDB/Kayak latency ratio `≈ 17.65`
- `180` docs:
  - Kayak exact `≈ 0.00302s`
  - LanceDB scan `≈ 0.03173s`
  - ratio `≈ 10.50`
- `360` docs:
  - Kayak exact `≈ 0.00461s`
  - LanceDB scan `≈ 0.03999s`
  - ratio `≈ 8.68`
- `720` docs:
  - Kayak exact `≈ 0.00812s`
  - LanceDB scan `≈ 0.05859s`
  - ratio `≈ 7.22`

Observed quality across those points stayed equal up to floating-point noise:
- Kayak exact `mean_ndcg@10 = 0.26234761964459435`
- LanceDB scan `mean_ndcg@10 = 0.26234761964459435`

Interpretation:
- the gold scale trend reproduces on evidence almost exactly
- under the current Python driver, Kayak exact now wins at every tested point
  on both slices
- LanceDB's advantage no longer reappears within this tested range; instead
  the size of Kayak's lead shrinks as the corpus grows
- because both slices agree, this is probably a property of the current Python
  exact harness rather than a property of one judged split

### Evidence Storage Engine Scale Sweep

Artifact:
- `.cache/kayak/storage_engine_browsecomp_plus_evidence_scale.json`

Measured:
- `90` docs / `12,778` stored vectors:
  - Kayak packed storage `6,543,578` bytes
  - LanceDB storage `7,801,876` bytes
  - LanceDB/Kayak byte ratio `≈ 1.19`
  - Kayak build `≈ 0.00368s`
  - LanceDB build `≈ 0.07080s`
  - build ratio `≈ 19.22`
- `180` docs / `25,450` stored vectors:
  - Kayak `13,034,511` bytes
  - LanceDB `14,293,826` bytes
  - ratio `≈ 1.10`
  - Kayak build `≈ 0.00733s`
  - LanceDB build `≈ 0.14136s`
  - build ratio `≈ 19.28`
- `360` docs / `51,032` stored vectors:
  - Kayak `26,138,235` bytes
  - LanceDB `27,399,297` bytes
  - ratio `≈ 1.05`
  - Kayak build `≈ 0.01513s`
  - LanceDB build `≈ 0.33118s`
  - build ratio `≈ 21.89`
- `720` docs / `102,220` stored vectors:
  - Kayak `52,357,987` bytes
  - LanceDB `53,622,736` bytes
  - ratio `≈ 1.02`
  - Kayak build `≈ 0.03077s`
  - LanceDB build `≈ 0.62606s`
  - build ratio `≈ 20.35`

Observed search on the reloaded native-storage path:
- `90` docs:
  - Kayak exact `≈ 0.00160s`
  - LanceDB scan `≈ 0.02743s`
  - ratio `≈ 17.14`
- `720` docs:
  - Kayak exact `≈ 0.01026s`
  - LanceDB scan `≈ 0.05907s`
  - ratio `≈ 5.76`

Interpretation:
- the storage result reproduces on evidence with almost the same byte ratios
  as gold
- the duplicated-distractor policy differs slightly because the protected set
  is larger on evidence, but the conclusion does not change
- on both slices, Kayak packed storage stayed smaller and materially cheaper to
  build than LanceDB throughout this tested range

### Why Python-Driven Kayak Exact Is So Much Slower Than Native Mojo

Verified path:
- [python/kayak_bridge/backend_dispatch.py](../../python/kayak_bridge/backend_dispatch.py)
  calls `_mojo_scores_for_query_and_index(...)`
- that helper now asks
  [python/kayak_bridge/prepared_index_cache.py](../../python/kayak_bridge/prepared_index_cache.py)
  for a cached native prepared packed-index object
- when the Mojo module supports it,
  [python/kayak_bridge/prepared_index_storage_artifact.py](../../python/kayak_bridge/prepared_index_storage_artifact.py)
  writes only `doc_ids.tsv`, `doc_offsets.bin`, and flat dim128
  `token_values.bin`
- [python/kayak_bridge/_mojo_exact_cpu_bindings.mojo](../../python/kayak_bridge/_mojo_exact_cpu_bindings.mojo)
  now loads that artifact into a native `PreparedPackedIndex` backed by
  `HybridFlatDim128Index`, then reuses it across later searches

Measured on the gold slice:
- at `720` gold documents / `126,146` vectors:
  - older direct in-memory preparation cost `≈ 0.77593s`
  - intermediate generic-storage `prepared_packed_index_object(...)` cost
    `≈ 0.13802s`
  - current flat-artifact write cost `≈ 0.01398s`
  - current native `prepare_packed_index_from_storage(...)` load cost
    `≈ 0.01532s`
  - current full cached `prepared_packed_index_object(...)` cost
    `≈ 0.02577s`
  - new first public `kayak.search_batch(...)` after clearing the cache cost
    `≈ 0.05556s` for the full `4`-query batch
  - steady-state public `kayak.search_batch(...)` then dropped to
    `≈ 0.03009s` per `4`-query batch
- cold-start scaling on gold was:
  - `90` docs:
    - prepare `≈ 0.00361s`
    - first search batch `≈ 0.00861s`
  - `180` docs:
    - prepare `≈ 0.00654s`
    - first search batch `≈ 0.01531s`
  - `360` docs:
    - prepare `≈ 0.01352s`
    - first search batch `≈ 0.02832s`
  - `720` docs:
    - prepare `≈ 0.02577s`
    - first search batch `≈ 0.05556s`
- cold-start scaling on evidence was nearly identical:
  - `90` docs:
    - prepare `≈ 0.00340s`
    - first search batch `≈ 0.00847s`
  - `180` docs:
    - prepare `≈ 0.00644s`
    - first search batch `≈ 0.01427s`
  - `360` docs:
    - prepare `≈ 0.01217s`
    - first search batch `≈ 0.02750s`
  - `720` docs:
    - prepare `≈ 0.02584s`
    - first search batch `≈ 0.05355s`
- repeated steady-state public `4`-query batch timings on gold were:
  - `90` docs: `≈ 0.00527s`
  - `180` docs: `≈ 0.00899s`
  - `360` docs: `≈ 0.01582s`
  - `720` docs: `≈ 0.03009s`
- native Mojo benchmark artifact remains `≈ 0.000768s` per query on the same
  gold slice

Interpretation:
- the hot steady-state gap was largely the repeated Python-to-Mojo index
  materialization, and the prepared-index path removes that cost
- the intermediate generic-storage path proved that Python list materialization
  was the main cold bottleneck, but it left a native load cost of
  `≈ 0.13802s` at `720` gold docs
- the current flat-dim128 prepared path cuts that cold prepare again by about
  `5.4x` relative to the intermediate path (`0.13802s -> 0.02577s`)
  and by about `30.1x` relative to the original nested-list path
  (`0.77593s -> 0.02577s`)
- first public `search_batch(...)` after a cache clear also drops by about
  `3.0x` relative to the intermediate path (`0.16870s -> 0.05556s`)
  and by about `14.3x` relative to the original path
  (`0.79333s -> 0.05556s`)
- steady-state Python search is now close to the native prepared-object search
  path, which means the remaining overhead is much smaller and mostly outside
  packed-index decoding
- artifact writing is no longer the bottleneck; it costs only about
  `0.011s` to `0.014s` at `720` gold docs on this host
- the remaining cold-start cost is no longer dominated by one obvious stage:
  the current split is roughly `0.014s` artifact write plus `0.015s` native
  load at `720` gold docs
- therefore the current Python exact path is now a materially stronger same-
  machine comparison harness than before, even though it still is not a
  faithful proxy for Kayak's native standalone exact engine

## Interpretation

Verified:
- the same encoded BrowseComp task can be consumed by LanceDB multivector
  search without re-encoding
- on the current small slices, LanceDB reproduces Kayak's exact judged
  retrieval quality when zero vectors are filtered explicitly
- this gives the repo a real same-machine external baseline artifact instead of
  a speculative comparison note
- the indexed LanceDB path is also runnable locally as a second operating point
- the indexed path changes judged quality enough that it should be frozen
  separately
- the indexed path also shows material rebuild variance, so the repo now uses a
  declared freeze policy instead of a single indexed run
- when LanceDB owns storage in both branches, Kayak exact from LanceDB is now
  faster than LanceDB scan on the base gold and evidence slices while keeping
  judged quality identical
- the measured reason is not speculation: the current Python exact bridge
  used to rebuild and re-decode index payloads per query and later per batch
- the existing batched public API, the native prepared-object path, and the
  new storage-backed cold-start path are all additive speedups that together
  flip the base same-storage result without changing retrieval semantics
- the synthetic scale trend is consistent across gold and evidence under that
  same Python driver
- on the same filtered vectors, Kayak packed storage is smaller than LanceDB
  across the tested `90` to `720` document range on both gold and evidence
- on the same filtered vectors, Kayak packed storage builds materially faster
  than LanceDB across that same range on this host
- the current freeze policy is:
  - indexed LanceDB summary = arithmetic mean across `5` rebuilds on the same
    encoded task
  - source min/max remain part of the variance artifact
  - scorecards consume the frozen summary, not a single indexed build

Not yet verified:
- larger-slice indexed LanceDB behavior
- filter-aware parity between Kayak and LanceDB
- whether `5` rebuilds is enough once the lane grows beyond the current small
  BrowseComp slices
- whether Mojo can ingest contiguous Python buffers directly enough to shrink
  the remaining `≈ 0.013s` to `≈ 0.015s` native cold-load cost without going
  through a temporary storage artifact
- whether LanceDB remains quality-matched once Kayak moves beyond exact
  full-scan comparison toward staged candidate-generation lanes

## Next Step

The next sound extension is now narrower:
- keep this exact same encoded-task comparison surface
- keep the indexed scorecards secondary to the variance artifacts when reading
  the results
- use the frozen gold scorecard, not the single indexed run, as the current
  Lane A indexed comparison surface
- if the same-storage Python path still needs lower first-use latency, test a
  direct contiguous-buffer ingest path into Mojo so the remaining native load
  cost can be compared against the new storage-backed prepare path
- revisit the rebuild budget only if larger-slice variance stays material
