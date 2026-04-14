# 2026-04-14 LanceDB Threshold Follow-Up

## Goal

Answer the next question left open by the first LanceDB comparison trace:

- does Kayak stay faster as document count keeps growing?
- does Kayak stay faster when document count is fixed but document vector
  count grows?
- does the answer change when LanceDB is only the storage layer and Kayak does
  the search?

This follow-up also tries to remove one obvious benchmark artifact:
- rebuilding a Kayak index from LanceDB rows via Python `to_pylist()`

## Why This Follow-Up Was Needed

The earlier trace already showed:
- Kayak exact was faster than LanceDB scan on the current small BrowseComp
  slices
- Kayak packed storage was slightly smaller and materially faster to build
- the Python exact cold path had been reduced materially

But two uncertainties remained:
- whether the latency gap would reverse at larger document counts
- whether "more vectors" might be the axis where LanceDB catches up or wins

Those are different questions and require different benchmark surfaces.

## Implementation

Added:
- [python/kayak_bridge/lancedb_arrow_bridge.py](../../python/kayak_bridge/lancedb_arrow_bridge.py)
  for columnar Arrow-to-Kayak packed-index conversion
- [python/kayak_bridge/task_vector_scale.py](../../python/kayak_bridge/task_vector_scale.py)
  for fixed-document-count vector-density sweeps
- [python/kayak_bridge/vector_density_comparison.py](../../python/kayak_bridge/vector_density_comparison.py)
  for Kayak-versus-LanceDB vector-density comparisons
- [python/scripts/bench_vector_density_compare.py](../../python/scripts/bench_vector_density_compare.py)
  for a CLI entrypoint over encoded task JSON
- [python/scripts/bench_lancedb_scale_sweep_filtered.py](../../python/scripts/bench_lancedb_scale_sweep_filtered.py)
  for the zero-filtered document-count sweep
- [python/tests/test_task_vector_scale.py](../../python/tests/test_task_vector_scale.py)
  for the deterministic vector-scaling helper

Changed:
- [python/kayak_bridge/lancedb_storage_comparison.py](../../python/kayak_bridge/lancedb_storage_comparison.py)
  now rebuilds a Kayak packed index from LanceDB through Arrow buffers instead
  of `table.to_arrow().to_pylist()`

Reason:
- the benchmark only needs `doc_id`, offsets, and contiguous vector values
- rebuilding Python row dicts and nested float lists was unrelated overhead
- the Arrow schema already exposes `list<fixed_size_list<float[128]>>`, which
  maps cleanly to Kayak's packed index layout

## Verification

Tested:

```bash
PYTHONPATH=python pixi run python -m unittest python/tests/test_task_vector_scale.py python/tests/test_prepared_index_cache.py python/tests/test_task_scale.py python/tests/test_kayak_task_benchmark.py python/tests/test_batch_api.py
PYTHONPATH=python pixi run python -m unittest python/tests/test_storage_scale_comparison.py
PYTHONPATH=python uv run --python 3.11 --with lancedb python python/scripts/bench_vector_density_compare.py --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json --output .cache/kayak/vector_density_browsecomp_plus_gold.json --db-root .cache/kayak/vector_density_browsecomp_plus_gold --table-prefix vector_density_gold --document-vector-multiplier 1 --document-vector-multiplier 2 --document-vector-multiplier 4 --document-vector-multiplier 8 --document-vector-multiplier 16 --warmup-iterations 1 --measurement-iterations 3
PYTHONPATH=python uv run --python 3.11 --with lancedb python python/scripts/bench_vector_density_compare.py --task .cache/kayak/browsecomp_plus_real_subset/python_task_evidence.json --output .cache/kayak/vector_density_browsecomp_plus_evidence.json --db-root .cache/kayak/vector_density_browsecomp_plus_evidence --table-prefix vector_density_evidence --document-vector-multiplier 1 --document-vector-multiplier 2 --document-vector-multiplier 4 --document-vector-multiplier 8 --document-vector-multiplier 16 --warmup-iterations 1 --measurement-iterations 3
PYTHONPATH=python uv run --python 3.11 --with lancedb python python/scripts/bench_lancedb_scale_sweep.py --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json --output .cache/kayak/lancedb_browsecomp_plus_gold_scale_sweep_far.json --db-root .cache/kayak/lancedb_browsecomp_plus_gold_scale_far --table-prefix browsecomp_plus_gold_scale_far --target-document-count 90 --target-document-count 180 --target-document-count 360 --target-document-count 720 --target-document-count 1440 --target-document-count 2880 --target-document-count 5760 --target-document-count 11520 --warmup-iterations 1 --measurement-iterations 3
PYTHONPATH=python uv run --python 3.11 --with lancedb python python/scripts/bench_lancedb_scale_sweep_filtered.py --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json --output .cache/kayak/lancedb_browsecomp_plus_gold_scale_sweep_filtered_far.json --db-root .cache/kayak/lancedb_browsecomp_plus_gold_scale_filtered_far --table-prefix browsecomp_plus_gold_scale_filtered_far --target-document-count 90 --target-document-count 180 --target-document-count 360 --target-document-count 720 --target-document-count 1440 --target-document-count 2880 --target-document-count 5760 --target-document-count 11520 --warmup-iterations 1 --measurement-iterations 3
PYTHONPATH=python uv run --python 3.11 --with lancedb python python/scripts/bench_lancedb_scale_sweep_filtered.py --task .cache/kayak/browsecomp_plus_real_subset/python_task_evidence.json --output .cache/kayak/lancedb_browsecomp_plus_evidence_scale_sweep_filtered_far.json --db-root .cache/kayak/lancedb_browsecomp_plus_evidence_scale_filtered_far --table-prefix browsecomp_plus_evidence_scale_filtered_far --target-document-count 90 --target-document-count 180 --target-document-count 360 --target-document-count 720 --target-document-count 1440 --target-document-count 2880 --target-document-count 5760 --target-document-count 11520 --warmup-iterations 1 --measurement-iterations 3
```

Observed:
- the new pure-Python tests passed
- the existing benchmark-lane Python tests still passed after the Arrow bridge
- the gold and evidence vector-density artifacts were written successfully
- the gold and evidence far search-sweep artifacts were written successfully
- the gold and evidence far filtered search-sweep artifacts were written
  successfully

Non-blocking environment note:
- the `uv --with lancedb` runs emitted an embedded-Python `encodings` warning
  before the benchmark body on this host
- the runs still completed successfully and wrote valid JSON artifacts
- therefore those warnings are treated as launcher noise, not benchmark
  failures

## Key Result 1: Arrow Removed The Same-Storage Load Bottleneck

Before this follow-up, rebuilding a Kayak index from a LanceDB table cost:
- gold base slice: `≈ 0.20881s`
- evidence base slice: `≈ 0.21454s`

After the Arrow bridge:
- gold base slice load dropped to `≈ 0.00264s`
- gold `720` docs load dropped to `≈ 0.01378s`
- gold `1440` docs load dropped to `≈ 0.02455s`
- gold `2880` docs load dropped to `≈ 0.04649s`

Interpretation:
- that load path is no longer dominated by Python row materialization
- the storage-controlled comparison is now measuring search much more directly

## Key Result 2: No Measured Crossover On Document Count Yet

Artifacts:
- `.cache/kayak/lancedb_browsecomp_plus_gold_scale_sweep_far.json`
- `.cache/kayak/lancedb_browsecomp_plus_evidence_scale_sweep_far.json`
- `.cache/kayak/lancedb_storage_compare_gold_threshold.json`

### Python-Driven Gold Scale Sweep

Measured:
- `5760` docs:
  - Kayak exact `≈ 0.05871s`
  - LanceDB scan `≈ 0.11147s`
  - ratio `≈ 1.90`
- `11520` docs:
  - Kayak exact `≈ 0.12720s`
  - LanceDB scan `≈ 0.15223s`
  - ratio `≈ 1.20`

### Python-Driven Evidence Scale Sweep

Measured:
- `5760` docs:
  - Kayak exact `≈ 0.04941s`
  - LanceDB scan `≈ 0.08744s`
  - ratio `≈ 1.77`
- `11520` docs:
  - Kayak exact `≈ 0.11372s`
  - LanceDB scan `≈ 0.14385s`
  - ratio `≈ 1.27`

Interpretation:
- the margin keeps shrinking as document count grows
- but Kayak still wins on both slices at `11520` documents
- therefore no measured document-count threshold was reached on this host

### Same LanceDB Storage, Gold Only

Measured after the Arrow bridge:
- `5760` docs / `820,942` stored vectors:
  - LanceDB scan `≈ 0.10610s`
  - Kayak exact from LanceDB `≈ 0.06256s`
  - ratio `≈ 1.70`
  - LanceDB-to-Kayak load `≈ 0.08498s`
- `11520` docs / `1,641,934` stored vectors:
  - LanceDB scan `≈ 0.16910s`
  - Kayak exact from LanceDB `≈ 0.08948s`
  - ratio `≈ 1.89`
  - LanceDB-to-Kayak load `≈ 0.17572s`

Interpretation:
- when both branches read the same LanceDB-stored vectors, Kayak still wins
  more comfortably than on the broader Python-vs-LanceDB lane
- this indicates that part of the narrowing on the general lane comes from the
  benchmark surface itself, not only from the raw search kernel

### Zero-Filtered Far Search Sweep

Artifacts:
- `.cache/kayak/lancedb_browsecomp_plus_gold_scale_sweep_filtered_far.json`
- `.cache/kayak/lancedb_browsecomp_plus_evidence_scale_sweep_filtered_far.json`

Measured on gold after filtering zero vectors before both branches:
- `5760` docs:
  - Kayak exact `≈ 0.04980s`
  - LanceDB scan `≈ 0.13266s`
  - ratio `≈ 2.66`
- `11520` docs:
  - Kayak exact `≈ 0.10850s`
  - LanceDB scan `≈ 0.16203s`
  - ratio `≈ 1.49`

Measured on evidence after the same filtering:
- `5760` docs:
  - Kayak exact `≈ 0.05209s`
  - LanceDB scan `≈ 0.13333s`
  - ratio `≈ 2.56`
- `11520` docs:
  - Kayak exact `≈ 0.14799s`
  - LanceDB scan `≈ 0.24758s`
  - ratio `≈ 1.67`

Interpretation:
- the earlier raw-lane near-crossover was partially an artifact of comparison
  shape
- LanceDB must drop zero vectors for cosine search, while raw Kayak exact was
  still scoring them
- once both branches consume the same zero-filtered task, Kayak's lead at the
  far points is materially larger than in the raw lane
- therefore the zero-vector treatment is a first-class benchmark contract
  decision, not a minor implementation detail

## Key Result 3: More Vectors Per Document Favors Kayak More, Not Less

Artifacts:
- `.cache/kayak/vector_density_browsecomp_plus_gold.json`
- `.cache/kayak/vector_density_browsecomp_plus_evidence.json`
- `.cache/kayak/lancedb_storage_compare_gold_vector_density.json`

Benchmark policy:
- document count stays fixed
- each document's vectors are repeated in place by a multiplier
- queries stay unchanged

Reason this is sound:
- MaxSim over duplicated identical document vectors is unchanged
- judged quality should therefore stay invariant, while search cost rises with
  total vector count

### Gold Vector Density Sweep

Measured:
- `x1` document vectors / `12,778` stored vectors:
  - Kayak exact `≈ 0.00115s`
  - LanceDB scan `≈ 0.02574s`
  - ratio `≈ 22.43`
- `x8` / `102,224` stored vectors:
  - Kayak `≈ 0.00664s`
  - LanceDB `≈ 0.22120s`
  - ratio `≈ 33.31`
- `x16` / `204,448` stored vectors:
  - Kayak `≈ 0.01395s`
  - LanceDB `≈ 0.39932s`
  - ratio `≈ 28.62`

### Evidence Vector Density Sweep

Measured:
- `x1` document vectors:
  - ratio `≈ 20.26`
- `x8`:
  - ratio `≈ 32.41`
- `x16`:
  - ratio `≈ 28.82`

Interpretation:
- increasing vector density does **not** move the benchmark toward a LanceDB
  win
- it moves the benchmark strongly toward Kayak
- therefore "more vectors per document" is not the threshold axis to worry
  about in this current lane

### Same LanceDB Storage, Gold Vector Density Sweep

Measured:
- `x1` vector density:
  - LanceDB scan `≈ 0.02558s`
  - Kayak exact from LanceDB `≈ 0.00109s`
  - ratio `≈ 23.49`
- `x8`:
  - LanceDB scan `≈ 0.19409s`
  - Kayak exact from LanceDB `≈ 0.00507s`
  - ratio `≈ 38.30`
- `x16`:
  - LanceDB scan `≈ 0.39051s`
  - Kayak exact from LanceDB `≈ 0.00991s`
  - ratio `≈ 39.42`

Interpretation:
- once the corpus is already LanceDB-backed, more vectors per document favors
  Kayak even more strongly
- the Arrow load path remains small enough that it does not erase that search
  advantage

## Threshold Discussion

Verified:
- no measured crossover was reached up to `11520` documents
- no measured crossover was reached up to `16x` document-vector density
- same-storage comparisons remain in Kayak's favor through the farthest tested
  points

Tentative, not verified:
- a naive log-log fit on the gold and evidence far search sweeps suggests the
  general Python-vs-LanceDB lane could cross somewhere beyond `11520` docs,
  roughly around the next octave (`≈ 23k` docs)
- that is an extrapolation from the current curve, not a measured result
- the filtered lane weakens that crossover story materially, because it leaves
  Kayak ahead by `≈ 1.49x` on gold and `≈ 1.67x` on evidence at `11520` docs
- the same-storage lane does **not** currently support the same crossover
  story, because it still leaves Kayak ahead by `≈ 1.89x` at `11520` docs

## Main Special Considerations

1. Zero-vector handling matters.
   Kayak can score the original cached task directly, while LanceDB filters
   zero vectors for cosine search. The far filtered sweep shows this is large
   enough to change the apparent threshold story, so filtered tasks should be
   explicit for serious comparisons.

2. Document count and document vector count are different axes.
   Document-count growth narrows the Kayak lead.
   Vector-density growth widens the Kayak lead.

3. Storage substrate and search engine must be separated.
   "LanceDB storage + Kayak search" and "LanceDB storage + LanceDB search" are
   not the same benchmark.

4. Host disk can become the limiting resource first.
   One `11520`-doc same-storage attempt failed with `No space left on device`
   before temporary benchmark databases were cleaned up.

## Current Conclusion

The strongest evidence today is:
- Kayak remains faster than LanceDB scan through `11520` documents on both
  BrowseComp slices in the current Python-driven lane
- Kayak remains faster through `11520` documents on the stricter zero-filtered
  lane, and the margin there is larger than on the raw lane
- Kayak remains faster than LanceDB scan when both branches read the same
  LanceDB-stored vectors
- more vectors per document make Kayak look better, not worse

So the most likely threshold axis is still document count, not vector density.
But that threshold has not yet been observed directly on this machine, and the
fairer filtered lane suggests it may be farther out than the raw lane implied.
