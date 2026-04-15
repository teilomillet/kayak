# 2026-04-15 LanceDB vs Kayak BrowseComp Gold Matrix

## Goal

Verify, on the current tree and on the same machine:

- how Kayak exact compares with LanceDB scan on the same encoded BrowseComp Gold
  task
- whether the gap persists when both branches use LanceDB as the storage base
- whether the storage and latency gaps persist as the corpus is scaled

Reason:
- the user question is not only "is Kayak faster on its own storage?"
- it is also "does the advantage survive when the data lives in LanceDB?"
- and "does that relationship still hold at larger corpus sizes?"

## Command

I used the repo quiet-wrapper because `AGENTS.md` requires quiet-gated
benchmarking by default:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 20 --force -- \
  uv run --python 3.11 --with numpy --with lancedb --with pyarrow --with faiss-cpu \
  python python/scripts/bench_task_comparison_suite.py \
    --task .cache/kayak/browsecomp_plus_real_subset/python_task_gold.json \
    --output-root .cache/kayak/browsecomp_plus_real_subset \
    --artifact-prefix browsecomp_gold_matrix_20260415 \
    --warmup-iterations 1 \
    --measurement-iterations 3 \
    --rebuild-count 5 \
    --target-document-count 90 \
    --target-document-count 180 \
    --target-document-count 360 \
    --target-document-count 720 \
    --target-document-count 1440
```

Quiet-wrapper metadata:
- `.cache/kayak/bench_quiet/20260415T153742Z/meta.txt`

## Verification

Code-level validation:

```bash
PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_task_comparison_bundle.py \
  python/tests/test_storage_scale_comparison.py \
  python/tests/test_comparison_scorecard.py \
  python/tests/test_kayak_task_benchmark.py -q
```

Observed:
- `Ran 6 tests ... OK`
- the full suite completed and wrote all expected artifacts

## Host-Noise Caveat

This was **not** a quiet-host run.

The wrapper timed out after `20s` waiting for the default `max_other_cpu=40`
threshold and force-ran the benchmark. Representative quiet-check snapshots:

- `.cache/kayak/bench_quiet/20260415T153742Z/quiet_check_run_1_sample_0.txt`
- `.cache/kayak/bench_quiet/20260415T153742Z/quiet_check_run_1_sample_10.txt`

Examples from those snapshots:

- `Zed` was near `98-99%` CPU
- several `modular-crashpad-handler` processes were each near `30-31%` CPU

Interpretation:
- the relative ratios are still useful because the suite ran end to end in one
  bounded pass on the same host
- the absolute latency numbers should be treated as host-contended evidence,
  not quiet-host production medians

## Results

### Base task, same encoded corpus

Source artifacts:

- `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_comparison_scorecard.json`
- `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_comparison_bundle.json`

Observed on the base `90`-document gold slice:

| system | primary value (`ndcg@10`) | mean search seconds | ratio vs Kayak |
| --- | ---: | ---: | ---: |
| Kayak exact | `0.28512677790387286` | `0.0009517015` | `1.00x` |
| LanceDB scan | `0.28512677790387286` | `0.0256809373` | `26.98x` |
| LanceDB `IVF_PQ` frozen | `0.24669230148616947` | `0.0239973646` | `25.22x` |

Interpretation:
- Kayak exact and LanceDB scan match on judged quality for this slice
- the frozen indexed LanceDB point is only slightly faster than LanceDB scan on
  this host-contended run
- that indexed point also loses quality, so it should remain a separate
  operating point rather than being described as a neutral acceleration

Indexed rebuild variance remains material:

- `primary_value_min = 0.22134512495984793`
- `primary_value_max = 0.2789948467118853`
- `mean_search_seconds_min = 0.021881475831226755`
- `mean_search_seconds_max = 0.025191197821792837`

Source:
- `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_lancedb_ivf_pq_variance.json`

### Same LanceDB storage, different search engines

Source artifact:
- `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_storage_compare.json`

Observed:

- LanceDB scan search time: `0.0255631318s`
- Kayak load-from-LanceDB time: `0.0045609580s`
- Kayak exact search time on those LanceDB-owned rows: `0.0010990729s`
- Kayak load + search total: `0.0056600309s`

Derived comparisons:

- search-only, LanceDB scan is `23.26x` slower than Kayak searching the same
  LanceDB materialized rows
- even if the Kayak branch pays the full measured load step each time, LanceDB
  scan is still `4.52x` slower than Kayak load + search on the same storage

Interpretation:
- the current gap is **not** explained only by Kayak owning a different storage
  format
- on this task, Kayak's search path still wins materially even when LanceDB is
  the storage base

### Same-task scale sweep

Source artifact:
- `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_scale_sweep.json`

Observed:

| docs | Kayak exact (s) | LanceDB scan (s) | LanceDB/Kayak ratio |
| ---: | ---: | ---: | ---: |
| `90` | `0.0012727672` | `0.0297038367` | `23.34x` |
| `180` | `0.0022064306` | `0.0337666353` | `15.30x` |
| `360` | `0.0038387431` | `0.0444166356` | `11.57x` |
| `720` | `0.0099053715` | `0.0641764898` | `6.48x` |
| `1440` | `0.0173540139` | `0.0827068369` | `4.77x` |

Derived trend:

- Kayak exact grows by `13.63x` from `90` to `1440` docs
- LanceDB scan grows by `2.78x` over the same sweep
- the relative gap narrows by about `4.90x`, but LanceDB scan is still slower
  at every measured point

Interpretation:
- on this synthetic duplication sweep, LanceDB scan has the flatter latency
  slope
- that does **not** erase the current disadvantage on the tested envelope
- the right statement is: the gap narrows with scale here, but it does not
  invert by `1440` docs

### Storage-engine scale sweep

Source artifact:
- `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_storage_scale.json`

Observed:

| docs | LanceDB/Kayak bytes | LanceDB/Kayak build | LanceDB/Kayak search |
| ---: | ---: | ---: | ---: |
| `90` | `1.1923x` | `25.30x` | `31.37x` |
| `180` | `1.0960x` | `18.95x` | `23.10x` |
| `360` | `1.0480x` | `18.13x` | `16.37x` |
| `720` | `1.0241x` | `22.20x` | `12.18x` |
| `1440` | `1.0121x` | `19.23x` | `7.94x` |

Interpretation:
- the storage-byte gap shrinks strongly with scale and is nearly flat by
  `1440` docs
- the build and search gaps remain large in favor of Kayak across the entire
  tested envelope
- the honest statement today is not "Kayak storage is always much smaller"
- it is "Kayak storage remains smaller here, but the stronger sustained win is
  build/search cost rather than byte size"

## Bundle Schema Update

The compact bundle was extended so downstream consumers do not need to reopen
multiple JSON files just to answer the core comparison questions.

Added fields:

- `kayak_exact_mean_search_seconds`
- `lancedb_scan_mean_search_seconds`
- `lancedb_indexed_frozen_mean_search_seconds`
- `storage_compare_kayak_load_from_lancedb_seconds`
- `storage_compare_lancedb_scan_mean_search_seconds`
- `storage_compare_kayak_exact_from_lancedb_mean_search_seconds`
- `base_storage_build_seconds_ratio_vs_kayak`
- `base_storage_search_seconds_ratio_vs_kayak`
- `largest_storage_build_seconds_ratio_vs_kayak`
- `largest_storage_search_seconds_ratio_vs_kayak`

Reason:
- these are the values the user repeatedly asked to compare directly
- exposing them in one bundle keeps the benchmark auditable without forcing
  every consumer to stitch together the scorecard, storage-compare, and
  storage-scale artifacts by hand

## Artifacts

- bundle:
  - `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_comparison_bundle.json`
- scorecard:
  - `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_comparison_scorecard.json`
- scale sweep:
  - `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_scale_sweep.json`
- same-storage comparison:
  - `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_storage_compare.json`
- storage-scale sweep:
  - `.cache/kayak/browsecomp_plus_real_subset/browsecomp_gold_matrix_20260415_storage_scale.json`
- quiet-wrapper logs:
  - `.cache/kayak/bench_quiet/20260415T153742Z`

## Bottom Line

Verified on the current tree:

- Kayak exact is substantially faster than LanceDB scan on the same BrowseComp
  Gold task while matching judged quality
- that advantage still holds when LanceDB is the storage base for both
  branches
- the latency gap narrows as the synthetic corpus grows, but it does not invert
  through `1440` documents on this run
- Kayak storage stays smaller here, but the more durable current advantage is
  build/search cost, not a dramatic byte-size lead
