# Storage Encoding Trace: `binary_f16_le` vs `binary_le`

## Objective

Verify whether native Kayak packed storage should switch from
`binary_le` to `binary_f16_le` for task-json benchmarks and storage
comparisons.

This required measurement, not inference, because the tradeoff is not
one-dimensional:

- `binary_f16_le` should reduce persisted vector bytes
- reduced bytes do not automatically imply faster load or search
- a default change is only justified if the full build/load/search
  tradeoff stays acceptable on real tasks

## Benchmark Surfaces Added

Additive surfaces only; no existing caller was refactored away.

- `benchmarks/task_json_storage_encoding_compare.mojo`
  Runs both encodings on the same task JSON and writes summary rows.
- `python/kayak_bridge/task_storage_encoding.py`
  Owns the benchmark launcher and the f16-vs-binary comparison bundle.
- `python/scripts/bench_task_storage_encoding.py`
  Thin CLI for fixed-slice storage-encoding benchmarks.
- `python/scripts/bench_storage_engine_scale_compare.py`
  Added `--kayak-vector-payload-encoding` so scaled storage-vs-LanceDB
  comparisons can use either encoding explicitly.
- `python/scripts/bench_task_comparison_suite.py`
  Added the same explicit encoding knob for storage sweeps.

## Runtime Note

The new benchmark path initially failed under `uv run` because Mojo was
bridging against the repo `.venv` Python instead of the Pixi Python
runtime. That was verified from the failing traceback and then fixed by
launching the existing `scripts/run_mojo_with_pixi_python.sh` wrapper
through `pixi run`, so both Python and Mojo come from the same runtime.

This matters because mixed runtimes would make storage measurements
unsound.

## Commands

Fixed-slice storage encoding:

```bash
PYTHONPATH=python uv run --python 3.11 python python/scripts/bench_task_storage_encoding.py --task .cache/kayak/bright_stackoverflow_real_subset/python_task.json
PYTHONPATH=python uv run --python 3.11 python python/scripts/bench_task_storage_encoding.py --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json
PYTHONPATH=python uv run --python 3.11 python python/scripts/bench_task_storage_encoding.py --task .cache/kayak/r2med_biology_real_subset/python_task.json
```

Scaled storage-vs-LanceDB comparison with explicit f16:

```bash
PYTHONPATH=python uv run --python 3.11 --with lancedb --with faiss-cpu python python/scripts/bench_storage_engine_scale_compare.py --task .cache/kayak/bright_stackoverflow_real_subset/python_task.json --output .cache/kayak/bright_stackoverflow_real_subset/bright_stackoverflow_real_subset_storage_scale_f16.json --db-root .cache/kayak/bright_stackoverflow_real_subset/bright_stackoverflow_real_subset_lancedb_storage_scale_f16 --table-prefix bright_stackoverflow_real_subset_f16 --kayak-vector-payload-encoding binary_f16_le

PYTHONPATH=python uv run --python 3.11 --with lancedb --with faiss-cpu python python/scripts/bench_storage_engine_scale_compare.py --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json --output .cache/kayak/lemb_narrativeqa_real_subset/lemb_narrativeqa_real_subset_storage_scale_f16.json --db-root .cache/kayak/lemb_narrativeqa_real_subset/lemb_narrativeqa_real_subset_lancedb_storage_scale_f16 --table-prefix lemb_narrativeqa_real_subset_f16 --kayak-vector-payload-encoding binary_f16_le

PYTHONPATH=python uv run --python 3.11 --with lancedb --with faiss-cpu python python/scripts/bench_storage_engine_scale_compare.py --task .cache/kayak/r2med_biology_real_subset/python_task.json --output .cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_storage_scale_f16.json --db-root .cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_lancedb_storage_scale_f16 --table-prefix r2med_biology_real_subset_f16 --kayak-vector-payload-encoding binary_f16_le
```

## Fixed-Slice Results

Artifacts:

- BRIGHT: `.cache/kayak/bright_stackoverflow_real_subset/bright_stackoverflow_real_subset_storage_encoding_bundle.json`
- LEMB: `.cache/kayak/lemb_narrativeqa_real_subset/lemb_narrativeqa_real_subset_storage_encoding_bundle.json`
- R2MED: `.cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_storage_encoding_bundle.json`
- LIMIT smoke: `.cache/kayak/limit_small_real_subset/limit_small_real_subset_storage_encoding_bundle.json`

Ratios are `binary_f16_le / binary_le`, so values below `1.0` are
smaller or faster for f16 and values above `1.0` are slower for f16.

| Dataset | Artifact Bytes Ratio | Build Seconds Ratio | Load Seconds Ratio | Search Seconds Ratio | Primary Delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| LIMIT | `0.500150` | `0.711699` | `1.133786` | `1.037093` | `0.0` |
| BRIGHT | `0.500270` | `0.893387` | `1.151132` | `1.005947` | `0.0` |
| LEMB | `0.500077` | `0.876615` | `1.130519` | `1.020210` | `0.0` |
| R2MED | `0.500432` | `0.855248` | `1.122703` | `0.946583` | `0.0` |

Observed pattern:

- persisted bytes are cut almost exactly in half on every measured task
- build time improves on every measured task
- load time regresses consistently by about `12%` to `15%`
- search time is near-neutral, slightly worse on LIMIT/BRIGHT/LEMB and
  slightly better on R2MED
- no metric loss was observed on these slices

## Storage-Vs-LanceDB at Scale

Artifacts:

- BRIGHT binary: `.cache/kayak/bright_stackoverflow_real_subset/bright_stackoverflow_real_subset_storage_scale.json`
- BRIGHT f16: `.cache/kayak/bright_stackoverflow_real_subset/bright_stackoverflow_real_subset_storage_scale_f16.json`
- LEMB binary: `.cache/kayak/lemb_narrativeqa_real_subset/lemb_narrativeqa_real_subset_storage_scale.json`
- LEMB f16: `.cache/kayak/lemb_narrativeqa_real_subset/lemb_narrativeqa_real_subset_storage_scale_f16.json`
- R2MED binary: `.cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_storage_scale.json`
- R2MED f16: `.cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_storage_scale_f16.json`

Largest measured point per dataset:

| Dataset | Docs | LanceDB/Kayak Bytes Ratio (`binary_le`) | LanceDB/Kayak Bytes Ratio (`binary_f16_le`) |
| --- | ---: | ---: | ---: |
| BRIGHT | `1200` | `1.049634` | `2.096608` |
| LEMB | `2840` | `5.943377` | `11.880608` |
| R2MED | `1424` | `1.011755` | `2.020832` |

Largest measured point, Kayak load/search impact:

| Dataset | Binary Load s | F16 Load s | Binary Search s | F16 Search s |
| --- | ---: | ---: | ---: | ---: |
| BRIGHT | `0.106972` | `0.122607` | `0.006324` | `0.006280` |
| LEMB | `0.344357` | `0.404493` | `0.018486` | `0.020006` |
| R2MED | `0.153159` | `0.173088` | `0.009554` | `0.009775` |

Observed pattern:

- the f16 storage advantage scales almost exactly as the fixed-slice
  benchmark predicts
- Kayak’s storage lead over LanceDB roughly doubles when switching from
  `binary_le` to `binary_f16_le`
- the load penalty remains visible at scale
- search remains near-neutral, not decisively better

## Decision

The evidence supports two conclusions:

1. `binary_f16_le` is a real and valuable storage option.
   The byte savings are stable, large, and quality-preserving on the
   measured tasks.

2. The evidence does **not** justify silently changing the default
   packed-index encoding from `binary_le` to `binary_f16_le`.
   The reason is the repeated load regression. The storage win is clear,
   but the default should not move unless we decide that startup/load
   latency is less important than artifact size for the common path.

That is why the implemented code change is narrower:

- expose the encoding as an explicit benchmark knob
- keep the default behavior unchanged
- let future comparisons choose `binary_f16_le` when storage is the
  priority being optimized

## Next Justified Step

If we want to revisit the default encoding later, the next evidence to
collect should be load-heavy end-to-end workloads rather than more
single-query search timings.

The open question is no longer “does f16 save storage?”.
That is verified.

The open question is “is the load penalty acceptable for the default
artifact format on realistic serve/startup workflows?”.
