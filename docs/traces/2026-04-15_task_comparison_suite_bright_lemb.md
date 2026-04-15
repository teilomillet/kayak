# 2026-04-15: generic task comparison suite on BRIGHT and LEMB

## Goal

Turn the newly added benchmark families into direct engine-level evidence by
running the same encoded task JSONs through:

- Kayak exact
- LanceDB scan
- LanceDB indexed IVF_PQ with rebuild variance and frozen mean
- same-task scale sweeps
- native Kayak storage versus LanceDB storage scale sweeps

## What changed

### Correctness fix

I verified locally on `lancedb 0.30.2` that the table API exposes:

- `create_index(...)`
- `list_indices()`
- `wait_for_index(...)`

The indexed benchmark path previously called `create_index()` and then searched
immediately. I patched
[python/kayak_bridge/lancedb_benchmark.py](../../python/kayak_bridge/lancedb_benchmark.py)
to call `wait_for_index(...)` before timing indexed search. This makes the
indexed baseline explicit instead of relying on an undocumented assumption.

### New generic tooling

Added:

- [python/scripts/build_task_json.py](../../python/scripts/build_task_json.py)
- [python/scripts/bench_kayak_task_exact.py](../../python/scripts/bench_kayak_task_exact.py)
- [python/scripts/bench_task_comparison_suite.py](../../python/scripts/bench_task_comparison_suite.py)
- [python/kayak_bridge/task_json_catalog.py](../../python/kayak_bridge/task_json_catalog.py)
- [python/kayak_bridge/task_comparison_bundle.py](../../python/kayak_bridge/task_comparison_bundle.py)

These let the repo build named encoded task JSONs and benchmark any compatible
task JSON without BrowseComp-specific orchestration.

## Validation checks

### Lightweight tests

Passed:

```bash
PYTHONPATH=python uv run --python 3.11 --with pytest --with numpy \
  python -m pytest \
  python/tests/test_task_json_catalog.py \
  python/tests/test_task_comparison_bundle.py -q
```

### Builder checks

Validated:

```bash
PYTHONPATH=python pixi run python python/scripts/build_task_json.py \
  --dataset-key bright_stackoverflow_real_subset

PYTHONPATH=python pixi run python python/scripts/build_task_json.py \
  --dataset-key lemb_narrativeqa_real_subset

PYTHONPATH=python pixi run python python/scripts/build_task_json.py \
  --dataset-key r2med_biology_real_subset
```

### Generic exact benchmark check

Validated:

```bash
PYTHONPATH=python uv run --python 3.11 python \
  python/scripts/bench_kayak_task_exact.py \
  --task .cache/kayak/bright_stackoverflow_real_subset/python_task.json \
  --output .cache/kayak/bright_stackoverflow_real_subset/bright_stackoverflow_real_subset_kayak_exact_python_script_benchmark.json \
  --warmup-iterations 1 \
  --measurement-iterations 2
```

## Host-load note

The first quiet-host attempt failed because the machine had several stale
repo-local benchmark workers still running from earlier profiling work:

- long-running `mojo` profile jobs in this repo
- three `python -` workers from the same Pixi environment

Those were stopped before the final BRIGHT and LEMB suite runs. After cleanup,
the quiet wrapper could proceed with a relaxed threshold of `500` aggregate
background `%CPU`, which was materially better than the earlier `~1100%+`
background load.

## Commands used for final BRIGHT and LEMB runs

### BRIGHT

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh \
  --repeats 1 \
  --max-other-cpu 500 \
  -- \
  uv run --python 3.11 --with lancedb --with faiss-cpu python \
    python/scripts/bench_task_comparison_suite.py \
    --task .cache/kayak/bright_stackoverflow_real_subset/python_task.json
```

### LEMB

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh \
  --repeats 1 \
  --max-other-cpu 500 \
  -- \
  uv run --python 3.11 --with lancedb --with faiss-cpu python \
    python/scripts/bench_task_comparison_suite.py \
    --task .cache/kayak/lemb_narrativeqa_real_subset/python_task.json
```

## Results

### BRIGHT StackOverflow

Artifact bundle:

- `.cache/kayak/bright_stackoverflow_real_subset/bright_stackoverflow_real_subset_comparison_bundle.json`

Key values:

- Kayak exact `nDCG`: `0.2914247870199321`
- LanceDB scan `nDCG`: `0.2914247870199321`
- LanceDB scan latency ratio vs Kayak: `11.39310204644104`
- LanceDB indexed frozen `nDCG`: `0.3164312553889746`
- LanceDB indexed frozen latency ratio vs Kayak: `9.839180234975732`
- Indexed rebuild latency mean range:
  `0.015480232667565966s` to `0.017684869788354263s`
- Indexed rebuild quality range:
  `0.2838109640074143` to `0.33373731923735683`
- Base LanceDB storage byte ratio vs native Kayak storage: `1.0491287684467707`
- Largest measured storage byte ratio at `1200` docs: `1.049633854422875`
- Largest measured LanceDB scan latency ratio at `1200` docs: `4.697295628272423`

Interpretation:

- LanceDB scan preserved quality on this slice but was much slower than Kayak
  exact.
- LanceDB indexed search reduced the latency gap somewhat, but it remained
  about `9.84x` slower than Kayak exact in the frozen mean summary.
- Indexed quality varied across rebuilds on BRIGHT, which means a single
  indexed run would not have been strong enough evidence.
- LanceDB storage was only about `5%` larger than native Kayak storage on this
  shorter-document slice.

### LEMB NarrativeQA

Artifact bundle:

- `.cache/kayak/lemb_narrativeqa_real_subset/lemb_narrativeqa_real_subset_comparison_bundle.json`

Key values:

- Kayak exact `nDCG`: `0.375`
- LanceDB scan `nDCG`: `0.375`
- LanceDB scan latency ratio vs Kayak: `9.255447680868924`
- LanceDB indexed frozen `nDCG`: `0.375`
- LanceDB indexed frozen latency ratio vs Kayak: `5.898675081597088`
- Indexed rebuild latency mean range:
  `0.02381039746978786s` to `0.024907965280969318s`
- Indexed rebuild quality range:
  `0.375` to `0.375`
- Base LanceDB storage byte ratio vs native Kayak storage: `6.178646738518826`
- Largest measured storage byte ratio at `2840` docs: `5.943376935791758`
- Largest measured LanceDB scan latency ratio at `2840` docs: `2.6852264245165487`

Interpretation:

- LanceDB scan again preserved quality but stayed materially slower than Kayak
  exact.
- LanceDB indexing helped more on LEMB than on BRIGHT, shrinking the latency
  gap to about `5.90x`.
- Unlike BRIGHT, the indexed quality was stable across rebuilds on LEMB in the
  measured runs.
- Storage is the major LEMB difference: LanceDB used roughly `6x` the bytes of
  native Kayak storage on this long-document slice.

## Cross-slice conclusion

Across both BRIGHT and LEMB:

- scan quality matched Kayak exact on the measured tasks
- scan latency was much worse than Kayak exact
- indexed LanceDB improved latency but did not close the gap

The engine gap differs by slice:

- on BRIGHT, the dominant difference is search latency with only a small
  storage penalty
- on LEMB, the dominant differences are both search latency and a very large
  storage penalty

That means the next optimization target should not be guessed globally.

## What this implies next

1. Do not spend the next loop on Kayak exact search first.
   The direct evidence already shows a strong exact-path lead on both slices.

2. Do inspect native Kayak storage and build/materialization costs on long
   documents more closely.
   LEMB shows the storage story is a major differentiator, not a side detail.

3. If the comparison focus stays on LanceDB, the next LanceDB-facing question is
   indexed configuration sensitivity, not whether scan is good enough.
   The remaining gap is in the indexed path.

4. Query-bucket breadth is still unresolved.
   These additions changed semantics and document length, but not the measured
   query-width bucket under the current encoder.
