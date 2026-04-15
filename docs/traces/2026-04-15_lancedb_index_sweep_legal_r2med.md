# Trace: LanceDB Indexed Configuration Sweep on Legal and R2MED

## Question

After exposing explicit LanceDB indexed query controls, do build-time IVF_PQ
settings matter on the harder legal and biomedical slices, or is the whole
story just query-time `refine_factor`?

## Why this follow-up was justified

The prior trace established two verified facts:

1. the default indexed LanceDB query path underperformed badly on the legal slice
2. query-time refine was the main lever for repairing that instability

That still left an open question: once refine is explicit, do index build
controls such as `num_partitions` or `num_sub_vectors` give a better
quality-latency tradeoff on the harder slices?

This trace answers that with a reusable sweep, not with ad hoc commands.

## Code changes

### Reusable indexed sweep path

Added:

- `python/kayak_bridge/lancedb_index_sweep.py`
- `python/scripts/bench_lancedb_index_sweep.py`

These own:

- explicit parsing of sweep configs such as
  `p8_refine2:index_num_partitions=8,indexed_refine_factor=2`
- writing one variance artifact and one frozen artifact per config
- writing one aggregate sweep summary with deltas versus Kayak exact and
  LanceDB scan

### Benchmark contract cleanup

Updated:

- `python/kayak_bridge/benchmark_variance.py`

Reason:

- variance summaries now carry `k`
- the sweep summary can therefore verify that every compared artifact describes
  the same ranking task instead of assuming that implicitly

### Bundle visibility

Updated:

- `python/kayak_bridge/task_comparison_bundle.py`
- `python/kayak_bridge/lancedb_lane_a_bundle.py`
- `python/scripts/bench_lancedb_lane_a.py`

Reason:

- once indexed settings are configurable, compact bundles must surface them
- otherwise "ivf_pq frozen" is no longer specific enough to audit

## Validation

Targeted tests:

```bash
env PYTHONPATH=python uv run --python 3.11 python -m unittest \
  python.tests.test_lancedb_index_controls \
  python.tests.test_benchmark_variance \
  python.tests.test_task_comparison_bundle \
  python.tests.test_lancedb_index_sweep \
  python.tests.test_lancedb_lane_a_bundle
```

Result: passed.

## Sweep grid

The same grid was used for both tasks:

- `default`
- `nprobe64:indexed_nprobes=64`
- `refine1:indexed_refine_factor=1`
- `refine2:indexed_refine_factor=2`
- `refine4:indexed_refine_factor=4`
- `p4_refine2:index_num_partitions=4,indexed_refine_factor=2`
- `p8_refine2:index_num_partitions=8,indexed_refine_factor=2`
- `sv16_refine2:index_num_sub_vectors=16,indexed_refine_factor=2`
- `sv32_refine2:index_num_sub_vectors=32,indexed_refine_factor=2`
- `p8_sv16_refine2:index_num_partitions=8,index_num_sub_vectors=16,indexed_refine_factor=2`

The choice of grid was deliberate:

- include the old default indexed path
- include a pure probe-budget test
- include refine-only baselines
- test whether partition count or PQ subvector count improves the refine-only
  baseline

## Legal results

Artifact:

- `.cache/kayak/legal_rag_bench_real_subset/index_sweep_full/legal_rag_bench_index_sweep_indexed_sweep_summary.json`

Baselines:

- Kayak exact: `mrr=0.16666666666666666`, `0.002465435788811495 s`
- LanceDB scan: `mrr=0.16666666666666666`, `0.03266078119243806 s`

Selected indexed rows:

- `default`
  - `mrr=0.10343253968253968`
  - `0.025933407619595526 s`
- `nprobe64`
  - `mrr=0.10390873015873014`
  - `0.02650987598365949 s`
- `refine1`
  - `mrr=0.1798809523809524`
  - `0.0293384597345721 s`
- `refine2`
  - `mrr=0.1847123015873016`
  - `0.029663825998432 s`
- `p4_refine2`
  - `mrr=0.18110119047619047`
  - `0.02633966697806803 s`
- `p8_refine2`
  - `mrr=0.1834722222222222`
  - `0.0269059174102343 s`
- `sv16_refine2`
  - `mrr=0.16944444444444443`
  - `0.02597146215266548 s`

Interpretation:

- `nprobes=64` alone still does not fix the legal indexed problem
- `refine_factor` is still the main quality lever
- build-time partitions matter after refine is enabled:
  - `p4_refine2` and `p8_refine2` recover most of the `refine2` quality gain
  - both do so at materially lower latency than refine-only `refine2`
- `sv16_refine2` is the fastest positive-delta legal config in this sweep, but
  it gives up much of the extra quality that the partitioned refine runs keep
- the combined `p8_sv16_refine2` setting was not useful here; it was slower and
  lower quality than the simpler partitioned refine variants

Practical legal frontier from this sweep:

- maximum quality: `refine2`
- best latency-quality balance: `p8_refine2`
- fastest config that still beats scan quality: `sv16_refine2`

## R2MED results

Artifact:

- `.cache/kayak/r2med_biology_real_subset/index_sweep_full/r2med_biology_index_sweep_indexed_sweep_summary.json`

Baselines:

- Kayak exact: `ndcg=0.8646255554317486`, `0.002566559002540695 s`
- LanceDB scan: `ndcg=0.8646255554317486`, `0.031081308993937757 s`

Selected indexed rows:

- `default`
  - `ndcg=0.8688616658092362`
  - `0.02461048921395559 s`
- `nprobe64`
  - `ndcg=0.869307408679145`
  - `0.024318125725646192 s`
- `refine1`
  - `ndcg=0.878253589622985`
  - `0.029058180541809028 s`
- `refine2`
  - `ndcg=0.8732438514688223`
  - `0.025722308306527945 s`
- `p4_refine2`
  - `ndcg=0.8708999071557898`
  - `0.025623656939327097 s`
- `p8_refine2`
  - `ndcg=0.8712962615242782`
  - `0.024162505195514918 s`
- `sv16_refine2`
  - `ndcg=0.8646255554317486`
  - `0.02269005555814753 s`

Interpretation:

- unlike legal, the default indexed path already beats scan slightly on this
  slice
- `refine1` gives the highest judged quality in the tested grid
- `p8_refine2` is the strongest quality-latency tradeoff among the build-time
  variants tested:
  - better quality than default or `nprobe64`
  - noticeably faster than refine-only `refine2`
- `p4_refine2` is weaker than `p8_refine2` here and is therefore not on the
  measured frontier
- `sv16_refine2` and `sv32_refine2` are the fastest indexed variants, but they
  collapse back to scan-level quality on this slice

Practical R2MED frontier from this sweep:

- maximum quality: `refine1`
- best latency-quality balance: `p8_refine2`
- fastest config while matching scan quality: `sv16_refine2`

## Cross-slice conclusions

Verified:

- the query-time refine story is real, but it is not the whole story
- build-time IVF partition choices matter on both harder slices once refine is
  explicit
- probe budget alone is not the useful knob on the legal slice
- PQ subvector compression mostly buys speed by giving back the extra quality
  that refine had recovered

The most stable cross-slice recommendation from this sweep is:

- if you want pure indexed quality, use refine
- if you want a better indexed quality-latency tradeoff than refine-only,
  partitioned refine settings such as `p8_refine2` are worth testing first
- if you want the fastest indexed path while merely matching scan quality,
  the smaller-subvector variants are candidates, but they are not the right
  choice when the goal is to maximize judged quality

## Caveats

- The legal slice is tiny, and LanceDB emitted empty-cluster warnings for some
  low-partition settings. Those warnings are evidence that very small slices can
  make IVF partition choices look noisier than they would on larger corpora.
- The legal sweep was run before the sweep script emitted a final parseable
  `Mean:` line for the quiet wrapper. The JSON artifacts were written correctly;
  the later script update only made the wrapper summary compatible and did not
  change the benchmark logic.

## Practical next step

For future harder-slice benchmarking, the reasonable LanceDB indexed candidates
are no longer just:

- default indexed
- refine-only indexed

They should now include at least one partitioned-refine configuration, because
that is where the better quality-latency tradeoffs appeared in this sweep.
