# Tachiom Candidate Pruning Sweep

Date: `2026-05-02`

## Claim Under Test

Tighter query-time candidate pruning can reduce residual-PQ rerank work for the
native streaming TAC/HNSW/PQ path.

Reason:
- the internal profile showed rerank scoring is the largest isolated stage on
  `docs10000_q128_c32768`
- candidate pruning changes the rerank candidate window without rebuilding the
  TAC/PQ index or HNSW graph
- the optimization is only valid if measured quality loss is explicit

## Implementation

Primary files:
- `python/kayak_bridge/tachiom_streaming_benchmark.py`
- `python/scripts/bench_tachiom_streaming_index.py`
- `python/scripts/sweep_tachiom_streaming_pruning.py`
- `python/tests/test_tachiom_streaming_index.py`

The benchmark now supports three query-time states:
- use the artifact's stored `candidate_pruning_alpha`
- override it with a positive alpha in `(0, 1)`
- disable pruning

The pruning sweep now builds exact MaxSim reference rankings once when
`--run-exact` is enabled and passes that reference to every alpha row. This
keeps exact-aware Pareto sweeps from redoing the same exact search for every
candidate-pruning setting.

Prepared Mojo readers keep their native prepared index handle. The pruning alpha
is passed per search call, so the override changes only the lightweight Python
metadata wrapper.

## Validation

Focused commands:

```bash
pixi run python -m py_compile \
  python/kayak_bridge/tachiom_streaming_benchmark.py \
  python/scripts/bench_tachiom_streaming_index.py \
  python/scripts/sweep_tachiom_streaming_pruning.py \
  python/tests/test_tachiom_streaming_index.py
```

```bash
env PYTHONPATH=python pixi run python -m unittest \
  python/tests/test_tachiom_streaming_index.py
```

Result: `3` tests passed.

## Speed Sweep

Command shape:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --quiet-checks 2 \
  --sleep-seconds 1 --timeout-seconds 120 --force -- \
  pixi run python python/scripts/sweep_tachiom_streaming_pruning.py \
    --snapshot <snapshot> \
    --index <streaming_tachiom_index_c32768> \
    --engine streaming_tac_hnsw_pq_mojo \
    --pruning-alphas artifact,disabled,0.05,0.1,0.2,0.35,0.5 \
    --warmup-iterations 1 \
    --measurement-iterations 3 \
    --sweep-repeats 2 \
    --output <pruning_sweep_hnsw_pq_mojo.json> \
    --emit-quiet-mean
```

Artifacts:
- `.cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768/pruning_sweep_hnsw_pq_mojo.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/pruning_sweep_hnsw_pq_mojo.json`
- `.cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768/pruning_sweep_alpha_0p3_hnsw_pq_mojo.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/pruning_sweep_alpha_0p3_hnsw_pq_mojo.json`

| Slice | Alpha | Median batch s | QPS | Judged MRR@10 | Mean candidate window |
| --- | ---: | ---: | ---: | ---: | ---: |
| `docs1500_q48_c32768` | artifact `0.35` | `0.3994217383333307` | `120.17372965299751` | `1.0` | `43.541666666666664` |
| `docs1500_q48_c32768` | `0.05` | `0.34260286999960954` | `140.1039051425772` | `1.0` | `15.229166666666666` |
| `docs1500_q48_c32768` | `0.2` | `0.36246580366666115` | `132.42628549903932` | `1.0` | `26.458333333333332` |
| `docs1500_q48_c32768` | `0.3` | `0.38400247233342577` | `124.9991952091445` | `1.0` | `37.0` |
| `docs1500_q48_c32768` | disabled | `2.226147235166233` | `21.56191614002373` | `1.0` | `1000.0` |
| `docs10000_q128_c32768` | artifact `0.35` | `1.9839862224998797` | `64.51657705501418` | `0.9895833333333334` | `193.171875` |
| `docs10000_q128_c32768` | `0.05` | `1.066742297166153` | `119.99149217204335` | `0.9895833333333334` | `17.328125` |
| `docs10000_q128_c32768` | `0.2` | `1.329732102167327` | `96.25999085934161` | `0.9895833333333334` | `68.921875` |
| `docs10000_q128_c32768` | `0.3` | `1.6979175160007192` | `75.38646535757029` | `0.9895833333333334` | `141.3515625` |
| `docs10000_q128_c32768` | disabled | `6.236496090000098` | `20.524345426150663` | `0.9895833333333334` | `1000.0` |

## Exact-Reference Check

The speed sweep above only proves judged metric preservation on these selected
positive slices. Exact top-10 overlap is a stricter diagnostic and does change.

Artifacts:
- `.cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768/search_hnsw_pq_mojo_prune_alpha_0p05_exact_summary.json`
- `.cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768/search_hnsw_pq_mojo_prune_alpha_0p2_exact_summary.json`
- `.cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768/search_hnsw_pq_mojo_prune_alpha_0p3_exact_summary.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/search_hnsw_pq_mojo_prune_alpha_0p05_exact_summary.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/search_hnsw_pq_mojo_prune_alpha_0p2_exact_summary.json`
- `.cache/kayak/tachiom_streaming_scale_docs10000_q128_c32768_mojo/search_hnsw_pq_mojo_prune_alpha_0p3_exact_summary.json`

| Slice | Alpha | Judged MRR@10 | Candidate recall@10 vs exact | Final recall@10 vs exact |
| --- | ---: | ---: | ---: | ---: |
| `docs1500_q48_c32768` | artifact `0.35` | `1.0` | `0.94375` | `0.9041666666666663` |
| `docs1500_q48_c32768` | `0.05` | `1.0` | `0.7270833333333333` | `0.7229166666666665` |
| `docs1500_q48_c32768` | `0.2` | `1.0` | `0.8687499999999999` | `0.8416666666666667` |
| `docs1500_q48_c32768` | `0.3` | `1.0` | `0.9312499999999998` | `0.895833333333333` |
| `docs10000_q128_c32768` | artifact `0.35` | `0.9895833333333334` | `0.9484375000000002` | `0.8867187500000006` |
| `docs10000_q128_c32768` | `0.05` | `0.9895833333333334` | `0.5632812500000002` | `0.5625000000000002` |
| `docs10000_q128_c32768` | `0.2` | `0.9895833333333334` | `0.8117187500000002` | `0.7867187500000001` |
| `docs10000_q128_c32768` | `0.3` | `0.9895833333333334` | `0.9156250000000002` | `0.8664062500000004` |

## Interpretation

Validated:
- candidate pruning is a real speed lever on the bounded native HNSW+PQ path
- alpha `0.05` roughly doubles docs10000 QPS while preserving judged MRR@10 on
  this selected-positive slice
- exact top-10 overlap falls sharply at alpha `0.05`, so it is not a safe
  default replacement
- alpha `0.3` is the measured compromise point here: smaller speed gain, much
  smaller exact-overlap loss

Open:
- these are still bounded slices, not full MS MARCO or LoTTE paper-scale runs
- the right alpha is a task policy choice, not a fixed implementation truth
- exact-reference computation is still repeated per benchmark call; a future
  sweep could cache exact rankings once per artifact when `--run-exact` is used
