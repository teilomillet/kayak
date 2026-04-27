# 2026-04-27: GPU I8 Hybrid Shortlist Rerank Scope

## Question

Can fused GPU centroid-posting scores be used as a shortlist generator, then
exact-reranked with the GPU address scorer, to keep address-window recall while
removing CPU candidate generation?

## Change

Added a benchmark-only hybrid primitive and FastPlaid policy-row wiring:

- `python/kayak_bridge/gpu_i8_fastplaid_hybrid_probe.py`
- `python/kayak_bridge/gpu_i8_fastplaid_hybrid_scope.py`
- `python/kayak_bridge/gpu_i8_fastplaid_hybrid_metrics.py`
- `python/kayak_bridge/gpu_i8_fastplaid_hybrid_reference.py`
- `--gpu-hybrid-shortlist-k` on the FastPlaid comparison scripts

The primitive performs:

1. fused GPU centroid-posting scoring with device top-k to produce a shortlist
2. GPU exact i8 address rerank over that shortlist
3. CPU validation of exact rerank order for the same shortlist

Reason: the previous fused final-top-k row was fast but too low-recall. This
tests fused scoring as a candidate generator rather than as final output.

## Measurement

Decision-quality default shortlist run:

```bash
pixi run compare_gpu_i8_fastplaid_policy
```

Artifacts:

- report: `.cache/kayak/gpu_i8_fastplaid_policy_compare/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T155138Z`

Exploratory shortlist sweeps:

```bash
pixi run compare_gpu_i8_fastplaid_policy_raw --gpu-hybrid-shortlist-k 64 --fastplaid-devices cpu --output .cache/kayak/gpu_i8_fastplaid_policy_compare/summary_hybrid_k64_cpu.json --report-root .cache/kayak/gpu_i8_fastplaid_policy_compare/reports_hybrid_k64_cpu
pixi run compare_gpu_i8_fastplaid_policy_raw --gpu-hybrid-shortlist-k 128 --fastplaid-devices cpu --output .cache/kayak/gpu_i8_fastplaid_policy_compare/summary_hybrid_k128_cpu.json --report-root .cache/kayak/gpu_i8_fastplaid_policy_compare/reports_hybrid_k128_cpu
pixi run compare_gpu_i8_fastplaid_policy_raw --gpu-hybrid-shortlist-k 192 --fastplaid-devices cpu --output .cache/kayak/gpu_i8_fastplaid_policy_compare/summary_hybrid_k192_cpu.json --report-root .cache/kayak/gpu_i8_fastplaid_policy_compare/reports_hybrid_k192_cpu
```

The raw sweeps are exploratory because one quiet CPU-only sweep encountered a
transient Mojo `gpu-query` NVML failure inside the FastPlaid/Torch process. The
comparison script now probes Mojo GPU capability before running FastPlaid so
capability detection is not ordered after external CUDA/NVML use.

## Default Shortlist Result

The default hybrid shortlist is `candidate_k` clipped to document count, so the
wide policy rows use `shortlist_k = 256`.

| case | FastPlaid device | address recall | hybrid recall | FastPlaid recall | hybrid s/window | hybrid / FastPlaid | exact rerank share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `cpu` | `0.7` | `0.7` | `0.35` | `0.005546778500502114` | `0.37570165778238473` | `0.033591570631649526` |
| `query_vectors32` | `cuda` | `0.7` | `0.7` | `0.5` | `0.0051522097492124885` | `1.9717180761782396` | `0.03391030403910094` |
| `doc_vectors64` | `cpu` | `0.65` | `0.65` | `0.6` | `0.005664044750119501` | `0.22826193051355484` | `0.05088995807813799` |
| `doc_vectors64` | `cuda` | `0.65` | `0.65` | `0.7` | `0.005675433751093806` | `0.9857047693278133` | `0.05084778762223186` |
| `query_batch4` | `cpu` | `0.7` | `0.7` | `0.55` | `0.005290223000883998` | `0.40839168622911554` | `0.02212543784927765` |
| `query_batch4` | `cuda` | `0.7` | `0.7` | `0.5249999999999999` | `0.0052954724997107405` | `1.312590999269168` | `0.022173469744005605` |

Verified:

- `gpu_hybrid_final_topk_position_agreement_min = 1.0`
- exact rerank is not the bottleneck; it is only about `2.2%` to `5.1%` of
  hybrid time on the quiet rows
- the high cost comes from producing a large fused device shortlist, not from
  exact address rerank

## Shortlist Sweep

CPU FastPlaid rows, raw exploratory timings:

| case | k64 recall / s | k128 recall / s | k192 recall / s | k256 recall / s |
| --- | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.3 / 0.0005598747502517654` | `0.5 / 0.0014988767507020384` | `0.55 / 0.0032660680008120835` | `0.7 / 0.005546778500502114` |
| `doc_vectors64` | `0.1 / 0.0005962874984106747` | `0.2 / 0.0016801894989839639` | `0.45 / 0.0033668017504169256` | `0.65 / 0.005664044750119501` |
| `query_batch4` | `0.2 / 0.0005662062503688503` | `0.4 / 0.0015683305009588366` | `0.55 / 0.0031624899993403233` | `0.7 / 0.005290223000883998` |

Interpretation:

- smaller shortlists are fast but lose too much recall
- recall only recovers near the full `candidate_k=256` window
- the fused score order is not strong enough to safely prune these rows to 64
  or 128 candidates
- at the shortlist size that preserves recall, fused device top-k dominates
  the hybrid path

## Decision

Do not spend the next optimization pass on exact rerank or host/device handoff.

Reason: the exact-rerank share is small on the measured rows, and smaller
shortlists falsify the hope that fused-score pruning alone can keep recall.
The next primitive should be candidate-window generation from selected
centroid postings, not large fused-score top-k.

The strongest existing evidence points to the selected-posting accumulation
path:

- `docs/traces/2026-04-27_gpu_i8_posting_accumulation_probe.md`
- non-full GPU accumulation plus host top-k cost about `0.146x` to `0.214x` of
  full CPU candidate generation while preserving exact score and candidate
  top-k agreement against the CPU i8 reference

That primitive still needs a serving-shaped output boundary that returns
candidate positions for exact rerank. Until that exists, the hybrid row remains
a useful negative/diagnostic primitive, not the preferred optimization target.
