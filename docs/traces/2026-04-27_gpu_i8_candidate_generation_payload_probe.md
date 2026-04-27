# 2026-04-27: GPU I8 Candidate-Generation Payload Probe

## Question

Is the centroid-posting payload small enough to plausibly keep resident on the
GPU before implementing posting accumulation kernels?

## Change

Added a benchmark-only payload preparation probe:

- `python/kayak_bridge/gpu_i8_candidate_generation_payload.py`
- `python/scripts/profile_gpu_i8_candidate_generation_payload.py`
- `profile_gpu_i8_candidate_generation_payload_raw`
- `profile_gpu_i8_candidate_generation_payload`

The probe copies the i8 candidate-generation payload tensors to the GPU and
reads them back for validation:

- centroid token indices
- centroid document offsets
- centroid document indices

Reason: a future GPU candidate generator needs these posting tensors resident.
Measuring the payload boundary first prevents us from writing a posting kernel
whose transfer cost or shape contract is unknown.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_candidate_generation_payload
```

Artifacts:

- report: `.cache/kayak/gpu_i8_candidate_generation_payload/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T122445Z`

Status:

- benchmark status: `ok`
- rows: `5 / 5`
- non-full candidate-generation rows: `3`
- GPU target: `nvidia:sm_89`
- device: `NVIDIA GeForce RTX 4070 Ti`

All rows validated:

- `copy_mismatch_count = 0`
- `offset_violation_count = 0`
- `doc_index_out_of_range_count = 0`

Non-full candidate-generation rows:

| case | posting count | payload bytes | CPU candidate s | H2D s | D2H s | H2D / CPU candidate | H2D + D2H / CPU candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `7689` | `63568` | `0.0004160156665117635` | `0.00000833109598323437` | `0.000007364455430296311` | `0.020025918862838773` | `0.0377282700556347` |
| `doc_vectors64` | `25252` | `204072` | `0.00035558333244504564` | `0.000013605872359754713` | `0.00001256860767859016` | `0.03826352676937539` | `0.07360997451248665` |
| `query_batch4` | `7706` | `63704` | `0.0005772229997091927` | `0.000008379481219341275` | `0.000007374456732539585` | `0.014516887275044294` | `0.02729263726465813` |

Summary:

- non-full H2D cost ranged from about `1.45%` to `3.83%` of CPU candidate
  generation time
- non-full H2D plus validation readback ranged from about `2.73%` to `7.36%`
  of CPU candidate generation time
- maximum measured payload was `204072` bytes

The two full-window rows are reported but are not the optimization target:
candidate generation is intentionally near-zero when `candidate_k` equals
`document_count`, so copy/CPU-candidate ratios are not meaningful for deciding
whether to move posting accumulation to the GPU.

## Interpretation

This validates the payload-residency prerequisite for the non-full rows. The
copy boundary alone is small relative to CPU candidate generation on the rows
where candidate generation is actually doing work.

This does not prove a GPU candidate-generation speedup. It only says the
posting payload is cheap enough to copy/validate in the measured synthetic
rows, so a benchmark-only posting accumulation kernel is now justified.

The extension-call field in this report is not serving evidence because the
Mojo function uses `benchmark.run` internally. The useful timings from this
probe are the separated host-ingest, H2D, and D2H means.

## Next Step

Add a benchmark-only GPU posting-accumulation probe for the non-full rows.

Initial scope:

- input: selected centroid ids per query and resident centroid-posting payload
- output: touched document ids or dense per-document touched flags
- correctness reference: CPU i8 candidate-generation profile for the same
  selected centroids
- keep final candidate top-k on CPU first

Reason: the measured payload copy is not the blocker. The next unknown is
whether GPU posting traversal can beat the CPU posting accumulation substep
without exploding synchronization or dense reset cost.
