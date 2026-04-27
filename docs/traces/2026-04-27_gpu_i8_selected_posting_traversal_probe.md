# 2026-04-27: GPU I8 Selected-Posting Traversal Probe

## Question

Can the GPU traverse selected i8 centroid posting lists correctly and cheaply
enough to justify a fused posting-accumulation kernel?

## Change

Added a benchmark-only selected-posting traversal probe:

- `python/kayak_bridge/gpu_i8_candidate_posting_traversal.py`
- `python/scripts/profile_gpu_i8_candidate_posting_traversal.py`
- `profile_gpu_i8_candidate_posting_traversal_raw`
- `profile_gpu_i8_candidate_posting_traversal`

The GPU kernel expands selected centroid ids into a flat stream of:

- document ids from the centroid posting lists
- the selected centroid proxy score repeated for each posting visit

Reason: this tests the next memory-access primitive after payload residency
without mixing in atomics, per-document max/reduce accumulation, or candidate
top-k. It is not a full candidate generator.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_candidate_posting_traversal
```

Artifacts:

- report: `.cache/kayak/gpu_i8_candidate_posting_traversal/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T124911Z`

Status:

- benchmark status: `ok`
- rows: `5 / 5`
- non-full candidate-generation rows: `3`
- GPU target: `nvidia:sm_89`
- device: `NVIDIA GeForce RTX 4070 Ti`

All rows validated:

- `doc_mismatch_count = 0`
- `doc_index_out_of_range_count = 0`
- `score_delta_max_abs = 0.0`
- `traversal_agreement_ok = true`

Non-full candidate-generation rows:

| case | selected centroids | expanded postings | CPU candidate s | CPU posting s | GPU kernel s | GPU all measured s | GPU all / CPU candidate | GPU selected path / CPU posting |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `256` | `16087` | `0.0004132309998870672` | `0.00006495819933333334` | `0.000019755204205684246` | `0.00004822149883808465` | `0.11669380770383443` | `0.6297499959274729` |
| `doc_vectors64` | `256` | `53949` | `0.0003714696661821411` | `0.00007139952` | `0.00004984263772330702` | `0.00010107559034583547` | `0.2720964847134396` | `1.2383358245530642` |
| `query_batch4` | `256` | `15686` | `0.0005846039997171223` | `0.00005525532118471132` | `0.000020629994144997415` | `0.00005065099981299205` | `0.0866415553733827` | `0.7829243450378431` |

Summary:

- non-full GPU all-measured path ranged from about `8.7%` to `27.2%` of full
  CPU candidate generation
- non-full GPU selected-H2D + kernel + validation-D2H ranged from about
  `0.63x` to `1.24x` of isolated CPU posting accumulation
- maximum expanded posting count was `53949`

The two full-window rows are reported but are not the optimization target:
candidate generation is intentionally near-zero when `candidate_k` equals
`document_count`.

## Interpretation

The probe validates the selected-posting traversal contract: selected centroid
ids, posting offsets, and posting doc ids can be moved through the GPU without
layout mismatch on the measured shapes.

It does not validate a production GPU candidate generator. The plain expansion
boundary materializes every posting visit and reads it back for validation. That
is useful for proving memory traversal, but the `doc_vectors64` row is slower
than isolated CPU posting accumulation when measured as selected H2D + kernel +
D2H.

The next GPU candidate-generation primitive should fuse traversal with
per-document accumulation/reduction, then return touched candidate ids or a
bounded candidate window. Keeping this as a separate step prevents us from
claiming a speedup from a kernel that only moves the bottleneck to readback.

## Next Step

Add a benchmark-only GPU posting-accumulation/reduction probe.

Initial scope:

- input: selected centroid ids/scores plus resident centroid posting payload
- output: document scores or touched document candidates, not the whole posting
  visit stream
- correctness reference: CPU i8 candidate-generation profile for the same
  selected centroids
- keep final candidate top-k on CPU first

Reason: the traversal probe says the GPU can walk the posting lists correctly,
but a competitive primitive needs to reduce visits on device before readback.
