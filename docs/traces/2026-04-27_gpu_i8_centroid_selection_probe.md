# 2026-04-27: GPU I8 Centroid Selection Probe

## Question

Can GPU scoring of sampled i8 centroids plus host centroid top-k selection beat
the CPU centroid scoring/selection substep in candidate generation?

## Change

Added a benchmark-only centroid-selection probe:

- `python/kayak_bridge/gpu_i8_centroid_selection.py`
- `python/scripts/profile_gpu_i8_centroid_selection.py`
- `profile_gpu_i8_centroid_selection_raw`
- `profile_gpu_i8_centroid_selection`

The GPU kernel assigns one lane to each
`[query, query_vector, centroid]` score, gathers the centroid token index from
the real Kayak i8 payload, computes the dim128 i8 dot product, and writes a
dense `[query_count, query_vector_count, centroid_count]` score tensor. Host
selection then chooses `centroids_per_query_vector` centroid ids per query
vector.

Reason: the previous accumulation envelope showed CPU centroid
scoring/selection as a remaining measured slice. This probe tests that slice
directly without mixing it with posting traversal or document-score
accumulation.

## Measurement

Command:

```bash
pixi run profile_gpu_i8_centroid_selection
```

Artifacts:

- report: `.cache/kayak/gpu_i8_centroid_selection/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T135640Z`
- rejected naive-host-selection quiet wrapper:
  `.cache/kayak/bench_quiet/20260427T135346Z`

Status:

- benchmark status: `ok`
- rows: `5 / 5`
- non-full candidate-generation rows: `3`
- GPU target: `nvidia:sm_89`
- device: `NVIDIA GeForce RTX 4070 Ti`

All rows validated:

- `selected_position_mismatch_count = 0`
- `selected_score_mismatch_count = 0`
- selected-score tolerance: `0.0001`
- max selected-score delta: `2.288818359375e-05`
- `centroid_token_out_of_range_count = 0`

Non-full candidate-generation rows:

| case | CPU centroid scoring + selection s | GPU kernel s | host selection s | resident payload s | cold payload s | resident / CPU centroid step | resident / CPU candidate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.00009688810258300944` | `0.000016412489714755897` | `0.000013324056710775046` | `0.0000393976966624338` | `0.00008897262812403579` | `0.4066309031976304` | `0.098966145360673` |
| `doc_vectors64` | `0.00004327223311652038` | `0.000010655066034391065` | `0.00000990175365934797` | `0.000028391188773605647` | `0.00020564040031206718` | `0.6561063926873356` | `0.076983033705886` |
| `query_batch4` | `0.00006063228818461679` | `0.000010621184590690208` | `0.000011425547532517215` | `0.00003276833615678412` | `0.0000822348112288896` | `0.5404436668629286` | `0.05699944709298083` |

Summary:

- non-full resident-payload GPU centroid selection ranged from about `0.407x`
  to `0.656x` of CPU centroid scoring plus selection
- non-full resident-payload GPU centroid selection ranged from about `0.057x`
  to `0.099x` of full CPU candidate generation
- non-full cold-payload GPU centroid selection ranged from about `0.143x` to
  `0.558x` of full CPU candidate generation

The two full-window rows are reported but are not optimization targets because
candidate generation is intentionally near-zero when `candidate_k` equals
`document_count`.

## Rejected Variant

The first host-selection implementation used repeated full scans over all
centroids for each selected rank. It preserved selected positions, but the host
selection step dominated the resident path. On the non-full rows, resident GPU
centroid selection measured about `1.29x`, `3.21x`, and `2.00x` of CPU
centroid scoring plus selection.

Reason for rejection: this measured a bad selection primitive, not an inherent
GPU centroid-scoring limit. Replacing it with a bounded worst-first heap mirrors
the CPU selection structure and makes the measurement more representative.

## Interpretation

The heap-backed probe validates GPU centroid scoring plus host centroid
selection as a useful internal primitive on the non-full rows. This is not a
public backend claim: the probe still reads all centroid scores back to host and
then sends selected centroid ids/scores to the accumulation primitive in a
separate later step.

The result changes the next optimization target. Earlier evidence pointed at
CPU centroid scoring/selection as a possible blocker. This probe shows that the
resident centroid-selection substep can beat CPU on the measured non-full rows,
but the current boundary still has two avoidable data movements:

- centroid-score D2H for host selection
- selected-centroid H2D before GPU posting accumulation

## Next Step

Design a fused selected-centroid-to-accumulation probe that keeps selected
centroid ids and scores on device.

Reason: the strongest current path is no longer "optimize centroid selection"
or "optimize accumulation" in isolation. The next falsifiable question is
whether a device-resident boundary can remove centroid-score readback and
selected-centroid upload while preserving exact selected positions and document
top-k agreement.
