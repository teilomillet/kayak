# 2026-04-27: GPU I8 Resident Selected-Posting Candidate Window

## Question

Can selected-posting candidate generation become a serving-shaped win if the
posting payload stays resident on GPU and dense scores are written into
caller-owned NumPy memory instead of returned as Python float objects?

## Change

Added a resident selected-posting payload handle:

- `prepare_i8_selected_posting_session_handle`
- `score_i8_selected_posting_session_handle_dense_scores_into`
- `release_i8_selected_posting_session_handle`
- Python wrapper:
  `score_i8_selected_posting_resident_dense_candidate_positions_addresses`

The handle keeps `centroid_doc_offsets` and `centroid_doc_indices` on device.
Each score call copies only selected centroid positions/scores, runs the
selected-posting accumulation/reduction kernels, copies dense document scores
back, writes those scores into a NumPy `Float32` buffer, and performs
deterministic host `candidate_k` selection.

Reason: the previous dense-score candidate boundary was correct but lost after
returning dense scores as Python float objects. The new boundary tests whether
the losing part was payload residency/output materialization rather than the
selected-posting kernels.

## Measurement

Commands:

```bash
pixi run profile_gpu_i8_candidate_posting_accumulation_raw \
  --case smoke:documents=64,document_vectors=8,queries=1,query_vectors=4,candidate_k=32 \
  --warmup-iterations 0 \
  --measurement-iterations 1 \
  --output .cache/kayak/gpu_i8_candidate_posting_accumulation/resident_dense_candidate_numpy_output_smoke.json

pixi run profile_gpu_i8_candidate_posting_accumulation
```

Artifacts:

- quiet report: `.cache/kayak/gpu_i8_candidate_posting_accumulation/summary.json`
- quiet wrapper: `.cache/kayak/bench_quiet/20260427T171615Z`
- smoke report:
  `.cache/kayak/gpu_i8_candidate_posting_accumulation/resident_dense_candidate_numpy_output_smoke.json`

Status:

- benchmark status: `ok`
- rows: `5 / 5`
- non-full candidate-generation rows: `3`
- GPU target: `nvidia:sm_89`
- device: `NVIDIA GeForce RTX 4070 Ti`

All resident candidate rows validated:

- `candidate_position_agreement = 1.0`
- `candidate_score_delta_max_abs = 0.0`
- `score_delta_max_abs = 0.0`
- `validation_reference_scores_sent_to_extension = false`
- no selected-position or doc-index range violations

## Result

Non-full rows from the accepted NumPy-output resident boundary:

| case | queries | query vectors | docs | doc vectors | candidate_k | expanded postings | CPU candidate s | resident score s | host selection s | resident score+selection s | score+selection / CPU | projected CPU selection + resident s | projected / CPU | projected with prepare / CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `2` | `32` | `512` | `16` | `256` | `16087` | `0.0004023139990749769` | `0.000026558998797554523` | `0.0001365550015179906` | `0.00016311400031554513` | `0.40543953402214705` | `0.0002669158557523611` | `0.663451573562116` | `0.757186817676649` |
| `doc_vectors64` | `2` | `8` | `512` | `64` | `256` | `53949` | `0.0003585056656447705` | `0.00003870199725497514` | `0.00012959200103068724` | `0.00016829399828566238` | `0.4694319069769415` | `0.00021250234274962154` | `0.5927447265510776` | `0.7421426475292772` |
| `query_batch4` | `4` | `8` | `512` | `16` | `256` | `15686` | `0.000574505332527527` | `0.0000284630004898645` | `0.00023225400218507275` | `0.00026071700267493725` | `0.4538112840970805` | `0.0003238016388021461` | `0.5636181606488941` | `0.6299552304096512` |

Summary:

- resident score+host-selection is `0.405x` to `0.469x` of CPU candidate
  generation on the non-full rows
- projected CPU centroid scoring/selection plus resident selected-posting
  candidate generation is `0.564x` to `0.663x` of CPU candidate generation
- even including handle prepare, the projected cold path is `0.630x` to
  `0.757x` of CPU candidate generation on the non-full rows
- full-window rows remain excluded from the optimization decision because CPU
  candidate generation is intentionally near-zero when
  `candidate_k == document_count`

Preceding negative check:

- before the NumPy-output path, the resident boundary still returned dense
  scores as Python float objects
- that run was exact, but projected CPU selection plus resident selected-posting
  was `1.059x` to `1.321x` of CPU candidate generation on the non-full rows
- replacing Python float-list output with caller-owned NumPy output changed the
  conclusion from "not yet a win" to "accepted internal primitive"

## Interpretation

The winning change is a boundary change, not a new scoring algorithm.

The selected-posting qv-doc reduce kernels were already fast in the lower-level
probe. The earlier serving-shaped boundary lost because it paid too much Python
object materialization and per-call payload movement. Keeping posting metadata
resident and writing dense scores into typed NumPy memory exposes the kernel
win to the benchmark-visible primitive.

This also falsifies two tempting directions:

- the block-parallel candidate selector is exact but still `9.11x` to `14.02x`
  of CPU candidate generation on non-full rows
- returning dense scores through Python float lists is exact but still
  `1.09x` to `1.36x` of CPU candidate generation on non-full rows

## Decision

Accept the resident selected-posting NumPy-output candidate boundary as the
next internal GPU candidate-generation primitive.

Do not expose it as a public search backend yet.

Reason: the primitive is correct and produces a measured non-full win against
CPU i8 candidate generation, but it still starts from CPU-selected centroids.
The next system comparison must account for the full pipeline shape and keep
FastPlaid as the external full-search baseline.

Next concrete steps:

- feed this resident selected-posting candidate window into the exact rerank
  comparison path
- add the projected resident-selected-posting row to the FastPlaid policy
  comparison so the same wide shapes report recall and timing against FastPlaid
- keep the lower-level accumulation and block-selector rows as diagnostics, not
  serving defaults

## FastPlaid Scope Follow-Up

The accepted resident selected-posting candidate boundary was added to the
FastPlaid policy matrix as an internal two-stage primitive:

```text
CPU selected centroids
  -> resident GPU selected-posting candidate_k window
  -> GPU exact i8 address rerank
```

Command:

```bash
pixi run compare_gpu_i8_fastplaid_policy --overwrite-index-root
```

Artifacts:

- quiet log: `.cache/kayak/bench_quiet/20260427T172856Z`
- summary report:
  `.cache/kayak/gpu_i8_fastplaid_policy_compare/summary.json`
- per-row reports:
  `.cache/kayak/gpu_i8_fastplaid_policy_compare/reports/`

Status:

- policy comparison status: `ok`
- rows: `6 / 6`
- FastPlaid devices: `cpu`, `cuda`
- candidate agreement minimum: `1.0`
- final top-k position agreement minimum: `1.0`

Resident selected-posting exact-rerank rows:

The headline row includes CPU selected-centroid scoring/selection, resident GPU
selected-posting candidate scoring plus host candidate selection, and GPU exact
i8 address rerank. The cold metric additionally includes per-window resident
payload prepare/release; it is tracked separately because the intended serving
shape keeps the selected-posting payload resident across query windows.

| case | FastPlaid device | selected+resident+exact s/window | selected+resident+exact / FastPlaid batch | recall | FastPlaid recall | recall delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `cpu` | `0.0006047690003470052` | `0.04682145535397505` | `0.7` | `0.45` | `0.25` |
| `query_vectors32` | `cuda` | `0.0006008210011714255` | `0.218733060397411` | `0.7` | `0.4` | `0.3` |
| `doc_vectors64` | `cpu` | `0.0006688427474728087` | `0.028418959770200584` | `0.65` | `0.7` | `-0.05` |
| `doc_vectors64` | `cuda` | `0.0006750444972567493` | `0.1391233042778215` | `0.65` | `0.6` | `0.05` |
| `query_batch4` | `cpu` | `0.0006103249988882453` | `0.04753502839552294` | `0.7` | `0.575` | `0.125` |
| `query_batch4` | `cuda` | `0.0006061732501621009` | `0.13219772834877314` | `0.7` | `0.575` | `0.125` |

Summary:

- mean resident selected-posting exact-rerank / FastPlaid batch:
  `0.10213825609061737`
- max resident selected-posting exact-rerank / FastPlaid batch:
  `0.218733060397411`
- max cold resident selected-posting exact-rerank / FastPlaid batch:
  `0.23550834938348914`
- minimum recall delta versus FastPlaid: `-0.04999999999999993`
- mean exact-rerank share of this new row:
  `0.3247089859445457`
- mean CPU selected-centroid share of this new row:
  `0.3432215448787619`
- mean resident candidate scoring/selection share of this new row:
  `0.3320694691766924`

Interpretation:

The new row is faster than FastPlaid full search on every measured CPU and CUDA
FastPlaid row in this scoped synthetic matrix. It does not yet dominate
FastPlaid on quality everywhere: the `doc_vectors64` CPU row measured
`0.65` recall for Kayak versus `0.70` for FastPlaid.

Decision:

Raw speed is no longer the blocker for this boundary. The next optimization
should target candidate coverage/quality for the shape-policy selected
centroid budget, especially `doc_vectors64`, before claiming the row dominates
FastPlaid everywhere.
