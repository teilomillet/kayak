# 2026-04-27: GPU I8 Candidate-Generation Cleanup

## Claim

After prepared-handle GPU top-k won the scoring boundary, CPU candidate
generation became the next visible limiter for the GPU rerank envelope.

Reason: the GPU primitive is only useful if the whole candidate-window rerank
boundary is competitive. Optimizing another kernel before measuring the
CPU-side window construction would risk improving the wrong step.

## Changed

- `plaid_i8_candidate_positions_for_query(...)` now returns every document
  position directly when `candidate_k >= document_count`.
- `top_positions_by_score(...)` now uses a bounded worst-first heap instead of
  repeatedly scanning all scores.
- Regression coverage now checks that full-window i8 candidate positions are
  identity-ordered and that equal scores keep lower document positions first.

Reasons:

- a full candidate window has no stage-1 pruning left to do, so proxy sorting
  cannot change the candidate set
- the heap keeps the previous score-descending, lower-position tie order while
  reducing selection from `O(k * n)` repeated scans to bounded heap maintenance
  plus ordered extraction
- the change stays CPU-side and measurable; it does not create a new GPU
  abstraction or a silent fallback path

## Measurement

Commands:

```bash
pixi run profile_gpu_i8_address_serve_wide_topk
pixi run profile_gpu_i8_address_serve_sweep
pixi run compare_gpu_i8_fastplaid_wide_candidate1024
pixi run compare_gpu_i8_fastplaid_wide_candidate1024_cuda
```

Artifacts:

- final wide sweep: `.cache/kayak/bench_quiet/20260427T075809Z`
- final default sweep: `.cache/kayak/bench_quiet/20260427T075856Z`
- final FastPlaid CPU comparison: `.cache/kayak/bench_quiet/20260427T075913Z`
- final FastPlaid CUDA comparison: `.cache/kayak/bench_quiet/20260427T075930Z`
- final wide report:
  `.cache/kayak/gpu_i8_address_serve_sweep/wide_topk_summary.json`
- final default report: `.cache/kayak/gpu_i8_address_serve_sweep/summary.json`
- final FastPlaid reports:
  `.cache/kayak/gpu_i8_fastplaid_compare/wide_candidate1024_summary.json`
  and
  `.cache/kayak/gpu_i8_fastplaid_compare/wide_candidate1024_cuda_summary.json`

Common wide controls:

- vector dim: `128`
- top-k: `10`
- warmup iterations: `1`
- measurement iterations: `3`
- resident windows: `4`
- Kayak PLAID payload: `i8`
- GPU target: `nvidia:sm_89`
- device: `NVIDIA GeForce RTX 4070 Ti`

## Wide Results

CPU candidate generation per window:

| case | docs | queries | query vecs | doc vecs | candidate k | before cleanup s | full-window shortcut only s | final s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| candidate512 | 512 | 2 | 8 | 16 | 512 | `0.0015524065001955023` | `0.0003331649166587643` | `0.0003354087499853146` |
| candidate1024 | 1024 | 2 | 8 | 16 | 1024 | `0.004614932416491986` | `0.0005732072499995411` | `0.0005739495832888982` |
| query_vectors32 | 512 | 2 | 32 | 16 | 256 | `0.0021934758333372883` | `0.0021859340833240517` | `0.0012884576667602232` |
| doc_vectors64 | 512 | 2 | 8 | 64 | 256 | `0.0011090663338109152` | `0.0011065703333012304` | `0.0005140262500541818` |
| query_batch4 | 512 | 4 | 8 | 16 | 256 | `0.002150007916801163` | `0.00217470983337383` | `0.0009644714167128162` |

CPU candidate generation plus GPU prepared-handle top-k per window:

| case | before cleanup s | final s | final / CPU candidate+CPU score | top-k agreement | max delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate512 | `0.0016615927506791195` | `0.00044736974988760875` | `0.3650090723588047` | `1.0` | `0.0000762939453125` |
| candidate1024 | `0.004795226166910045` | `0.000755711083532636` | `0.3306988261359973` | `1.0` | `0.0000762939453125` |
| query_vectors32 | `0.0023785658331689774` | `0.0014756991670310526` | `0.5228260910368743` | `1.0` | `0.000244140625` |
| doc_vectors64 | `0.001402285584845231` | `0.000807182500163132` | `0.4544372919011802` | `1.0` | `0.00006103515625` |
| query_batch4 | `0.0022776184171865075` | `0.0010920494167597403` | `0.5597503227234745` | `1.0` | `0.0000762939453125` |

Final wide summary:

- status: `ok`
- ok cases: `5 / 5`
- best CPU-candidate-plus-top-k ratio: `0.3306988261359973`
- worst CPU-candidate-plus-top-k ratio: `0.5597503227234745`
- best prepared-handle top-k score ratio: `0.10621595784990773`
- worst prepared-handle top-k score ratio: `0.23225854378549696`

## Default Sweep Check

The default six-case sweep still passed after the candidate-generation cleanup.

Selected final rows:

| case | CPU candidate generation/window s | CPU candidates + GPU top-k/window s | final / CPU candidate+CPU score |
| --- | ---: | ---: | ---: |
| baseline | `0.0003286590833037432` | `0.00039021908325291105` | `0.6245570535160144` |
| candidate32 | `0.0002531483332859352` | `0.00029740308309555985` | `0.7502785024622507` |
| candidate256 | `0.00020767533328580612` | `0.00028698058349618805` | `0.4124098992651909` |
| query_vectors16 | `0.0005559910833502121` | `0.0006412628332176004` | `0.5983582993078763` |
| doc_vectors32 | `0.0003343238333097058` | `0.00044127033326428017` | `0.587727722815865` |
| documents512 | `0.00038901824996173673` | `0.00045206850018075784` | `0.6571037673464278` |

## FastPlaid Context

Shape:

- documents: `1024`
- document vectors per document: `16`
- queries per window: `2`
- query vectors per query: `8`
- candidate window: `1024`
- candidate scores per window: `2048`
- top-k: `10`
- FastPlaid version: `1.4.6.2110`

| FastPlaid device | FastPlaid batch s | FastPlaid recall@10 | Kayak i8 CPU batch s | Kayak i8 recall@10 | GPU top-k/window s | CPU candidates + GPU top-k/window s | top-k/FastPlaid batch | candidates+top-k/FastPlaid batch | top-k agreement | max delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU | `0.008554922000257648` | `0.5` | `0.0010891000001720386` | `1.0` | `0.0002139034995707334` | `0.0007838207495751703` | `0.02500355930355546` | `0.09162219708742687` | `1.0` | `0.0000762939453125` |
| CUDA | `0.002270741999382153` | `0.44999999999999996` | `0.0010966729996653157` | `1.0` | `0.00022827050020168826` | `0.0007932682501632371` | `0.1005268323146348` | `0.34934318842875045` | `1.0` | `0.0000762939453125` |

Scope warning: this is still not an apples-to-apples backend comparison.
FastPlaid is timed as full search. The Kayak GPU row starts from CPU-generated
candidate windows and measures an internal prepared-handle rerank/top-k
boundary.

## Interpretation

Verified:

- the full-window shortcut removes the old proxy-sort cost for
  `candidate_k >= document_count`
- bounded heap selection improves the non-full-window wide cases without
  changing top-k agreement
- the final wide GPU boundary is faster than CPU candidate generation plus CPU
  scoring on all five wide cases
- the final default sweep remains correct on all six cases
- on the explicit wide `candidate1024` shape, CPU candidate generation plus GPU
  top-k is faster than the measured FastPlaid CPU and CUDA full-search rows in
  this scope-limited comparison

Still open:

- this does not prove a public GPU search backend speedup
- candidate generation is still CPU-side
- the current GPU top-k row still performs host-side top-k inside the Mojo
  extension after score production
- FastPlaid recall on this deterministic synthetic shape is lower than Kayak i8
  recall, so speed cannot be interpreted without the recall row

## Decision

Keep GPU work focused on an internal candidate-window rerank/search boundary,
not a public `gpu=True` API.

Reason: the measured primitive is now competitive enough to justify integration
work, but the remaining uncertainty is boundary ownership and search-path
composition, not the existence of a GPU device or another isolated kernel
micro-optimization.

## Validation

Ran:

```bash
pixi run mojo format kayak/search/plaid_approx_dim128.mojo kayak/search/plaid_i8_approx_dim128.mojo
pixi run python -m py_compile python/tests/test_fastplaid_speed_track.py python/tests/test_gpu_i8_rerank_contract.py
PYTHONPATH=python pixi run python -m unittest python.tests.test_fastplaid_speed_track -v
PYTHONPATH=python pixi run python -m unittest python.tests.test_gpu_i8_rerank_contract -v
pixi run profile_gpu_i8_address_serve_wide_topk
pixi run profile_gpu_i8_address_serve_sweep
pixi run compare_gpu_i8_fastplaid_wide_candidate1024
pixi run compare_gpu_i8_fastplaid_wide_candidate1024_cuda
```

Observed:

- FastPlaid speed-track tests: `10/10` passed
- GPU i8 rerank contract tests: `36/36` passed
- final wide quiet sweep status: `ok`
- final default quiet sweep status: `ok`
- final FastPlaid CPU and CUDA comparisons status: `ok`
