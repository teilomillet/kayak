# 2026-04-27: GPU I8 Next Optimization Priority

## Question

Should the next work optimize GPU top-k/readback, or should it target
candidate generation?

## Evidence

Latest shape-policy FastPlaid comparison:

- report: `.cache/kayak/gpu_i8_fastplaid_policy_compare/summary.json`
- status: `ok`, `6 / 6` rows
- mean scoped envelope / FastPlaid batch: `0.1044423715919968`
- mean CPU candidate-generation share of scoped envelope:
  `0.6925531844894505`
- mean GPU no-reference top-k share of scoped envelope:
  `0.3074468155105495`

Per-row scoped envelope shares:

| case | CPU candidate s/window | GPU top-k s/window | candidate share | GPU top-k share |
| --- | ---: | ---: | ---: | ---: |
| `query_vectors32` CPU | `0.00037759950009785825` | `0.00018359499972575577` | `0.672849609567698` | `0.327150390432302` |
| `query_vectors32` CUDA | `0.00037980600018272526` | `0.0001693007498033694` | `0.6916797147227626` | `0.3083202852772374` |
| `doc_vectors64` CPU | `0.0003881667498717434` | `0.0002870202499707375` | `0.5749025824879649` | `0.4250974175120351` |
| `doc_vectors64` CUDA | `0.0003945109999676788` | `0.0002868477499760047` | `0.5790062870701909` | `0.42099371292980914` |
| `query_batch4` CPU | `0.0005176909999136114` | `0.00011679800036290544` | `0.8159180059670007` | `0.18408199403299924` |
| `query_batch4` CUDA | `0.0005381065000165108` | `0.00011735125008272007` | `0.8209629071210859` | `0.1790370928789141` |

Candidate-generation breakdown:

- report:
  `.cache/kayak/gpu_i8_candidate_generation_breakdown/policy_summary.json`
- non-full rows show posting accumulation, centroid selection, and final
  candidate top-k as the relevant substeps
- smaller `candidate_k` and positive-centroid postings were tested separately
  and were not promoted because recall or envelope evidence did not support
  them

## Decision

The highest-leverage next target is candidate generation, not GPU top-k
readback.

Reason: even a perfect GPU top-k/readback optimization can only remove the
smaller `18-43%` top-k share on these rows. Candidate generation owns
`57-82%` of the scoped envelope and is the consistent limiter after the
prepared GPU rerank handle.

## Concrete Next Step

Expose the i8 centroid-posting payload needed by a future GPU candidate
generation primitive:

- centroid token indices
- centroid document offsets
- centroid document indices

Reason: a GPU candidate generator cannot be designed or measured honestly
without the exact resident payload it would use. This is a low-risk enabling
step that keeps the backend boundary explicit and avoids pretending the
current CPU candidate loop is already a GPU-ready primitive.

Observed candidate-generation payload sizes for the wide non-full policy rows:

| case | centroid count | posting count | candidate-generation bytes |
| --- | ---: | ---: | ---: |
| `query_vectors32` | `128` | `7676` | `63464` |
| `doc_vectors64` | `128` | `25212` | `203752` |
| `query_batch4` | `128` | `7689` | `63568` |

Interpretation:

- the posting payload is small enough to be a plausible resident GPU index
  component on these synthetic rows
- this does not prove a GPU candidate-generation speedup
- the next validation step is a benchmark-only GPU candidate-generation
  payload/preparation probe before implementing posting accumulation kernels

## Follow-Up: Hybrid Shortlist Rerank

The fused-shortlist exact-rerank probe confirms that the next target is still
candidate-window generation.

Latest artifacts:

- default quiet run: `.cache/kayak/bench_quiet/20260427T155138Z`
- default report: `.cache/kayak/gpu_i8_fastplaid_policy_compare/summary.json`
- trace:
  `docs/traces/2026-04-27_gpu_i8_hybrid_shortlist_rerank_scope.md`

Result:

- `shortlist_k=256` recovers address-window recall on the wide rows and exact
  rerank agreement is `1.0`
- exact rerank is only about `2.2%` to `5.1%` of hybrid time
- smaller raw exploratory shortlists at `64`, `128`, and `192` are faster but
  lose too much recall

Decision update: do not optimize exact rerank next. The next useful primitive
is a serving-shaped selected-posting candidate generator that returns candidate
positions for exact rerank.
