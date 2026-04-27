# 2026-04-27: GPU I8 Unordered Candidate Window

## Claim

The GPU rerank pipeline does not need CPU candidate windows ordered by
approximate score; it needs the retained candidate set.

Reason: GPU top-k reranks candidates with i8 MaxSim scores. Approximate-score
order is useful for diagnostics and public candidate inspection, but it is not
required for scoring the same retained documents.

## Change

Added an explicit internal unordered candidate-window path:

- `top_unordered_positions_by_score(...)`
- `plaid_i8_candidate_positions_for_query_unordered(...)`
- `KayakPlaidApproxIndex.i8_candidate_positions_batch_unordered(...)`
- `--kayak-i8-candidate-order {ordered,unordered}`

The public ordered candidate API remains unchanged. The policy FastPlaid
comparison now defaults to `unordered` for the internal GPU pipeline.

Reason: this keeps candidate order as an explicit measurement/control knob
instead of silently changing public candidate semantics.

## Measurement

Commands:

```bash
pixi run env PYTHONPATH=python python -m unittest python/tests/test_gpu_i8_candidate_generation_breakdown.py python/tests/test_gpu_i8_fastplaid_policy_compare.py
pixi run env PYTHONPATH=python python -m unittest python/tests/test_fastplaid_speed_track.py
pixi run profile_gpu_i8_candidate_generation_breakdown_policy
pixi run compare_gpu_i8_fastplaid_policy
```

Artifacts:

- candidate breakdown quiet log: `.cache/kayak/bench_quiet/20260427T104620Z`
- FastPlaid policy quiet log: `.cache/kayak/bench_quiet/20260427T105039Z`
- FastPlaid policy summary:
  `.cache/kayak/gpu_i8_fastplaid_policy_compare/summary.json`

## Results

Candidate-generation breakdown, policy budgets, wide non-full rows:

| case | ordered candidate batch s | unordered candidate batch s | unordered / ordered | candidate set agreement |
| --- | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.00024551876988711447` | `0.0002198763876916095` | `0.8955583631862651` | `1.0` |
| `doc_vectors64` | `0.00018408207623251066` | `0.00016077031383931185` | `0.8733621280772922` | `1.0` |
| `query_batch4` | `0.0002523595122390538` | `0.00019051949865747477` | `0.7549527139559552` | `1.0` |

Automated FastPlaid policy comparison with unordered internal candidate
windows:

- status: `ok`, `6 / 6` rows
- minimum top-k position agreement: `1.0`
- minimum Kayak recall delta versus FastPlaid: `1.1102230246251565e-16`
- mean scoped envelope / FastPlaid batch: `0.09869997626737713`
- max scoped envelope / FastPlaid batch: `0.21780632157235916`
- mean CPU candidate-generation share of scoped envelope:
  `0.6723469052696253`
- mean GPU no-reference top-k share of scoped envelope:
  `0.3276530947303748`

Per-window CPU candidate generation in the FastPlaid policy run:

- `query_vectors32`: about `0.000371s/window`
- `doc_vectors64`: about `0.000324s/window`
- `query_batch4`: about `0.000500s/window`

## Negative Results

The reusable full workspace path was not promoted. It preserved candidate
positions but measured slower on the non-full policy rows.

The fused centroid scoring/selection variant was also removed before commit.
It preserved the intended work but made the non-full policy candidate path
slower in the quiet profile.

Unordered centroid selection was measured but not promoted. It preserved the
selected-centroid set in the candidate-generation breakdown and was faster as
an isolated substep, but a full FastPlaid policy comparison did not confirm an
end-to-end envelope improvement.

Artifacts:

- candidate breakdown quiet log: `.cache/kayak/bench_quiet/20260427T114436Z`
- FastPlaid policy quiet log with hot-path unordered centroid selection:
  `.cache/kayak/bench_quiet/20260427T114041Z`

Latest policy-budget breakdown, after restoring production centroid selection:

| case | ordered candidate batch s | unordered candidate batch s | unordered / ordered | unordered centroid selection / ordered centroid selection | set agreement |
| --- | ---: | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.0002470551531272893` | `0.00021484063226374743` | `0.8696059545580734` | `0.8208490485396996` | `1.0` |
| `doc_vectors64` | `0.0001852820134791447` | `0.00015366722621421117` | `0.8293693668841089` | `0.7723661071111159` | `1.0` |
| `query_batch4` | `0.00025705754796460467` | `0.00019225225989148183` | `0.7478957977065659` | `0.8361471905946738` | `1.0` |

The rejected hot-path FastPlaid policy run stayed correct but did not improve
the scoped envelope:

- status: `ok`, `6 / 6` rows
- minimum top-k position agreement: `1.0`
- minimum Kayak recall delta versus FastPlaid: `0.0`
- mean scoped envelope / FastPlaid batch: `0.1044423715919968`
- mean CPU candidate-generation share of scoped envelope:
  `0.6925531844894505`

Reason: both ideas added hot-path complexity without decision-quality speed
evidence.

## Decision

Keep unordered candidate windows as an internal GPU-pipeline option and keep
ordered candidate windows as the public/default candidate API.

Reason: the unordered path has set-equivalence guards, improves the measured
candidate-generation boundary on the useful non-full rows, and keeps the
candidate-order tradeoff explicit for future profiling.
