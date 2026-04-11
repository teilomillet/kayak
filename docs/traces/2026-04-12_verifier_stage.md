# Verifier Stage Trace

Date: `2026-04-12`

## Scope

This trace records the first empirical pass for the new verifier stage:
- public API and evaluation integration
- correctness checks against the exact baseline
- real-subset timing shape for stage-3 reranking

## Commands

Correctness:

```bash
pixi run test_verifier
pixi run test_eval_battle
pixi run test_storage
pixi run test_hybrid_flat_dim128
```

Performance:

Strict quiet run attempted first:

```bash
pixi run bench_profile_cpu_verifier_real_subset
```

The strict quiet wrapper timed out after `120s` with sustained competing load above the default `max_other_cpu=40` threshold.
Representative quiet-check values were roughly `81-262%` other CPU.
The latest snapshot showed GUI and browser processes plus OS background agents dominating the machine.

Quiet-run log directory:

```text
.cache/kayak/bench_quiet/20260411T230224Z
```

Because the host never entered the default quiet envelope, the timing fallback for this trace used the raw benchmark path.
Those numbers are useful for regression tracking and shape analysis, but they are not decision-grade absolute performance claims.

Raw benchmark command:

```bash
pixi run bench_profile_cpu_verifier_real_subset_raw
```

## Correctness Result

Verified:
- `no_verifier` preserves the exact-search baseline
- `exact_late_interaction_verifier` can reorder a bad candidate window correctly
- `evaluate_task_with_verifier` matches `evaluate_task` when stage 1 is already exact

## Raw Timing Result

All timings below are raw means in seconds.

### SciFact

| Section | Mean (s) |
| --- | ---: |
| `search_exact` | `0.0004407707437661221` |
| `rerank noop verifier`, candidate `10` | `2.3036440149361995e-07` |
| `rerank exact late interaction`, candidate `10` | `0.0005191495827624859` |
| `search_exact_with_verifier(exact late interaction)`, candidate `10` | `0.0010517235227272728` |
| `rerank noop verifier`, candidate `40` | `2.3080714614128953e-07` |
| `rerank exact late interaction`, candidate `40` | `0.0020822673684210524` |
| `search_exact_with_verifier(exact late interaction)`, candidate `40` | `0.0025885006566850536` |

### FIQA

| Section | Mean (s) |
| --- | ---: |
| `search_exact` | `0.0004722063828568154` |
| `rerank noop verifier`, candidate `10` | `2.3085195342950314e-07` |
| `rerank exact late interaction`, candidate `10` | `0.0004163971076702948` |
| `search_exact_with_verifier(exact late interaction)`, candidate `10` | `0.0008980372709390205` |
| `rerank noop verifier`, candidate `40` | `2.2876869360467247e-07` |
| `rerank exact late interaction`, candidate `40` | `0.0015145496678010046` |
| `search_exact_with_verifier(exact late interaction)`, candidate `40` | `0.002012217391304348` |

## Interpretation

What the evidence supports:
- the verifier interface is mechanically correct
- the noop verifier cost is negligible
- exact late-interaction reranking scales upward with candidate window size as expected
- on these small exact-stage baselines, end-to-end verifier search is slower than plain `search_exact`

What the evidence does not support:
- enabling exact verifier reranking by default on top of an already exact stage-1 scorer
- making absolute performance claims from this run, because the strict quiet wrapper could not secure a low-noise host

Current conclusion:
- keep the verifier stage optional
- treat it as infrastructure for future approximate stage-1 retrieval, hybrid candidate generation, or later text-level reranking once storage includes raw text
