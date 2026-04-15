# Document Proxy Stage-2 Investigation on BrowseComp-Plus Gold

## Claim

The next justified optimization target after the earlier exact-rerank dispatch
work was the `document_proxy` operating point on `browsecomp_plus_gold`.

The working claim for this slice was:

- `document_proxy` stage-2 exact rerank still dominates that operating point
- if it dominates, the next win must come from the scorer kernel itself rather
  than from candidate resolution or hit assembly

This trace records the checks I ran to validate or debunk the obvious next
optimizations before touching production code.

## Commands

```bash
pixi run mojo -I . benchmarks/profile_document_proxy_search_breakdown_browsecomp_gold.mojo
pixi run mojo -I . benchmarks/profile_document_proxy_exact_stage_breakdown_browsecomp_gold.mojo
pixi run mojo -I . benchmarks/profile_document_proxy_hybrid_stage2_browsecomp_gold.mojo
pixi run mojo -I . benchmarks/profile_document_proxy_stage2_partition_sweep_browsecomp_gold.mojo
pixi run mojo -I . benchmarks/profile_document_proxy_stage2_dispatch_compare_browsecomp_gold.mojo
pixi run mojo -I . tests/test_maxsim.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
bash scripts/run_bench_quiet.sh --repeats 3 --timeout-seconds 20 --force -- pixi run mojo -I . benchmarks/bench_document_proxy_stage2_default_browsecomp_gold.mojo
```

## Stage split

From:

- `.cache/kayak/profile_document_proxy_search_breakdown_browsecomp_gold.tsv`
- `.cache/kayak/profile_document_proxy_exact_stage_breakdown_browsecomp_gold.tsv`

Measured on the gold slice:

- `stage1_document_proxy`: `1.7493324878078243e-05 s`
- `stage2_from_candidates`: `0.0006926189695550351 s`
- `search_document_proxy`: `0.0007375159010600706 s`

Exact-stage internal split:

- `resolve_candidate_window`: `8.398549159977249e-07 s`
- `score_resolved_candidate_window_for_cpu`: `0.000689044106167057 s`
- `assemble_topk_hits`: `5.249428422775461e-07 s`

Interpretation:

- stage-1 proxy generation is negligible on this slice
- inside stage-2, candidate resolution and final top-k assembly are also
  negligible
- the remaining cost is almost entirely the exact scoring kernel

That ruled out bookkeeping-oriented changes as the next lever.

## Hybrid-flat check

From:

- `.cache/kayak/profile_document_proxy_hybrid_stage2_browsecomp_gold.tsv`

Measured scorer-only means:

- current nested scorer: `0.0007914600468384075 s`
- hybrid-flat scorer: `0.0023294905168326223 s`
- hybrid-flat scorer with flat query: `0.002324741754385965 s`

Derived ratios:

- hybrid-flat vs nested: `2.94x` slower
- hybrid-flat plus flat query vs nested: `2.94x` slower

Interpretation:

- on these actual candidate windows, wiring the existing hybrid-flat dim128
  path into stage-2 would be a regression
- the hybrid-flat artifact remains useful infrastructure, but it is not the
  next CPU stage-2 win here

## Parallel-policy sweep

From:

- `.cache/kayak/profile_document_proxy_stage2_partition_sweep_browsecomp_gold.tsv`

Measured stage-2 means by forced work-item count:

- `1`: `0.0025277685393258428 s`
- `2`: `0.001635859649122807 s`
- `3`: `0.0010235033660743846 s`
- `4`: `0.0008536763614481999 s`
- `5`: `0.000987344935064935 s`
- `6`: `0.0010377046396441928 s`
- `8`: `0.0023157509998709843 s`

Interpretation:

- the best explicit setting for this shape is `4`
- that matches the current heuristic outcome for a `40`-document candidate
  window with this vector budget
- changing the work-item policy would not produce a verified win on this slice

## Kernel-shape check

From:

- `.cache/kayak/profile_document_proxy_stage2_dispatch_compare_browsecomp_gold.tsv`

Measured scorer-only means:

- current scorer: `0.0007464237288135593 s`
- legacy per-document-dispatch reimplementation: `0.0007311225790432047 s`
- document-token-outer alternative: `0.0007400546218487395 s`

Interpretation:

- the three variants are all in the same narrow band on this workload
- the document-token-outer loop is not a clear improvement over the current
  scorer
- a manual dispatch specialization that I prototyped during this loop was also
  not kept, because isolated A/B benchmarking did not verify a stable win

The important point is not that the current scorer is globally optimal; it is
that the obvious local rewrites tested here did not earn production adoption.

## Quiet wrapper note

Quiet-wrapper runs were recorded under:

- `.cache/kayak/bench_quiet/20260415T044026Z`
- `.cache/kayak/bench_quiet/20260415T044700Z`

Both runs had to force past the quiet wait because the host was heavily loaded
by unrelated work. Those logs remain useful as environment records, but they
were too noisy to support a trustworthy before/after production-speed claim by
themselves.

That is why the decision in this trace relies primarily on the tighter
single-process comparison artifacts above.

## Bottom line

What is verified:

- `document_proxy` stage-2 on `browsecomp_plus_gold` is still scorer-bound
- the next obvious non-kernel ideas were debunked:
  - candidate resolution is not the bottleneck
  - final hit assembly is not the bottleneck
  - hybrid-flat exact scoring is slower here
  - the current parallel policy is already at the best tested work-item count
  - a document-token-outer scorer rewrite is not a clear win

What I did **not** verify:

- a new production scorer kernel that is meaningfully faster than the current
  exact path on this slice

Current conclusion:

- keep production exact stage unchanged
- keep the new diagnostic benchmarks, because they make the remaining search
  space explicit
- the next serious speed attempt should target a genuinely different scorer
  kernel, not another stage-2 orchestration tweak
