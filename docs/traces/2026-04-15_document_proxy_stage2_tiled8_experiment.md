# Document Proxy Stage-2 Tiled8 Experiment

## Claim

After widening the current dim128 exact-stage fast path across measured query
counts, the next remaining stage-2 question was narrower:

- on the real `q=32`, `dim=128` `document_proxy` seam, can a wider tiled
  microkernel beat the current production `tiled4` path

The concrete experiment here was:

- add a benchmark-only `tiled8` flat-query scorer
- compare it against the current production scorer and the existing benchmark
  `tiled4` helper on the same real resolved candidate windows
- do **not** change production unless the result is clear

## Benchmark seam

Extended:

- `benchmarks/profile_document_proxy_stage2_flat_query_tiled4_compare_browsecomp_gold.mojo`

The benchmark now validates exact score equivalence for:

- current production scorer
- benchmark `tiled4` helper
- benchmark `tiled8` helper

on the real BrowseComp-Plus gold resolved candidate windows.

Measured operating point:

- dataset: `BrowseComp-Plus Gold`
- slice: `browsecomp_plus_gold_slice`
- query vectors: `32`
- document vectors: `175`
- `candidate_k = 40`
- mean candidate documents: `40.0`
- mean candidate vectors: `6974.75`

## Raw exploratory run

Command:

```bash
pixi run mojo -I . benchmarks/profile_document_proxy_stage2_flat_query_tiled4_compare_browsecomp_gold.mojo
```

Exploratory means:

- current scorer: `0.0009621514853647881 s`
- `build_flat_query_dim128`: `3.096404315422527e-06 s`
- prebuilt `tiled4`: `0.0010900445657116872 s`
- `build_flat_query + tiled4`: `0.0010830939325842696 s`
- prebuilt `tiled8`: `0.001065778642455685 s`
- `build_flat_query + tiled8`: `0.0008668464864864865 s`

Exploratory interpretation:

- `build_flat_query + tiled8` looked about `9.9%` faster than current
- but that same run also had prebuilt `tiled8` slower than current

That inconsistency was enough to reject the raw run as decision-quality evidence
by itself.

## Quiet-wrapper rerun

Command:

```bash
bash scripts/run_bench_quiet.sh --repeats 2 --timeout-seconds 20 --force -- \
  pixi run mojo -I . benchmarks/profile_document_proxy_stage2_flat_query_tiled4_compare_browsecomp_gold.mojo
```

Artifact:

- `.cache/kayak/bench_quiet/20260415T115231Z/`

Host note:

- the quiet wrapper never obtained a quiet host
- both runs were force-started after timeout
- competing host CPU stayed above `1000%`
- another Mojo benchmark process was active during the rerun

That means these logs are still host-contended. They are useful for direction,
not for fine-grained latency claims.

## Quiet-wrapper run 1

Source:

- `.cache/kayak/bench_quiet/20260415T115231Z/run_1.txt`

Measured means:

- current scorer: `0.0008770770332480818 s`
- prebuilt `tiled4`: `0.0007891824870466321 s`
- `build_flat_query + tiled4`: `0.0009011533161953728 s`
- prebuilt `tiled8`: `0.0010148275566966196 s`
- `build_flat_query + tiled8`: `0.000948411592241768 s`

Derived ratios versus current:

- prebuilt `tiled4`: `0.899787`
- `build_flat_query + tiled4`: `1.027451`
- prebuilt `tiled8`: `1.157056`
- `build_flat_query + tiled8`: `1.081332`

Interpretation:

- run 1 does **not** support rolling out `tiled8`
- both `tiled8` measurements lost to current

## Quiet-wrapper run 2

Source:

- `.cache/kayak/bench_quiet/20260415T115231Z/run_2.txt`

Run 2 was interrupted before the final `build_flat_query + tiled8` section
finished, so it is incomplete. The partial measurements recorded before
termination were:

- current scorer: `0.0008832403801576264 s`
- prebuilt `tiled4`: `0.0008217377823881973 s`
- `build_flat_query + tiled4`: `0.0010155288738183312 s`
- prebuilt `tiled8`: `0.0007571002094240838 s`

Derived partial ratios versus current:

- prebuilt `tiled4`: `0.930367`
- `build_flat_query + tiled4`: `1.149776`
- prebuilt `tiled8`: `0.857185`

Interpretation:

- run 2 does show a prebuilt `tiled8` win
- but the build-inclusive `tiled8` result is missing, so the run cannot settle
  the production question

## What this does and does not establish

What is verified:

- the benchmark-only `tiled8` scorer is exact on this seam
- the current host was too contended to make a narrow production decision from
  one exploratory run
- the repeated evidence is mixed rather than clearly favorable

What is **not** verified:

- a stable build-inclusive `tiled8` win over the current production scorer
- a planner-visible stage-2 improvement large enough to justify rollout

## Decision

Do **not** productionize `tiled8` from this evidence.

Reason:

- the exploratory raw run suggested a possible win
- quiet-wrapper run 1 contradicted that and showed both `tiled8` measurements
  losing to current
- quiet-wrapper run 2 was incomplete and therefore insufficient to rescue the
  production case

That is not a strong enough basis for changing production code.

## Current conclusion

- keep the production exact-stage kernel unchanged
- keep the extended benchmark because it makes the next kernel comparison cheap
- if stage-2 work continues, require a tighter benchmark harness or a clearly
  different kernel idea before attempting another rollout
