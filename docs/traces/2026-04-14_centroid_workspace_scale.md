# Centroid Workspace Scale Check

## Claim

The remaining hypothesis after the first public-slice A/B was narrow:

- if the centroid-family reusable workspace really removes an
  `O(document_count)` setup tax
- then a controlled benchmark that scales document count while keeping the
  exact retrieval policy fixed should show the reusable path improving as the
  corpus grows

This trace checks that claim directly instead of inferring it from a mixed
end-to-end slice.

## Why this fixture

I used the single-core scale fixture rather than another public slice because
it changes one main axis at a time:

- `document_count` scales from `64` to `16384`
- `nominal_query_vector_count` stays at `4`
- `nominal_document_vector_count` stays at `8`
- `vector_dim` stays at `64`
- `candidate_k` stays at `16`
- centroid count stays at `64`

That makes it a better probe for the removed per-query dense setup cost than a
public benchmark where retrieval mix, posting structure, and slice-specific
variance all move together.

## Benchmark surface

Added:

- `benchmarks/profile_centroid_workspace_scale.mojo`

The benchmark compares:

- `fresh_workspace`
  - `candidate_generation_for_plan(...)`
- `reused_workspace`
  - `candidate_generation_for_plan_with_workspace(...)`

Measured generator kinds:

- `centroid_postings`
- `centroid_postings_flat`

Artifacts:

- `.cache/kayak/profile_centroid_workspace_scale.tsv`
- `.cache/kayak/bench_quiet/20260414T183857Z`

## Commands run

```bash
pixi run mojo -I . benchmarks/profile_centroid_workspace_scale.mojo
bash scripts/run_bench_quiet.sh --timeout-seconds 20 --force --repeats 1 -- pixi run mojo -I . benchmarks/profile_centroid_workspace_scale.mojo
```

The quiet wrapper timed out waiting for a quiet host and force-ran the command.
Observed pre-run competing load samples ranged from roughly `539.60` to
`836.60` aggregate `%CPU` from other processes, so the wrapped run should be
treated as host-contented evidence, not decision-grade quiet-host evidence.

## Result

Quiet-wrapper artifact:

- `.cache/kayak/profile_centroid_workspace_scale.tsv`

Measured means and reuse ratios (`reused / fresh`):

- `docs_64`
  - `centroid_postings`: `1.5347313142581107e-05 s` -> `1.3211249978788033e-05 s`
    - ratio: `0.860818`
  - `centroid_postings_flat`: `2.390595356400759e-05 s` -> `2.2136475177304967e-05 s`
    - ratio: `0.925982`
- `docs_256`
  - `centroid_postings`: `1.8486847531518794e-05 s` -> `1.589932168788592e-05 s`
    - ratio: `0.860034`
  - `centroid_postings_flat`: `2.7468755367755675e-05 s` -> `2.4804923712749758e-05 s`
    - ratio: `0.903023`
- `docs_1024`
  - `centroid_postings`: `3.175700559664984e-05 s` -> `2.6566192275398825e-05 s`
    - ratio: `0.836546`
  - `centroid_postings_flat`: `4.040003705648025e-05 s` -> `3.5468306449381624e-05 s`
    - ratio: `0.877928`
- `docs_4096`
  - `centroid_postings`: `7.96221761289558e-05 s` -> `7.698323458835021e-05 s`
    - ratio: `0.966857`
  - `centroid_postings_flat`: `8.986404596072876e-05 s` -> `8.291910657520496e-05 s`
    - ratio: `0.922717`
- `docs_16384`
  - `centroid_postings`: `2.781770991264169e-04 s` -> `2.771684706353979e-04 s`
    - ratio: `0.996374`
  - `centroid_postings_flat`: `2.9071625528129865e-04 s` -> `2.5885065409389735e-04 s`
    - ratio: `0.890389`

## Interpretation

What is verified:

- on the controlled scale fixture, workspace reuse is faster at every measured
  document count for `centroid_postings_flat`
- on the same fixture, workspace reuse is also faster at every measured
  document count for `centroid_postings`
- the strongest clear plain-path improvement appears around `docs_1024`
  with ratio `0.836546`

What is still uncertain:

- how much of the remaining spread at `docs_4096` and especially
  `docs_16384 / centroid_postings` is true algorithmic saturation versus host
  contention
- whether the same direction survives across the broader imputed centroid
  family, because this scale benchmark intentionally focused on the plain exact
  variants first

So the sound conclusion is:

- the reusable workspace now has direct benchmark evidence that is consistent
  with reducing the stage-1 document-count setup tax on controlled exact
  centroid paths
- the evidence is stronger than the earlier mixed SciFact public-slice result,
  but the quiet-wrapper run was still host-contended and should not be
  overclaimed as a clean final latency number

## Relation to the earlier trace

This trace does **not** overturn the earlier public-slice note in
`docs/traces/2026-04-14_centroid_stage1_workspace_reuse.md`.

The two traces support different statements:

- `2026-04-14_centroid_stage1_workspace_reuse.md`
  - end-to-end public-slice A/B was mixed, so no broad latency claim was
    justified there
- this trace
  - controlled document-count scaling does show the reusable workspace helping
    on the exact centroid variants, which is the narrower claim the code change
    was meant to test
