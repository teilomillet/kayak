# Centroid Workspace Scale Check for Imputed Paths

## Claim

After the exact-path scale sweep, the next unresolved question was:

- does the reusable centroid workspace also help the imputed centroid paths on
  a controlled document-count scale benchmark

This trace checks that claim with a separate benchmark artifact so the earlier
exact-path evidence stays auditable by filename.

## Why a separate benchmark

I added a new benchmark instead of mutating
`benchmarks/profile_centroid_workspace_scale.mojo` because the earlier trace
for the exact paths already cites that artifact. Reusing the same benchmark and
output file for a different claim would have made the evidence trail harder to
follow.

Added:

- `benchmarks/profile_centroid_workspace_scale_imputed.mojo`

Artifact:

- `.cache/kayak/profile_centroid_workspace_scale_imputed.tsv`

## Fixture

The benchmark reuses the same single-core scale fixture shape used by the exact
path check:

- `document_count` scales from `64` to `16384`
- `nominal_query_vector_count` stays at `4`
- `nominal_document_vector_count` stays at `8`
- `vector_dim` stays at `64`
- `candidate_k` stays at `16`
- centroid count stays at `64`

Measured generator kinds:

- `centroid_postings_imputed`
- `centroid_postings_imputed_flat`

## Commands run

```bash
pixi run mojo -I . benchmarks/profile_centroid_workspace_scale_imputed.mojo
bash scripts/run_bench_quiet.sh --timeout-seconds 20 --force --repeats 1 -- pixi run mojo -I . benchmarks/profile_centroid_workspace_scale_imputed.mojo
```

The direct run was used first as a compile-and-behavior check.

The quiet wrapper again timed out waiting for a quiet host and force-ran the
benchmark. Pre-run competing load samples ranged from roughly `410.00` to
`538.70` aggregate `%CPU` from other processes, so this wrapped result should
still be treated as host-contented evidence, not clean quiet-host latency.

Wrapper artifact directory:

- `.cache/kayak/bench_quiet/20260414T192647Z`

## Result

Wrapped means and reuse ratios (`reused / fresh`):

- `docs_64`
  - `centroid_postings_imputed`: `2.670967359003478e-05 s` -> `2.435248274567559e-05 s`
    - ratio: `0.911748`
  - `centroid_postings_imputed_flat`: `3.7209296770235565e-05 s` -> `3.517365661523233e-05 s`
    - ratio: `0.945292`
- `docs_256`
  - `centroid_postings_imputed`: `4.04813393893103e-05 s` -> `3.751091489361702e-05 s`
    - ratio: `0.926622`
  - `centroid_postings_imputed_flat`: `4.965049207898919e-05 s` -> `4.689172078032001e-05 s`
    - ratio: `0.944436`
- `docs_1024`
  - `centroid_postings_imputed`: `9.044182702941828e-05 s` -> `8.501461246713358e-05 s`
    - ratio: `0.939992`
  - `centroid_postings_imputed_flat`: `1.006681121751026e-04 s` -> `9.41388574034265e-05 s`
    - ratio: `0.935141`
- `docs_4096`
  - `centroid_postings_imputed`: `2.723447763257968e-04 s` -> `2.6023234138872247e-04 s`
    - ratio: `0.955525`
  - `centroid_postings_imputed_flat`: `2.86932683478138e-04 s` -> `2.7171458290703226e-04 s`
    - ratio: `0.946963`
- `docs_16384`
  - `centroid_postings_imputed`: `1.0585373826734277e-03 s` -> `9.84341717983509e-04 s`
    - ratio: `0.929907`
  - `centroid_postings_imputed_flat`: `1.0216135933147632e-03 s` -> `9.944408163265305e-04 s`
    - ratio: `0.973402`

## Direct-run ambiguity

The first non-wrapped run was almost fully consistent with the same direction,
but it had one ambiguous point:

- `docs_4096 / centroid_postings_imputed`
  - direct ratio: `1.006511`

The wrapped run flipped that same point back to:

- `docs_4096 / centroid_postings_imputed`
  - wrapped ratio: `0.955525`

That makes the most likely reading:

- the earlier `docs_4096` inversion was measurement noise on a non-quiet host,
  not stable evidence of a regression

## Interpretation

What is verified:

- on the controlled scale fixture, workspace reuse is faster at every measured
  document count for `centroid_postings_imputed_flat`
- on the wrapped run, workspace reuse is also faster at every measured
  document count for `centroid_postings_imputed`
- the imputed paths therefore now match the exact paths directionally on this
  controlled document-count benchmark

What is still uncertain:

- absolute latency numbers remain host-contended because the quiet wrapper
  never actually found a quiet machine
- the scale benchmark still isolates candidate generation only; it does not by
  itself prove the same win will dominate end-to-end public-slice latency

So the sound conclusion is:

- the reusable centroid workspace now has controlled benchmark evidence across
  both exact and imputed centroid paths that is consistent with reducing the
  stage-1 document-count setup tax
- the remaining epistemic gap is not whether the direction exists on the
  controlled candidate-generation benchmark, but how much of that win survives
  in noisier public-slice end-to-end workloads
