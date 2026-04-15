# Quiet Benchmark Multi-Section Summaries

## Claim

The quiet benchmark wrapper was good enough for single-metric runs, but it was
not good enough for the newer stage and serving benchmarks that emit many named
benchmark sections in one invocation.

Before this change, `scripts/run_bench_quiet.sh` only did two things reliably:

- wait for a quieter host before each repeat
- summarize repeated runs when exactly one `Mean:` appeared per run

For multi-section runs it fell back to:

- `summary runs=N multi_section=1 scalar_summary=skipped`

That was not enough for decision-grade repeated measurements because it forced
manual pairing of section labels to repeated means. The sound fix was:

- keep the existing single-metric behavior unchanged
- parse ordered `== section ==` labels for multi-section runs
- only summarize cross-run medians when section ordering is actually verified

## Why this was justified

The recently added centroid prepared-snapshot mirror benchmark is exactly the
kind of benchmark that exposed the gap:

- benchmark:
  [profile_prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold.mojo](/Users/teilomillet/Code/kayak/benchmarks/profile_prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold.mojo:1)
- it emits `22` benchmark sections in one run
- the earlier wrapper therefore preserved raw logs, but not machine-readable
  cross-run per-section medians

That meant the benchmark seam existed, but the evidence lane was still
partially manual.

## Implementation

Updated wrapper:

- [run_bench_quiet.sh](/Users/teilomillet/Code/kayak/scripts/run_bench_quiet.sh:1)

What changed:

- the wrapper now accepts `KAYAK_BENCH_QUIET_LOG_ROOT` so tests or isolated
  workflows can redirect artifacts without mutating repo state
- each run now writes `run_<n>_sections.tsv` with:
  - `section_index`
  - normalized `label`
  - per-run `mean`
- the wrapper now verifies section alignment across repeats before producing a
  cross-run summary
- when alignment holds, the wrapper writes:
  - `section_summary.tsv`
  - `section_summary.txt`
- when alignment does **not** hold, the wrapper refuses to summarize and writes
  `section_alignment_error.txt`

Reason:

- medians are only meaningful if run `i` and run `j` refer to the same section
- that alignment must be checked, not assumed

Added focused parser checks:

- fixture:
  [run_bench_quiet_multisection_fixture.sh](/Users/teilomillet/Code/kayak/tests/fixtures/run_bench_quiet_multisection_fixture.sh:1)
- test:
  [test_run_bench_quiet_multisection.sh](/Users/teilomillet/Code/kayak/tests/test_run_bench_quiet_multisection.sh:1)

What those checks verify:

- aligned multi-section output produces `section_summary.tsv`
- deliberately re-ordered sections do **not** produce a summary and instead
  surface an alignment failure

## Validation

Commands run:

```bash
bash tests/test_run_bench_quiet_multisection.sh

bash scripts/run_bench_quiet.sh --repeats 2 --max-other-cpu 40 --timeout-seconds 10 --force -- \
  pixi run mojo -I . benchmarks/profile_prepared_snapshot_centroid_segment_mirror_reuse_browsecomp_gold.mojo

KAYAK_BENCH_QUIET_LOG_ROOT="$(mktemp -d /tmp/kayak-bench-quiet-single.XXXXXX)" \
  bash scripts/run_bench_quiet.sh \
    --repeats 2 \
    --max-other-cpu 100000 \
    --quiet-checks 1 \
    --sleep-seconds 0 \
    --timeout-seconds 1 \
    -- \
    bash -lc 'printf "== single ==\n"; printf "Benchmark Report (s)\n"; printf "Mean: 1.25\n"'
```

Observed results:

- `bash tests/test_run_bench_quiet_multisection.sh`: passed
- real centroid mirror benchmark:
  - parsed `22` sections on run 1
  - parsed `22` sections on run 2
  - produced:
    - `.cache/kayak/bench_quiet/20260415T183312Z/section_summary.tsv`
    - `.cache/kayak/bench_quiet/20260415T183312Z/section_summary.txt`
- single-metric compatibility probe:
  - still reported the legacy scalar summary:
    `summary runs=2 min=1.25 median=1.25 mean=1.25 max=1.25`

## Real benchmark artifact

Updated quiet-wrapper directory:

- `.cache/kayak/bench_quiet/20260415T183312Z`

Wrapper summary:

- `.cache/kayak/bench_quiet/20260415T183312Z/summary.txt`

This now reports:

- `summary runs=2 multi_section=1 section_count=22 ...`

The new section summary TSV makes the centroid mirror comparison inspectable
without manual log pairing. A few representative rows:

- prepare snapshot without mirrors:
  - run 1: `0.008230473684210525`
  - run 2: `0.007984473684210527`
  - median: `0.00810747`
- prepare snapshot with mirrors:
  - run 1: `0.009146315789473683`
  - run 2: `0.009049684210526316`
  - median: `0.009098`
- `centroid_postings_imputed` direct prepared search:
  - baseline median: `0.0007663`
  - mirror median: `0.000757705`
- `centroid_postings_imputed_flat` direct prepared search:
  - baseline median: `0.000602304`
  - mirror median: `0.000595902`

The same artifact also makes the noisy rows obvious instead of hiding them in
two long raw logs. For example:

- `centroid_postings_imputed_flat` batch worker_count `8`
  - baseline runs: `0.0073939444444444434`, `0.008295235294117647`
  - mirror runs: `0.007834411764705881`, `0.01198975`

That is exactly the kind of variance we needed the wrapper to surface
mechanically.

## Conclusion

What is now verified:

- the quiet wrapper can produce machine-readable repeated summaries for
  multi-section benchmarks
- those summaries are only emitted when section identity is verified across
  repeats
- the legacy single-metric path still behaves as before

Why this matters:

- the repo can now treat many of the newer multi-section stage and serving
  benchmarks as proper repeated measurement lanes instead of partially manual
  log interpretation exercises
- this does not make the underlying host quieter, but it does make noisy
  evidence much more inspectable and falsifiable
