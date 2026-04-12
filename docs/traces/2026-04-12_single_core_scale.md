# 2026-04-12 Single-Core Scale Benchmark

## Goal

Implement Phase I1 from
[docs/late_interaction_efficiency_roadmap.md](../late_interaction_efficiency_roadmap.md):

- add one reproducible single-core scaling benchmark
- keep query vectors, document vectors, candidate budget, and stage policy
  explicit
- record measured results instead of repeating the qualitative efficiency claim

## Files Added Or Changed

- `kayak/benchmarks/single_core_scale_fixture.mojo`
- `kayak/benchmarks/stage_aware_json.mojo`
- `benchmarks/single_core_scale.mojo`
- `tests/test_single_core_scale_fixture.mojo`
- `tests/test_stage_aware_benchmark_json.mojo`
- `pyproject.toml`
- `README.md`

## Design Choice

Reason:
- the repo already had a real collection/search-plan benchmark path through
  `StageAwareSearchSummary`
- the missing piece was a fixed-shape synthetic scale fixture and explicit
  stage-density reporting

Decision:
- keep the benchmark on the real collection stack instead of the old
  `workload_matrix` exact-kernel path
- extend `StageAwareSearchSummary` with:
  - nominal query/document vector counts
  - candidate-stage bytes, vectors, and densities
  - exact-stage bytes and vectors
- use a deterministic grouped-concept synthetic workload so the exact
  full-scan path remains easy to validate

## Verification Commands

Targeted correctness checks:

```bash
pixi run test_stage_aware_benchmark_json
pixi run test_single_core_scale_fixture
```

Benchmark smoke run:

```bash
pixi run bench_single_core_scale_raw
```

Repeated benchmark run with quiet-wrapper bookkeeping:

```bash
bash scripts/run_bench_quiet.sh --timeout-seconds 5 --force --repeats 3 -- pixi run bench_single_core_scale_raw
```

## Measurement Context

The synthetic sweep held these parameters fixed:

- `query_count = 8`
- `nominal_query_vector_count = 4`
- `nominal_document_vector_count = 8`
- `vector_dim = 64`
- `final_k = 2`
- `candidate_k = 16`
- `centroid_head_posting_cap = 16`

Corpus sizes swept:

- `64`
- `256`
- `1024`
- `4096`

Measured plans:

- `exact_full_scan`
- `document_proxy`
- `centroid_postings`
- `centroid_heads`
- `centroid_postings_head_auto`

Artifacts:

- JSON output: `.cache/kayak/single_core_scale.json`
- quiet-run logs: `.cache/kayak/bench_quiet/20260412T201937Z/`

## Important Caveat

The repeated quiet-wrapper run did **not** find a genuinely quiet host.

Verified from the captured snapshots:
- another Mojo benchmark process was already consuming about one full core:
  `benchmarks/profile_stage1_blockmax_real_subset.mojo`
- editor and system processes were also active
- the wrapper proceeded only because `--force` was supplied after a short
  timeout

Inference:
- these numbers are useful local evidence
- they are not clean idle-host publication numbers

## Result Snapshot

Values below are from the final emitted
`.cache/kayak/single_core_scale.json` after the repeated run.

### Exact Full Scan

- `64 docs`: `0.0000160s`
- `256 docs`: `0.000049625s`
- `1024 docs`: `0.0001835s`
- `4096 docs`: `0.0007645s`

Observation:
- exact full scan grew by about `47.8x` while document count grew by `64x`

### Document Proxy

- `64 docs`: `0.00002975s`, candidate recall `1.0`
- `256 docs`: `0.0000340s`, candidate recall `1.0`
- `1024 docs`: `0.000050375s`, candidate recall `1.0`
- `4096 docs`: `0.000104375s`, candidate recall `1.0`

Observation:
- latency grew much more slowly than exact full scan on this synthetic slice
- candidate-stage vector count still scaled with documents:
  `64 -> 256 -> 1024 -> 4096`

### Centroid Postings

- `64 docs`: `0.00003525s`, candidate recall `1.0`
- `256 docs`: `0.00004175s`, candidate recall `1.0`
- `1024 docs`: `0.000060375s`, candidate recall `1.0`
- `4096 docs`: `0.0001185s`, candidate recall `1.0`

Observation:
- candidate-stage centroid vector count stayed fixed at `64`
- candidate-stage byte size still grew substantially because postings grew with
  corpus size:
  - `19920`
  - `29026`
  - `66235`
  - `238321`

### Centroid Heads

- `64 docs`: `0.000035375s`, candidate recall `1.0`
- `256 docs`: `0.000039875s`, candidate recall `1.0`
- `1024 docs`: `0.000056125s`, candidate recall `0.5625`
- `4096 docs`: `0.00010275s`, candidate recall `0.125`

### Centroid Postings Head Auto

- `64 docs`: `0.00003575s`, candidate recall `1.0`
- `256 docs`: `0.00003975s`, candidate recall `1.0`
- `1024 docs`: `0.000057875s`, candidate recall `0.5625`
- `4096 docs`: `0.000103s`, candidate recall `0.125`

Observation:
- the capped native candidate engines stayed near-flat in latency relative to
  exact full scan
- they also lost faithfulness as corpus size increased on this workload

## What This Verifies

Verified locally:
- Kayak now has a real single-core scaling benchmark over increasing corpus
  sizes
- the benchmark runs through the current collection/search-plan stack rather
  than a kernel-only shortcut
- the repo can now measure latency, recall, vectors, and bytes together on the
  same artifact
- on this synthetic slice, exact full scan scales much worse in latency than
  `document_proxy` and the native candidate engines

Verified tradeoff:
- lower-latency capped native engines can lose stage-1 recall as the corpus
  grows
- uncapped centroid postings preserve recall on this slice but pay growing
  posting-storage cost

## What This Still Does Not Prove

Not verified by this work:
- under-`200ms` single-core search at multi-billion-token scale
- `6 bytes/vector` storage viability
- any `sqrt(m)` pruning law
- behavior on harder public benchmark families
- a cleaner idle-host benchmark curve

## Takeaway

Phase I1 is now implemented and locally measured.

The repo should stop speaking about single-core efficiency as a purely
qualitative claim. It can now point to one explicit scaling artifact and one
explicit failure mode:

- exact full scan grows much faster than the tighter candidate engines
- tighter candidate engines need explicit recall accounting because speed alone
  is not enough
