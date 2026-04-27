# 2026-04-27: GPU I8 Candidate-Generation Breakdown

## Claim

The next GPU optimization should be guided by a candidate-generation breakdown,
not by another local CPU loop edit.

Reason: the previous hot-path experiments preserved correctness but did not
produce a stable speed win across default and wide sweeps. A profiler-friendly
boundary is needed before deciding whether to reduce posting work on CPU, fuse
substeps, or design a GPU candidate-generation primitive.

## Change

Added a benchmark-only CPU i8 candidate-generation breakdown:

- `kayak/search/plaid_i8_candidate_profile_types_dim128.mojo`
- `kayak/search/plaid_i8_candidate_profile_dim128.mojo`
- `python/scripts/profile_gpu_i8_candidate_generation_breakdown.py`
- `profile_gpu_i8_candidate_generation_breakdown_raw`
- `profile_gpu_i8_candidate_generation_breakdown`

The profiler reports these substeps:

- full production candidate generation
- centroid scoring
- centroid selection
- posting accumulation
- final candidate top-k

It also reports the work shape explicitly:

- query count and query vector count
- document count, regular document vector count, and total document vectors
- centroid count and centroids per query vector
- candidate window size
- selected centroid count
- posting visits
- touched documents
- output candidate count

Reason: these fields make vector count and posting fanout visible beside timing,
which is necessary before moving any candidate-generation work to GPU.

## Measurement

Commands:

```bash
pixi run python -m py_compile python/scripts/profile_gpu_i8_candidate_generation_breakdown.py
pixi run python python/scripts/profile_gpu_i8_candidate_generation_breakdown.py --case smoke:document_count=32,document_vector_count=8,query_count=1,query_vector_count=4,candidate_k=16 --measurement-iterations 1 --output .cache/kayak/gpu_i8_candidate_generation_breakdown/smoke.json
pixi run env PYTHONPATH=python python -m unittest python/tests/test_gpu_i8_rerank_contract.py python/tests/test_fastplaid_speed_track.py
pixi run profile_gpu_i8_candidate_generation_breakdown
```

Artifacts:

- smoke report:
  `.cache/kayak/gpu_i8_candidate_generation_breakdown/smoke.json`
- quiet log: `.cache/kayak/bench_quiet/20260427T091452Z`
- quiet report:
  `.cache/kayak/gpu_i8_candidate_generation_breakdown/summary.json`

## Results

Quiet wide breakdown status: `ok`, `5 / 5` cases.

| case | query vectors | document vectors/doc | candidate k | full candidate batch s | centroid selection batch s | posting accumulation batch s | final top-k batch s | posting visits | touched docs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `candidate512` | `8` | `16` | `512` | `0.0000002822967567045884` | `0.000050936771584373354` | `0.000056230499402819874` | `0.00008878472673127708` | `30942` | `8116` |
| `candidate1024` | `8` | `16` | `1024` | `0.0000005185445138136711` | `0.000050749678637494926` | `0.0001033273074998432` | `0.0001994089152741093` | `60433` | `16182` |
| `query_vectors32` | `32` | `16` | `256` | `0.00067530914` | `0.00020648110901083611` | `0.0002133741182704914` | `0.00006485040433333334` | `124625` | `32463` |
| `doc_vectors64` | `8` | `64` | `256` | `0.000228263237502696` | `0.00005164550476899205` | `0.00010008920845060168` | `0.00006344441133333333` | `104893` | `8192` |
| `query_batch4` | `8` | `16` | `256` | `0.00037112947427510594` | `0.00010341687325687252` | `0.00011238458935632989` | `0.00012734926266666667` | `61809` | `16202` |

Full-window rows:

- `candidate512` and `candidate1024` use the production full-window shortcut,
  so full candidate generation returns all document ids instead of running the
  decomposed PLAID path.
- The decomposed substep timings in those rows are explanatory only; they do
  not describe the production full-window path.

Non-full rows:

- `query_vectors32`: centroid selection and posting accumulation each account
  for about `0.31x` of full candidate time.
- `doc_vectors64`: posting accumulation accounts for about `0.44x` of full
  candidate time, with final top-k about `0.28x`.
- `query_batch4`: posting accumulation is about `0.30x`, centroid selection is
  about `0.28x`, and final top-k is about `0.34x`.

## Interpretation

Verified:

- the benchmark-only binding compiles and runs through the prepared i8 index
  handle
- the report records explicit vector counts and posting fanout
- non-full candidate generation is dominated by centroid selection, posting
  accumulation, and final candidate top-k rather than centroid scoring

Not claimed:

- this does not prove that GPU candidate generation is the next implementation
  step
- this does not prove the decomposed substeps sum to production candidate time
- this does not change public search behavior

## Decision

Keep the profiling scaffold and use it to guide the next optimization.

Reason: the evidence points to work-shape problems: selected centroid fanout,
posting visits, touched document count, and final candidate top-k. The next
implementation should either reduce that work before it exists or make those
substeps a first-class primitive. Moving the current loop to GPU directly would
mix algorithmic fanout with device-transfer and synchronization questions.

## Validation

Observed:

- Python syntax check passed
- smoke profiler status: `ok`
- focused GPU/FastPlaid tests: `49 / 49` passed
- quiet wide breakdown status: `ok`
- quiet wide breakdown parsed sections: `25`

## Follow-Up: Workspace And Unordered Windows

Added benchmark-only workspace and unordered candidate-window measurements.

Latest quiet artifact:

- quiet log: `.cache/kayak/bench_quiet/20260427T104620Z`
- report:
  `.cache/kayak/gpu_i8_candidate_generation_breakdown/policy_summary.json`

Results on the policy-budget wide non-full rows:

| case | ordered candidate batch s | workspace / ordered | unordered / ordered | unordered set agreement |
| --- | ---: | ---: | ---: | ---: |
| `query_vectors32` | `0.00024551876988711447` | `1.0120847542301457` | `0.8955583631862651` | `1.0` |
| `doc_vectors64` | `0.00018408207623251066` | `1.2769317380880978` | `0.8733621280772922` | `1.0` |
| `query_batch4` | `0.0002523595122390538` | `1.0342718331121212` | `0.7549527139559552` | `1.0` |

Interpretation:

- the reusable full workspace was falsified as a speed optimization on these
  rows, so it remains benchmark-only
- unordered retained candidate sets are a useful internal GPU-pipeline option
  because rerank consumes the candidate set rather than approximate-score order
- ordered candidate windows remain the public/default API
