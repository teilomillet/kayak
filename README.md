# kayak

`kayak` is a Mojo-first late-interaction retrieval engine.

The current scaffold is intentionally narrow:
- encoder output is treated as an external boundary
- indexing and exact MaxSim scoring live in Mojo
- CPU exact search is the first verified path
- benchmarks and tests are first-class, not an afterthought

## Current Layout

- `kayak/contracts/`: validated query/document contracts
- `kayak/numeric/`: centralized scalar aliases and storage-format constants
- `kayak/index/`: packed index layout and optional flat `dim128` document layouts
- `kayak/scoring/`: exact MaxSim scoring kernels
- `kayak/runtime/`: backend boundary, CPU backend first
- `kayak/search/`: top-k search orchestration
- `kayak/verifier/`: optional candidate-window reranking and verifier pipeline
- `kayak/benchmarks/`: deterministic workload profiles and proxy benchmark tasks
- `kayak/eval/`: judged tasks and lightweight retrieval metrics
- `kayak/interop/`: Python bridge for external encoders and real public subsets
- `kayak/storage/`: persisted judged tasks, packed indexes, and optional derived layouts
- `benchmarks/`: runnable benchmark entrypoints
- `python/`: small Python bridge modules for ColBERT and dataset loading
- `tests/`: runnable unit-test entrypoints using `std.testing.TestSuite`

## Why This Shape

This layout is justified by the current project goal:
- keep the encoder boundary swappable for MAX or another transformer stack
- keep the retrieval core in Mojo
- keep scalar choices centralized so vector/score dtypes can evolve without a rewrite
- keep hot paths isolated so they are easy to profile and later replace with GPU kernels

The code keeps vector counts explicit because search quality and systems cost both depend on:
- query vector count
- document vector count
- related sparse-attention or pruning budgets

## Benchmark Coverage

The benchmark layer now has two complementary pieces:
- workload profiles for system timings across benchmark families
- tiny judged proxy tasks for fast retrieval-quality checks

The included slices are inspired by public benchmark families that are relevant to late interaction:
- `LoTTE`: domain-specific forum retrieval in the ColBERT ecosystem
- `BEIR`: heterogeneous factual retrieval across domains
- `MS MARCO`: short passage retrieval
- `BRIGHT`: reasoning-heavy retrieval
- `MIRACL`: multilingual retrieval

Important epistemic boundary:
- these shipped tasks are proxies, not official benchmark reproductions
- they are intended to keep the code runnable, fast, and easy to scale later
- official full-benchmark claims still require running the public datasets and their evaluation protocols

The source rationale for those families is recorded in [docs/benchmark_rationale.md](/Users/teilomillet/Code/kayak/docs/benchmark_rationale.md).

## Profiling Benchmarks

The repo now has profiling-oriented microbenchmarks that separate:

- raw `dot_product`
- per-document exact MaxSim
- backend `score_all`
- full `search_exact`

These benches sweep explicit shapes so vector count stays first-class in the output.
The current CPU path uses a SIMD `dot_product` kernel, a narrow `128`-dim
fast path for ColBERT-shaped embeddings, and vector-balanced document
partitioning for larger exact-search workloads.
Both CPU optimizations are now explicitly configurable through
`ExactScoringConfig`, so you can disable the `128`-dim fast path or
document-level parallel scoring when profiling or comparing kernels.
Parallel work-item oversubscription can also be disabled explicitly when you
want a strict `worker_count` partitioning policy, and the work-item count can
be overridden directly for scalability sweeps and USL fitting.

## Robustness Layer

The repo now includes a small robustness layer inspired by property-first testing and mutation-quality checks:

- `tests/test_battle.mojo`: randomized differential and metamorphic checks for the late-interaction core
- `tests/test_storage_invariants.mojo`: corruption and compatibility checks for persisted artifacts
- `tests/test_score_partitions.mojo`: vector-balanced partitioning checks for parallel exact scoring
- `tests/test_eval_battle.mojo`: metric reference and evaluation invariants
- `python/scripts/mutation_smoke.py`: curated mutation-smoke harness for core kernels, metrics, and storage guards

This is documented in [docs/robustness_testing.md](/Users/teilomillet/Code/kayak/docs/robustness_testing.md).

## Real Subset Bridge

The first real public end-to-end path uses:
- `colbert-ai` for ColBERTv2 token embeddings on CPU
- Mojo for packing, exact search, and evaluation
- small `BEIR/SciFact` and `BEIR/FIQA` subsets as real benchmark slices
- repo-local storage so repeated runs can reload encoded tasks and packed indexes

This is a deliberate first real subset, not a claim of full benchmark reproduction.
`LoTTE` remains a target, but the straightforward official loader path currently pulls a 3.58 GB archive, which is too heavy for the fast smoke workflow this repo wants.

The persisted artifacts live under `.cache/kayak/scifact_real_subset/` and `.cache/kayak/fiqa_real_subset/`. The manifest records:
- storage format version
- vector scalar type
- dataset id
- model name
- judged task payload
- packed index payload

Storage format `v2` keeps manifests and lightweight metadata in text, but stores
the hot vector payloads in binary little-endian form.
That is a deliberate compromise:
- metadata stays easy to inspect by eye
- vector payloads stop paying TSV parse and size overhead on every reload
- legacy `v1` text payloads still load for compatibility

The repo also supports an optional persisted `hybrid_flat_dim128_index` artifact.
This is a derived layout for `128`-dim document embeddings:
- it keeps `doc_ids` and `doc_offsets`
- it stores document token values as one flat scalar buffer
- it is opt-in, not the default exact-search path

For the same late-interaction path, the repo also supports an optional
`FlatQueryDim128` query layout.
This still preserves the full multi-vector query representation.
It changes query memory layout only; it does not collapse retrieval into a
single-vector search.

That choice is deliberate.
The current measurements support keeping it as a first-class optional artifact,
but they do not yet support silently replacing the default CPU search path.

## Verifier Stage

The repo now includes a narrow third-stage verifier interface:
- `no_verifier`: preserves the current exact-search path
- `exact_late_interaction_verifier(candidate_k)`: reranks a candidate window in Mojo with exact MaxSim

This stage is intentionally vector-only in `v0.1`.
That is an epistemic boundary, not a missing buzzword: the current stored artifacts contain token embeddings and ids, but not the raw text needed for an honest cross-encoder reranker.
If we want a text-level verifier later, the storage layer must first persist the necessary text payload explicitly.

## Commands

Run the demo:

```bash
pixi run demo
pixi run demo_scifact
pixi run demo_fiqa
```

Run tests:

```bash
pixi run test_index
pixi run test_maxsim
pixi run test_eval
pixi run test_proxies
pixi run test_python_bridge
pixi run test_storage
pixi run test_storage_compat
pixi run test_storage_invariants
pixi run test_score_partitions
pixi run test_hybrid_flat_dim128
pixi run test_verifier
pixi run test_battle
pixi run test_eval_battle
```

Run the exact CPU benchmark:

```bash
pixi run bench_exact
pixi run bench_profile_exact
pixi run bench_profile_cpu_micro
pixi run bench_profile_cpu_structural
pixi run bench_profile_cpu_structural_real_subset
pixi run bench_profile_cpu_hybrid_real_subset
pixi run bench_profile_cpu_verifier_real_subset
pixi run bench_profile_storage_real_subset
pixi run bench_profile_cpu_configs
pixi run bench_profile_cpu_usl
pixi run fit_usl
```

For lower-noise comparisons on a busy machine, use the quiet benchmark wrapper:

```bash
bash scripts/run_bench_quiet.sh --repeats 3 --max-other-cpu 40 -- pixi run bench_scifact
```

The default `pixi run bench_*` tasks for performance-sensitive benchmarks now
use this quiet wrapper automatically. Use the corresponding `*_raw` tasks only
for quick smoke checks when you explicitly do not want quiet-run gating.

Run the workload matrix and the proxy evaluation matrix:

```bash
pixi run bench_matrix
pixi run eval_matrix
pixi run bench_scifact
pixi run bench_fiqa
pixi run bench_real_subset_policies
pixi run bench_real_subset_breakdown
```

Run the curated mutation-smoke check:

```bash
pixi run mutate_smoke
```

Compile the package:

```bash
pixi run package_mojo
```

The compiled Mojo package is written to `dist/kayak.mojopkg`.
