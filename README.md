# kayak

`kayak` is a Mojo-first late-interaction retrieval engine.

For developers using the Python SDK, the supported entrypoint is `import kayak`.
The current monorepo keeps the Python SDK and Mojo engine together on purpose.
The documented public SDK boundary is narrower than the full repo surface.

The current scaffold is intentionally narrow:
- encoder output is treated as an external boundary
- indexing and exact MaxSim scoring live in Mojo
- CPU exact search is the first verified path
- benchmarks and tests are first-class, not an afterthought

## Current Layout

- `kayak/contracts/`: validated query/document contracts
- `kayak/numeric/`: centralized scalar aliases and storage-format constants
- `kayak/collections/`: serving-side collection, segment, snapshot, and compaction contracts
- `kayak/filters/`: typed filter-expression contracts for service-side retrieval requests
- `kayak/service/`: canonical service requests and responses, before transport adapters
- `kayak/planning/`: explicit search plans, candidate sets, collection-level explain data
- `kayak/index/`: packed index layout and optional flat `dim128` document layouts
- `kayak/scoring/`: exact MaxSim scoring kernels
- `kayak/runtime/`: backend boundary, CPU backend first
- `kayak/search/`: top-k search orchestration
- `kayak/verifier/`: optional candidate-window rescoring and verifier pipeline
- `kayak/benchmarks/`: deterministic workload profiles and proxy benchmark tasks
- `kayak/eval/`: judged tasks and lightweight retrieval metrics
- `kayak/interop/`: Python bridge for external encoders and real public subsets
- `kayak/storage/`: persisted judged tasks, packed indexes, and optional derived layouts
- `benchmarks/`: runnable benchmark entrypoints
- `python/`: Python bridge modules, explicit late-interaction objects, and a reference exact backend
- `tests/`: runnable unit-test entrypoints using `std.testing.TestSuite`

The Phase A storage boundary for hosted collections is documented in
[docs/architecture/segment_storage.md](docs/architecture/segment_storage.md).
The matching service-boundary draft is documented in
[docs/architecture/service_api.md](docs/architecture/service_api.md).
The tenant-layout and filter-planning note is documented in
[docs/architecture/tenant_layout.md](docs/architecture/tenant_layout.md).
That boundary now includes persisted collection, segment, snapshot, and
document-text-corpus manifest codecs under `kayak/collections/`.
It also includes explicit snapshot resolution and collection storage reporting,
both keyed by an explicit snapshot id rather than a hidden "current snapshot"
convention.

There is now a collection-level explain example for the real SciFact slice:

```bash
pixi run explain_scifact_collection
```

The typed service-boundary contracts now also live under `kayak/service/`.
There is also a machine-readable storage report example for the same collection:

```bash
pixi run report_scifact_collection_storage
```

For the public smoke suite, there is also an aggregated storage-density report:

```bash
pixi run bench_real_subset_collection_storage_json
```

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

## Python Late-Interaction Layer

The repo now includes an additive Python-facing late-interaction layer under
`python/kayak_bridge/` plus a light `python/kayak/` facade for `import kayak`.
The packaging config points Python packaging at the existing `python/` tree
instead of introducing a second `src/kayak` tree on top of the repo's Mojo
package layout.

What it owns:
- explicit `LateQuery`, `LateDocuments`, `LateIndex`, and `LateScores` objects
- explicit `LateQueryBatch` plus batch scoring/search helpers
- explicit layout conversions for `flat_dim128` queries and `hybrid_flat_dim128` indexes
- exact `maxsim` and `search` operations over those objects
- backend capability inspection through `available_backends()` and `backend_info(...)`
- NumPy and PyTorch input ergonomics without pretending the data is a dense `B x T x D` tensor problem
- a pip-installable Python package surface rooted at `import kayak`

What it does not claim yet:
- hidden approximation, implicit layout conversion, or overloaded tensor algebra
- a published wheel that bundles the Mojo toolchain itself

Current backend boundary:
- `numpy_reference`: correctness-oriented NumPy reference path
- `mojo_exact_cpu`: Mojo-backed exact CPU scoring through a compiled Python extension module

Current packaging boundary:
- `pip install .` from a source checkout is verified
- repo-head builds now stage bundled engine sources under
  `kayak_bridge/_engine/kayak` inside the Python distribution
- when Mojo is available at build time, repo-head builds also bundle
  `kayak_bridge/_artifacts/kayak.mojopkg`
- the current package still expects a local Mojo toolchain at runtime for `mojo_exact_cpu`
- early fresh-consumer validation on `2026-04-12` verified published `kayak 0.1.1` for `numpy_reference` through:
  - `python -m pip install kayak` in a fresh Python `3.11` environment
  - `uv add kayak` in a fresh project constrained to Python `>=3.11,<3.12`
  - `pixi add --pypi kayak` in a fresh Pixi project with `python=3.11`
- plain `pixi add kayak` did not work because no conda package was found for `kayak`
- the published package did not contain `kayak_bridge/_artifacts/kayak.mojopkg`
- because of that missing artifact, `mojo_exact_cpu` failed after published installs, including in a fresh Pixi environment that already had `mojo`
- follow-up fresh-consumer validation on `2026-04-12` then verified published
  `kayak 0.1.2` for `mojo_exact_cpu` through:
  - `python -m pip install kayak` in an environment that already had a usable
    `mojo` CLI
  - `pixi add --pypi kayak` in a fresh Pixi project with `python=3.11` and
    `mojo`
- `uv add kayak` remains verified for `numpy_reference`; `mojo_exact_cpu` has
  not yet been re-verified through that consumer path
- local repo-head validation on `2026-04-12` verified `uv build` produced:
  - an sdist that includes the top-level `kayak/` Mojo sources
  - a wheel that includes both `kayak_bridge/_artifacts/kayak.mojopkg` and
    `kayak_bridge/_engine/kayak/...`
- fresh-consumer validation on `2026-04-12` verified `mojo_exact_cpu` from that
  locally built wheel after:
  - `pixi init .`
  - `pixi add python=3.11 mojo`
  - `pixi run python -m ensurepip --upgrade`
  - `pixi run python -m pip install /path/to/kayak-<version>-py3-none-any.whl`

Supported public Python boundary:
- import from `kayak`
- treat `kayak_bridge` as internal and unstable
- treat the top-level Mojo package `kayak/` as engine code, not as the Python SDK

The package-scoped Python README lives at
[python/kayak/README.md](python/kayak/README.md).

The detailed SDK boundary, install paths, and quickstarts are documented in
[docs/python_sdk.md](docs/python_sdk.md).
The product positioning for Kayak Python versus the hosted engine is documented
in [docs/python_sdk_charter.md](docs/python_sdk_charter.md).
The execution plan for that SDK is documented in
[docs/python_sdk_roadmap.md](docs/python_sdk_roadmap.md).

Example:

```python
import kayak

q = kayak.query(query_vectors)
docs = kayak.documents(["doc-a", "doc-b"], document_vectors)
index = docs.pack().to_layout("hybrid_flat_dim128")

scores = kayak.maxsim(q, docs.pack(), backend=kayak.MOJO_EXACT_CPU_BACKEND)
hits = kayak.search(
    q.to_layout("flat_dim128"),
    index,
    k=10,
    backend=kayak.NUMPY_REFERENCE_BACKEND,
)
```

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

The source rationale for those families is recorded in
[docs/benchmark_rationale.md](docs/benchmark_rationale.md).

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

This is documented in [docs/robustness_testing.md](docs/robustness_testing.md).

## Real Subset Bridge

The real public benchmark path now uses:
- `colbert-ai` for ColBERTv2 token embeddings on CPU
- Mojo for packing, exact search, and evaluation
- small `BEIR/SciFact` and `BEIR/FIQA` subsets as real benchmark slices
- an official `LIMIT-small` slice with the full `46`-document corpus and a light `32`-query subset
- light `BrowseComp-Plus` evidence and gold slices built from official decrypted queries, human-verified evidence documents, gold answer documents, and the benchmark's curated hard negatives
- a query-diagnostic benchmark that scores the same BrowseComp ranking against both evidence and gold qrels
- repo-local storage so repeated runs can reload encoded tasks and packed indexes

This is still a deliberate smoke-oriented public suite, not a claim of full benchmark reproduction.
`LoTTE` remains a target, but the straightforward official loader path currently pulls a 3.58 GB archive, which is too heavy for the fast smoke workflow this repo wants.
`BrowseComp-Plus` is also intentionally sliced:
- the official benchmark uses a fixed corpus of about `100k` documents and an agent loop
- the repo keeps the retrieval core honest by using the benchmark's evidence docs, gold docs, and hard negatives
- the repo does not yet claim full agent-benchmark reproduction inside `kayak`

The persisted artifacts live under:
- `.cache/kayak/scifact_real_subset/`
- `.cache/kayak/fiqa_real_subset/`
- `.cache/kayak/limit_small_real_subset/`
- `.cache/kayak/browsecomp_plus_real_subset/`

Machine-readable benchmark artifacts now also land under `.cache/kayak/`:
- `public_real_slice_benchmarks.json`
- `scifact_real_subset_benchmark.json`
- `fiqa_real_subset_benchmark.json`
- `limit_small_real_subset_benchmark.json`
- `browsecomp_plus_evidence_benchmark.json`
- `browsecomp_plus_gold_benchmark.json`
- `public_real_slice_collection_storage.json`
- `public_candidate_window_sweep.json`
- `public_vector_budget_sweep.json`
- `public_partition_policy_benchmarks.json`
- `public_search_breakdown.json`
- `proxy_eval_matrix.json`
- `workload_matrix.json`

The collection mirror path now persists search-native sidecars per sealed
segment alongside the packed late-interaction index. Today those sidecars are:
- `document_proxy/`
- `centroid_postings/`

This is deliberately a staged progression, not a claim of a full native
PLAID/WARP/GEM-class engine yet:
- stage 1 can now use `document_proxy` or `centroid_postings`
- stage 2 now reranks the shortlisted documents with exact late interaction
- stage-1 recall is measured against an exact full-corpus oracle in the public
  candidate-window and vector-budget artifacts

Current epistemic status:
- `exact_full_scan` remains the correctness anchor
- `document_proxy` is still the stronger light baseline on the current public
  slices at equal `candidate_k`
- `centroid_postings` is the first search-native centroid/posting baseline and
  is now benchmarked on the same axes
- heavier native candidate-generation work should now be driven by these
  recall-vs-budget traces instead of by assumption

The next heavier native-index step is documented in
[docs/architecture/native_candidate_generation_next.md](docs/architecture/native_candidate_generation_next.md).

For the BrowseComp-Plus slices, the raw Python tasks are also materialized once at:
- `.cache/kayak/browsecomp_plus_real_subset/python_task_evidence.json`
- `.cache/kayak/browsecomp_plus_real_subset/python_task_gold.json`

The manifest records:
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

Packed-index storage now also has an opt-in `binary_f16_le` payload encoding.
This is intentionally storage-only:
- the manifest still records the runtime scalar type separately
- payloads decode back into the current `VectorScalar` on load
- the exact-search kernels keep the same in-memory semantics
- the lossy tradeoff is explicit and non-default through `save_stored_packed_index_with_encoding(...)`

Current repo measurements from `pixi run bench_profile_storage_real_subset_raw`
show the expected byte win but not a load-latency win yet:
- SciFact packed index: `4,936,043` bytes in `binary_le` vs `2,468,463` bytes in `binary_f16_le`
- FIQA packed index: `5,202,408` bytes in `binary_le` vs `2,601,708` bytes in `binary_f16_le`
- reload stays slightly slower with `binary_f16_le` today because load still expands half precision back to the runtime scalar type

The repo also supports an optional persisted `hybrid_flat_dim128_index` artifact.
This is a derived layout for `128`-dim document embeddings:
- it keeps `doc_ids` and `doc_offsets`
- it stores document token values as one flat scalar buffer
- it is opt-in, not the default exact-search path

For the same late-interaction path, the repo also supports an optional
`FlatQueryDim128` query layout.

The exact late-interaction runtime now has an explicit backend contract in
`kayak/runtime/ExactScoringBackend`.
That keeps the current CPU path unchanged while making the replacement seam for
future GPU scoring explicit in search, evaluation, verifier, and collection
planning code.
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

There is now a narrow BrowseComp clause-text reranker prototype for
benchmarking.
It loads document text from the already-materialized BrowseComp JSON task cache
instead of pretending the generic verifier pipeline has text available.
That is deliberate:
- the prototype measures whether text-aware reranking can recover answer-bearing
  docs already present in the candidate window
- it is not yet the generic default verifier path

## Commands

Run the demo:

```bash
pixi run demo
pixi run demo_scifact
pixi run demo_fiqa
pixi run demo_python_sdk
pixi run demo_python_sdk_mojo
```

Run tests:

```bash
pixi run test_index
pixi run test_maxsim
pixi run test_eval
pixi run test_proxies
pixi run test_python_bridge
pixi run test_python_api
pixi run test_storage
pixi run test_storage_compat
pixi run test_storage_invariants
pixi run test_score_partitions
pixi run test_hybrid_flat_dim128
pixi run test_verifier
pixi run test_battle
pixi run test_eval_battle
```

Install the Python package from a source checkout:

```bash
python -m pip install .
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
pixi run bench_limit_small
pixi run bench_browsecomp_plus
pixi run bench_browsecomp_plus_gold
pixi run bench_browsecomp_plus_diag
pixi run bench_browsecomp_plus_ranks
pixi run bench_browsecomp_plus_clause
pixi run bench_real_subset_policies
pixi run bench_real_subset_breakdown
```

Materialize the BrowseComp-Plus task json explicitly if you want to separate the
plain-Python encoding step from the Mojo benchmark run:

```bash
pixi run build_browsecomp_plus_task_json
```

That command now materializes both BrowseComp-Plus retrieval variants:
- evidence qrels
- gold qrels

Run the curated mutation-smoke check:

```bash
pixi run mutate_smoke
```

Compile the package:

```bash
pixi run package_mojo
```

The compiled Mojo package is written to `dist/kayak.mojopkg`.
