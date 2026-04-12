# Flat Query Dim128 Trace

Date: `2026-04-12`

## Scope

This trace records the first query-side layout specialization for the hybrid
flat `dim128` path.

What changed:
- production `FlatQueryDim128` type under `kayak/contracts/`
- production hybrid flat scorers and search entrypoints that accept flat queries
- tests proving equivalence against the existing multi-vector query path

Important epistemic boundary:
- this is still exact multi-vector late interaction
- every query token remains part of MaxSim
- no query is collapsed to a single embedding

## Validation

Commands:

```bash
pixi run test_hybrid_flat_dim128
pixi run test_storage
pixi run test_storage_invariants
```

All passed.

Verified:
- flat-query hybrid scores match nested-query hybrid scores
- flat-query hybrid hits match nested-query hybrid hits
- existing storage paths remain valid

## Benchmark Command

```bash
pixi run bench_profile_cpu_hybrid_real_subset_raw
```

This was an exploratory raw run on the stored SciFact and FIQA real subsets.
Use it for shape and regression tracking, not as a final absolute-performance claim.

## Result

### SciFact

| Section | Mean (s) |
| --- | ---: |
| `build_hybrid_flat_dim128_index` | `0.0005027334593572778` |
| `load_stored_hybrid_flat_dim128_index` | `0.001044479188405797` |
| `build_flat_query_dim128` | `2.1328084254445734e-06` |
| `nested score_all` | `0.00048475636256273985` |
| `hybrid score_all` | `0.0005274087203119462` |
| `hybrid score_all with flat query` | `0.0004974976613657625` |
| `nested search_exact` | `0.0004995772912982499` |
| `hybrid search_exact` | `0.0005003106159083876` |
| `hybrid search_exact with flat query` | `0.0005017280669599218` |

### FIQA

| Section | Mean (s) |
| --- | ---: |
| `build_hybrid_flat_dim128_index` | `0.0005214873718429607` |
| `load_stored_hybrid_flat_dim128_index` | `0.0011095037369207772` |
| `build_flat_query_dim128` | `2.1387901198327736e-06` |
| `nested score_all` | `0.0004662769666100736` |
| `hybrid score_all` | `0.00047826146010186753` |
| `hybrid score_all with flat query` | `0.0005868761018609207` |
| `nested search_exact` | `0.0005292084413426387` |
| `hybrid search_exact` | `0.0005967183703703703` |
| `hybrid search_exact with flat query` | `0.0005299905527638191` |

## Interpretation

What the evidence supports:
- flattening the query is almost free to build on these workloads (`~2.1 µs`)
- query-side flattening preserves exact multi-vector behavior
- on SciFact, flat-query hybrid scoring reduces some hybrid overhead, but does not produce a clear search win
- on FIQA, flat-query hybrid search recovers most of the regression from the older hybrid search path and lands near the nested baseline

What the evidence does not support:
- claiming a universal flat-query search win on CPU
- switching the default exact path to hybrid-flat-plus-flat-query

Current conclusion:
- keep `FlatQueryDim128` as an explicit optimization hook for multi-vector serving
- use it together with the persisted hybrid flat document artifact when profiling larger shapes or backend-specific paths
- continue tuning the hybrid path only if the next measurements are on richer multi-vector workloads rather than single-vector-like small slices
