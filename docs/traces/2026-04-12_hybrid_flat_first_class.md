# Hybrid Flat Dim128 Trace

Date: `2026-04-12`

## Scope

This trace records the step where the `dim128` hybrid flat layout stopped being benchmark-only code and became a first-class optional artifact in `kayak`.

What changed:
- production `HybridFlatDim128Index` type under `kayak/index/`
- production scoring and search functions under `kayak/scoring/` and `kayak/search/`
- persisted `hybrid_flat_dim128_index` artifact under `kayak/storage/`
- correctness and storage-invariant tests against that artifact

## Validation

Commands:

```bash
pixi run test_hybrid_flat_dim128
pixi run test_storage
pixi run test_storage_compat
pixi run test_storage_invariants
```

All passed.

Verified:
- hybrid flat scoring still matches the default exact backend
- hybrid flat search still matches default `search_exact`
- persisted hybrid flat artifacts roundtrip correctly
- corrupted hybrid flat scalar payloads are rejected

## Benchmark Command

```bash
pixi run bench_profile_cpu_hybrid_real_subset_raw
```

This was an exploratory raw run on the stored SciFact and FIQA real subsets.
It is useful for shape and regression tracking, not for high-confidence absolute CPU claims.

## Artifact Size

Persisted hybrid flat artifacts:

| Dataset | On-disk size |
| --- | ---: |
| SciFact | `4.7M` |
| FIQA | `5.0M` |

Representative contents:
- `manifest.tsv`
- `doc_ids.tsv`
- `doc_offsets.tsv`
- `token_values.bin`

## Result

### SciFact

| Section | Mean (s) |
| --- | ---: |
| `build_hybrid_flat_dim128_index` | `0.0004580996800529567` |
| `load_stored_hybrid_flat_dim128_index` | `0.0011516594368340943` |
| `nested score_all` | `0.00043832504269854824` |
| `hybrid score_all` | `0.0004208643900410624` |
| `nested search_exact` | `0.00042320427994902294` |
| `hybrid search_exact` | `0.00043091822782084414` |

### FIQA

| Section | Mean (s) |
| --- | ---: |
| `build_hybrid_flat_dim128_index` | `0.00048609689501026945` |
| `load_stored_hybrid_flat_dim128_index` | `0.0011758276661514681` |
| `nested score_all` | `0.0004460117823479006` |
| `hybrid score_all` | `0.0004523247579212743` |
| `nested search_exact` | `0.0004486438773250664` |
| `hybrid search_exact` | `0.00045308732190561` |

## Interpretation

What the evidence supports:
- the hybrid flat `dim128` path is now a real, persisted, test-covered option
- direct load of the hybrid artifact is cheap enough to be practical
- the artifact size is compact and almost entirely in the flat scalar buffer
- the path is appropriate infrastructure for future backend specialization and GPU-oriented layouts

What the evidence does not support:
- switching the default CPU exact search path to hybrid flat right now
- claiming a universal search-speed win from the current CPU kernels

Important nuance:
- if you already have a nested `PackedIndex` in memory, rebuilding the hybrid flat layout is cheaper than loading it from disk in this benchmark
- the reason to persist the artifact is different: it gives a direct flat recovery path from storage without reconstructing nested token vectors first

Inference from this trace plus the earlier storage `v2` trace:
- hybrid flat load (`~1.15 ms`) is materially cheaper than loading the full packed nested index (`~4.8-5.0 ms`) on these same real subsets
- that makes the persisted artifact useful for layout-specific serving paths, even though the current CPU scorer is not yet a default-search win

Current conclusion:
- keep the hybrid flat artifact explicit and opt-in
- use it for profiling, backend experiments, and future layout-specialized serving
- do not replace the default exact path until stronger search-time evidence exists
