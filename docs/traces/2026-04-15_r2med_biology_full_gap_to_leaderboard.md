# 2026-04-15 R2MED Biology Full Benchmark and Gap to Leaderboard

## Question

Where does Kayak currently sit on one public full benchmark with an externally
visible reference, and is it far from the current top result?

## Why This Benchmark

`R2MED/Biology` was chosen instead of full `BrowseComp-Plus` for the first
full-corpus comparison because the decision could be justified with local
measurements:

- full `BrowseComp-Plus` on the current Python exact path would require about
  `100,195` documents, around `18M` document vectors, roughly `9.2 GB` for the
  packed matrix alone, and roughly `18+ GB` once the Mojo prepared artifact is
  included
- sample encode cost on this machine was about `0.308s` per BrowseComp
  document, which implies a many-hour full build
- `R2MED/Biology` is still reasoning-heavy, publicly hosted, and externally
  comparable, but its full corpus is only `57,359` documents

That makes `R2MED/Biology` the first sound full-corpus benchmark for a real
Kayak-vs-public-result comparison in this environment.

## Implementation Added

Added:

- [directory_snapshot_builder.py](../../python/kayak_bridge/directory_snapshot_builder.py)
  for one-pass `DirectoryLateStore`-compatible packed snapshots
- [r2med_biology_full.py](../../python/kayak_bridge/r2med_biology_full.py)
  for full-corpus R2MED loading, query encoding, exact benchmarking, and run
  export
- [bench_r2med_biology_full.py](../../python/scripts/bench_r2med_biology_full.py)
  as the reproducible benchmark entrypoint
- segmented large prepared-artifact support in
  [prepared_index_storage_artifact.py](../../python/kayak_bridge/prepared_index_storage_artifact.py)
  and
  [_mojo_exact_cpu_bindings.mojo](../../python/kayak_bridge/_mojo_exact_cpu_bindings.mojo)

Why the prepared-artifact change was necessary:

- the first full R2MED attempt successfully built the snapshot, then failed
  inside `prepare_packed_index_from_storage(...)`
- direct inspection showed the generated `token_values.bin` was
  `2,373,852,672` bytes, which is above the signed-32-bit `2,147,483,647`
  boundary
- after segmenting `token_values` into ordered parts and teaching the Mojo
  loader to concatenate them, the exact same full snapshot loaded and searched
  successfully

This was validated with:

- [test_prepared_index_cache.py](../../python/tests/test_prepared_index_cache.py)
- [test_directory_snapshot_builder.py](../../python/tests/test_directory_snapshot_builder.py)
- [test_r2med_biology_full.py](../../python/tests/test_r2med_biology_full.py)
- [test_colbert_encoder.py](../../python/tests/test_colbert_encoder.py)

## Commands

Smoke path:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 1200 -- \
  uv run --python 3.11 python python/scripts/bench_r2med_biology_full.py \
    --artifact-root .cache/kayak/r2med_biology_full_smoke3 \
    --document-limit 2048 \
    --query-limit 1 \
    --warmup-iterations 0 \
    --measurement-iterations 1
```

Full build attempt that exposed the large prepared-artifact failure:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 1200 -- \
  uv run --python 3.11 python python/scripts/bench_r2med_biology_full.py \
    --artifact-root .cache/kayak/r2med_biology_full_main \
    --warmup-iterations 1 \
    --measurement-iterations 3
```

Full rerun after the segmented prepared-artifact fix, reusing the built
snapshot:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 1200 -- \
  uv run --python 3.11 python python/scripts/bench_r2med_biology_full.py \
    --artifact-root .cache/kayak/r2med_biology_full_main \
    --warmup-iterations 1 \
    --measurement-iterations 3
```

## Result

Artifact:

- [.cache/kayak/r2med_biology_full_main/docs_full_queries_full/r2med_biology_full_summary.json](../../.cache/kayak/r2med_biology_full_main/docs_full_queries_full/r2med_biology_full_summary.json)
- [.cache/kayak/r2med_biology_full_main/docs_full_queries_full/r2med_biology_full.run](../../.cache/kayak/r2med_biology_full_main/docs_full_queries_full/r2med_biology_full.run)

Measured full-corpus exact result:

- dataset: `R2MED/Biology`
- corpus: `57,359` documents
- queries: `103`
- total document vectors: `4,636,431`
- mean document vectors per document: about `81`
- metric: `nDCG@10 = 0.09529319923295203`
  - equivalently about `9.53` on the leaderboard-style `0-100` scale
- exact steady-state search time: `0.21992620523615128s/query`
- cold first search including prepared-index load: `34.001511416980065s`
- persisted snapshot bytes: `5,289,545,217`

Correctness cross-check on the full snapshot:

- Mojo exact and NumPy reference returned identical top-10 document ids for the
  first two full-corpus queries
- that check was run directly against the same
  `.cache/kayak/r2med_biology_full_main/docs_full_queries_full/store`
  snapshot after the large-artifact loader fix

## External Comparison

Official source:

- `https://r2med.github.io/`

Observed public reference values on April 15, 2026:

- the project page says the original study found that even the best model in
  the paper reached only `31.4 nDCG@10` on the benchmark average across all
  eight datasets
- the current leaderboard top average is `43.18`
- for the specific `Biology` column, the current top public score shown is
  `54.01`

Comparison boundary:

- Kayak was measured on the single public `Biology` subset, not the average
  across all eight R2MED datasets
- so the sound direct comparison is:
  - Kayak Biology: about `9.53`
  - current top public Biology result: `54.01`

That means Kayak is currently far from the public R2MED Biology frontier:

- absolute gap to the current top public Biology result: about `44.48` nDCG@10
  points
- ratio: Kayak is at about `17.6%` of that public Biology score

## Optimization Follow-Up

Measured encoder microbenchmark on a `64`-document R2MED sample:

- one-document-at-a-time `encode_document_text(...)`:
  `3.386942332959734s` total, about `0.0529s/doc`
- batched `encode_document_texts(..., batch_size=8)` with explicit zero-padding
  trim:
  `1.4556330000050366s` total, about `0.0227s/doc`
- measured speedup: about `2.33x`
- vector counts matched exactly after trimming zero-padded trailing rows

This optimization is now wired into the large-corpus R2MED snapshot builder for
future rebuilds. The full benchmark result above still comes from the earlier
one-document snapshot that was already in flight when this improvement was
validated, so the measured quality number is unchanged, but the next full
rebuild should be materially faster.

## Conclusion

Verified facts:

- Kayak now has one reproducible full-corpus public benchmark path on
  `R2MED/Biology`
- the full exact Mojo path can handle this corpus after segmenting the prepared
  token-value artifact
- current full-corpus exact quality on this benchmark is about `9.53 nDCG@10`
  on the leaderboard scale
- that is far below the current public `Biology` frontier of `54.01`

Open next step:

- improving quality now matters more than shaving a few milliseconds from exact
  search on this benchmark, because the current gap is dominated by retrieval
  effectiveness rather than runtime
