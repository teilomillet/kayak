# 2026-04-12: Compact `centroid_heads` sidecar

## Why this exists

`centroid_postings_head` was already a useful non-default stage-1 baseline, but it
was still reading from the full `centroid_postings` artifact. That meant we were
not measuring its storage footprint honestly, and we could not persist a compact
search-native artifact per sealed segment.

This step makes that representation explicit:

- `centroid_heads` is a separate sealed-segment sidecar.
- It stores the same centroid vectors as `centroid_postings`.
- For each centroid it keeps only the first `posting_cap` postings from the
  weight-sorted posting list.
- Benchmarks continue to score stage-1 faithfulness against exact stage-2
  results via `explain_collection_search()` and `candidate_recall_at_final_k`.

## Design choices

- `posting_cap` is persisted in storage metadata and in the sidecar manifest.
- `centroid_heads` uses its own artifact kind: `centroid_head_index`.
- The new sidecar reuses the existing `CentroidPostingIndex` payload shape so the
  storage and scoring logic stay easy to browse.
- The runtime path keeps this sidecar non-default for now. That is deliberate:
  the public-slice evidence below supports it as an additional frontier point,
  not yet as a universal replacement for full centroid postings.

## What was added

- New index helper: `build_centroid_head_index(...)`
- New storage module: `kayak/storage/centroid_heads_store.mojo`
- New candidate generator + plan: `centroid_heads`
- Resolver + sealed-segment manifest support for optional `centroid_heads_root`
- Public-slice benchmark coverage in:
  - `benchmarks/real_subset_candidate_window_sweep.mojo`
  - `benchmarks/real_subset_vector_budget_sweep.mojo`

## Evidence

Benchmarks were regenerated with `CENTROID_HEAD_POSTING_CAP = 16`.

### Full-budget candidate-window behavior

At full centroid budgets, `centroid_heads` is usually only modestly smaller than
full `centroid_postings`, and latency is mixed rather than uniformly better.

Representative artifact sizes:

- SciFact: `74,048 B` vs `78,313 B` (`-5.4%`)
- FiQA: `75,260 B` vs `79,945 B` (`-5.9%`)
- LIMIT-small: `63,807 B` vs `67,423 B` (`-5.4%`)
- BrowseComp evidence slice: `75,838 B` vs `84,145 B` (`-9.9%`)

That is not enough to justify replacing the full sidecar by default.

### Vector-budget frontier

The useful result appears when vector budget is treated as a first-class axis.
At the same centroid budget, `centroid_heads` often removes `17-30%` of sidecar
bytes and can improve or preserve stage-1 recall against the exact oracle.

Representative same-budget comparisons:

- SciFact, centroid budget `8`:
  - `centroid_postings`: recall `0.8167`, size `6,419 B`
  - `centroid_heads`: recall `0.8500`, size `5,257 B`
- FiQA, centroid budget `16`:
  - `centroid_postings`: recall `0.8333`, size `12,537 B`
  - `centroid_heads`: recall `0.8500`, size `10,138 B`
- LIMIT-small, centroid budget `64`:
  - `centroid_postings`: recall `0.9031`, size `42,005 B`
  - `centroid_heads`: recall `0.9156`, size `38,403 B`
- BrowseComp evidence slice, centroid budget `8`:
  - `centroid_postings`: recall `0.7000`, size `7,592 B`
  - `centroid_heads`: recall `0.7500`, size `5,303 B`
- BrowseComp evidence slice, centroid budget `32`:
  - `centroid_postings`: recall `0.8000`, size `25,655 B`
  - `centroid_heads`: recall `0.8000`, size `19,924 B`

This is the real gain from the compact sidecar: it creates a better compressed
multi-vector operating point, especially when centroid budget is tight.

## Conclusion

The sound conclusion is:

- keep `centroid_heads` as a real, persisted, non-default stage-1 generator
- use it as a low-budget frontier option
- do **not** treat it as the new universal default yet

## Next decision

The next meaningful branch is not “make this default everywhere”.
The next meaningful branch is one of:

- sweep `posting_cap` explicitly to see whether `16` is actually near-optimal
- move into heavier native compression / WARP-GEM-style work only after that

The current evidence supports the first option.
