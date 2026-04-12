# 2026-04-12 BrowseComp-Plus Gold Vector Pruning

## Goal

Close Phase I3 from
[docs/late_interaction_efficiency_roadmap.md](../late_interaction_efficiency_roadmap.md):

- vary retained vectors/document directly
- measure latency and storage on the pruned exact representation
- compare the pruned exact ranking against the full exact reference
- state clearly whether a `sqrt(m)`-style budget is supported, unsupported, or
  still ambiguous on this slice

## Files Added Or Changed

- `kayak/benchmarks/vector_pruning_json.mojo`
- `benchmarks/browsecomp_plus_gold_vector_pruning.mojo`
- `tests/test_vector_pruning_json.mojo`
- `pyproject.toml`

## Design Choice

Reason:
- the earlier vector-budget sweeps operated on stage-1 sidecars
- Phase I3 needed a more direct test of vectors/document reduction itself

Decision:
- prune the exact document representation by keeping only the first
  `document_vector_budget` token vectors per document
- compare the resulting exact search output against the full exact reference

Important boundary:
- this is a naive prefix-pruning baseline
- it is intentionally not a learned pruning policy

That makes the result epistemically useful:
- if naive pruning already works well, the redundancy claim gets stronger
- if naive pruning fails early, the repo should not pretend `sqrt(m)` is
  already supported

## Verification Commands

```bash
pixi run test_vector_pruning_json
pixi run bench_browsecomp_plus_gold_vector_pruning_raw
```

Artifact:

```text
.cache/kayak/browsecomp_plus_gold_vector_pruning.json
```

## Measurement Context

Slice:

- `family = "browsecomp_plus"`
- `slice = "browsecomp_plus_gold_slice"`
- `document_count = 90`
- `full_vector_count = 15756`
- average full vectors/document ≈ `175.07`
- `sqrt(175.07) ≈ 13.23`

Budgets swept:

- `4`
- `8`
- `16`
- `32`
- `64`
- `128`
- `175`

## Result Snapshot

- budget `4`
  - `reference recall@10 = 0.60`
  - `nDCG@10 = 0.1150`
  - `search = 0.0000267 s`
- budget `8`
  - `reference recall@10 = 0.625`
  - `nDCG@10 = 0.1413`
  - `search = 0.0000476 s`
- budget `16`
  - `reference recall@10 = 0.625`
  - `nDCG@10 = 0.0802`
  - `search = 0.0000940 s`
- budget `32`
  - `reference recall@10 = 0.675`
  - `nDCG@10 = 0.1801`
  - `search = 0.0001341 s`
- budget `64`
  - `reference recall@10 = 0.775`
  - `nDCG@10 = 0.3950`
  - `search = 0.0002952 s`
- budget `128`
  - `reference recall@10 = 0.90`
  - `nDCG@10 = 0.3342`
  - `search = 0.0005416 s`
- budget `175`
  - `reference recall@10 = 1.0`
  - `nDCG@10 = 0.2851`
  - `search = 0.0006985 s`

Across the sweep, persisted bytes/vector stayed roughly flat around
`512 bytes/vector`.

Reason:
- this benchmark changes vectors/document, not the scalar encoding
- the storage format itself stayed `binary_le`

## Interpretation

The `sqrt(m)`-style regime is **unsupported** for this naive prefix-pruning
baseline on BrowseComp gold.

Concrete evidence:
- the `sqrt(m)` neighborhood is about `13`
- the closest measured budget, `16`, preserved only `62.5%` of the full exact
  top-`10`
- judged `nDCG@10` also fell sharply to `0.0802`

What this does **not** imply:
- that all smarter pruning laws are impossible
- that no learned or structure-aware pruning policy could do better

What it does imply:
- the repo should stop speaking as if `sqrt(m)` is already plausible by
  default
- a stronger pruning claim now needs a better pruning policy than
  "keep the first N vectors"

## What This Verifies

Verified locally:
- Kayak now has one direct vectors/document benchmark family
- the repo can compare pruned exact search against the full exact reference on
  the same public slice
- Phase I3 now has a measured answer for one concrete pruning policy

## What This Still Does Not Prove

Not verified by this work:
- a better token-selection policy
- a learned pruning method
- any broad impossibility result for vector-count laws

## Takeaway

Phase I3 is now grounded by a falsification result:
- naive exact prefix pruning does buy latency
- it does **not** support a `sqrt(m)`-style budget on BrowseComp gold

That is useful progress because it narrows the real next step:
- better pruning policy
- not more rhetoric about token redundancy
