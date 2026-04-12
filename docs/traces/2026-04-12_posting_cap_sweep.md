# 2026-04-12: explicit `posting_cap` sweep for `centroid_heads`

## Why this step

After adding the persisted `centroid_heads` sidecar, the next epistemically sound
question was not "is 16 good enough?" but "what does the `posting_cap` frontier
actually look like on public slices?"

This step adds an explicit benchmark axis for `posting_cap` and measures:

- stage-1 candidate generation latency
- stage-1 recall against exact stage-2 results
- stage-1 storage footprint
- end-to-end retrieval quality (`ndcg`, `mrr`, `recall`, `success`)

## What was added

- New benchmark module:
  - `kayak/benchmarks/posting_cap_json.mojo`
- New public-slice sweep:
  - `benchmarks/real_subset_posting_cap_sweep.mojo`
- New test:
  - `tests/test_posting_cap_json.mojo`
- New Pixi task:
  - `pixi run bench_real_subset_posting_caps`

The sweep writes:

- `.cache/kayak/public_posting_cap_sweep.json`

## Sweep definition

For each public slice:

- centroid budgets: the same standard document-vector budget ladder already used
  by the vector-budget sweep
- query budgets: the same standard query-vector budget ladder
- posting caps: `[1, 2, 4, 8, 16, 32, document_count]`
- stage-1 generator: `centroid_heads`
- faithfulness measurement: exact stage-2 comparison through
  `explain_collection_search()` and `candidate_recall_at_final_k`

## Main result

`posting_cap = 16` is **not** a universal optimum.

The correct conclusion is:

- `posting_cap` must remain configurable
- the best cap depends strongly on dataset and centroid budget
- a fixed cap of `16` is often reasonable, but sometimes too small and
  sometimes unnecessarily large

## Representative evidence

### SciFact

- centroid budget `8`
  - best frontier point reaches recall `0.8667` at cap `8`
  - cap `16` is unnecessary there
- centroid budget `32`
  - cap `2` already reaches recall `0.9000`
- centroid budget `128`
  - cap `32` reaches recall `0.9500`
  - cap `16` only reaches recall `0.9167`

Interpretation:

- on SciFact, larger centroid budgets can still benefit from larger posting caps
- a fixed cap of `16` leaves recall on the table at high budget

### FiQA

- centroid budget `8`
  - cap `1` already gives recall `0.8167`
- centroid budget `16`
  - cap `16` gives recall `0.8500`
- centroid budget `64`
  - cap `32` gives recall `0.9000`
  - cap `16` only gives `0.8667`

Interpretation:

- FiQA prefers a cap that grows with centroid budget

### LIMIT-small

- centroid budget `16`
  - cap `8` gives recall `0.9281`
- centroid budget `64`
  - cap `16` gives recall `0.9156`
- centroid budget `128`
  - cap `16` gives recall `0.9188`

Interpretation:

- LIMIT-small is the strongest case for the current default-like choice:
  `16` is consistently competitive there

### BrowseComp evidence / gold slices

- centroid budget `8`
  - cap `16` is the best measured point at recall `0.7500`
- centroid budget `64`
  - cap `16` gives recall `0.8500`
  - full cap (`90`) gives recall `0.9000`
  - bytes grow from about `39.2 KB` to `47.2 KB`

Interpretation:

- `16` is a good compressed point
- but it is still a real recall tradeoff on the harder BrowseComp slice

## Decision

The sound decision is:

- keep the benchmark axis
- do **not** hard-code a single global `posting_cap` policy as "solved"

The next meaningful choice is between:

- an adaptive policy that picks `posting_cap` from corpus / centroid-budget
  signals
- deeper native-engine work only after that policy is justified

Given the current evidence, the first option is the better next step.
