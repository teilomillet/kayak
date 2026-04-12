# 2026-04-12 BrowseComp-Plus Gold Faithfulness Frontier

## Goal

Mirror the new frontier benchmark on one harder public slice instead of
stopping at the synthetic scale fixture.

This trace measures the same variables as the synthetic frontier:

- candidate-generation time
- end-to-end search time
- candidate recall against exact full scan
- judged retrieval quality
- stage-1 storage context

The chosen public slice is `BrowseComp-Plus` gold because it is already one of
the repo's harder inspected public slices and it carries real ranking pressure.

## Verification Command

```bash
pixi run bench_browsecomp_plus_gold_faithfulness_frontier_raw
```

Artifact:

```text
.cache/kayak/browsecomp_plus_gold_faithfulness_frontier.json
```

## First-Run Note

This run was executed in a fresh worktree with no prior `.cache`.

Verified from the command output:
- the helper materialized the task JSON locally
- the first run downloaded the public parquet shards required by the Python
  subset builder
- the first run also downloaded the ColBERT-side model assets used by that
  builder

Important boundary:
- those first-run downloads affect wall-clock setup time
- they do **not** contaminate the emitted `mean_search_seconds` values, which
  come from the Mojo benchmark artifact itself

## Measurement Context

Slice:

- `family = "browsecomp_plus"`
- `slice = "browsecomp_plus_gold_slice"`
- `document_count = 90`
- `token_count = 15756`
- `vector_count = 15756`
- `final_k = 10`

Measured generator families:

- `exact_full_scan`
- `document_proxy`
- `centroid_postings`
- `centroid_heads`
- `centroid_postings_head_auto`

Swept knobs:

- `candidate_k = 10, 20, 40, 80, 90`
- `posting_cap = 1, 2, 4, 8, 16, 32, 90`

## Result Snapshot

### Exact Baseline

- exact full scan
  - `search = 5.71475 ms`
  - `candidate recall = 1.0`
  - `nDCG@10 = 0.2851`
  - `stage1 bytes/doc = 91104.76`

### Cheapest Overall Point

- document proxy, `candidate_k = 10`
  - `search = 0.7840 ms`
  - `candidate recall = 0.725`
  - `nDCG@10 = 0.1736`
  - `stage1 bytes/doc = 521.57`

Interpretation:
- the fastest point on the frontier is not faithful enough for this slice
- the public slice therefore rejects any claim that raw latency alone is the
  right benchmark output

### Cheapest Full-Recall Point

- document proxy, `candidate_k = 40`
  - `search = 3.52875 ms`
  - `candidate recall = 1.0`
  - `nDCG@10 = 0.2851`
  - `stage1 bytes/doc = 521.57`

Interpretation:
- on this `90`-document gold slice, `document_proxy` is the cheapest full-recall
  path that was measured
- relative to exact full scan, that point is about `1.62x` faster while using
  a much smaller stage-1 sidecar

### Native Centroid Family

- centroid postings, `candidate_k = 80`
  - `search = 8.16075 ms`
  - `candidate recall = 0.975`
  - `nDCG@10 = 0.3190`
  - `stage1 bytes/doc = 935.08`
- centroid heads, `candidate_k = 80`, `posting_cap = 8`
  - `search = 8.09325 ms`
  - `candidate recall = 1.0`
  - `nDCG@10 = 0.2851`
  - `stage1 bytes/doc = 802.60`
- centroid postings head auto, `candidate_k = 80`, `posting_cap = 8`
  - `search = 8.2085 ms`
  - `candidate recall = 0.975`
  - `nDCG@10 = 0.3190`
  - `stage1 bytes/doc = 935.08`

Interpretation:
- on this small public slice, the current centroid-family implementations do
  **not** beat exact full-scan latency once they approach full recall
- they still expose a meaningful frontier:
  - lower budgets are fast but lose faithfulness
  - higher budgets recover recall but lose their latency advantage
- judged quality can exceed the exact baseline before exact-reference recall is
  fully recovered, which again shows that these are different measurements

## What This Verifies

Verified locally:
- the new frontier summary works on a real public slice, not only on the
  synthetic fixture
- `candidate_k` and `posting_cap` materially change latency, faithfulness, and
  storage on that slice
- a public hard-recall mirror can contradict a simplistic "native candidates
  always beat exact" story
- the current `document_proxy` path is stronger than the current centroid
  family on this particular small gold slice when full recall is required

## What This Still Does Not Prove

Not verified by this work:
- that `document_proxy` or any centroid family is the best choice on larger or
  harder public corpora
- broader asymptotic wins for the current centroid implementations
- any claim about multi-billion-token behavior
- any claim about compressed token storage viability

## Takeaway

The public mirror confirms that the frontier work is worth doing, but it also
cuts against an over-broad efficiency narrative.

On this slice:
- `document_proxy` wins the cheapest faithful point
- centroid-family variants still expose interesting storage and judged-quality
  tradeoffs
- exact full scan remains a competitive baseline on small corpora

That is the right epistemic outcome: the benchmark sharpens which claims are
supported, and which ones still need better engines or larger datasets.
