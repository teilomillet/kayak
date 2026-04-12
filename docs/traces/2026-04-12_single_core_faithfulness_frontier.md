# 2026-04-12 Single-Core Faithfulness Frontier

## Goal

Continue Phase I4 from
[docs/late_interaction_efficiency_roadmap.md](../late_interaction_efficiency_roadmap.md):

- keep the single-core scale sweep
- add explicit frontier sweeps over `candidate_k` and `posting_cap`
- record latency, candidate recall, judged quality, and stage-1 storage in one
  artifact

This trace exists to verify a narrower claim than the Omar-style headline:
Kayak can now show the latency-recall-quality-storage frontier for multiple
candidate engines on the same synthetic scale slice.

## Files Added Or Changed

- `kayak/benchmarks/faithfulness_frontier_json.mojo`
- `benchmarks/single_core_faithfulness_frontier.mojo`
- `tests/test_faithfulness_frontier_json.mojo`
- `kayak/benchmarks/__init__.mojo`
- `pyproject.toml`

## Design Choice

Reason:
- the repo already had stage-aware search summaries and a deterministic
  synthetic scale fixture
- the missing piece was one reusable frontier summary that keeps stage-1 bytes,
  vectors, and faithfulness explicit while sweeping candidate knobs

Decision:
- keep this benchmark on the real collection/search-plan stack
- measure both:
  - candidate-generation time
  - end-to-end search time after exact stage-2 rescoring
- emit one machine-readable summary per `(engine, candidate_k, posting_cap,
  slice)` point

That choice is justified because a frontier claim is only meaningful when the
same artifact reports:
- latency
- exact-reference candidate recall
- judged retrieval quality
- stage-1 storage context

## Verification Commands

Targeted correctness checks:

```bash
pixi run test_faithfulness_frontier_json
pixi run test_stage_aware_benchmark_json
pixi run test_single_core_scale_fixture
```

Benchmark run:

```bash
pixi run bench_single_core_faithfulness_frontier_raw
```

Artifact:

```text
.cache/kayak/single_core_faithfulness_frontier.json
```

## Measurement Context

The benchmark reuses the deterministic single-core scale profiles:

- `query_count = 8`
- `nominal_query_vector_count = 4`
- `nominal_document_vector_count = 8`
- `vector_dim = 64`
- `final_k = 2`

Corpus sizes:

- `64`
- `256`
- `1024`
- `4096`

Measured generator families:

- `exact_full_scan`
- `document_proxy`
- `centroid_postings`
- `centroid_heads`
- `centroid_postings_head_auto`

Swept knobs:

- `candidate_k`
- `posting_cap`

## Result Snapshot

The most decision-relevant points are below.

### Docs 4096

- exact full scan
  - `search = 0.709125 ms`
  - `candidate recall = 1.0`
  - `stage1 bytes/doc = 2389.56`
- document proxy, `candidate_k = 2`
  - `search = 0.0425 ms`
  - `candidate recall = 1.0`
  - `stage1 bytes/doc = 266.87`
- centroid postings, `candidate_k = 2`
  - `search = 0.0560 ms`
  - `candidate recall = 1.0`
  - `stage1 bytes/doc = 58.18`
- centroid heads, `candidate_k = 2`, `posting_cap = 4`
  - `search = 0.0395 ms`
  - `candidate recall = 0.125`
  - `stage1 bytes/doc = 4.64`

Interpretation:
- the synthetic slice is easy enough that `document_proxy` and
  `centroid_postings` recover full exact-reference recall even at `candidate_k
  = 2`
- at `4096` docs that makes them materially faster than exact full scan:
  - `document_proxy`: about `16.7x`
  - `centroid_postings`: about `12.7x`
- the tighter head-capped path pushes stage-1 bytes/doc down to single digits,
  but it does so by collapsing faithfulness

### Docs 1024

- centroid heads, `candidate_k = 2`, `posting_cap = 1`
  - `search = 0.0190 ms`
  - `candidate recall = 0.125`
  - `nDCG@2 = 0.125`
  - `stage1 bytes/doc = 17.32`
- centroid postings head auto, `candidate_k = 16`, `posting_cap = 1`
  - `search = 0.058375 ms`
  - `candidate recall = 0.5625`
  - `nDCG@2 = 0.6615`
  - `stage1 bytes/doc = 64.68`

Interpretation:
- hybrid head/posting variants can buy back some faithfulness at moderate
  storage cost
- they do not remove the tradeoff; they only move to a different point on the
  frontier

## What This Verifies

Verified locally:
- Kayak now has a frontier benchmark, not only a scale benchmark
- the repo can compare latency, exact-reference recall, judged quality, and
  stage-1 storage in one artifact
- on this synthetic slice, cheap full-recall candidate engines exist
- more aggressive storage reduction can be directly seen to break faithfulness

## What This Still Does Not Prove

Not verified by this work:
- any multi-billion-token single-core claim
- that this synthetic frontier transfers to harder public slices
- any `6 bytes/vector` storage claim
- any `sqrt(m)` vectors/document law

## Takeaway

Phase I4 now has a synthetic frontier artifact instead of only a qualitative
story.

The main lesson is structural:
- there is real room to lower stage-1 cost
- the winning point depends on whether the workload tolerates faithfulness loss
- Kayak now measures that tradeoff explicitly instead of hiding it behind one
  headline latency number
