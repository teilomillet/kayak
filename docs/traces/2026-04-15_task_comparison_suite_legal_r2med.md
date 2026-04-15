# 2026-04-15: generic task comparison suite on Legal RAG Bench and R2MED

## Goal

Continue the cross-engine research on harder public slices after adding the new
legal benchmark.

The concrete question in this pass is:

- does the Kayak-versus-LanceDB gap survive on a hard legal slice and a hard
  medical slice?
- does that remain true even when both branches use LanceDB as the storage
  base?

## Benchmark surface

Both datasets were run through the existing generic comparison suite:

- Kayak exact
- LanceDB scan
- LanceDB indexed IVF_PQ with rebuild variance and frozen mean
- same-storage LanceDB-versus-Kayak search comparison
- same-task scale sweep
- native Kayak storage versus LanceDB storage scale sweep

The suite is owned by:

- [python/scripts/bench_task_comparison_suite.py](../../python/scripts/bench_task_comparison_suite.py)

## Inputs

Task JSONs:

- `.cache/kayak/legal_rag_bench_real_subset/python_task.json`
- `.cache/kayak/r2med_biology_real_subset/python_task.json`

The legal task JSON was produced earlier in this loop from the new legal subset
builder:

- [python/kayak_bridge/legal_rag_bench_subset.py](../../python/kayak_bridge/legal_rag_bench_subset.py)

## Commands

### Legal

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh \
  --repeats 1 \
  --max-other-cpu 500 \
  -- \
  uv run --python 3.11 --with lancedb --with faiss-cpu python \
    python/scripts/bench_task_comparison_suite.py \
    --task .cache/kayak/legal_rag_bench_real_subset/python_task.json
```

### R2MED

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh \
  --repeats 1 \
  --max-other-cpu 500 \
  -- \
  uv run --python 3.11 --with lancedb --with faiss-cpu python \
    python/scripts/bench_task_comparison_suite.py \
    --task .cache/kayak/r2med_biology_real_subset/python_task.json
```

## Host-load note

Both runs completed under the quiet wrapper without using `--force`.

The background host load was still nontrivial, roughly `300%` to `340%`
aggregate other-process CPU, but it stayed under the explicit `500%` threshold.
That makes these runs materially more comparable than the earlier forced
single-benchmark legal exact run.

## Results

### Legal RAG Bench

Artifact bundle:

- `.cache/kayak/legal_rag_bench_real_subset/legal_rag_bench_real_subset_comparison_bundle.json`

Key values:

- Kayak exact `MRR`: `0.16666666666666666`
- Kayak exact latency: `0.0013409774158693228s`
- LanceDB scan `MRR`: `0.16666666666666666`
- LanceDB scan latency: `0.02845272211319146s`
- LanceDB scan latency ratio vs Kayak: `21.21789806187471`
- LanceDB indexed frozen `MRR`: `0.09694444444444444`
- LanceDB indexed frozen latency: `0.022955144839943386s`
- LanceDB indexed frozen latency ratio vs Kayak: `17.118218821800312`
- Indexed rebuild quality range:
  `0.041666666666666664` to `0.18055555555555555`
- Indexed rebuild latency range:
  `0.022086208414596815s` to `0.025236267407308333s`

Same LanceDB storage, different search engines:

- artifact:
  `.cache/kayak/legal_rag_bench_real_subset/legal_rag_bench_real_subset_storage_compare.json`
- LanceDB scan search time: `0.02857026382116601s`
- Kayak load-from-LanceDB time: `0.004797375062480569s`
- Kayak exact search time on LanceDB-owned rows: `0.0011261510032151516s`
- LanceDB scan vs Kayak search-only ratio: `25.369833831873475`
- LanceDB scan vs Kayak load-plus-search ratio: `4.823185296106302`
- judged quality remained equal on both branches

Scale sweep:

- at `136` docs, LanceDB scan is `24.600714122839648x` slower than Kayak exact
- at `1088` docs, LanceDB scan is still `9.024805677630617x` slower than Kayak
  exact

Storage sweep:

- base LanceDB/Kayak bytes ratio: `1.0197893600623658`
- largest measured bytes ratio at `1088` docs: `1.019276157777919`
- largest measured LanceDB/Kayak search ratio in the storage sweep:
  `30.85549843071133` at `136` docs
- LanceDB/Kayak search ratio at `1088` docs: `10.783632906741632`

Interpretation:

- legal behaves like a strong “search latency gap, not storage blow-up” slice
- LanceDB scan preserves exact quality on this legal task
- indexed LanceDB is not just slower than Kayak here, it is also quality-unstable
- the Kayak advantage survives even when LanceDB is the storage base for both
  branches

### R2MED Biology

Artifact bundle:

- `.cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_comparison_bundle.json`

Key values:

- Kayak exact `nDCG`: `0.8646255554317486`
- Kayak exact latency: `0.0012007517846844469s`
- LanceDB scan `nDCG`: `0.8646255554317486`
- LanceDB scan latency: `0.02659444438177161s`
- LanceDB scan latency ratio vs Kayak: `22.148161444340914`
- LanceDB indexed frozen `nDCG`: `0.862877846985671`
- LanceDB indexed frozen latency: `0.020444743057790524s`
- LanceDB indexed frozen latency ratio vs Kayak: `17.02661892204751`
- Indexed rebuild quality range:
  `0.8554227902591527` to `0.8773711308443374`
- Indexed rebuild latency range:
  `0.020155687365331687s` to `0.02100993929586063s`

Same LanceDB storage, different search engines:

- artifact:
  `.cache/kayak/r2med_biology_real_subset/r2med_biology_real_subset_storage_compare.json`
- LanceDB scan search time: `0.026922067734024797s`
- Kayak load-from-LanceDB time: `0.0023100828984752297s`
- Kayak exact search time on LanceDB-owned rows: `0.0014462309191003442s`
- LanceDB scan vs Kayak search-only ratio: `18.61533132673736`
- LanceDB scan vs Kayak load-plus-search ratio: `7.167150840288691`
- judged quality remained equal on both branches

Scale sweep:

- at `178` docs, LanceDB scan is already much slower than Kayak exact
- at `1424` docs, LanceDB scan is still `6.841237236637102x` slower than Kayak
  exact

Storage sweep:

- base LanceDB/Kayak bytes ratio: `1.0209843875918503`
- largest measured bytes ratio in this sweep stayed close to parity
- LanceDB/Kayak bytes ratio at `1424` docs: `1.0117552212339231`
- LanceDB/Kayak search ratio at `1424` docs: `6.856388960429278`

Interpretation:

- R2MED shows the same broad pattern as legal:
  near-quality parity for LanceDB scan, but a large search-latency gap
- unlike legal, the indexed LanceDB point is much more stable on quality
- the storage gap is again small, about `1%` to `2%`, so the dominant
  differentiator is search time rather than storage size
- the Kayak advantage also survives when LanceDB is the storage base

## Cross-domain conclusion

Across both a hard legal slice and a hard medical slice:

- LanceDB scan matched Kayak exact on judged quality
- LanceDB scan remained about `21x` to `22x` slower than Kayak exact on the
  base slices
- when LanceDB owns storage in both branches, Kayak exact from LanceDB is still
  materially faster
- the storage-size difference is small on both slices, about `1%` to `2%`

That means the current evidence does **not** support the claim that the
remaining gap is mainly due to Kayak owning a custom storage format. On these
two harder domain slices, the larger differentiator is still search execution.

## Comparison to earlier slices

These two new results look much more like the earlier BRIGHT result than the
earlier LEMB result:

- BRIGHT: large search gap, small storage gap
- LEMB: large search gap, very large storage gap
- Legal and R2MED: large search gap, small storage gap

So the current evidence suggests:

- long-document regimes are where storage inflation matters most
- domain-specific hard retrieval can still favor Kayak strongly even without a
  large storage-space advantage

## What this changes

The next research step should not be “prove storage is everything.”

The more justified next steps are:

1. Study why LanceDB indexed search is unstable on the legal slice.
2. Extend the same suite to one contradiction-adjacent slice once a sound
   real-data formulation exists.
3. If we want to optimize Kayak further, focus on stage-1 and exact search
   execution rather than assuming storage is the dominant reason for the gap on
   hard legal or medical retrieval.
