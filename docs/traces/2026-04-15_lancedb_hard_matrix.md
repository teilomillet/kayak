# 2026-04-15 LanceDB Hard Matrix Across BRIGHT, LEMB, Legal, and R2MED

## Question

Do the LanceDB indexed tradeoffs we saw on individual hard slices survive when
we evaluate the same sweep and selection policy across a broader hard-benchmark
matrix, instead of choosing one dataset at a time?

This is the narrower claim I wanted to test:

- not "LanceDB is better than Kayak"
- not "one sweep result generalizes everywhere"
- but:
  - "the same indexed sweep and auto-selection workflow behaves coherently on a
    broader set of hard slices"

## Why this step was justified

The repo already had strong single-slice evidence:

- indexed sweeps on legal and biomedical slices
- automatic selected-candidate reruns from those sweeps

But that still left an epistemic gap:

- the workflow was modular in code
- yet the evidence was still fragmented by dataset

That meant we still did not know whether the LanceDB indexed-vs-scan story held
across:

- reasoning-heavy retrieval
- long-document retrieval
- legal retrieval
- biomedical reasoning retrieval

## Benchmark choice

The matrix uses four already supported hard slices:

- `BRIGHT / StackOverflow`
  - source: https://arxiv.org/abs/2407.12883
- `LEMB / NarrativeQA`
  - source: https://aclanthology.org/2024.emnlp-main.47/
- `Legal RAG Bench`
  - source: https://arxiv.org/abs/2408.10343
- `R2MED / Biology`
  - source: https://arxiv.org/abs/2505.14558

I chose these because they cover meaningfully different failure modes while
already fitting the repo's encoded task JSON workflow.

## Code added

Added:

- [python/kayak_bridge/lancedb_indexed_matrix.py](../../python/kayak_bridge/lancedb_indexed_matrix.py)
- [python/scripts/bench_lancedb_hard_matrix.py](../../python/scripts/bench_lancedb_hard_matrix.py)
- [python/tests/test_lancedb_indexed_matrix.py](../../python/tests/test_lancedb_indexed_matrix.py)

Design choice:

- keep the existing single-task sweep logic and selected-candidate logic intact
- add one matrix runner above them instead of refactoring the existing scripts

Reason:

- the current sweep and selection surfaces were already tested
- this keeps the new claim about benchmark breadth separate from unrelated code
  movement

## Validation

Focused tests:

```bash
env PYTHONPATH=python uv run --python 3.11 python -m unittest \
  python.tests.test_lancedb_indexed_matrix \
  python.tests.test_lancedb_indexed_selection \
  python.tests.test_lancedb_indexed_candidate_bundle \
  python.tests.test_lancedb_index_sweep \
  python.tests.test_benchmark_variance
```

Observed:

- passed

Syntax and patch hygiene:

```bash
env PYTHONPATH=python uv run --python 3.11 python -m py_compile \
  python/kayak_bridge/lancedb_indexed_matrix.py \
  python/scripts/bench_lancedb_hard_matrix.py \
  python/tests/test_lancedb_indexed_matrix.py

git diff --check -- \
  python/kayak_bridge/lancedb_indexed_matrix.py \
  python/scripts/bench_lancedb_hard_matrix.py \
  python/tests/test_lancedb_indexed_matrix.py
```

Observed:

- passed

Smoke run before the full matrix:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 1200 -- \
  uv run --python 3.11 --with lancedb --with faiss-cpu python \
    python/scripts/bench_lancedb_hard_matrix.py \
    --dataset-key legal_rag_bench_real_subset \
    --output-root .cache/kayak/lancedb_hard_matrix_smoke \
    --artifact-prefix lancedb_hard_matrix_smoke \
    --rebuild-count 1 \
    --warmup-iterations 1 \
    --measurement-iterations 2
```

Observed:

- the full `sweep -> auto-select -> rerun` flow completed successfully on one
  real hard slice before the larger run

## Full matrix run

Command:

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 1200 -- \
  uv run --python 3.11 --with lancedb --with faiss-cpu python \
    python/scripts/bench_lancedb_hard_matrix.py \
    --output-root .cache/kayak/lancedb_hard_matrix \
    --artifact-prefix lancedb_hard_matrix \
    --rebuild-count 5 \
    --warmup-iterations 1 \
    --measurement-iterations 3
```

Aggregate artifact:

- `.cache/kayak/lancedb_hard_matrix/lancedb_hard_matrix_summary.json`

Selection policy:

- `include_default_plus_best_quality_plus_fastest_quality_improving`

Shared sweep configs:

- `default`
- `nprobe64`
- `refine1`
- `refine2`
- `refine4`
- `p4_refine2`
- `p8_refine2`
- `sv16_refine2`

## Results

### BRIGHT StackOverflow

- queries: `8`
- documents: `150`
- Kayak exact:
  - `ndcg=0.2914247870199321`
  - `0.0014183246239554137 s`
- LanceDB scan:
  - `ndcg=0.2914247870199321`
  - `0.019327301992840756 s`
- best indexed rerun candidate:
  - `lancedb_indexed_nprobe64`
  - `ndcg=0.31298226376474886`
  - `0.015340730573128288 s`
- fastest quality-improving rerun candidate:
  - `lancedb_indexed_default`
  - `ndcg=0.3088192738516368`
  - `0.015325855894479901 s`

Interpretation:

- on this slice, `nprobe64` mattered more than refine
- the indexed candidates beat scan on both quality and latency

### LEMB NarrativeQA

- queries: `8`
- documents: `355`
- Kayak exact:
  - `ndcg=0.375`
  - `0.005577329876056562 s`
- LanceDB scan:
  - `ndcg=0.375`
  - `0.03798123262822628 s`
- best indexed rerun candidate:
  - `lancedb_indexed_p8_refine2`
  - `ndcg=0.39085908322493285`
  - `0.026730969127189988 s`

Interpretation:

- the long-document slice preferred a partitioned refined indexed setting
- the same indexed winner was both best-quality and fastest quality-improving

### Legal RAG Bench

- queries: `8`
- documents: `136`
- Kayak exact:
  - `mrr=0.16666666666666666`
  - `0.0012987934169359505 s`
- LanceDB scan:
  - `mrr=0.16666666666666666`
  - `0.029529901085576665 s`
- best indexed rerun candidate:
  - `lancedb_indexed_refine2`
  - `mrr=0.17506944444444444`
  - `0.028006109758280218 s`
- fastest quality-improving rerun candidate:
  - `lancedb_indexed_refine1`
  - `mrr=0.1738095238095238`
  - `0.02434242329133364 s`

Interpretation:

- the legal slice continued to prefer refine over the default indexed setting
- `refine1` became the fastest quality-improving rerun candidate, while
  `refine2` remained the quality winner

### R2MED Biology

- queries: `8`
- documents: `178`
- Kayak exact:
  - `ndcg=0.8646255554317486`
  - `0.001289899291199011 s`
- LanceDB scan:
  - `ndcg=0.8646255554317486`
  - `0.027171484456630424 s`
- best indexed rerun candidate:
  - `lancedb_indexed_refine1`
  - `ndcg=0.8798420480856977`
  - `0.021681299607735127 s`
- fastest quality-improving rerun candidate:
  - `lancedb_indexed_default`
  - `ndcg=0.8725237264314342`
  - `0.020561488895327783 s`

Interpretation:

- the biomedical slice still favored `refine1` on quality
- default indexed remained the lowest-latency quality-improving option

## Cross-slice interpretation

What the evidence supports:

- the same indexed sweep and auto-selection workflow works across all four hard
  slices without benchmark-specific code branches
- in every measured slice, the best indexed rerun candidate beat LanceDB scan
  on quality
- in every measured slice, the best indexed rerun candidate was also faster
  than LanceDB scan
- the preferred indexed config is not universal:
  - BRIGHT preferred `nprobe64`
  - LEMB preferred `p8_refine2`
  - legal preferred `refine2`
  - R2MED preferred `refine1`

What the evidence does **not** support:

- that one LanceDB indexed config should be hard-coded as globally best
- that LanceDB now beats Kayak overall
- that these 8-query subsets are equivalent to full official benchmark
  reproductions

An important caution:

- Kayak exact remained much faster than every LanceDB path on all four slices
- the matrix result is about LanceDB indexed-versus-scan behavior across hard
  slices, not about replacing the native Kayak engine result

## Practical conclusion

The main useful outcome is not one winning LanceDB parameter.

It is that the repo now has one reproducible hard-matrix benchmark path that:

1. reuses the same sweep policy across multiple hard slice families
2. removes manual candidate picking
3. records which indexed settings actually win per slice

That gives us a stronger base for the next step:

- either compare more slices under the same workflow
- or tune a narrower default LanceDB indexed policy only after seeing repeated
  matrix evidence rather than from one benchmark alone
