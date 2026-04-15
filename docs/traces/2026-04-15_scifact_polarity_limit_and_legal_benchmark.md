# 2026-04-15 SciFact polarity limit and legal benchmark addition

## Goal

Continue the harder-benchmark work without claiming a real contradiction
benchmark unless the local evidence supports that claim.

## Primary sources

- SciFact:
  https://aclanthology.org/2020.emnlp-main.609.pdf
- A Reasoning-Focused Legal Retrieval Benchmark:
  https://law.stanford.edu/wp-content/uploads/2025/03/3709025.3712219.pdf

## Verified local facts

I checked the current in-repo benchmark path and the raw cached SciFact files.

1. The current BEIR/SciFact loader is binary-only:
   [python/kayak_bridge/beir_subset.py](../../python/kayak_bridge/beir_subset.py)
   converts `qrel.relevance > 0` into `relevant_doc_ids`.
2. The judged-task storage format is also binary-only:
   [kayak/storage/judged_task_store.mojo](../../kayak/storage/judged_task_store.mojo)
   persists only `(query_id, doc_id)` qrels.
3. The raw cached SciFact `queries.json` file still contains evidence labels:

```bash
python3 - <<'PY'
import json
from pathlib import Path
path = Path('.cache/ir_datasets/beir/scifact/queries.json')
for line in path.read_text().splitlines()[:5]:
    print(line)
PY
```

The local file includes evidence metadata like:

- `"label": "SUPPORT"`
- `"label": "CONTRADICT"`

4. The raw SciFact cache contains many high-overlap opposite-claim pairs, but
   they usually share the same evidence paper rather than distinct papers.

Measured with a local token-overlap probe over the cached `queries.json`:

- `63` support-vs-contradict pairs with token-set Jaccard `>= 0.82`
- `0` such pairs with disjoint evidence sets
- only `2` such pairs with any query-specific unique evidence doc on either
  side

## Decision

Do not present SciFact as a sound doc-level contradiction benchmark under the
current retrieval abstraction.

Reason:

- the opposite-claim signal exists
- but for most near-duplicate opposite claims, the evidence object is the same
  paper
- that means the distinction is primarily a support-vs-contradict verification
  problem, not a document-retrieval discrimination problem

This does **not** invalidate the synthetic
`contradiction_hard_recall` lane. It only means that the current real-data
follow-up should not pretend SciFact already measures the same failure mode.

## Next benchmark addition

Add an opt-in legal retrieval slice instead.

Why this is justified:

- the legal benchmark is a true retrieval packaging with query, corpus, and
  gold passage ids
- it fits the existing `build_retrieval_subset_task(...)` path
- it adds a harder domain slice without widening the default public benchmark
  matrix

## Verified local legal packaging

I verified the dataset loading path inside the project environment:

```bash
pixi run python - <<'PY'
from datasets import load_dataset

qa = load_dataset('isaacus/legal-rag-bench', 'qa', split='test')
corpus = load_dataset('isaacus/legal-rag-bench', 'corpus', split='test')

print('qa', len(qa), qa.features)
print('corpus', len(corpus), corpus.features)
print(qa[0])
print(corpus[0])
PY
```

Observed locally:

- `qa`: `100` questions with one `relevant_passage_id` per question
- `corpus`: `4,876` legal passages

I also verified the retrieval packaging:

```bash
pixi run python - <<'PY'
from collections import Counter, defaultdict
from datasets import load_dataset

qrels = load_dataset('isaacus/mleb-legal-rag-bench', 'default', split='test')
by_query = defaultdict(list)
for row in qrels:
    by_query[str(row['query-id'])].append(str(row['corpus-id']))

print('query_count', len(by_query))
print('relevant_docs_per_query_hist', Counter(len(v) for v in by_query.values()))
PY
```

Observed locally:

- `100` retrieval queries
- exactly `1` judged relevant passage per query

That is why the new slice uses `primary_metric = "mrr"` rather than `ndcg`.

## Code added

- [python/kayak_bridge/legal_rag_bench_subset.py](../../python/kayak_bridge/legal_rag_bench_subset.py)
- [kayak/interop/legal_rag_bench_subset.mojo](../../kayak/interop/legal_rag_bench_subset.mojo)
- [kayak/storage/legal_rag_bench_cache.mojo](../../kayak/storage/legal_rag_bench_cache.mojo)
- [benchmarks/legal_rag_bench_real_subset_search.mojo](../../benchmarks/legal_rag_bench_real_subset_search.mojo)

Registry wiring was also added so the slice can be loaded through:

- the Python task-json catalog
- the Mojo public benchmark dataset registry
- the exact real-slice benchmark task list

## Important implementation choice

The legal documents are encoded as `title + "\n" + text` and omit the
`footnotes` field.

Reason:

- the judged relevant object is the passage text itself
- footnotes are citation scaffolding, not the primary retrieval target
- including them would inflate document vector count and storage without clear
  evidence that they improve passage-level retrieval for this benchmark

This is a measured design choice to keep the slice closer to the legal passage
retrieval target, not a claim that footnotes are never useful.

## Validation

### Focused correctness check

I verified that every judged relevant passage is actually present in the
encoded subset:

```bash
PYTHONPATH=python pixi run python - <<'PY'
from kayak_bridge.legal_rag_bench_subset import build_legal_rag_bench_colbert_subset

task = build_legal_rag_bench_colbert_subset()
doc_ids = {doc["doc_id"] for doc in task["documents"]}
for query in task["queries"]:
    for doc_id in query["relevant_doc_ids"]:
        assert doc_id in doc_ids, (query["query_id"], doc_id)
print("all_gold_docs_present", len(task["queries"]))
PY
```

Observed locally:

- `all_gold_docs_present 8`

This matters because the benchmark quality numbers below are low. The check
debunks the easy failure mode where low exact-search quality would be caused by
missing gold documents.

### Builder cost

Measured with:

```bash
PYTHONPATH=python pixi run python - <<'PY'
import time
from kayak_bridge.legal_rag_bench_subset import build_legal_rag_bench_colbert_subset

start = time.perf_counter()
task = build_legal_rag_bench_colbert_subset()
print('build_seconds', round(time.perf_counter() - start, 3))
print('queries', len(task['queries']))
print('documents', len(task['documents']))
print('primary_metric', task['primary_metric'])
print('nominal_query_vector_count', task['nominal_query_vector_count'])
print('nominal_document_vector_count', task['nominal_document_vector_count'])
PY
```

Observed locally:

- `build_seconds 9.817`
- `queries 8`
- `documents 136`
- `primary_metric mrr`
- `nominal_query_vector_count 32`
- `nominal_document_vector_count 157`

### Exact benchmark

Validated through the new Mojo entrypoint:

```bash
pixi run bench_legal_rag_bench_raw
```

and then through the quiet wrapper with an explicit forced timeout because the
host never became quiet enough for an unqualified quiet run:

```bash
bash scripts/run_bench_quiet.sh --timeout-seconds 1 --force --repeats 1 -- \
  pixi run bench_legal_rag_bench_raw
```

Observed artifact:

- `.cache/kayak/legal_rag_bench_real_subset_benchmark.json`

Observed wrapped search mean on the current host:

- `mean_search_seconds = 0.0014947177989130434`

Observed exact-search quality from the emitted JSON:

- `primary_metric = "mrr"`
- `primary_value = 0.16666666666666666`
- `mean_ndcg_at_k = 0.1875`
- `mean_reciprocal_rank = 0.16666666666666666`
- `mean_recall_at_k = 0.25`
- `success_rate_at_k = 0.25`

Interpretation:

- this is a genuinely hard legal retrieval slice even after shrinking the
  corpus to `136` passages
- the low exact-search score is not a subset-assembly bug because all gold
  passages were verified present
- that makes the slice useful as an opt-in “hard legal reasoning retrieval”
  benchmark even before adding approximate-frontier scripts for it
