# Benchmark Ladder

This note defines the benchmark ladder that Kayak should maintain instead of
growing ad hoc benchmark scripts.

The goal is not to maximize the number of datasets.

The goal is to keep one explicit evaluation ladder where each rung answers a
different systems question and has a concrete exit criterion.

## Why A Ladder

Reason:
- the repository already has several useful benchmark surfaces
- without an explicit ladder it is too easy to mix:
  - smoke tests
  - hard-recall comparisons
  - stronger local ceilings
  - future code or multimodal ambitions

Decision:
- every new benchmark family should justify which rung it belongs to
- every rung should state what evidence is required before it counts as
  implemented

## The Rungs

### 1. Trivial Sanity

Current families:
- `SciFact`
- `FIQA`
- `LIMIT-small`

Question this rung answers:
- does the exact late-interaction loop still work end to end on real judged
  data?

Exit criteria:
- the dataset loads through the repo’s existing benchmark path
- exact late interaction runs end to end
- judged quality and stage-aware JSON are emitted
- the family is treated as a regression/sanity slice, not as the strongest
  hard-recall proof

### 2. Harder Public Text

Current families:
- `BrowseComp-Plus` evidence slice
- `BrowseComp-Plus` gold slice

Question this rung answers:
- how much exact-reference candidate recall do current stage-1 plans lose on a
  materially harder public text retrieval slice?

Exit criteria:
- the dataset is mirrored into the explicit collection/snapshot path
- stage-aware output reports candidate recall against exact full scan
- vector counts, byte counts, and candidate budgets are explicit

### 3. Scalable Synthetic Hard Recall

Current families:
- `synthetic_hard_recall`
- `contradiction_hard_recall`
- `long_document_hard_recall`

Question this rung answers:
- how do stage-1 failures move once the corpus shape or document shape becomes
  harder than the current public slices?

Exit criteria:
- query vector count and nominal document vector count are explicit
- the family scales beyond the tiny public slices
- at least one non-exact stage-1 plan measurably loses exact-reference recall
  at bounded `candidate_k`
- exact late-interaction reranking can still recover when the candidate window
  fully covers the oracle

Important distinction:
- `synthetic_hard_recall` stresses conjunction-style ambiguity
- `contradiction_hard_recall` stresses same-topic opposite-polarity confusion
- `long_document_hard_recall` stresses long noisy prefixes with late exact
  evidence

### 4. Stronger Local Ceiling

Current path:
- `exact_clause_text_ceiling`

Question this rung answers:
- how much quality exists above exact vector-only late interaction when Kayak
  uses one explicitly more expensive local path?

Exit criteria:
- the expensive path is labeled by the actual execution path used
- candidate generation, stage-2 reference, and stage-3 verifier semantics stay
  explicit in the output
- latency is reported alongside quality

Naming rule:
- a clause-text reranker is a `local_stronger_ceiling`
- it is not a cross-encoder ceiling
- it is not a long-context LLM ceiling

### 5. Corpus-Scale PLAID Serving

Current status:
- scaffolded, not yet a decision-quality full-corpus benchmark

Current scaffold:
- `python/scripts/profile_task_plaid_i8_candidate_generation.py`
- `python/scripts/sweep_task_plaid_i8_candidate_generation.py`
- `python/scripts/build_msmarco_passage_task_json.py`
- `python/scripts/build_lemb_narrativeqa_task_json.py`

Target families:
- MS MARCO passage full corpus
- one long-context corpus: LoCoV1, LEMB/NarrativeQA, or local arXiv chunks

Question this rung answers:
- does Kayak's advantage come from candidate generation, pruning, compact
  storage, and residual/exact materialization rather than from isolated MaxSim
  kernels?

Exit criteria:
- corpus size and vector counts are explicit: document count, query count,
  total document vectors, query-vector distribution, document-vector
  distribution
- candidate generation is measured separately from exact rerank
- residual or i8 token payload materialization is measured or estimated as its
  own field
- external baselines include FastPlaid where the comparison is batch/offline
  and NextPlaid where the comparison is serving/index lifecycle
- full-corpus runs use the quiet benchmark wrapper, and any sampled smoke run
  is labeled as a pipeline validation only

Reason:
- the current GPU/FastPlaid rows are useful primitive tests, but they are too
  small to prove corpus-scale PLAID serving behavior. Long documents and large
  corpora are where posting fanout, candidate pruning, and residual
  materialization become the system bottleneck.

### 6. Future Code Or Multimodal Lane

Current status:
- not implemented

Question this rung will answer:
- can the same stage-aware reporting surface support code or beyond-text
  retrieval without hiding encoder or representation assumptions?

Exit criteria:
- one real loader or fixture exists through the normal benchmark path
- encoder identity remains explicit
- the resulting artifact uses the same stage-aware JSON surface as the text
  families

## Current Commands

Sanity and public hard-recall paths:

```bash
pixi run bench_hard_recall_real_subset
```

Synthetic hard-recall paths:

```bash
pixi run bench_synthetic_hard_recall_stage_aware
pixi run bench_contradiction_hard_recall_stage_aware
pixi run bench_long_document_hard_recall_stage_aware
```

Local stronger ceiling:

```bash
pixi run bench_browsecomp_plus_gold_ceiling_comparison
```

Corpus-scale scaffold on an already encoded task JSON:

```bash
PYTHONPATH=python python python/scripts/build_msmarco_passage_task_json.py \
  --collection /data/msmarco/collection.tsv \
  --queries /data/msmarco/queries.dev.tsv \
  --qrels /data/msmarco/qrels.dev.tsv \
  --document-limit 10000 \
  --query-limit 16 \
  --include-document-token-ids \
  --document-batch-size 16 \
  --output .cache/kayak/msmarco_passage_smoke/python_task.json

PYTHONPATH=python python python/scripts/profile_task_plaid_i8_candidate_generation.py \
  --task .cache/kayak/msmarco_passage_smoke/python_task.json \
  --query-limit 4 \
  --candidate-k 256 \
  --emit-quiet-mean

PYTHONPATH=python python python/scripts/bench_tachiom_task.py \
  --task .cache/kayak/msmarco_passage_smoke/python_task.json \
  --engine tachiom_tac_hnsw_pq \
  --tac-centroid-count 32768 \
  --tac-micro-token-threshold 128 \
  --tac-small-token-threshold 256 \
  --tac-active-token-floor 4 \
  --tac-min-vectors-per-centroid 39 \
  --tac-kmeans-iterations 10 \
  --tac-centroids-per-query-vector 120 \
  --tac-candidate-k 1000 \
  --tac-candidate-pruning-alpha 0.35 \
  --hnsw-max-neighbors 32 \
  --hnsw-ef-construction 1500 \
  --hnsw-ef-search 180 \
  --pq-subspace-count 32 \
  --pq-codebook-size 256 \
  --pq-kmeans-iterations 10 \
  --output .cache/kayak/msmarco_passage_smoke/tachiom_task_summary.json
```

Paper-scale MS MARCO materialization should use the sharded binary snapshot
path, not task JSON:

Canonical claim boundary:
- [docs/tachiom_reproduction_status.md](tachiom_reproduction_status.md)

```bash
PYTHONPATH=python python python/scripts/materialize_msmarco_colbert_snapshot.py \
  --collection /data/msmarco/collection.tsv \
  --queries /data/msmarco/dev/small/queries.tsv \
  --qrels /data/msmarco/dev/small/qrels \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --document-batch-size 16 \
  --query-batch-size 64 \
  --shard-max-vectors 4000000 \
  --resume
```

The current MS MARCO paper-scale estimate for this format is
`155550734592` document payload bytes with `float16` document vectors and
`uint32` token ids. This keeps vector count explicit: `598000000` document
token vectors at dim `128`.

For bounded judged snapshot slices, add `--include-query-positives` so selected
query positives are present even when they fall outside the document prefix.
Without that flag, a prefix snapshot is a pipeline smoke only and judged MRR can
be zero by construction.

Tachiom can be built from a selected snapshot slice without task JSON:

```bash
PYTHONPATH=python python python/scripts/bench_tachiom_snapshot.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --engine tachiom_tac_hnsw_pq \
  --document-limit 10000 \
  --query-limit 128 \
  --max-vector-count 1000000 \
  --tac-centroid-count 32768 \
  --tac-micro-token-threshold 128 \
  --tac-small-token-threshold 256 \
  --tac-active-token-floor 4 \
  --tac-min-vectors-per-centroid 39 \
  --tac-kmeans-iterations 10 \
  --tac-centroids-per-query-vector 120 \
  --tac-candidate-k 1000 \
  --tac-candidate-pruning-alpha 0.35 \
  --hnsw-max-neighbors 32 \
  --hnsw-ef-construction 1500 \
  --hnsw-ef-search 180 \
  --pq-subspace-count 32 \
  --pq-codebook-size 256 \
  --pq-kmeans-iterations 10 \
  --pq-training-sample-count 32768 \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot/tachiom_slice_summary.json
```

This is a selected-slice benchmark bridge. It avoids task JSON, but still loads
the selected documents into one in-memory matrix before invoking the current
reference index.

Streaming TAC/PQ artifact construction avoids both task JSON and full selected
document-matrix materialization:

```bash
PYTHONPATH=python python python/scripts/build_tachiom_streaming_index.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_index \
  --tac-centroid-count 4000000 \
  --tac-micro-token-threshold 128 \
  --tac-small-token-threshold 256 \
  --tac-active-token-floor 4 \
  --tac-min-vectors-per-centroid 39 \
  --tac-kmeans-iterations 10 \
  --tac-centroids-per-query-vector 120 \
  --tac-candidate-k 1000 \
  --tac-candidate-pruning-alpha 0.35 \
  --centroid-samples-per-centroid 4 \
  --kmeans-max-centroids-per-token 256 \
  --kmeans-max-samples-per-token 4096 \
  --assignment-vector-chunk-size 2048 \
  --assignment-centroid-chunk-size 4096 \
  --posting-partition-count 128 \
  --pq-subspace-count 32 \
  --pq-codebook-size 256 \
  --pq-kmeans-iterations 10 \
  --pq-training-sample-count 32768
```

The resulting streaming artifact can be searched directly through memmaps:

```bash
PYTHONPATH=python python python/scripts/bench_tachiom_streaming_index.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --index .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_index \
  --query-limit 6980 \
  --warmup-iterations 1 \
  --measurement-iterations 3 \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_summary.json \
  --emit-quiet-mean
```

For dim128 artifacts, the native Mojo residual-PQ reader avoids the Python
query-time scoring loops while using the same streaming index files:

```bash
PYTHONPATH=python python python/scripts/bench_tachiom_streaming_index.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --index .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_index \
  --engine streaming_tac_pq_mojo \
  --query-limit 6980 \
  --warmup-iterations 2 \
  --measurement-iterations 5 \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_mojo_summary.json \
  --emit-quiet-mean
```

A persisted centroid HNSW graph can be built as a sidecar:

```bash
PYTHONPATH=python python python/scripts/build_tachiom_streaming_hnsw.py \
  --index .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_index \
  --hnsw-max-neighbors 32 \
  --hnsw-ef-construction 1500 \
  --hnsw-ef-search 180 \
  --hnsw-level-probability 0.0625
```

The HNSW sidecar can then be used for streaming artifact search:

```bash
PYTHONPATH=python python python/scripts/bench_tachiom_streaming_index.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --index .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_index \
  --engine streaming_tac_hnsw_pq \
  --query-limit 6980 \
  --warmup-iterations 1 \
  --measurement-iterations 3 \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_hnsw_summary.json \
  --emit-quiet-mean
```

For dim128 artifacts, the native HNSW+PQ reader uses the same sidecar and
reranks candidate-window document tokens without a full query-by-centroid score
table:

```bash
PYTHONPATH=python python python/scripts/bench_tachiom_streaming_index.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --index .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_index \
  --engine streaming_tac_hnsw_pq_mojo \
  --query-limit 6980 \
  --warmup-iterations 2 \
  --measurement-iterations 5 \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_hnsw_mojo_summary.json \
  --emit-quiet-mean
```

To tune the native reader's same-shape query batch cap, run the USL sweep
against the already-built artifact. The sweep changes only batching; document
vectors, query vectors, candidate budgets, rankings, and index bytes stay
fixed. Use the best measured median batch cap as an input to later benchmark
runs rather than treating the fitted USL peak as proof:

```bash
bash scripts/run_bench_quiet.sh --repeats 1 --timeout-seconds 60 --force -- \
  pixi run env PYTHONPATH=python python \
  python/scripts/sweep_tachiom_streaming_usl.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --index .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_index \
  --engine streaming_tac_hnsw_pq_mojo \
  --batch-sizes 1,2,4,8,16,32,64,128 \
  --warmup-iterations 1 \
  --measurement-iterations 3 \
  --sweep-repeats 3 \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_index/usl_hnsw_pq_mojo_batch_sweep.json \
  --emit-quiet-mean
```

Then pass the chosen cap explicitly:

```bash
PYTHONPATH=python python python/scripts/bench_tachiom_streaming_index.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --index .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_index \
  --engine streaming_tac_hnsw_pq_mojo \
  --max-query-batch-size 64 \
  --query-limit 6980 \
  --warmup-iterations 2 \
  --measurement-iterations 5 \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_hnsw_mojo_batch64_summary.json \
  --emit-quiet-mean
```

For larger artifacts where Python-list materialization becomes the bottleneck,
use the address-backed variant. It reads the same memmap-backed arrays by
address and keeps query-time results comparable, but it is currently slower
than the List-backed native engine on the bounded local slices:

```bash
PYTHONPATH=python python python/scripts/bench_tachiom_streaming_index.py \
  --snapshot .cache/kayak/msmarco_colbertv2_f16_snapshot \
  --index .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_index \
  --engine streaming_tac_hnsw_pq_mojo_address \
  --query-limit 6980 \
  --warmup-iterations 2 \
  --measurement-iterations 5 \
  --output .cache/kayak/msmarco_colbertv2_f16_snapshot/streaming_tachiom_hnsw_mojo_address_summary.json \
  --emit-quiet-mean
```

This is still not a paper-throughput command by itself: the current graph
builder is the Python reference and has not been validated at paper-scale
centroid counts.

For repeatable bounded scale gates, use the combined runner. It materializes a
judged-positive snapshot, builds the streaming TAC/PQ artifact, optionally
benchmarks the native Mojo reader, optionally builds the HNSW sidecar, and
writes one summary. The runner emits per-stage progress and records
`stage_timings_seconds` because snapshot materialization can dominate wall
time on larger local slices:

```bash
PYTHONPATH=python python python/scripts/run_tachiom_streaming_scale_gate.py \
  --collection /data/msmarco/collection.tsv \
  --queries /data/msmarco/dev/small/queries.tsv \
  --qrels /data/msmarco/dev/small/qrels \
  --output-root .cache/kayak/tachiom_streaming_scale_docs1500_q48_c32768 \
  --document-limit 1500 \
  --query-limit 48 \
  --tac-centroid-count 32768 \
  --max-exact-vector-count 150000 \
  --include-address-mojo \
  --overwrite
```

## What This Note Does Not Claim

This note does not claim:
- that the current ladder is final
- that every rung already has the best possible dataset
- that a local stronger ceiling is a substitute for a verified cross-attention
  or long-context ceiling

It only sets the current contract:
- keep the ladder explicit
- keep vector counts explicit
- keep stronger ceilings labeled by the actual path used
