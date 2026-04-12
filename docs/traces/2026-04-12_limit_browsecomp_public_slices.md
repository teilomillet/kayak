# Public Slice Expansion: LIMIT-small and BrowseComp-Plus

Date: April 12, 2026

## Claim

`kayak` should cover at least one more adversarial public retrieval benchmark and one more reasoning-heavy public retrieval benchmark without abandoning the smoke-test workflow.

## Sources Checked

- Google DeepMind `LIMIT` official repo:
  - `LIMIT-small` has `46` docs, `1,000` queries, and `2,000` qrels.
  - the official repo says the data is in MTEB format and can be loaded with Hugging Face datasets.
- MTEB retrieval overview:
  - retrieval tasks commonly report `ndcg_at_10`, which justified adding `nDCG@k` to the evaluation layer.
- BrowseComp-Plus official repo:
  - the benchmark uses a fixed corpus and publishes retrieval-only evaluation with `recall@5,100,1000` and `ndcg@10`.
  - the repo exposes both evidence and gold qrels, and the evidence qrels are the more natural retrieval target for the current smoke slice.
- Hugging Face dataset builders checked locally on April 12, 2026:
  - `Tevatron/browsecomp-plus`: `830` queries
  - `Tevatron/browsecomp-plus-corpus`: `100,195` docs

## Design Chosen

- Added `nDCG@k` to the Mojo eval layer and validated it against a reference implementation.
- Added `LIMIT-small` as an official light public slice:
  - full `46`-document corpus
  - `32` encoded queries
  - exact late interaction end to end in Mojo after the encoder boundary
- Added `BrowseComp-Plus` as light retrieval slices:
  - `4` decrypted official queries
  - an evidence task with official human-verified evidence docs as positives
  - a gold task with official answer-containing gold docs as positives
  - official curated hard negatives from the benchmark rows
  - supporting evidence docs remain in the gold slice corpus as realistic distractors
  - this is intentionally not the full `100k`-document agent benchmark

## Important Boundary

The embedded Mojo Python bridge hit a torch shared-memory failure on the heavy BrowseComp-Plus build path.

Evidence:
- direct Python build of the same slice succeeded
- embedded Mojo build failed with `unable to open shared memory object </torch_...>`

Resolution:
- materialize the BrowseComp-Plus tasks once as:
  - `.cache/kayak/browsecomp_plus_real_subset/python_task_evidence.json`
  - `.cache/kayak/browsecomp_plus_real_subset/python_task_gold.json`
  in a plain Python process
- keep task decoding, storage, packing, search, and evaluation in Mojo

This is a systems workaround, not a benchmark change.
The retrieved documents, queries, vectors, and evaluation path are unchanged by the JSON materialization step.

## Validation Commands

```bash
pixi run test_eval
pixi run test_eval_battle
pixi run test_python_bridge
pixi run test_storage
pixi run bench_limit_small_raw
pixi run build_browsecomp_plus_task_json
pixi run bench_browsecomp_plus_raw
pixi run bench_browsecomp_plus_gold_raw
pixi run bench_browsecomp_plus_diag
```

Quiet-wrapper follow-up attempted:

```bash
pixi run bench_limit_small
pixi run bench_browsecomp_plus
```

Result:
- both quiet runs timed out after `120s`
- competing host CPU stayed far above the `40%` threshold
- observed `other_cpu` stayed roughly in the `323%` to `596%` range during the wait window

So the latency numbers below should be treated as exploratory raw timings, not low-noise decision-quality measurements.

## Results

### LIMIT-small

- shape:
  - `32` queries
  - `46` docs
  - query vectors about `32`
  - doc vectors about `162`
  - dim `128`
- retrieval quality:
  - `ndcg@10 = 0.9674217039448095`
  - `mrr@10 = 0.9479166666666666`
  - `recall@10 = 1.0`
  - `success@10 = 1.0`
- search timing:
  - raw mean `0.0004704584110828386 s`

### BrowseComp-Plus evidence slice

- shape:
  - `4` queries
  - `90` docs
  - query vectors about `32`
  - doc vectors about `175`
  - dim `128`
- retrieval quality:
  - `ndcg@10 = 0.26234761965070796`
  - `mrr@10 = 0.4652777777777778`
  - `recall@10 = 0.2722222222222222`
  - `success@10 = 1.0`
- search timing:
  - raw mean `0.0009456092072934178 s`

### BrowseComp-Plus gold slice

- shape:
  - `4` queries
  - `90` docs
  - query vectors about `32`
  - doc vectors about `175`
  - dim `128`
- retrieval quality:
  - `ndcg@10 = 0.2851267779084149`
  - `mrr@10 = 0.4375`
  - `recall@10 = 0.30833333333333335`
  - `success@10 = 0.75`
- search timing:
  - raw mean `0.0016518212535014006 s`

## Interpretation

- `LIMIT-small` is almost solved by exact ColBERTv2 on this light slice, which makes it useful as a compact adversarial regression target.
- `BrowseComp-Plus` is materially harder even after slicing, which is good evidence that it adds real pressure rather than duplicating BEIR-style smoke tests.
- On this current `4`-query light slice, the gold task is slightly better on `nDCG@10` and recall than the evidence task, but worse on `success@10`. That is useful signal: the two qrel sets are not interchangeable, so carrying both is justified.
- The BrowseComp-Plus slice is still light enough for iterative CPU profiling while preserving benchmark-native hard negatives.

## Query-Level Diagnostic Follow-Up

Command:

```bash
pixi run bench_browsecomp_plus_diag
```

Purpose:
- hold the ranked retrieval results fixed
- score the same retrieved list against both evidence and gold qrels
- distinguish low-rank partial hits from complete misses

Observed on April 12, 2026:
- query `769` is a partial-hit case:
  - the top two hits are relevant under both qrel sets
  - the remaining score loss comes from additional relevant documents staying below the top `10`
- queries `770` and `771` are low-rank cases:
  - the first relevant hit appears at rank `4` and rank `2`
  - retrieval finds some signal, but not early enough
- query `772` is the clean miss:
  - `gold success@10 = 0.0`
  - none of the top `10` docs are gold-relevant
  - the evidence slice still shows a weak late hit, which explains why evidence and gold disagree most strongly on this query

Interpretation:
- the current BrowseComp failure mode is not uniform
- some queries need earlier concentration of already-retrieved relevant evidence
- at least one query needs better candidate generation entirely because the gold-positive document never enters the top `10`
- that makes the diagnostic benchmark a justified next baseline for any retrieval-side changes

## Rank-Coverage Follow-Up

Command:

```bash
pixi run bench_browsecomp_plus_ranks
```

Purpose:
- score the full `90`-document slice instead of truncating at `k=10`
- verify that evidence and gold caches produce the same exact ranking
- measure how far the missed gold-positive docs sit below the current top-`10` cutoff

Observed on April 12, 2026:
- the evidence and gold rankings are identical for every query in the slice
  - that matters because it confirms the current difference is qrels, not index drift
- query `772` is a near miss rather than a deep-corpus miss:
  - `best_evidence_rank = 9`
  - `best_gold_rank = 18`
  - the only gold-positive doc is `93372`
  - `score(93372) = 12.640364`
  - `top10_cutoff = 13.685546`
  - `delta_to_top10 = -1.0451822`
- the evidence side of the same query shows the model already retrieves township-context evidence:
  - evidence doc `11848` reaches rank `9`
  - evidence doc `92455` is just outside the cutoff at rank `11`
  - the gold doc `93372` is present but later at rank `18`
- the candidate-window implication is concrete:
  - gold `hits@5 = 0 / 1`
  - gold `hits@10 = 0 / 1`
  - gold `hits@20 = 1 / 1`
  - gold `hits@50 = 1 / 1`

Interpretation:
- this does not look like a hopeless first-stage failure
- it looks like a clause-selection failure where the retriever locks onto general Gugulethu context before the school-specific answer document
- a future richer second-stage verifier with `candidate_k >= 20` could, in principle, recover this query because the answer document is already in the candidate set by rank `18`
- the current exact late-interaction verifier does not solve that class of miss by itself because it reranks with the same exact MaxSim signal rather than a richer text-level model

## Clause-Text Prototype

Command:

```bash
pixi run bench_browsecomp_plus_clause
```

Prototype design:
- keep the main verifier pipeline unchanged and vector-only
- load document text from the existing BrowseComp JSON task cache
- rerank a `candidate_k = 20` exact-search window with a small clause-aware lexical boost
- weight the final answer-bearing clause more heavily than earlier setup clauses

This is a benchmarked prototype, not a claim that the generic verifier path is now text-aware.

Observed on April 12, 2026:
- the focus miss on query `772` is recovered inside the existing candidate window:
  - baseline gold rank for doc `93372`: `18`
  - clause-reranked gold rank for doc `93372`: `4`
- gold slice metrics improve materially:
  - baseline `ndcg@10 = 0.2851267779084149`
  - clause-text `ndcg@10 = 0.3638261862358497`
  - baseline `mrr@10 = 0.4375`
  - clause-text `mrr@10 = 0.4375`
  - baseline `recall@10 = 0.30833333333333335`
  - clause-text `recall@10 = 0.5583333333333333`
  - baseline `success@10 = 0.75`
  - clause-text `success@10 = 1.0`
- evidence slice is mixed rather than uniformly better:
  - baseline `ndcg@10 = 0.26234761965070796`
  - clause-text `ndcg@10 = 0.23928446246331353`
  - baseline `mrr@10 = 0.4652777777777778`
  - clause-text `mrr@10 = 0.4375`
  - baseline `recall@10 = 0.2722222222222222`
  - clause-text `recall@10 = 0.28055555555555556`
  - baseline `success@10 = 1.0`
  - clause-text `success@10 = 1.0`

Interpretation:
- the prototype validates the underlying hypothesis:
  - a text-aware reranker can recover the answer-bearing gold document for the current hard miss because the document is already inside the top-`20` candidate window
- the prototype is not yet a global improvement:
  - it helps the gold-answer objective
  - it slightly regresses the evidence objective
- that means the next sound step is not "turn it on everywhere"
- the next sound step is to keep it as an explicit benchmark path and either:
  - tune the clause scorer more carefully
  - or replace it with a stronger text-level reranker once storage exposes text as a first-class artifact
