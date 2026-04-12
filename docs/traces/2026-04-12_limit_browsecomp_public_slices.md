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
