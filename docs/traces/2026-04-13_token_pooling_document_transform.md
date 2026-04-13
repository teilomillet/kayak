## Token Pooling As A Document Transform

Date: April 13, 2026

## Claim

Token pooling should land as a document-representation transform on the sealed
segment boundary, not as a new stage-1 engine family.

The evidence needed was:

- a concrete transform execution path over stored documents
- preservation through sealing and compaction without double-application
- a real benchmark on the BrowseComp-Plus gold slice showing what pooled exact
  search actually does to vector count, storage, latency, and task metrics

## Sources Checked

- `docs/architecture/extensibility_wall.md`
- `docs/traces/2026-04-13_document_representation_transform_contract.md`
- `kayak/collections/segment_builder.mojo`
- `kayak/collections/compaction_runtime.mojo`
- `kayak/collections/document_representation_transform.mojo`
- `kayak/collections/document_representation_transform_runtime.mojo`
- `.cache/kayak/browsecomp_plus_gold_benchmark.json`
- `.cache/kayak/browsecomp_plus_gold_token_pooling.json`
- `.cache/kayak/bench_quiet/20260413T160510Z/`
- `.cache/kayak/bench_quiet/20260413T160605Z/`

## Design Chosen

- implement token pooling as a document-side transform over `EncodedDocument`
- support two explicit policies:
  - `hierarchical`
  - `sequential`
- preserve transform provenance on sealed segments
- require transform-chain consistency across source segments during compaction
- keep the exact late-interaction scorer unchanged

Reason:

- this isolates the paper-shaped change at the representation seam instead of
  entangling it with planner semantics or stage-1 engine contracts
- it lets us compare pooled exact search against the current exact reference
  path before making any claims about staged retrieval

## Validation Commands

```bash
pixi run mojo -I . tests/test_document_representation_transform_runtime.mojo
pixi run mojo -I . tests/test_segment_builder_policy.mojo
pixi run mojo -I . tests/test_collection_storage.mojo
pixi run mojo -I . tests/test_collection_compaction.mojo
pixi run mojo -I . tests/test_document_representation_transform.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
pixi run mojo -I . tests/test_snapshot_inventory.mojo
pixi run mojo -I . tests/test_token_pooling_json.mojo
pixi run build_browsecomp_plus_task_json
bash scripts/run_bench_quiet.sh --timeout-seconds 5 --force --repeats 1 -- pixi run bench_browsecomp_plus_gold_token_pooling_raw
bash scripts/run_bench_quiet.sh --timeout-seconds 5 --force --repeats 1 -- pixi run bench_browsecomp_plus_gold_raw
```

## Important Benchmark Caveat

The quiet-wrapper runs timed out after `5s` on this host and had to continue
with `--force`.

Observed competing host CPU during the two runs stayed roughly in the
`372%` to `621%` range.

So:

- the absolute timings below are not quiet-host deployment numbers
- the relative comparisons are still worth recording because the baseline and
  pooled runs used the same host, the same cached gold slice, and the same
  wrapper protocol

## Results

### Baseline exact gold slice

- dataset: `Tevatron/browsecomp-plus/gold-slice`
- queries: `4`
- documents: `90`
- full stored vectors: `15,756`
- packed-index storage: `8,068,326` bytes
- `ndcg@10 = 0.2851267779084149`
- `mrr@10 = 0.4375`
- `recall@10 = 0.30833333333333335`
- `success@10 = 0.75`
- mean exact-search time: `0.0011110242268041236 s`

### Hierarchical pooling

#### Factor `2`

- pooled vectors: `7,879`
- vector ratio vs baseline: `0.500063`
- storage bytes: `4,035,260`
- storage ratio vs baseline: `0.500136`
- mean reference top-`10` recall vs exact baseline ranking: `0.975`
- `ndcg@10 = 0.2851267779084149`
- mean exact-search time: `0.00053779 s`
- same-host speedup vs baseline: `2.06591x`

Interpretation:

- this is the cleanest first positive result
- roughly half the stored vectors and half the storage preserved the current
  gold-slice task metric exactly on this slice
- the small drop in reference overlap (`0.975`) shows the ranking is not
  identical even when the top-line task metric is unchanged

#### Factor `3`

- pooled vectors: `5,253`
- vector ratio vs baseline: `0.333397`
- storage bytes: `2,690,742`
- storage ratio vs baseline: `0.333494`
- mean reference top-`10` recall vs exact baseline ranking: `0.9`
- `ndcg@10 = 0.3333708080832098`
- mean exact-search time: `0.0003044406779661017 s`
- same-host speedup vs baseline: `3.64939x`

Interpretation:

- this run improved the small-slice task metric while becoming less faithful to
  the exact baseline ranking
- that is a real measured divergence between:
  - faithfulness to the exact late-interaction oracle
  - benchmark utility on this `4`-query slice
- this does **not** prove pooled semantics are globally better
- it does prove that exact-reference overlap and final task metric are
  different axes, which justifies keeping the exact stage-2 oracle explicit

#### Factors `4`, `6`, and `8`

- factor `4`:
  - pooled vectors: `3,940`
  - reference recall: `0.925`
  - `ndcg@10 = 0.24339620066315637`
  - mean search time: `0.00024364257028112448 s`
- factor `6`:
  - pooled vectors: `2,628`
  - reference recall: `0.8250000000000001`
  - `ndcg@10 = 0.2532139753355904`
  - mean search time: `0.00015416716417910448 s`
- factor `8`:
  - pooled vectors: `2,014`
  - reference recall: `0.8`
  - `ndcg@10 = 0.2523491038749828`
  - mean search time: `0.00013333983286908078 s`
  - same-host speedup vs baseline: `8.33228x`

Interpretation:

- the latency curve continued to improve as vector count fell
- quality on this slice did not collapse immediately, but the drop in reference
  overlap became material
- these stronger pooling factors look usable only if a later exact stage is
  still available to restore reference semantics

### Sequential pooling baseline

At the same factor and storage budget, hierarchical pooling beat sequential
pooling on this slice.

At factor `2`:

- both policies stored `7,879` vectors and `4,035,260` bytes
- hierarchical `ndcg@10 = 0.2851267779084149`
- sequential `ndcg@10 = 0.26723303363941064`
- `ndcg` delta = `0.0178937`
- hierarchical reference recall = `0.975`
- sequential reference recall = `0.875`
- reference-recall delta = `0.1`
- speed ratio was nearly neutral:
  - sequential / hierarchical search time = `0.970007`

Interpretation:

- the quality gap came from the pooling policy, not from a different vector
  budget
- that makes the hierarchical policy the justified default for the first
  token-pooling implementation

### Storage Observation

The pooled artifacts stayed at roughly `512` bytes per vector across all
configurations.

That is expected from the current storage layout:

- vectors are still stored as raw `128 x float32`
- token pooling reduces vector count
- token pooling does **not** compress per-vector representation

So token pooling and storage compression remain separate work items.

## What This Validates

- the document-transform seam was the right integration point
- token pooling can be executed at segment build time and preserved across
  compaction without changing the exact scorer
- a pooled exact index is already enough to measure concrete tradeoffs on a
  real benchmark slice
- hierarchical pooling is a better first-class policy than sequential pooling
  at the same budget on this slice

## What This Does Not Yet Validate

- quiet-host production latency
- transform build cost, because this benchmark records search-time effects but
  not transform-time cost
- larger-slice or larger-corpus generalization
- a staged retrieval story where pooled stage 1 is followed by exact stage 2

## Next Step

The next justified step is not more transform plumbing.

It is a staged benchmark that separates:

- pooled candidate generation or pooled exact prefiltering
- exact late-interaction stage-2 restoration
- final task quality after restoration

Reason:

- the factor-`3` result already shows that slice metric and exact-reference
  faithfulness can move in different directions
- that is precisely the boundary where staged late interaction becomes the
  right next measurement surface
