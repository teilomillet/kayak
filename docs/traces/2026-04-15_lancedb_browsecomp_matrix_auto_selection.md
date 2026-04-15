# 2026-04-15 LanceDB BrowseComp Matrix with Indexed Auto-Selection

## Question

If we apply the same LanceDB workflow used in the new hard-matrix run to the
repo's BrowseComp-Plus gold and evidence slices, do we still see the older
pattern from the previous one-point LanceDB baseline, or do broader indexed
sweeps find better operating points?

The narrower claim tested here is:

- not "BrowseComp proves everything"
- but:
  - "the same `sweep -> auto-select -> rerun` workflow used on BRIGHT, LEMB,
    legal, and R2MED also behaves coherently on BrowseComp-Plus gold and
    evidence"

## Why this step was justified

The older BrowseComp comparison in
[2026-04-15_lancedb_kayak_browsecomp_gold_matrix.md](2026-04-15_lancedb_kayak_browsecomp_gold_matrix.md)
used one frozen LanceDB indexed point inside the broader task-comparison suite.

That result was useful, but limited:

- the indexed point slightly improved latency versus LanceDB scan
- it lost quality on gold
- it did not answer whether a broader indexed sweep would find a better
  LanceDB operating point on the same slices

The newer hard-matrix workflow is better for that question because it keeps the
selection policy explicit and reuses the same sweep surface across slices.

## Command

```bash
env PYTHONPATH=python bash scripts/run_bench_quiet.sh --repeats 1 --max-other-cpu 1200 -- \
  uv run --python 3.11 --with lancedb --with faiss-cpu python \
    python/scripts/bench_lancedb_hard_matrix.py \
    --dataset-key browsecomp_plus_gold \
    --dataset-key browsecomp_plus_evidence \
    --output-root .cache/kayak/lancedb_browsecomp_matrix \
    --artifact-prefix lancedb_browsecomp_matrix \
    --rebuild-count 5 \
    --warmup-iterations 1 \
    --measurement-iterations 3
```

Aggregate artifact:

- `.cache/kayak/lancedb_browsecomp_matrix/lancedb_browsecomp_matrix_summary.json`

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

## Dataset shape

Both slices use the same encoded task profile:

- queries: `4`
- documents: `90`
- nominal query vectors: `32`
- nominal document vectors: `175`
- primary metric: `ndcg`

## Results

### BrowseComp-Plus Gold

Sources:

- `.cache/kayak/lancedb_browsecomp_matrix/browsecomp_plus_gold/index_sweep/browsecomp_plus_gold_indexed_sweep_summary.json`
- `.cache/kayak/lancedb_browsecomp_matrix/browsecomp_plus_gold/candidate_compare_auto/browsecomp_plus_gold_candidate_selection.json`
- `.cache/kayak/lancedb_browsecomp_matrix/browsecomp_plus_gold/candidate_compare_auto/browsecomp_plus_gold_candidate_bundle.json`

Baselines:

- Kayak exact:
  - `ndcg=0.28512677790387286`
  - `0.0010202049161307514 s`
- LanceDB scan:
  - `ndcg=0.28512677790387286`
  - `0.025608777747644734 s`

Sweep top rows by frozen quality:

- `refine1`
  - `ndcg=0.30609858759938265`
  - `0.023288153668665968 s`
  - `+0.020971809695509794` vs scan
- `nprobe64`
  - `ndcg=0.29573680885984174`
  - `0.022123361232942343 s`
  - `+0.010610030955968878` vs scan

Auto-selected candidates:

- `default`
- `refine1`
- `nprobe64`

Rerun bundle winners:

- best indexed quality:
  - `lancedb_indexed_refine1`
  - `ndcg=0.29258089529952214`
  - `0.02455334033196171 s`
- fastest quality-improving indexed candidate:
  - `lancedb_indexed_refine1`
  - same row on the rerun

Interpretation:

- the older one-point indexed baseline was too pessimistic
- on gold, the broader sweep does find an indexed setting that beats scan on
  both quality and latency
- `refine1` is the strongest rerun point even though the sweep also selected
  `nprobe64` as the fastest quality-improving candidate

That last point matters:

- sweep-time ordering is useful for candidate selection
- rerun-time ordering still needs to be checked rather than assumed

### BrowseComp-Plus Evidence

Sources:

- `.cache/kayak/lancedb_browsecomp_matrix/browsecomp_plus_evidence/index_sweep/browsecomp_plus_evidence_indexed_sweep_summary.json`
- `.cache/kayak/lancedb_browsecomp_matrix/browsecomp_plus_evidence/candidate_compare_auto/browsecomp_plus_evidence_candidate_selection.json`
- `.cache/kayak/lancedb_browsecomp_matrix/browsecomp_plus_evidence/candidate_compare_auto/browsecomp_plus_evidence_candidate_bundle.json`

Baselines:

- Kayak exact:
  - `ndcg=0.26234761964459435`
  - `0.0010200485897560914 s`
- LanceDB scan:
  - `ndcg=0.26234761964459435`
  - `0.02836557966656983 s`

Sweep top rows by frozen quality:

- `refine1`
  - `ndcg=0.3303035134406606`
  - `0.023379502321841817 s`
  - `+0.06795589379606624` vs scan
- `nprobe64`
  - `ndcg=0.30016545851933345`
  - `0.022255731584504134 s`
  - `+0.0378178388747391` vs scan
- `default`
  - `ndcg=0.28301185252006684`
  - `0.02218480776622842 s`
  - `+0.020664232875472487` vs scan

Auto-selected candidates:

- `default`
- `refine1`

Rerun bundle winners:

- best indexed quality:
  - `lancedb_indexed_refine1`
  - `ndcg=0.3160418704842516`
  - `0.023918159721263994 s`
- fastest quality-improving indexed candidate:
  - `lancedb_indexed_default`
  - `ndcg=0.2936076431552892`
  - `0.0229412013432011 s`

Interpretation:

- evidence is the stronger LanceDB upside slice here
- `refine1` gives a substantial quality lift relative to scan
- default indexed still matters because it is the lower-latency improving point

## Comparison with the older BrowseComp Gold trace

Earlier gold result from
[2026-04-15_lancedb_kayak_browsecomp_gold_matrix.md](2026-04-15_lancedb_kayak_browsecomp_gold_matrix.md):

- LanceDB `IVF_PQ` frozen:
  - `ndcg=0.24669230148616947`
  - `0.0239973646 s`

Current gold rerun winner from this sweep-driven workflow:

- `lancedb_indexed_refine1`
  - `ndcg=0.29258089529952214`
  - `0.02455334033196171 s`

What changed:

- the earlier benchmark tested one indexed operating point
- the current workflow tested eight explicit indexed configs, then reran the
  selected candidates

What this supports:

- the older result should not be treated as evidence that indexed LanceDB is
  inherently quality-regressing on BrowseComp gold
- the stronger statement is narrower and better justified:
  - a broad indexed sweep is necessary to find the useful LanceDB tradeoff on
    BrowseComp

## Practical conclusion

What the evidence supports:

- the hard-matrix LanceDB workflow generalizes cleanly to BrowseComp gold and
  evidence
- on both BrowseComp slices, the best rerun indexed candidate beats LanceDB
  scan on quality and is still faster than scan
- `refine1` is the strongest quality setting on both slices in this run
- evidence shows the larger quality upside for LanceDB indexed search

What the evidence does **not** support:

- that LanceDB beats Kayak overall on BrowseComp
- that one indexed config should now be hard-coded globally
- that a sweep summary alone is enough without the rerun bundle

The current honest statement is:

- Kayak exact remains much faster than the LanceDB paths
- but the best LanceDB indexed operating point on BrowseComp is meaningfully
  stronger than the old one-point baseline had suggested
