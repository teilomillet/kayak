# 2026-04-13: Mirror Text Sidecars For Planner Evidence

## Why This Step Exists

The repository already had:

- one-segment collection mirrors for public benchmark slices
- explicit stage-2 operators, including `clause_text`
- planner-evidence summaries with a `stage2_kind` field

But the mirror path only persisted:

- packed late-interaction vectors
- search-native stage-1 sidecars

It did **not** persist a sealed-segment `text_corpus`.

That made the planner-evidence surface incomplete:

- `exact_late_interaction` could be benchmarked on mirrored public slices
- `clause_text` could not, even though the engine already supported it

The missing piece was not a new stage-2 algorithm. It was the storage seam that
lets a mirrored snapshot carry the same optional text sidecar as a hosted
sealed segment.

## What Changed

### Collection mirror now accepts optional text sidecars

Updated:

- [kayak/collections/mirror.mojo](../../kayak/collections/mirror.mojo)

`ensure_one_segment_collection_mirror(...)` now has a text-aware overload that
accepts a `DocumentTextCorpus`.

The mirror path now:

- validates that text-corpus doc ids match the packed-index doc id order
- rejects misaligned corpora instead of silently saving ambiguous text
- persists `text_corpus` when at least one document has non-empty text
- includes text-sidecar bytes in segment and snapshot byte-size accounting

This keeps the mirror epistemically aligned with the actual packed index.

### BrowseComp text-corpus loaders are explicit

Updated:

- [kayak/interop/browsecomp_plus_subset.mojo](../../kayak/interop/browsecomp_plus_subset.mojo)
- [kayak/interop/browsecomp_plus_gold_subset.mojo](../../kayak/interop/browsecomp_plus_gold_subset.mojo)
- [kayak/interop/__init__.mojo](../../kayak/interop/__init__.mojo)

Added:

- `load_browsecomp_plus_real_subset_document_text_corpus(...)`
- `load_browsecomp_plus_gold_real_subset_document_text_corpus(...)`

Reason:

- benchmark callers should not have to hardcode JSON cache paths
- the interop layer already owns the task-shape decoding logic

### Gold planner-evidence benchmark now measures two stage-2 families

Updated:

- [benchmarks/browsecomp_plus_gold_planner_evidence.mojo](../../benchmarks/browsecomp_plus_gold_planner_evidence.mojo)

The benchmark now:

- loads BrowseComp gold document texts through the new interop helper
- mirrors them into the sealed benchmark collection
- emits planner-evidence summaries for:
  - `exact_late_interaction`
  - `clause_text`

This changes the gold planner-evidence artifact from:

- `20` summaries

to:

- `40` summaries

with stage-2 kinds:

- `exact_late_interaction`
- `clause_text`

### Existing clause-text consumers were rewired to the helper

Updated:

- [benchmarks/browsecomp_plus_gold_ceiling_comparison.mojo](../../benchmarks/browsecomp_plus_gold_ceiling_comparison.mojo)
- [benchmarks/browsecomp_plus_clause_text_verifier.mojo](../../benchmarks/browsecomp_plus_clause_text_verifier.mojo)

The ceiling comparison now also mirrors the text sidecar into its base
collection. The verifier and ceiling scripts no longer depend on raw cache-path
knowledge for document texts.

## Validation

Low-level mirror tests:

```bash
pixi run mojo -I . tests/test_collection_mirror.mojo
```

Planner-evidence tests:

```bash
pixi run mojo -I . tests/test_planner_evidence_json.mojo
```

Public planner-evidence artifact:

```bash
pixi run bench_browsecomp_plus_gold_planner_evidence_raw
```

Clause-text consumer sanity checks:

```bash
pixi run mojo -I . benchmarks/browsecomp_plus_gold_ceiling_comparison.mojo
pixi run mojo -I . benchmarks/browsecomp_plus_clause_text_verifier.mojo
```

Observed result:

- all listed tests and scripts passed locally
- `.cache/kayak/browsecomp_plus_gold_planner_evidence.json` was regenerated
- the regenerated artifact contains:
  - `40` summaries
  - stage-2 kinds `clause_text` and `exact_late_interaction`

Observed clause-text verifier signal on BrowseComp gold:

- `candidate_k = 20`
- `ndcg@10 = 0.3638261862358497`
- `recall@10 = 0.5583333333333333`
- `success@10 = 1.0`

## Epistemic Boundary

This step proves:

- mirrored public collections can now carry a first-class text sidecar
- planner evidence can measure `clause_text` on mirrored public slices
- clause-text benchmark consumers no longer rely on hidden path knowledge

This step does **not** prove:

- that `clause_text` should be the default stage-2 operator
- that current clause-text quality is globally optimal
- that all public benchmark families should now mirror text sidecars by default

It only proves that the substrate exists and works, which is the right
primitive before making any stronger planner or benchmark-policy claims.
