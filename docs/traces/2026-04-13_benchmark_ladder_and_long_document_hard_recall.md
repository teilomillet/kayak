# 2026-04-13 Benchmark Ladder And Long-Document Hard Recall

## Goal

Close the remaining Priority 8 benchmark-ladder items from
[TODO.md](../../TODO.md):

- write the benchmark ladder down explicitly
- add one code- or long-document-shaped hard-recall lane through the existing
  stage-aware JSON pipeline
- keep stronger-ceiling artifacts labeled by the actual path used

## Why This Tranche

Local code and doc inspection before the change showed:
- the repo already had:
  - real public text hard-recall slices
  - one scalable synthetic conjunction-style hard-recall family
  - one local clause-text stronger ceiling
- but it still did not have:
  - one explicit benchmark ladder note with exit criteria
  - one long-document-shaped hard-recall family on the same stage-aware output
  - one structured ceiling label that said what expensive path was actually
    used

Reason:
- without those pieces, the repo could benchmark harder tasks, but the
  progression between benchmark rungs remained implicit
- the clause-text ceiling was described honestly in prose, but the machine
  output still exposed only `method_kind`

## Changes

Added:
- [docs/benchmark_ladder.md](../benchmark_ladder.md)
- [kayak/benchmarks/long_document_hard_recall_fixture.mojo](../../kayak/benchmarks/long_document_hard_recall_fixture.mojo)
- [benchmarks/long_document_hard_recall_stage_aware.mojo](../../benchmarks/long_document_hard_recall_stage_aware.mojo)
- [tests/test_long_document_hard_recall_fixture.mojo](../../tests/test_long_document_hard_recall_fixture.mojo)

Changed:
- [kayak/benchmarks/ceiling_comparison_json.mojo](../../kayak/benchmarks/ceiling_comparison_json.mojo)
- [tests/test_ceiling_comparison_json.mojo](../../tests/test_ceiling_comparison_json.mojo)
- [docs/hard_recall_evaluation.md](../hard_recall_evaluation.md)
- [docs/late_interaction_2030.md](../late_interaction_2030.md)
- [TODO.md](../../TODO.md)
- [pyproject.toml](../../pyproject.toml)

## Design Decision

Decision:
- implement the missing lane as a deterministic long-document synthetic family,
  not as a real code dataset yet

Reason:
- the repo already had a deterministic synthetic hard-recall substrate
- the repo does not yet have a code-data loader or code-specific encoder
  contract that would be honest enough for a real code retrieval lane
- a long-document lane still addresses Omar’s benchmark point by adding a
  different retrieval failure mode through the same stage-aware surface

The new family is shaped so that:
- documents have a long noisy prefix of common filler concepts
- the exact matching concepts appear only at the end
- `document_proxy` is stressed mechanically because it averages only the first
  `document_vector_budget` vectors
- low `centroid_budget` also loses the late exact dimensions because the common
  filler dimensions dominate global token counts

## Verification Commands

```bash
pixi run mojo -I . tests/test_long_document_hard_recall_fixture.mojo
pixi run mojo -I . tests/test_ceiling_comparison_json.mojo
pixi run mojo -I . benchmarks/long_document_hard_recall_stage_aware.mojo
```

## Verified Results

### Long-document fixture tests

`tests/test_long_document_hard_recall_fixture.mojo`
- `5/5` passed

Verified properties:
- exact full scan remains perfect on the new family
- the family reports explicit query and document vector counts
- a budgeted `document_proxy` path loses stage-1 recall
- exact reranking recovers when the candidate window fully covers the oracle

### Long-document stage-aware artifact

Artifact:

```text
.cache/kayak/long_document_hard_recall_stage_aware_search.json
```

Measured profiles:

1. `late_suffix_docs544_vec100`
   - exact full scan: `candidate_recall = 1.0`
   - all tested non-exact stage-1 plans stayed at `0.0` through
     `candidate_k = 32`
   - the tested plans reached only `0.125` at `candidate_k = 64`
   - the tested plans reached only `0.25` at `candidate_k = 128`

2. `late_suffix_docs2088_vec133`
   - exact full scan: `candidate_recall = 1.0`
   - all tested non-exact stage-1 plans stayed at `0.0` through
     `candidate_k = 128`

Interpretation:
- this long-document family is substantially harsher than the earlier
  conjunction-style synthetic family for the current budgeted stage-1 sidecars
- it gives the repo one explicit late-evidence benchmark lane instead of only a
  conjunction-style lane

### Ceiling labeling

`tests/test_ceiling_comparison_json.mojo`
- `3/3` passed

The structured ceiling summary now emits:
- `comparison_role`
- `execution_path_kind`

Example:
- the clause-text path is labeled as
  `comparison_role = "local_stronger_ceiling"`
- its execution path is labeled as
  `execution_path_kind = "exact_then_clause_text_rerank"`

That keeps the artifact honest:
- it is a local stronger ceiling
- it is not a cross-encoder claim
- it is not a long-context LLM claim

## What This Does Not Claim

This change does not claim:
- that the long-document synthetic family is a substitute for a real code
  retrieval benchmark
- that the current clause-text path is the best possible expensive reference
  path
- that the current stage-1 families are irreparably weak on long documents

It only establishes:
- the benchmark ladder is now explicit in-repo
- the repo now has a second scalable hard-recall lane with a different failure
  mode
- stronger-ceiling artifacts now label the actual execution path they used
