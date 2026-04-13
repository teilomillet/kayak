# Stage-2 Primitives

Status: `architecture note`  
Date: `2026-04-13`

This note records the current repository state after a fresh code scan and
captures the stage-2 boundary that has now been implemented from first
principles.

The purpose was not to replace the current exact rerank path immediately.
The implemented goal was to make the stage-2 contract explicit enough that:

- exact late interaction remains the default correctness anchor
- richer rerankers can attach without becoming side paths
- the Python SDK can expose one stable public model instead of re-exporting
  whichever engine detail happens to exist today

## Sources Checked

Repository scan:

- recent `git log --oneline --decorate -15`
- [kayak/planning/search_plan.mojo](../../kayak/planning/search_plan.mojo)
- [kayak/planning/exact_stage.mojo](../../kayak/planning/exact_stage.mojo)
- [kayak/planning/execution.mojo](../../kayak/planning/execution.mojo)
- [kayak/planning/explain.mojo](../../kayak/planning/explain.mojo)
- [kayak/verifier/pipeline.mojo](../../kayak/verifier/pipeline.mojo)
- [kayak/verifier/exact_late_interaction.mojo](../../kayak/verifier/exact_late_interaction.mojo)
- [kayak/verifier/clause_text.mojo](../../kayak/verifier/clause_text.mojo)
- [kayak/service/search_contracts.mojo](../../kayak/service/search_contracts.mojo)
- [python/kayak/__init__.py](../../python/kayak/__init__.py)
- [python/kayak_bridge/planned_search.py](../../python/kayak_bridge/planned_search.py)
- [python/kayak_bridge/search_plan.py](../../python/kayak_bridge/search_plan.py)
- [docs/hosted_engine_grand_plan.md](../hosted_engine_grand_plan.md)
- [docs/python_sdk_charter.md](../python_sdk_charter.md)

## New Things Observed

The current repository has moved a lot since the first Python SDK pass.

Recent verified changes from the local history include:

- hosted lifecycle control-plane work
- reclaim planning and execution
- exact metadata filter support
- full GEM-family stage-1 work
- GEM frontier instrumentation

Inference:

- most current design pressure has been on stage 1 and hosted engine
  operations
- stage 2 has remained stable enough to work, but not yet explicit enough to
  serve as the long-lived public refinement contract

## Verified Facts

These statements are checked against the current codebase.

1. Stage-1 engine identity is now explicit and family-aware.
   Evidence:
   - [kayak/planning/candidate_generator.mojo](../../kayak/planning/candidate_generator.mojo)
   - [kayak/planning/execution.mojo](../../kayak/planning/execution.mojo)

2. `SearchPlan` now owns an explicit stage-2 contract.
   Evidence:
   - [kayak/planning/search_plan.mojo](../../kayak/planning/search_plan.mojo)
   - [kayak/planning/stage2_operator.mojo](../../kayak/planning/stage2_operator.mojo)
   - [kayak/planning/execution_stage2.mojo](../../kayak/planning/execution_stage2.mojo)
   - compatibility fields still exist, but they are now derived from
     `stage2_operator`

3. The implemented stage-2 boundary already supports multiple explicit
   operators.
   Evidence:
   - [kayak/planning/exact_stage.mojo](../../kayak/planning/exact_stage.mojo)
   - [kayak/planning/clause_text_stage.mojo](../../kayak/planning/clause_text_stage.mojo)
   - [kayak/planning/stage2_operator.mojo](../../kayak/planning/stage2_operator.mojo)
   - current built-ins are:
     - `noop_topk`
     - `exact_late_interaction`
     - `clause_text`

4. The current clause-text refinement path is now routed through the same
   `SearchPlan` contract instead of living only in a side verifier path.
   Evidence:
   - [kayak/verifier/clause_text.mojo](../../kayak/verifier/clause_text.mojo)
   - [kayak/planning/clause_text_stage.mojo](../../kayak/planning/clause_text_stage.mojo)
   - [kayak/service/runtime.mojo](../../kayak/service/runtime.mojo)

5. The Python SDK mirrors the same narrowing.
   Evidence:
   - [python/kayak_bridge/planned_search.py](../../python/kayak_bridge/planned_search.py)
   - [python/kayak_bridge/stage2_operator.py](../../python/kayak_bridge/stage2_operator.py)
   - [python/kayak_bridge/clause_text.py](../../python/kayak_bridge/clause_text.py)
   - Python `SearchPlan` now carries `stage2_operator`
   - Python `LateQuery` and `LateIndex` can carry `query.text` and
     `doc_texts` for text-family stage 2

6. The service boundary depends on `SearchPlan`, so this is not only a local
   SDK concern.
   Evidence:
   - [kayak/service/search_contracts.mojo](../../kayak/service/search_contracts.mojo)
   - [kayak/service/runtime.mojo](../../kayak/service/runtime.mojo)
   - hosted requests now validate `query_text` when the chosen stage-2
     operator requires it

## Problem Statement

The repository currently has a clean stage-1 family boundary, but stage 2 is
split across two different ideas:

- the main planning path assumes one exact late-interaction stage
- the verifier path hosts optional rerankers and stronger ceilings

That split is mechanically workable, but it is not the right long-term
primitive boundary.

Reason:

- exact late interaction is one valid stage-2 operator, not the only possible
  stage-2 operator
- text-aware reranking, hybrid refinement, or future cross-attention reranking
  should not remain a side pipeline forever
- the Python SDK should not freeze around one concrete stage-2 technique if the
  real invariant is "explicit refinement of a candidate window"

## Decision

Define stage 2 from first principles as:

**an explicit candidate-window refinement stage with declared artifact
requirements and explicit scoring semantics**

The correct stable abstraction is **not**:

- `"exact_late_interaction"` as a hardcoded stage kind
- plus a separate verifier/reranker escape hatch

The correct stable abstraction is narrower and more general:

- one candidate window coming out of stage 1
- one explicit stage-2 operator acting on that window
- one explicit profile/result surface for what that operator consumed and
  produced

## First-Principles Stage-2 Model

### 1. Stage 2 refines a window, not a corpus

Stage 2 should own:

- reranking or rescoring a bounded candidate window
- materializing only the artifacts needed for that window
- producing the final ranked hits for that search plan

Stage 2 should not own:

- corpus-wide candidate generation
- hidden fallback to stage 1
- implicit retrieval over the full snapshot

Reason:

- that keeps the stage boundary measurable
- it makes stage-1 recall and stage-2 refinement costs separable

### 2. Stage 2 must declare what artifacts it needs

Different refinement operators need different evidence:

- exact late interaction needs packed late-interaction vectors
- clause-text reranking needs document text
- future metadata-aware refinement may need metadata sidecars
- future multimodal or cross-attention refinement may need other artifact
  families entirely

Therefore the stable stage-2 contract should declare required artifact
families explicitly rather than assuming packed vectors only.

### 3. Stage 2 must declare its scoring family

The repo should distinguish at least:

- late-interaction refinement
- text reranking
- hybrid or blended refinement

Reason:

- these have different semantics, costs, and artifact requirements
- treating them all as `"reranker_kind"` would be as misleading as treating
  WARP and GEM as one stage-1 family

### 4. Stage 2 must keep exact-reference semantics explicit

Exact late interaction should remain a first-class stage-2 operator because it
is the current correctness anchor.

But exactness should be represented as an operator property, not as the entire
definition of stage 2.

## Stable Primitives To Add

This section is intentionally about contracts, not final implementation names.

### `Stage2Operator`

Minimal fields the repo should eventually carry:

- `kind`
- `family`
- `required_artifact_families`
- `is_exact_reference`

Initial operator kinds that fit the current repo:

- `exact_late_interaction`
- `clause_text`
- `noop_topk`

Why these first:

- they already correspond to real code paths
- they let the repo unify the current exact path and current local ceiling path
  without inventing speculative operators

### `Stage2Result`

Minimal fields:

- final hits
- artifact counts or bytes consumed for the candidate window
- operator-specific stage profile

This can likely reuse most of the current
[SearchStageProfile](../../kayak/planning/stage_profile.mojo) shape rather than
creating a second profiling object.

### `Stage2ArtifactMaterialization`

The exact stage already materializes a packed candidate index in
[kayak/planning/exact_stage.mojo](../../kayak/planning/exact_stage.mojo).

The more general version should make explicit:

- which candidate-window artifacts were materialized
- how many documents or vectors they contain
- what byte cost they imply

That keeps stage 2 measurable even when it is not vector-only.

## SearchPlan Migration

The current `SearchPlan` fields:

- `exact_stage_kind`
- `reranker_kind`

should eventually collapse into one explicit stage-2 operator field.

Recommended target shape:

- `candidate_generator`
- `candidate_budget`
- `stage2_operator`
- `faithfulness_policy`

Reason:

- this matches the real plan geometry better
- it avoids the current split where one exact stage is "inside" the plan but
  richer refinement is "outside" the plan

## Python SDK Implication

The public Python SDK should converge on the same boundary.

Today:

- `python/kayak/__init__.py` is mostly a stable re-export layer
- `python/kayak_bridge/planned_search.py` always runs exact MaxSim on the
  shortlisted index as stage 2

That is acceptable as a first pass, but it is not the final stable public
surface if the repo wants Python to be the canonical programmable layer.

The Python side should eventually expose:

- `Stage2Operator`
- explicit `search_with_plan(...)` using that operator
- backend capability reporting for stage-2 operators

Reason:

- users should program against stable refinement primitives
- not against whichever internal rerank path is wired today

## What Not To Do

1. Do **not** keep exact late interaction and richer reranking in permanently
   separate planning universes.
2. Do **not** make stage 2 a generic opaque callback that hides artifact
   requirements.
3. Do **not** auto-select stage-2 behavior from artifact presence.
4. Do **not** collapse stage 2 into the Python `maxsim(...)` helper alone.
5. Do **not** bury future text or hybrid rerankers under a verifier-only API if
   they are meant to be first-class search-plan stages.

## Immediate Phases

### Phase S2-1

Add contract types only.

Deliverables:

- a stage-2 operator contract
- explicit artifact-requirement constants or types
- no behavior change yet

Exit criterion:

- the repo can name stage-2 operators and requirements without overloading
  `SearchPlan`

### Phase S2-2

Adapt exact late interaction to the new contract.

Deliverables:

- route the current exact rerank through the stage-2 operator boundary
- preserve all current search, explain, and faithfulness behavior

Exit criterion:

- no behavior regression on current tests and traces

### Phase S2-3

Move the current clause-text reranker onto the same contract.

Deliverables:

- one text-family stage-2 operator
- explicit artifact requirement on document text
- one comparison trace through the unified stage-2 path

Exit criterion:

- stronger-ceiling comparisons no longer depend on a separate verifier-only
  execution path

### Phase S2-4

Expose the same operator boundary in Python.

Deliverables:

- Python `Stage2Operator`
- SDK examples for exact late interaction and one text-aware refinement path

Exit criterion:

- the Python SDK reflects the same stage-2 semantics as the engine contract

## Conclusion

The repository does **not** need to abandon exact late interaction as stage 2.

It does need to stop treating exact late interaction as the whole definition of
stage 2.

The stable primitive is:

- stage 1 chooses a candidate window
- stage 2 explicitly refines that window using declared evidence and declared
  semantics

That boundary is compatible with:

- current exact MaxSim refinement
- current clause-text ceiling work
- future hybrid or stronger rerankers
- a more principled Python SDK surface
