# 2026-04-13: merge-readiness semantic audit

This note records a repo pass aimed at one question:

- where are the remaining semantic seams still too narrow or too duplicated for
  a safe merge of ongoing worktree changes?

It is intentionally epistemic.

The goal is not to pretend every file was reread line by line.
The goal is to:

- scan every major folder
- read the contract-setting docs and entry points
- identify where hardcoding is acceptable
- identify where hardcoding is now the real merge risk

## Scope Reviewed

### Docs and plans

- `docs/architecture/*.md`
- `docs/hosted_engine_grand_plan.md`
- `docs/late_interaction_2030.md`
- `docs/late_interaction_efficiency_roadmap.md`
- `docs/epistemic_status.md`
- `docs/python_sdk*.md`
- trace inventory under `docs/traces/`

### Code folders scanned

- `kayak/collections/`
- `kayak/planning/`
- `kayak/service/`
- `kayak/storage/`
- `kayak/index/`
- `kayak/verifier/`
- `kayak/contracts/`
- `kayak/search/`
- `kayak/filters/`
- `kayak/text/`
- `kayak/interop/`
- `kayak/benchmarks/`
- `python/kayak_bridge/`

### Key files read directly

- `kayak/__init__.mojo`
- `kayak/collections/__init__.mojo`
- `kayak/planning/__init__.mojo`
- `kayak/service/runtime.mojo`
- `kayak/service/search_contracts.mojo`
- `kayak/collections/segment.mojo`
- `kayak/collections/resolved_snapshot.mojo`
- `kayak/collections/search_artifact.mojo`
- `kayak/collections/search_artifact_policy.mojo`
- `kayak/collections/search_artifact_builders.mojo`
- `kayak/planning/candidate_generator.mojo`
- `kayak/planning/stage1_capabilities.mojo`
- `kayak/planning/centroid_execution_contract.mojo`
- `kayak/planning/planner_registry.mojo`
- `kayak/planning/search_plan.mojo`
- `kayak/planning/stage2_operator.mojo`
- `kayak/collections/resolution_requirements.mojo`
- `python/kayak_bridge/candidate_generator.py`
- `python/kayak_bridge/search_plan.py`
- `python/kayak_bridge/stage2_operator.py`

## Repo-Level Read

The repository is not missing abstractions everywhere.

It already has strong folder separation:

- `collections/` owns hosted storage and snapshot semantics
- `planning/` owns stage-aware retrieval semantics
- `service/` owns the hosted boundary
- `storage/` owns artifact codecs
- `index/` owns family-specific index construction
- `verifier/` owns stronger or alternate refinement paths

That broad structure is good.

The real remaining problem is narrower:

- the same semantic fact is still encoded in multiple modules at once
- some of those encodings are convenience-level
- some are architectural
- today they are not always cleanly separated

That is the real merge risk.

## Folder-By-Folder Assessment

### `docs/`

Strength:

- the repository has unusually strong written situational awareness
- architecture notes and traces are rich enough to justify decisions

Weakness:

- the docs are fragmented
- it is not always obvious which note is normative and which note is historical
- many traces are excellent evidence, but they are not a single canonical map
  of current invariants

Improvement:

- keep architecture docs as the stable contract layer
- keep traces as evidence only
- add a small canonical index of "current normative docs" when merge work
  settles

### `kayak/collections/`

Strength:

- collection, segment, snapshot, publish, reclaim, report, and resolution are
  first-class concepts
- the folder already expresses a hosted engine rather than a benchmark cache

Weakness:

- search artifacts are generic at the manifest level, but not generic enough at
  the loaded and builder levels
- `segment.mojo` still keeps convenience constructors and helper accessors for
  specific artifact families
- `resolved_snapshot.mojo` keeps one typed payload field per artifact family
- `search_artifact_builders.mojo` and `search_artifact_policy.mojo` still
  branch by artifact family explicitly

Meaning:

- new artifact families are possible
- but adding one still requires several semantic edits across the same folder

Verdict:

- this folder is structurally right
- but its semantic core is still family-aware in too many places

### `kayak/planning/`

Strength:

- stage 1 and stage 2 are explicit
- family dispatch exists
- faithfulness and profiling are first-class

Weakness:

- generator meaning is duplicated across:
  - `candidate_generator.mojo`
  - `stage1_capabilities.mojo`
  - `planner_registry.mojo`
  - `centroid_execution_contract.mojo`
  - `search_plan.mojo`
- stage-2 meaning is still encoded through a kind-dispatch file rather than a
  reusable descriptor surface
- `search_plan.mojo` still exposes many per-kind convenience constructors
  directly in the semantic core

Meaning:

- adding or changing a generator kind is still too much of a multi-file edit
- some hardcoding here is acceptable inside family implementations
- the duplication across contract files is not

Verdict:

- this is the single highest-leverage merge-risk folder

### `kayak/service/`

Strength:

- service contracts are explicit
- hosted lifecycle and search APIs are real, not implicit benchmark glue

Weakness:

- request validation still depends on low-level plan field semantics and string
  identity
- `same_search_plan(...)` is field-by-field and kind-string-aware rather than
  descriptor-aware
- planned requests still carry `stage2_operator_kind` as a raw string

Meaning:

- this folder is not the origin of the narrow semantics
- but it reflects them directly

Verdict:

- mostly healthy
- should consume cleaner plan descriptors once planning is refactored

### `kayak/storage/`

Strength:

- storage codecs are separated from hosted collection semantics
- per-artifact stores are appropriately explicit

Weakness:

- some duplication is unavoidable because each artifact family has its own
  codec
- however, family-specific manifest assumptions still leak upward into builder
  logic in `collections/`

Verdict:

- acceptable hardcoding here is normal
- this folder is not the main semantic problem

### `kayak/index/`

Strength:

- family-local algorithms live where they should
- `document_proxy`, centroid, and GEM code are isolated

Weakness:

- none of the important merge risk comes from the existence of family-local
  code here
- the real risk would be moving semantic registration logic into these files

Verdict:

- family-specific hardcoding is acceptable here
- do not try to over-generalize these algorithm modules

### `kayak/verifier/`

Strength:

- stronger ceilings and alternate rerankers are isolated

Weakness:

- only if stage-2 semantics remain duplicated between verifier paths and
  planning paths

Verdict:

- keep this folder narrow
- let planning own stage-2 operator descriptors, not verifier internals

### `kayak/contracts/`, `kayak/search/`, `kayak/filters/`, `kayak/text/`

Strength:

- these are mostly leaf or substrate folders
- their abstractions are already narrow in the good sense

Verdict:

- not the current merge blocker

### `kayak/interop/` and `kayak/benchmarks/`

Strength:

- benchmark and workload plumbing is explicit and rich

Weakness:

- as usual, these can accidentally fossilize experimental names or defaults if
  the semantic core below them is not stable

Verdict:

- they should consume stable descriptors from the engine
- they should not define them

### `python/kayak_bridge/`

Strength:

- Python is intentionally narrower than Mojo core

Weakness:

- Python re-encodes semantics independently
- `candidate_generator.py` only supports `exact_full_scan` and
  `document_proxy`
- `stage2_operator.py` duplicates the stage-2 kind logic in Python
- `search_plan.py` is a Python-specific subset of the Mojo semantic model

Meaning:

- the Python bridge is currently a product subset, not just a wrapper
- that is acceptable only if it is documented as an intentional subset
- it is dangerous if it drifts while the Mojo core evolves

Verdict:

- this folder is a secondary merge-risk area because semantic drift can appear
  across language boundaries

## Where Hardcoding Is Fine

Not all hardcoding is bad.

The following are good places for explicit family-local code:

- `kayak/index/`
- `kayak/storage/*_store.mojo`
- `kayak/planning/execution_*_family.mojo`
- specific verifier implementations

Reason:

- these are implementation modules
- their job is to know family-local details

## Where Hardcoding Is Dangerous

The dangerous hardcoding is semantic duplication across contract layers.

The current hotspots are:

1. stage-1 generator meaning
   Files:
   - `candidate_generator.mojo`
   - `stage1_capabilities.mojo`
   - `planner_registry.mojo`
   - `centroid_execution_contract.mojo`
   - `search_plan.mojo`

2. search-artifact family meaning
   Files:
   - `search_artifact.mojo`
   - `search_artifact_policy.mojo`
   - `search_artifact_builders.mojo`
   - `segment.mojo`
   - `resolved_snapshot.mojo`

3. stage-2 operator meaning
   Files:
   - `stage2_operator.mojo`
   - `execution_stage2.mojo`
   - `service/search_contracts.mojo`
   - Python bridge stage-2 files

4. cross-language semantic duplication
   Files:
   - Mojo planning contracts
   - Python bridge `candidate_generator.py`
   - Python bridge `search_plan.py`
   - Python bridge `stage2_operator.py`

## Most Important Conclusion

The repo does **not** primarily need more generic math or more generic storage.

The repo primarily needs:

- semantic descriptor layers that become the single source of truth for
  planning and artifact meaning

That is the real next substrate step before a major merge.

## Recommended Merge-Safe Refactor Order

### 1. Stage-1 generator descriptor registry

Create one descriptor source for:

- generator kind
- family
- required artifact families
- exactness
- filter support
- planner status and priorities
- optional family-local planner metadata where appropriate

Then make these consumers derive from it:

- `candidate_generator.mojo`
- `stage1_capabilities.mojo`
- `planner_registry.mojo`

This is the highest-leverage first step.

### 2. Stage-2 operator descriptor registry

Create one descriptor source for:

- operator kind
- family
- required artifact families
- query-text requirement
- exact-reference status
- execution kind

Then make these consumers derive from it:

- `stage2_operator.mojo`
- `service/search_contracts.mojo`
- Python bridge stage-2 surface

### 3. Search-artifact family descriptor registry

Create one descriptor source for:

- artifact family
- config keys
- segment-sealing support
- loader/storage identity

Then make these consumers derive from it:

- `search_artifact_policy.mojo`
- `search_artifact_builders.mojo`
- `resolved_snapshot.mojo`

This is harder than steps 1 and 2 and should follow them.

### 4. Document-representation transform contract

Add an explicit contract for:

- pooling
- pruning
- compression-aware transforms
- future document-side representation rewrites

This should sit at the segment or packed-index provenance layer, not be faked
as just another engine family.

### 5. Python bridge policy

Decide explicitly between two options:

- Python is a stable intentional subset
- Python is a faithful mirror of Mojo planning semantics

Either option is fine.
What is not fine is drifting between them.

## Merge Guidance Right Now

Safe to merge:

- family-local implementation improvements inside `index/`, `storage/`, or one
  execution-family module
- hosted operational work that does not redefine generator or artifact meaning
- benchmark additions that only consume existing semantics

Not safe to merge without descriptor cleanup:

- new generator kinds that require edits across multiple planning contract files
- new artifact families that require edits across multiple collections contract
  files
- new stage-2 semantics defined separately in Mojo and Python
- any work that mixes model semantics, representation transforms, and stage-1
  engine semantics in the same change

## Bottom Line

The user's concern is correct.

The main gap is no longer "we need more architecture".

The main gap is:

- the semantic interactions are still too duplicated across contract files
- that duplication is what will make future merges brittle

If one thing should happen before a large merge, it is this:

- make stage-1 and stage-2 meaning come from descriptor registries instead of
  from several parallel `if kind == ...` surfaces
