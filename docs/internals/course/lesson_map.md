# Lesson Map

This is the current internal sequence for a course shaped around one broad user
problem:

- "my RAG system still cannot reliably find the evidence I need"

The sequence is ordered so that broad pains come first and primitives appear
only when the learner needs them.

The delivery style should follow one additional rule:

- lessons should feel like realistic troubleshooting sessions
- primitives should appear as names for things the learner just observed
- explicit epistemic framing should appear mainly when a claim or number needs
  to be trusted

Before any lesson ordering, producers should also have a symptom-first entry
surface:

- [symptom_first_issue_catalog.md](symptom_first_issue_catalog.md)

Reason:
- many learners will enter through a complaint, not through a concept

## Lesson 1: Retrieval Misses The Evidence

Problem:
- the answer exists in the corpus, but retrieval still surfaces the wrong text

Why this speaks to many teams:
- this is the most common practical complaint about RAG quality

Natural entry:
- start from "why did this obvious answer not come back?"
- let the learner compare two rankings before naming MaxSim

Kayak primitives introduced:
- `LateQuery`
- `LateDocuments`
- `LateIndex`
- exact `search(...)`
- MaxSim as a query-token-by-document-token interaction

Current evidence:
- deterministic internal notebook and smoke test in this folder
- broader motivation from hard-recall notes in
  [../../hard_recall_evaluation.md](../../hard_recall_evaluation.md)

Open gap:
- add a compact real-data retrieval debugging walkthrough on one judged slice

Complaint-first judged-slice bridge:

- [mini_lab_01_answer_bearing_page_on_bright.md](mini_lab_01_answer_bearing_page_on_bright.md)
  for "it retrieves related material, not the answer-bearing page"

## Lesson 2: Chunking Is A Modeling Decision

Problem:
- chunking changes results, but the team does not know why

Why this speaks to many teams:
- many RAG failures are actually representation failures

Natural entry:
- start from "I changed chunking and everything moved"
- only after the learner sees the movement, name retrieval-unit design as the variable

Kayak primitives introduced:
- vector count as a first-class variable
- document grouping
- packing and layouts

Current evidence:
- hard-recall benchmark framing in
  [../../hard_recall_evaluation.md](../../hard_recall_evaluation.md)
- vector-budget caution and falsification in
  [../../epistemic_status.md](../../epistemic_status.md)
- long-document hard-recall evidence referenced from that note

Open gap:
- add one compact notebook that shows document-unit changes on a real or
  semi-real slice

Internal bridge:
- [lesson_02_failure_patterns.md](lesson_02_failure_patterns.md)
- [diagnosis_playbook.md](diagnosis_playbook.md)

## Lesson 3: Exact First, Fast Second

Problem:
- the team is optimizing before it has a correctness anchor

Why this speaks to many teams:
- retrieval debugging is much harder without a reference path

Natural entry:
- start from "I made it faster but I no longer know what I broke"
- then introduce exact search as the thing that makes comparison possible

Kayak primitives introduced:
- `NUMPY_REFERENCE_BACKEND`
- `MOJO_EXACT_CPU_BACKEND`
- `query_batch(...)`
- `search_batch(...)`

Current evidence:
- batch API correctness tests
- Python SDK batch fast-path trace in
  [../../traces/2026-04-12_python_sdk_batch_fast_path.md](../../traces/2026-04-12_python_sdk_batch_fast_path.md)

Open gap:
- add one lesson-specific notebook cell that shows exploratory timing without
  presenting it as benchmark evidence

Real-data bridge:
- [lesson_03_real_slice_proxy_diagnosis.md](lesson_03_real_slice_proxy_diagnosis.md)
- [notebooks/real_slice_proxy_diagnosis.ipynb](notebooks/real_slice_proxy_diagnosis.ipynb)
- [mini_lab_02_shortlist_loss_on_bright.md](mini_lab_02_shortlist_loss_on_bright.md)
  for a judged-slice version of "widening `k` helps"

## Lesson 4: Storage Is Not Search

Problem:
- teams think they must replace their vector DB before they can improve
  retrieval quality

Why this speaks to many teams:
- adoption anxiety is one of the biggest blockers

Natural entry:
- start from "do I need to rewrite my stack to try this?"
- answer that fear before discussing architecture

Kayak primitives introduced:
- `open_store(...)`
- `load_index(...)`
- reusable `LateIndex`
- candidate stage versus exact stage

Current evidence:
- public storage guidance in
  [../../../public/docs/storage-and-search.md](../../../public/docs/storage-and-search.md)
- internal LanceDB comparison contract in
  [../../lancedb_comparison_benchmark_contract.md](../../lancedb_comparison_benchmark_contract.md)
- measured LanceDB storage-handoff traces referenced there

Open gap:
- add one internal teaching note that separates "store adapter" from
  "late-interaction serving system"

Additional course lesson:
- [lesson_04_classic_chunking_and_one_vector.md](lesson_04_classic_chunking_and_one_vector.md)
  because many teams start from those baselines before they ever think in late
  interaction

## Lesson 5: Where Kayak Works, And Where It Does Not Yet Work

Problem:
- learners need to know whether they are seeing a frontier tool, a reference
  tool, or an unfinished research system

Why this matters:
- trust comes from explicit limits, not only success cases

Natural entry:
- wait until the learner already sees real value
- then show the limits plainly so the trust feels earned rather than defensive

Kayak primitives introduced:
- benchmark surfaces
- evidence status labels
- negative results as first-class engineering outcomes

Current evidence:
- epistemic status note in [../../epistemic_status.md](../../epistemic_status.md)
- public full-benchmark gap note in
  [../../traces/2026-04-15_r2med_biology_full_gap_to_leaderboard.md](../../traces/2026-04-15_r2med_biology_full_gap_to_leaderboard.md)
- negative-result trace in
  [../../traces/2026-04-15_centroid_exact_stage_topk_fusion_negative_result.md](../../traces/2026-04-15_centroid_exact_stage_topk_fusion_negative_result.md)

Open gap:
- add one short instructor note on how to talk about unresolved claims without
  weakening the course

Complaint-first judged-slice bridge:

- [mini_lab_03_long_document_pressure.md](mini_lab_03_long_document_pressure.md)
  for the long-document complaint with an explicit counterexample boundary
