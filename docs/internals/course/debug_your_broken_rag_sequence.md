# Debug Your Broken RAG With Kayak

This note is the main internal sequence for turning the existing lesson notes
and notebooks into one coherent course.

It is the answer to:

- what is the learner journey?
- what is the main notebook?
- what order should the ideas appear in?

For complaint-first entrypoints, pair this sequence with:

- [symptom_first_issue_catalog.md](symptom_first_issue_catalog.md)

The intended learner is still:

- an engineer or data scientist who cannot make retrieval-augmented generation
  reliably find the right evidence

## Course Promise

By the end of this sequence, the learner should be able to:

- recognize whether retrieval is failing because of scoring, retrieval-unit
  design, or shortlist loss
- compare exact late interaction against one-vector and chunked one-vector
  baselines without changing the encoder at the same time
- use one judged slice to debug quality before touching system-level tuning
- understand when Kayak is a diagnosis and control tool rather than a claim of
  universal superiority

## Complaint-First Openers

If this is delivered to a broad audience, the course should open from one of
these recognizable complaints:

- "the answer is in the corpus, but the wrong page keeps winning"
- "we changed chunking and everything moved"
- "widening `k` helps, but reranking still does not fix it"
- "long documents are where the system really breaks"
- "we made it faster and now we don't know what broke"

Why:

- these are more recognizable than "today we will learn late interaction"
- they still map cleanly to the verified failure families in this folder
- they preserve the practical feel of debugging before theory

## One Sentence Story

The learner starts from:

- "my RAG still misses the evidence"

and ends at:

- "I can tell which simplification is hurting me, and I can verify the fix with
  Kayak before I optimize anything else"

## Main Narrative

The sequence should feel like one debugging case, not a pile of unrelated IR
lectures.

It should also feel like real use:

- the learner pokes at a failure
- something visibly changes
- only then do we name the mechanism
- only when a conclusion hardens do we surface the evidence boundary

Use this narrative:

1. My RAG misses the answer.
2. I first need a correctness anchor.
3. I compare exact late interaction to the simpler baselines people actually
   ship.
4. I learn whether the problem is:
   - compressed scoring
   - chunking / retrieval-unit design
   - stage-1 shortlist loss
5. Only after that do I discuss performance and deployment shape.

That order matters because:

- most teams reach for system tuning before they know where quality was lost
- the course should teach diagnosis before optimization
- most users naturally play with a system before they are ready for a research-style frame

This means the producer should be free to start from different complaints while
keeping the same internal spine:

1. visible miss
2. exact reference path
3. comparison against the simplification the learner already uses
4. narrow conclusion
5. only then performance or architecture

## Main Notebook

The primary notebook entrypoint for the sequence should be:

- [notebooks/debug_your_broken_rag_with_kayak.ipynb](notebooks/debug_your_broken_rag_with_kayak.ipynb)

That notebook should be the thing a course producer or instructor opens first.

Its primary maintenance hook should be:

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_sequence_smoke -v
```

Reason:
- the course needs one obvious rerun path for the main story
- that test keeps the deterministic spine and the cached `LIMIT-small` bridge checked together

Its direct execution suite should also exist:

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_notebook_execution -v
```

Reason:
- the integrated notebook is part of the teaching artifact, not just a derivation target
- the current environment lacks notebook execution tooling, so the repo needs a plain-Python fallback
- the same suite now also executes the supporting course notebooks when their cached task JSONs are present

The other notebooks remain supporting labs:

- [notebooks/rag_retrieval_debugging_with_kayak.ipynb](notebooks/rag_retrieval_debugging_with_kayak.ipynb)
- [notebooks/retrieval_failure_pattern_catalog.ipynb](notebooks/retrieval_failure_pattern_catalog.ipynb)
- [notebooks/real_slice_proxy_diagnosis.ipynb](notebooks/real_slice_proxy_diagnosis.ipynb)
- [notebooks/classic_chunking_and_one_vector.ipynb](notebooks/classic_chunking_and_one_vector.ipynb)

## Recommended Teaching Order

### Part 1. The First Honest Question

Ask:

- if I keep the same token vectors, what changes when I compare:
  - exact late interaction
  - one vector per whole document
  - one vector per fixed-size chunk

Why:
- this keeps the comparison epistemically clean
- it matches how many teams currently think about retrieval

Primary material:
- [lesson_01_rag_retrieval_debugging.md](lesson_01_rag_retrieval_debugging.md)
- [lesson_04_classic_chunking_and_one_vector.md](lesson_04_classic_chunking_and_one_vector.md)

### Part 2. Failure Patterns

Teach the three failure classes explicitly:

- scorer failure
- retrieval-unit failure
- shortlist failure

Primary material:
- [lesson_02_failure_patterns.md](lesson_02_failure_patterns.md)
- [diagnosis_playbook.md](diagnosis_playbook.md)

### Part 3. Real Judged Slice

Move to one cached judged slice.

Default choice:
- `LIMIT-small`

Reason:
- tiny enough to inspect
- locally cached
- already justified in the benchmark rationale as adversarial to
  single-vector retrieval

Primary material:
- [lesson_03_real_slice_proxy_diagnosis.md](lesson_03_real_slice_proxy_diagnosis.md)

Complaint-first judged-slice mini labs:

- [mini_lab_01_answer_bearing_page_on_bright.md](mini_lab_01_answer_bearing_page_on_bright.md)
- [mini_lab_02_shortlist_loss_on_bright.md](mini_lab_02_shortlist_loss_on_bright.md)
- [mini_lab_03_long_document_pressure.md](mini_lab_03_long_document_pressure.md)

Reason:
- these give producers a faster path from a familiar complaint to one measured
  real-slice reproduction
- they broaden recognition without replacing the main notebook spine

### Part 4. Practical Systems Boundary

Only after the learner sees the quality boundary should the course discuss:

- repeated-query fast paths
- store handoff
- exact-versus-approximate stage design

Primary references:
- [../../../public/docs/storage-and-search.md](../../../public/docs/storage-and-search.md)
- [../../traces/2026-04-12_python_sdk_batch_fast_path.md](../../traces/2026-04-12_python_sdk_batch_fast_path.md)

### Part 5. Explicit Limits

Close with:

- where the evidence is strong
- where it is conditional
- where Kayak is still far from the public frontier

Primary references:
- [../../epistemic_status.md](../../epistemic_status.md)
- [../../traces/2026-04-15_r2med_biology_full_gap_to_leaderboard.md](../../traces/2026-04-15_r2med_biology_full_gap_to_leaderboard.md)

## Sequence Shape

If this becomes a course, the recommended shape is:

1. one main conceptual notebook
2. three supporting lab notebooks
3. one instructor note on evidence boundaries

That is better than:

- five separate lessons with no visible spine

because the learner should feel they are solving one problem, not enrolling in
five unrelated modules.

## Production Guidance

When this becomes recorded or public-facing material:

- open with the user pain, not the API
- teach the baselines people already know
- name the simplification being removed at each step
- keep every strong claim tied to one notebook cell, test, or trace
- let the learner feel the behavior before giving the formal explanation
- reserve explicit methodology language for the moments where certainty matters

The sequence should **not** open with:

- Mojo
- kernel details
- service architecture
- benchmark grand narratives

Those belong later.
