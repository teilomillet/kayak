# Internal Course Scaffold

This folder is the internal scaffold for a possible course about making
retrieval-augmented generation work with Kayak.

The target learner is:

- the engineer or data scientist whose RAG stack is not retrieving the right
  evidence
- the researcher who wants late interaction as an explicit mental model rather
  than as an opaque reranker

The course thesis is intentionally narrow:

- start from problems that many teams already feel
- use those problems to introduce late-interaction primitives
- keep every strong claim tied to local evidence, a benchmark note, or a small
  reproducible example

The delivery principle should also stay explicit:

- the learner experience should feel like real debugging and natural play
- the epistemic machinery is mostly for us, not for the learner
- certainty should surface when a claim hardens, not as constant classroom
  ceremony

This is not public-facing documentation. It is an internal production scaffold.

## Why This Exists

The repo already has two kinds of useful material:

- public Python SDK usage docs
- internal traces and benchmark notes

What it did not yet have was a bridge between those two:

- a course-shaped internal map
- explicit epistemic rules for teaching claims
- one small notebook that shows the retrieval-debugging mindset in code
- one canonical smoke path for keeping that main notebook honest

That bridge is what this folder owns.

## File Map

- [epistemic_standard.md](epistemic_standard.md)
  The rules each lesson must satisfy before it is taught as fact.
- [lesson_map.md](lesson_map.md)
  The proposed sequence, the mass-market problem each lesson speaks to, and the
  primitives each lesson should teach.
- [debug_your_broken_rag_sequence.md](debug_your_broken_rag_sequence.md)
  The main learner journey and the recommended order for the course.
- [instructor_notes.md](instructor_notes.md)
  Delivery guidance for keeping the course epistemically clean.
- [evidence_register.md](evidence_register.md)
  The current evidence inventory for course-worthy claims.
- [lesson_01_rag_retrieval_debugging.md](lesson_01_rag_retrieval_debugging.md)
  The first lesson draft centered on retrieval debugging.
- [lesson_02_failure_patterns.md](lesson_02_failure_patterns.md)
  The second lesson draft centered on recognizing which layer is failing.
- [lesson_03_real_slice_proxy_diagnosis.md](lesson_03_real_slice_proxy_diagnosis.md)
  The first real judged-slice diagnosis lesson.
- [lesson_04_classic_chunking_and_one_vector.md](lesson_04_classic_chunking_and_one_vector.md)
  A dedicated comparison against classic one-vector and chunked one-vector retrieval.
- [mini_lab_01_answer_bearing_page_on_bright.md](mini_lab_01_answer_bearing_page_on_bright.md)
  A judged-slice mini lab for "it retrieves related material, not the answer-bearing page."
- [mini_lab_02_shortlist_loss_on_bright.md](mini_lab_02_shortlist_loss_on_bright.md)
  A judged-slice mini lab for "widening `k` helps because stage 1 dropped the answer."
- [mini_lab_03_long_document_pressure.md](mini_lab_03_long_document_pressure.md)
  A judged-slice mini lab for "long documents are a harsher regime."
- [raw_text_chunk_sweep.md](raw_text_chunk_sweep.md)
  The realistic raw-text chunking benchmark path, its chunk-embedding geometry,
  and the first measured smoke and compression-diagnostic results.
- [openai_like_dense_chunk_baseline.md](openai_like_dense_chunk_baseline.md)
  The OpenAI-like dense one-vector chunk baseline path for comparing common
  production chunk retrieval recipes against Kayak exact late interaction.
- [open_source_dense_chunk_baseline.md](open_source_dense_chunk_baseline.md)
  The first live local dense chunk benchmark using an open-source embedding
  model that can run entirely in the current environment.
- [chunk_coverage_vs_joint_scoring.md](chunk_coverage_vs_joint_scoring.md)
  A broad explanation of why chunking the whole document does not automatically imply joint document scoring.
- [assets/chunk_coverage_vs_joint_scoring.svg](assets/chunk_coverage_vs_joint_scoring.svg)
  A reusable visual asset for slides or notebook cells explaining chunk coverage versus joint scoring.
- [diagnosis_playbook.md](diagnosis_playbook.md)
  A symptom-to-cause-to-intervention map for retrieval debugging.
- [symptom_first_issue_catalog.md](symptom_first_issue_catalog.md)
  A broader complaint-first map for matching the course to what users actually say.
- [notebooks/rag_retrieval_debugging_with_kayak.ipynb](notebooks/rag_retrieval_debugging_with_kayak.ipynb)
  Runnable local notebook for the first lesson.
- [notebooks/retrieval_failure_pattern_catalog.ipynb](notebooks/retrieval_failure_pattern_catalog.ipynb)
  Runnable local notebook for the failure-pattern lab.
- [notebooks/real_slice_proxy_diagnosis.ipynb](notebooks/real_slice_proxy_diagnosis.ipynb)
  Runnable local notebook for real judged-slice exact-versus-proxy diagnosis.
- [notebooks/classic_chunking_and_one_vector.ipynb](notebooks/classic_chunking_and_one_vector.ipynb)
  Runnable local notebook comparing exact late interaction to one-vector and chunked one-vector baselines.
- [notebooks/chunk_coverage_vs_joint_scoring.ipynb](notebooks/chunk_coverage_vs_joint_scoring.ipynb)
  Runnable local notebook showing why chunk coverage does not automatically imply joint document scoring.
- [notebooks/debug_your_broken_rag_with_kayak.ipynb](notebooks/debug_your_broken_rag_with_kayak.ipynb)
  The main integrated notebook entrypoint for the full debugging sequence.
- [../../../python/tests/test_course_sequence_smoke.py](../../../python/tests/test_course_sequence_smoke.py)
  The canonical smoke coverage for the integrated notebook's teaching spine.
- [../../../python/tests/test_course_complaint_mini_labs.py](../../../python/tests/test_course_complaint_mini_labs.py)
  Focused smoke coverage for the judged-slice complaint-first mini labs.
- [../../../python/tests/test_raw_text_chunk_sweep.py](../../../python/tests/test_raw_text_chunk_sweep.py)
  Unit coverage for the raw-text token chunking benchmark path.
- [../../../python/tests/test_single_vector_chunk_baseline.py](../../../python/tests/test_single_vector_chunk_baseline.py)
  Unit coverage for the generic dense one-vector chunk baseline path.
- [../../../python/tests/test_openai_dense_baseline.py](../../../python/tests/test_openai_dense_baseline.py)
  Focused adapter coverage for the OpenAI-like embedding cache and request path.
- [../../../python/tests/test_hf_dense_baseline.py](../../../python/tests/test_hf_dense_baseline.py)
  Focused adapter coverage for the local Hugging Face dense embedding path.

## Canonical Rerun

For the main integrated notebook, the first maintenance command should be:

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_sequence_smoke -v
```

Reason:
- it covers the deterministic course spine in one place
- it also checks the cached `LIMIT-small` bridge when that local task JSON is present

For direct notebook execution coverage across the course notebooks, use:

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_notebook_execution -v
```

Reason:
- the environment does not currently ship Jupyter or `nbclient`
- this still verifies that the actual notebook cells execute sequentially
- the real-data notebooks remain explicit about which cached judged slices they require

For the complaint-first judged-slice mini labs, use:

```bash
PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_complaint_mini_labs -v
```

Reason:
- these mini labs are only safe to teach if their slice-level claims remain
  mechanically checked
- the test keeps the complaint-first notes tied to the current cached evidence

## Promotion Rule

No lesson should move into a real course unless it has:

- one clear user problem
- one explicit list of primitives it teaches
- one evidence table with statuses
- one rerun path for the strongest local claims
- one section called `Where This Does Not Yet Generalize`

Reason:
- the course should teach a way of thinking, not only a happy-path API tour
- the epistemic standard has to stay at least as strict as the benchmark notes
- but that rigor should mostly live backstage in the production scaffold, not
  as constant learner-facing ritual
