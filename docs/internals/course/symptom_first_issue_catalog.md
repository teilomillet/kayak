# Symptom-First Issue Catalog

This note translates the course into the kinds of things people actually say
when retrieval is going wrong.

It is meant to increase the chance that a learner thinks:

- "yes, that happened to me too"

without pretending that every complaint already has a separate end-to-end
reproduction.

## How To Use This

Start from the user's complaint, not from the lesson number.

For each complaint cluster below, the course should answer four questions:

1. what is the likely failure family?
2. what is the first Kayak check?
3. what evidence level do we currently have?
4. what should we avoid claiming too strongly?

## Evidence Tiers

The complaint surface can be broader than the current benchmark surface, but
the status still has to stay explicit.

- `Verified locally`
  Use this when the repository contains a direct local reproduction or smoke
  path for the claim.
- `Supported on a defined surface`
  Use this when the repo has named judged slices or traces that support the
  family of complaint, even if not every phrasing has its own reproduction.
- `Instrumented but unresolved`
  Use this when the repo can inspect the issue, but the course should not teach
  a settled conclusion yet.

## Complaint Clusters

### 1. "The wrong thing keeps winning."

What users often say:

- "the answer is in the corpus, but the wrong doc wins"
- "it retrieves something related, not the page that actually answers"
- "the top result looks topical, but not evidential"
- "we keep getting near misses"

Likely failure family:

- compressed scoring hides evidence structure
- same-topic distractors outrank the evidence-bearing document

First Kayak check:

- compare exact late interaction against a one-vector baseline on the same
  token vectors
- inspect whether the answer-bearing document is missing entirely or only
  ranked too low

Evidence status:

- core "wrong doc wins because compression hides the conjunction" claim:
  `Verified locally`
- broader "topical but not answer-bearing" family:
  `Supported on a defined surface`

Best starting artifacts:

- [lesson_01_rag_retrieval_debugging.md](lesson_01_rag_retrieval_debugging.md)
- [mini_lab_01_answer_bearing_page_on_bright.md](mini_lab_01_answer_bearing_page_on_bright.md)
- [notebooks/rag_retrieval_debugging_with_kayak.ipynb](notebooks/rag_retrieval_debugging_with_kayak.ipynb)
- [../../traces/2026-04-12_limit_browsecomp_public_slices.md](../../traces/2026-04-12_limit_browsecomp_public_slices.md)

What we can say honestly:

- There is at least one clean deterministic case where collapsing to one vector
  hides the evidence and exact late interaction restores it.
- We also have a real judged-slice surface where topical context is retrieved
  but the answer-bearing document lands too low.

### 2. "We changed chunking and everything moved."

What users often say:

- "changing chunk size changed the ranking"
- "full-page indexing and chunk indexing behave like different systems"
- "it got better on one query and worse on another"
- "we don't know whether chunking helped or just changed the failure"

Likely failure family:

- retrieval-unit design / chunking regime change

First Kayak check:

- compare exact, one-vector-per-document, and chunked-one-vector behavior
  before changing anything else
- keep vector count and retrieval unit explicit

Evidence status:

- `Verified locally` and supported on named cached slices

Best starting artifacts:

- [lesson_04_classic_chunking_and_one_vector.md](lesson_04_classic_chunking_and_one_vector.md)
- [chunk_coverage_vs_joint_scoring.md](chunk_coverage_vs_joint_scoring.md)
- [notebooks/classic_chunking_and_one_vector.ipynb](notebooks/classic_chunking_and_one_vector.ipynb)

What we can say honestly:

- Chunking is neither universally good nor universally bad.
- It changes which failure you get, and that is exactly the comparison the
  course should make visible.

### 3. "The answer seems split across pieces."

What users often say:

- "one chunk has half the answer and another chunk has the rest"
- "exact search still gives partial hits"
- "nothing looks fully wrong, but nothing is complete either"
- "the evidence spans boundaries"

Likely failure family:

- retrieval-unit failure rather than scorer failure

First Kayak check:

- hold the token vectors fixed and compare the same content indexed as split
  chunks versus grouped retrieval units

Evidence status:

- `Verified locally`

Best starting artifacts:

- [lesson_02_failure_patterns.md](lesson_02_failure_patterns.md)
- [notebooks/retrieval_failure_pattern_catalog.ipynb](notebooks/retrieval_failure_pattern_catalog.ipynb)

What we can say honestly:

- Some misses are not mainly about the scorer.
- If the evidence is split across retrieval units, the retriever cannot award a
  full score to a unit that does not actually contain the full match.

### 4. "Widening `k` helps, and reranking does not save it."

What users often say:

- "if I widen `k`, the right doc suddenly appears"
- "the reranker didn't help"
- "the answer is somewhere deeper in the list"
- "stage 2 looks smart, but it never sees the right page"

Likely failure family:

- shortlist / stage-1 recall pressure

First Kayak check:

- inspect whether exact search or a wider candidate window recovers the oracle
- if the answer-bearing document never enters stage 2, stop blaming the
  reranker first

Evidence status:

- local shortlist boundary: `Verified locally`
- broader family: `Supported on a defined surface`

Best starting artifacts:

- [lesson_02_failure_patterns.md](lesson_02_failure_patterns.md)
- [mini_lab_02_shortlist_loss_on_bright.md](mini_lab_02_shortlist_loss_on_bright.md)
- [diagnosis_playbook.md](diagnosis_playbook.md)
- [../../hard_recall_evaluation.md](../../hard_recall_evaluation.md)
- [notebooks/retrieval_failure_pattern_catalog.ipynb](notebooks/retrieval_failure_pattern_catalog.ipynb)

What we can say honestly:

- A reranker cannot recover a document that never entered the candidate set.
- Some misses are shortlist failures, not reranking failures.

### 5. "Long documents are where it really breaks."

What users often say:

- "short snippets work, manuals don't"
- "our long PDFs fail more than short docs"
- "policies, legal text, and reference pages are much worse"
- "late evidence seems to disappear in long documents"

Likely failure family:

- long-document stage-1 pressure
- retrieval-unit choices that are too coarse or too lossy for long evidence

First Kayak check:

- compare exact behavior against budgeted candidate paths with vector counts
  explicit
- inspect whether the answer-bearing long document is absent from the shortlist
  or merely ranked low

Evidence status:

- `Supported on a defined surface`

Best starting artifacts:

- [lesson_02_failure_patterns.md](lesson_02_failure_patterns.md)
- [mini_lab_03_long_document_pressure.md](mini_lab_03_long_document_pressure.md)
- [../../traces/2026-04-13_benchmark_ladder_and_long_document_hard_recall.md](../../traces/2026-04-13_benchmark_ladder_and_long_document_hard_recall.md)

What we can say honestly:

- Long late-evidence documents are a harsher regime.
- They should be taught as a distinct stress case rather than folded into a
  generic retrieval story.

### 6. "The cheap path is worse than the exact path."

What users often say:

- "the fast version feels lossy"
- "the exact path works, but the approximation is worse"
- "we compressed it and quality dropped"
- "we made it cheaper and now the answer disappears"

Likely failure family:

- compression / proxy faithfulness gap

First Kayak check:

- run exact versus proxy on the same cached judged slice

Evidence status:

- `Verified locally` on named cached slices

Best starting artifacts:

- [lesson_03_real_slice_proxy_diagnosis.md](lesson_03_real_slice_proxy_diagnosis.md)
- [notebooks/real_slice_proxy_diagnosis.ipynb](notebooks/real_slice_proxy_diagnosis.ipynb)

What we can say honestly:

- On several cached judged slices, exact late interaction outperforms the
  compressed proxy path.
- The course can show the gap and teach how to measure it, not promise that one
  proxy budget is universally safe.

### 7. "One vector looked fine on one dataset and bad on another."

What users often say:

- "our baseline looked fine in one eval and terrible in another"
- "one vector per doc seemed okay until it wasn't"
- "chunking helped one workload and hurt another"
- "I don't know whether I should keep the simple baseline"

Likely failure family:

- workload-dependent simplification cost

First Kayak check:

- compare exact, one-vector, and chunked baselines across more than one slice

Evidence status:

- `Verified locally` and supported on named cached slices

Best starting artifacts:

- [lesson_04_classic_chunking_and_one_vector.md](lesson_04_classic_chunking_and_one_vector.md)
- [notebooks/classic_chunking_and_one_vector.ipynb](notebooks/classic_chunking_and_one_vector.ipynb)

What we can say honestly:

- Simplification cost is workload-shaped.
- The course should not promise one global answer for one-vector or chunking
  baselines.

### 8. "We made it faster and now we don't know what broke."

What users often say:

- "we optimized first and now debugging is impossible"
- "quality got weird after we made it faster"
- "we don't know which approximation caused the drop"
- "everything changed at once"

Likely failure family:

- optimization happened before a correctness anchor existed

First Kayak check:

- freeze one exact reference path, then compare each simplification against it

Evidence status:

- `Supported on a defined surface`

Best starting artifacts:

- [debug_your_broken_rag_sequence.md](debug_your_broken_rag_sequence.md)
- [diagnosis_playbook.md](diagnosis_playbook.md)
- [../../../public/docs/storage-and-search.md](../../../public/docs/storage-and-search.md)

What we can say honestly:

- Exact-first, fast-second is the right workflow when the quality boundary is
  still unclear.
- The main value is diagnostic control, not an automatic speed-quality recipe.

### 9. "I think the LLM is hallucinating because retrieval never shows the evidence."

What users often say:

- "the model invents answers because retrieval is bad"
- "it feels like hallucination, but I think retrieval is the upstream issue"
- "the citations are wrong because the right page never comes back"

Likely failure family:

- possible retrieval-side upstream failure, but generation can also be at fault

First Kayak check:

- first check whether exact retrieval on the intended retrieval unit surfaces
  the right document at all

Evidence status:

- `Instrumented but unresolved`

Best starting artifacts:

- [diagnosis_playbook.md](diagnosis_playbook.md)
- [notebooks/debug_your_broken_rag_with_kayak.ipynb](notebooks/debug_your_broken_rag_with_kayak.ipynb)

What we can say honestly:

- This is a valid entry complaint.
- We should not teach it as a retrieval-only fact without generation-side
  isolation.

### 10. "I don't want to replace my vector DB just to try this."

What users often say:

- "do I need to rewrite my stack"
- "can I use this without changing storage first"
- "I want to diagnose retrieval quality, not replatform everything"

Likely failure family:

- store-boundary anxiety rather than a retrieval algorithm problem

First Kayak check:

- separate storage handoff from retrieval diagnosis

Evidence status:

- `Supported on a defined surface`

Best starting artifacts:

- [../../../public/docs/storage-and-search.md](../../../public/docs/storage-and-search.md)
- [diagnosis_playbook.md](diagnosis_playbook.md)

What we can say honestly:

- Kayak can be taught as a diagnosis and serving layer without claiming that
  users must rewrite their whole storage stack.

## What This Adds To The Course

This note broadens the course entry surface in a way that is still safe:

- more learners can recognize themselves in the opening complaint
- producers can choose a starting complaint that matches the audience
- the evidence bar stays explicit instead of being blurred by relatability

## What This Does Not Yet Add

This note still does not give us:

- a measured reproduction for every complaint phrasing above
- a production-frequency study showing which complaint is most common
- generation-side isolation for the hallucination complaint

So this should be used as:

- a symptom-first entry map

not as:

- proof that the course already covers every real retrieval failure a team can
  have
