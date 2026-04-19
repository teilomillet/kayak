# Evidence Register

This note maps proposed course claims to the best currently available evidence.

It is the internal answer to:

- what can we teach now?
- what should stay tentative?
- what still needs a dedicated reproduction?

## Claim Register

| Claim | Status | Evidence | Why it is useful in the course |
| --- | --- | --- | --- |
| Late interaction can recover an evidence-bearing document that a naive mean-pooled dense baseline misses on a deterministic toy case. | Verified locally | [lesson_01_rag_retrieval_debugging.md](lesson_01_rag_retrieval_debugging.md), [notebooks/rag_retrieval_debugging_with_kayak.ipynb](notebooks/rag_retrieval_debugging_with_kayak.ipynb), [../../../python/tests/test_course_rag_debugging_smoke.py](../../../python/tests/test_course_rag_debugging_smoke.py) | This is the cleanest first explanation for why token-level interaction matters. |
| If the evidence is split across different retrieval units, exact late interaction still cannot award a full-document match; regrouping the unit can fix the result. | Verified locally | [lesson_02_failure_patterns.md](lesson_02_failure_patterns.md), [diagnosis_playbook.md](diagnosis_playbook.md), [notebooks/retrieval_failure_pattern_catalog.ipynb](notebooks/retrieval_failure_pattern_catalog.ipynb), [../../../python/tests/test_course_failure_patterns_smoke.py](../../../python/tests/test_course_failure_patterns_smoke.py) | This teaches the critical distinction between a scoring problem and a retrieval-unit problem. |
| A narrow candidate stage can drop the oracle before reranking begins; widening the candidate set can let exact stage 2 recover it. | Verified locally for the toy case | [diagnosis_playbook.md](diagnosis_playbook.md), [notebooks/retrieval_failure_pattern_catalog.ipynb](notebooks/retrieval_failure_pattern_catalog.ipynb), [../../../python/tests/test_course_failure_patterns_smoke.py](../../../python/tests/test_course_failure_patterns_smoke.py) | This teaches why "better reranking" is not the right first response to every miss. |
| On several cached real judged slices, exact late interaction outperforms the compressed document-proxy path, with a particularly clear gap on `LIMIT-small`. | Verified locally on named cached slices | [lesson_03_real_slice_proxy_diagnosis.md](lesson_03_real_slice_proxy_diagnosis.md), [notebooks/real_slice_proxy_diagnosis.ipynb](notebooks/real_slice_proxy_diagnosis.ipynb) | This is the first non-toy bridge from the diagnosis lessons to real judged data. |
| On a judged Bright StackOverflow query, exact late interaction surfaces the answer-bearing pandas page while the one-vector baseline retrieves only non-relevant technical material in the top 10. | Verified locally | [mini_lab_01_answer_bearing_page_on_bright.md](mini_lab_01_answer_bearing_page_on_bright.md), [../../../python/tests/test_course_complaint_mini_labs.py](../../../python/tests/test_course_complaint_mini_labs.py) | This gives the course one realistic "related but not answer-bearing" complaint on a readable query. |
| On that same judged Bright StackOverflow query, widening the proxy candidate window from 10 to 20 admits a judged relevant page that stage 2 then reranks to the top. | Verified locally | [mini_lab_02_shortlist_loss_on_bright.md](mini_lab_02_shortlist_loss_on_bright.md), [../../../python/tests/test_course_complaint_mini_labs.py](../../../python/tests/test_course_complaint_mini_labs.py) | This gives the course a real judged-slice version of the "widening `k` helps" complaint. |
| Several longer-document judged slices are harsher for simplified retrieval paths than the lighter `r2med_biology_real_subset`, but the effect is not uniform across slices. | Verified locally on named cached slices | [mini_lab_03_long_document_pressure.md](mini_lab_03_long_document_pressure.md), [../../../python/tests/test_course_complaint_mini_labs.py](../../../python/tests/test_course_complaint_mini_labs.py) | This is a safer way to teach the long-document complaint without turning it into a universal slogan. |
| The repo's `document_proxy` path is the internal one-vector-per-document baseline under the same token vectors. | Verified locally | [lesson_04_classic_chunking_and_one_vector.md](lesson_04_classic_chunking_and_one_vector.md), [../../../python/tests/test_course_chunking_baselines_smoke.py](../../../python/tests/test_course_chunking_baselines_smoke.py) | This lets the course speak the user's language without changing the underlying measurement. |
| Chunked one-vector retrieval can help on local evidence pockets but hurt when the evidence spans chunk boundaries. | Verified locally and supported on named cached slices | [lesson_04_classic_chunking_and_one_vector.md](lesson_04_classic_chunking_and_one_vector.md), [notebooks/classic_chunking_and_one_vector.ipynb](notebooks/classic_chunking_and_one_vector.ipynb), [../../../python/tests/test_course_chunking_baselines_smoke.py](../../../python/tests/test_course_chunking_baselines_smoke.py) | This is the right nuanced replacement for any simplistic \"chunking good\" or \"chunking bad\" story. |
| The current course chunking comparison is a vector-level geometry comparison, not yet a realistic production token-chunk benchmark. | Verified by direct code and artifact inspection | [lesson_04_classic_chunking_and_one_vector.md](lesson_04_classic_chunking_and_one_vector.md), [../../../python/kayak_bridge/retrieval_task_builder.py](../../../python/kayak_bridge/retrieval_task_builder.py), [../../../python/kayak_bridge/colbert_encoder.py](../../../python/kayak_bridge/colbert_encoder.py), cached judged-slice stats in this folder | This keeps the course from overclaiming that `chunk16` / `chunk32` already represent modern RAG chunking practice. |
| A raw-text chunked one-vector baseline now exists for production-shaped source-token chunk sizes, and on BRIGHT it improves over one-vector-per-document while remaining far below late interaction over full chunk token matrices. | Verified locally on one judged slice | [raw_text_chunk_sweep.md](raw_text_chunk_sweep.md), [../../../python/kayak_bridge/raw_text_chunk_sweep.py](../../../python/kayak_bridge/raw_text_chunk_sweep.py), [../../../python/tests/test_raw_text_chunk_sweep.py](../../../python/tests/test_raw_text_chunk_sweep.py) | This is the right bridge from pedagogical chunking examples to more realistic chunk-size comparisons without hiding the compression loss. |
| On the BRIGHT `384 / 128` raw-text chunk setting, most of the quality loss comes from compressing each encoded chunk to one vector, not from chunking by itself. | Verified locally on one judged slice | [raw_text_chunk_sweep.md](raw_text_chunk_sweep.md) | This lets the course answer a common user question directly: "is chunking the issue, or is it how we embedded the chunks?" |
| The current ColBERT path uses `doc_maxlen = 180`, and many `256` to `512` source-token chunks on BRIGHT reach that cap. | Verified locally on one judged slice | [raw_text_chunk_sweep.md](raw_text_chunk_sweep.md), [../../../python/kayak_bridge/raw_text_chunk_sweep.py](../../../python/kayak_bridge/raw_text_chunk_sweep.py) | This prevents the course from overselling those chunk sweeps as faithful full-chunk representations under the current encoder. |
| An OpenAI-like dense one-vector chunk baseline path now exists, with explicit chunking, OpenAI embedding hooks, cosine-style normalization, and parent aggregation, but it has not yet been live-measured locally. | Implemented locally, not yet live-measured | [openai_like_dense_chunk_baseline.md](openai_like_dense_chunk_baseline.md), [../../../python/kayak_bridge/single_vector_chunk_baseline.py](../../../python/kayak_bridge/single_vector_chunk_baseline.py), [../../../python/kayak_bridge/openai_dense_baseline.py](../../../python/kayak_bridge/openai_dense_baseline.py), [../../../python/tests/test_single_vector_chunk_baseline.py](../../../python/tests/test_single_vector_chunk_baseline.py), [../../../python/tests/test_openai_dense_baseline.py](../../../python/tests/test_openai_dense_baseline.py) | This gives the course an honest lane for comparing Kayak against a common dense chunk retrieval recipe without pretending the current ColBERT tasks already answer that question. |
| A locally runnable open-source dense one-vector chunk baseline using `all-MiniLM-L6-v2` scores far above the exact full-document ColBERT baseline on the cached BRIGHT StackOverflow slice. | Verified locally on one judged slice | [open_source_dense_chunk_baseline.md](open_source_dense_chunk_baseline.md), [../../../python/kayak_bridge/hf_dense_baseline.py](../../../python/kayak_bridge/hf_dense_baseline.py), [../../../python/scripts/bench_hf_dense_chunk_baseline.py](../../../python/scripts/bench_hf_dense_chunk_baseline.py), [../../../python/tests/test_hf_dense_baseline.py](../../../python/tests/test_hf_dense_baseline.py) | This keeps the course from teaching the wrong slogan: the real issue is not "late interaction always wins," but which retrieval units and encoder budgets are actually being compared. |
| Stage-1 recall pressure matters on the repo's defined hard-recall surfaces. | Supported on a defined surface | [../../hard_recall_evaluation.md](../../hard_recall_evaluation.md) and the traces it cites | This justifies teaching retrieval debugging as more than a reranking problem. |
| Reusing one loaded index is the current repeated-query fast path in the Python SDK. | Supported on a defined surface | [../../../public/docs/storage-and-search.md](../../../public/docs/storage-and-search.md), [../../traces/2026-04-12_python_sdk_batch_fast_path.md](../../traces/2026-04-12_python_sdk_batch_fast_path.md), [../../../python/tests/test_batch_api.py](../../../python/tests/test_batch_api.py) | This gives the course one practical systems lesson without overselling service-level performance. |
| Kayak can coexist with an external store instead of replacing it. | Verified in interface and supported on defined surfaces | [../../../python/kayak/retrievers/factory.py](../../../python/kayak/retrievers/factory.py), [../../../public/docs/storage-and-search.md](../../../public/docs/storage-and-search.md), [../../lancedb_comparison_benchmark_contract.md](../../lancedb_comparison_benchmark_contract.md) | This lowers adoption anxiety and clarifies the store-versus-search boundary. |
| A simple "keep only about sqrt(m) vectors" story is not justified today. | Narrowly falsified | [../../epistemic_status.md](../../epistemic_status.md) | This is a strong teaching moment for Lesson 2 because it warns against easy chunking or pruning slogans. |
| Some plausible optimizations should be taught as negative results, not hidden. | Verified locally on a defined seam | [../../traces/2026-04-15_centroid_exact_stage_topk_fusion_negative_result.md](../../traces/2026-04-15_centroid_exact_stage_topk_fusion_negative_result.md) | This helps the course model scientific thinking rather than only product confidence. |
| Kayak is still far from the public frontier on at least one full benchmark. | Verified locally | [../../traces/2026-04-15_r2med_biology_full_gap_to_leaderboard.md](../../traces/2026-04-15_r2med_biology_full_gap_to_leaderboard.md) | This is the cleanest internal source for "where it does not work yet" without hand-waving. |

## Rerun Hooks

These are the concrete rerun hooks behind the strongest lesson claims.

- Main integrated course sequence:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_sequence_smoke -v`
- Main integrated notebook execution:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_notebook_execution -v`
- Lesson 1 toy retrieval debugging:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_rag_debugging_smoke -v`
- Lesson 2 failure patterns:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_failure_patterns_smoke -v`
- Lesson 4 chunking versus one-vector baselines:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_chunking_baselines_smoke -v`
- Complaint-first judged-slice mini labs:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_course_complaint_mini_labs -v`
- Raw-text chunk sweep unit coverage:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_raw_text_chunk_sweep -v`
- OpenAI-like dense baseline unit coverage:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_single_vector_chunk_baseline python.tests.test_openai_dense_baseline -v`
- Open-source dense baseline unit coverage:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_raw_text_chunk_sweep python.tests.test_single_vector_chunk_baseline python.tests.test_hf_dense_baseline -v`
- Batch API correctness:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_batch_api -v`
- Text retriever public workflow:
  - `PYTHONPATH=python ./.venv/bin/python -m unittest python.tests.test_text_retriever_api -v`

The larger benchmark traces already record their own commands and artifact
paths. Those should be reused rather than copied into each lesson.

## Missing Reproductions

The main remaining gap before a polished course is:

- notebook execution is now covered, but rendered notebook output is still not
  compared mechanically; the guardrail is sequential execution plus the lesson
  smoke tests, not screenshot-level or cell-output snapshot verification
