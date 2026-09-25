# RAG integration research and challenges

## Retrieved context as decision state

Checked 2026-09-25. The typed-response path uses the existing Kayak decision
interface: retrieval supplies query/source context as text `state`; named
questions produce native answers. The [48-line example](../../examples/rag_decisions.py)
demonstrates that path. Optional text generation lives in a
[standalone application recipe](../../examples/rag_quickstart.py); the package keeps
the typed decision interface and evaluation of recorded stages.

| Primary source | Observed calling shape | Kayak application |
| --- | --- | --- |
| [Jev: classifying RAG passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages) | Sends a query and retrieved passage as state with multiple Noul questions; application code routes the evidence, and a separate model generates prose. | Keep the retrieved content and typed questions explicit. Filtering policy belongs to the application. |
| [Jev: line-by-line search](https://docs.typesafe.ai/cookbooks/semantic_find) | Uses document state with Choice to identify source lines and Noul to assess whether an answer exists. | A closed-set answer and an evidence judgment can share a request; one is not proof of the other. |
| [Laya: RAG relevance benchmark](https://github.com/NandhaKishorM/laya/blob/main/research/scripts/bench_apps.py) | Supplies query/passage pairs from MS MARCO and a relevance Noul question. This is supplied-context judging, not an executed retriever or generator. | Pass original query/source text to `Client.judge` or `Model.judge`; retain native typed results. |

These sources support composition through state and questions. They do not
establish a need for a new model constructor, implicit retrieval hook, or another
result type. Source reading
does not establish equivalent model quality or transferable confidence thresholds.

## Experiment execution and scoring

Reviewed 2026-09-25. This records the basis for [RAG experiments](../rag-experiments.md).
The scope is integrating existing applications, configuring real experiment
changes, preserving observations, and making evaluation decisions inspectable.
It does not establish a state-of-the-art quality or speed ranking.

## Primary-source comparison

| Current source | Observation | Kayak decision |
| --- | --- | --- |
| [Ragas experiments](https://docs.ragas.io/en/stable/concepts/experimentation/) | Experiments run application endpoints over data, apply metrics, retain results, and accept configuration parameters; examples include async functions. | Ordinary sync/async callables, explicit effective settings, retained attempts, and separate scoring. No decorator or framework dependency is required. |
| [LangSmith evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts) | Evaluation distinguishes example inputs/references, application outputs, evaluators, and experiments. | The pipeline receives only `RAGInput`; the separate reviewer receives references and observed outputs. Dataset and configuration snapshots travel with reports. |
| [LangSmith repetitions](https://docs.langchain.com/langsmith/repetition) | Repeated executions expose variation on the same examples. | Retain every case/repetition pair, not just an average. Report descriptive dispersion and changed output hashes without treating repeats as independent labeled cases. |
| [DeepEval flags and configuration](https://deepeval.com/docs/evaluation-flags-and-configs) | Concurrency, error handling, cache behavior, and result persistence are explicit evaluation concerns. | Bound async workers and keep pipeline/judge errors separate. Caller-owned completion hooks permit journaling. There is no implicit retry, cache, or hidden persistence. |
| [Phoenix batch evaluation](https://arize.com/docs/phoenix/evaluation/how-to-evals/batch-evaluations) | Evaluations distinguish sync/async execution, input mappings, scores, and failed judgments. | Validate adapters at the boundary; a missing or failed judgment remains unknown, with observation coverage separate from quality. |
| [OpenInference semantic conventions](https://arize-ai.github.io/openinference/spec/semantic_conventions.html) | Reranker input documents and top-K output documents are separate observations, with IDs, text, and scores. | An explicitly incomplete reranking may return a prefix of its inputs. Stable IDs preserve lineage; unobserved tail ranks are not fabricated. |

These are readings of published documentation, not executions or performance
measurements of competing packages. Integration policy is Kayak's choice, not a
claim that another system follows the same policy. The existing
[engineering conventions](../engineering.md) apply Go's explicit ownership and
Jane Street's emphasis on useful types and readers through ordinary Python.

## Hypotheses and observed challenges

| Hypothesis | Challenge and observed result |
| --- | --- |
| A small callable boundary fits independent RAG implementations. | Executed lexical/extractive, async rewritten-query, and mocked HTTP examples. All map observations into public records without inheritance or optional inference libraries. This supports those integrations, not automatic compatibility with every framework. |
| Configuration must change actual execution, not only metadata. | Changed retrieval/context limits in the callable example; returned sources and rendered context changed. With no packed evidence, answer and source-coverage gates failed. Effective defaults/overrides are retained with attempts. |
| Input and review identity must include application parameters and references. | An independent audit reused a correct answer after changing tenant parameters and the reference. An early candidate accepted it. Attempts now retain complete input/system settings, and reviews bind the full case plus final and intermediate outputs. Regression probes reject changed references and settings. |
| Python equality is insufficient for JSON setting identity. | A probe changed `true` to `1` and initially retained a passing report. Comparisons now preserve JSON scalar types and array order while ignoring object key order. Existing artifact digest algorithms remain unchanged. |
| A foreign-language reviewer should not reproduce Python serialization. | Exported reviewer inputs with an explicit fingerprint; a separate JSON flow reordered keys, omitted optional defaults, retained Unicode, copied that fingerprint, and scored successfully. Rehashing a foreign serializer's bytes is not the exchange contract. |
| Unknown quality and excluded diagnostics must not appear as a pass. | Missing attempts, missing reviews, pipeline/judge failures, all-diagnostic runs, and mixed ordinary/diagnostic runs were probed. Configured gates cannot pass these experiments; ordinary-subset descriptive metrics remain available. |
| Top-K APIs need an honest observation boundary. | Tested incomplete rankings against completions of their unobserved tail. Only values established across completions remain known. An explicit regression fixture preserves the previous default trace digest. |
| Async ownership must survive adverse histories. | Event-controlled tests and independent probes exercised bounded calls, out-of-order completions, input/reviewer mutation, callback failure, cancellation, and hook failure. Owned workers stop and are awaited; completed hook records are detached. Work hidden inside an external backend remains caller-owned. |
| A newcomer or coding assistant should discover the integration without searching implementation files. | A fresh agent followed AGENTS → example catalog → experiment guide → sync/async/HTTP source and the prefix guide. It ran all three integrations, changed only a JSON retrieval setting to change actual outputs and gates, and completed external review exchange. Two of three reviews produced mean correctness 1.0 but coverage 2/3 and CLI exit 1; a forged fingerprint was rejected. A misleading no-context fallback was found and corrected. No remaining blocker was found within these tasks. |

Tests are grounded in the public contracts and negative cases. They support the
observed implementation within those cases; they do not certify judges, causality,
model accuracy, device performance, or durable recovery from machine failure.
Report checks establish consistency with retained evidence, not authenticity.

The code adds no inference dependencies. JSON exchange exposes explicit mappings
for other languages; framework-specific objects still need adapters. Intermediate
traces preserve observations without introducing a graph runtime. Final-query
labels never silently score rewritten intermediate queries.
