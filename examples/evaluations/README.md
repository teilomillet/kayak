# Evaluate a Kayak workflow

These three small datasets teach different evaluation patterns: independent
questions, permission-filtered tool ranking, and document reranking. They retain
17 fictional starter cases. They are teaching data, not a representative quality
benchmark. Review the labels and add real, held-out cases for your application.

| Pattern | Dataset | What to inspect |
| --- | --- | --- |
| Several questions about one input | [feedback.json](feedback.json) | Topic membership, mixed sentiment, and whole-case agreement |
| Rank already permitted tools | [route_tools.json](route_tools.json) | Relevant tool IDs; [the application](../route_tools.py) owns eligibility |
| Rerank retrieved documents | [rerank_documents.json](rerank_documents.json) | Relevant IDs and improvement over the original shortlist order |

The [runner](../evaluate_use_cases.py) submits each stored request, not the
linked example program. Align your application's exact questions and candidates
with its evaluation data. Expected labels stay outside inference inputs.

## Use from Python

```python
from pathlib import Path
from kayak import Client
from examples.evaluate_use_cases import evaluate, read_dataset, summarize

dataset, digest = read_dataset(Path("examples/evaluations/feedback.json"))
with Client(base_url="http://127.0.0.1:8000") as client:
    cases = evaluate(client, dataset)
print(summarize(dataset, cases))
```

The example adapts the public `evaluate_cases` callback API. For fixed-question
classification suites, use the [Python evaluation workflow](../../docs/evaluation-python.md)
and [support suite](../suites/support.json). For native or external ranking,
partial judgments, and RAG stages, use the [case and RAG guide](../../docs/rag-evaluation.md).

For a public customer-support benchmark with explicit out-of-scope cases, use
the [HINT3 v2 workflow](../../docs/hint3-evaluation.md). It supports all three
domains and keeps official test rows separate from training-derived development.

The [support-routing pilot](../../docs/support-routing.md) is the first application
acceptance exercise. It uses a separate four-outcome suite, explicit provisional
targets, and human review for every suggestion, including service failures.

## Validate, simulate, then measure

```sh
# Neither command below loads a model or measures semantic quality.
uv run -m examples.evaluate_use_cases examples/evaluations/*.json --validate
uv run -m examples.evaluate_use_cases examples/evaluations/*.json --simulate

# With a Kayak service running:
uv run -m examples.evaluate_use_cases examples/evaluations/*.json > results.jsonl
```

HTTP mode reads `KAYAK_BASE_URL` and optional `KAYAK_API_KEY`. See
[service setup](../../docs/serving.md) and [hardware requirements](../../docs/validation.md).
Simulation deliberately selects the first candidate independently of labels;
mismatches are expected. Validation checks contracts, not label correctness or
tokenizer limits. Calls are sequential with no retries or interrupted-run resume.

Each output line records one dataset: its hash, every case result and model
identity, failures, and summary. Case hashes bind requests, labels, and cutoffs;
summary validation recomputes metrics. Preserve these files and the code/model
revision when comparing changes. A model identity change counts as failure.

## Interpret the evidence

- Choice `question_accuracy` counts matching questions; `exact_match` requires
  every question in a case to match. A label may list multiple acceptable IDs.
  Failures count against every expected answer and the whole case.
- Ranking `top1`, `hit_at_k`, `recall_at_k`, and `mrr` describe only the supplied
  shortlist. Failed calls contribute zero. The `input_order_*` metrics score
  the unchanged candidate order against the same labels, without another call.
  Preserve retrieval order to see whether reranking helped or hurt.
- Ranking cases require at least one relevant supplied ID. For an all-poor
  shortlist, include and label a meaningful fallback or test the application's
  no-match path separately. A high relative share does not establish relevance.
- CLI evaluation returns 1 for failed calls, incomplete Choice agreement, or
  an irrelevant first-ranked candidate; invalid input returns 2. A zero exit
  status supports only these particular cases.

Check the surrounding application as well. [Tool tests](../../tests/test_search_examples.py)
exercise permission filtering before inference; [file and HTTP tests](../../tests/test_examples.py)
check record preservation and failures; [retry tests](../../tests/test_retry_example.py)
distinguish rejected requests from uncertain completion. Controlled responses
establish integration behavior, not model accuracy.

For generated answers, use [RAG experiments](../../docs/rag-experiments.md): retain
the actual retrieved text, context, answer, and separate review. Selecting a
source does not establish that the answer is correct or supported. Noul/Score
quality and thresholds need their own task-specific labels; these Choice/ranking
files do not validate calibration or direct Noul/Score quality.
