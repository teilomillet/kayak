# Evaluate your own task in Python

Use `Suite` for your labeled cases, `evaluate` to collect predictions, and
`compare` to inspect what a change fixed or broke. A resident `Model`, a `Client`,
or an adapter implementing `decide` uses the same evaluation function. You own
the backend's lifetime and choose when inference runs.

## Define the task

```python
from kayak import Choice
from kayak.eval import Example, Suite

suite = Suite(
    name="support-routing",
    split="dev",
    question=Choice(
        instructions="Choose the team that should handle this request.",
        criteria={
            "billing": "Charges, invoices, payments, and refunds",
            "account": "Login, passwords, and account access",
        },
    ),
    examples=[
        Example(id="duplicate", text="I was charged twice.", label="billing"),
        Example(id="login", text="My password no longer works.", label="account"),
    ],
    provenance={"source": "constructed demonstration; replace with reviewed cases"},
)
```

For data maintained in a file, use the same schema:

```python
from pathlib import Path
from kayak.eval import load_suite

Path("support.json").write_text(suite.model_dump_json(indent=2), encoding="utf-8")
suite = load_suite("support.json")
```

The loader validates the whole file before any inference. It preserves candidate
order, text, labels, and provenance, and rejects duplicate JSON keys, duplicate
example IDs, unknown labels, empty suites, and invalid requests. Invalid content
raises `ValueError` with the file and validation details; file-access errors
remain `OSError`. No model is loaded or downloaded. Token limits are checked
when the actual backend tokenizes a request.

## Measure one change

With an already running Kayak service:

```python
from kayak import Client
from kayak.eval import compare, evaluate, render_comparison

# Deep-copy before changing descriptions: the baseline keeps its own values.
candidate = suite.model_copy(deep=True)
candidate.question.criteria["account"] = "Cannot log in or reset a forgotten password"

with Client(base_url="http://127.0.0.1:8000") as client:
    baseline = evaluate(client, suite, output=".benchmarks/support-before")
    changed = evaluate(client, candidate, output=".benchmarks/support-after")

result = compare(
    ".benchmarks/support-before",
    ".benchmarks/support-after",
    allow_recipe_change=True,
)
regressions = [case for case in result["cases"] if case["outcome"] == "regressed"]
for case in regressions:
    print(case["id"], case["text"], case["expected"], case["candidate_choice"])
print(render_comparison(result))
```

Each output directory must be new. `evaluate` saves the validated suite,
predictions, failures, model identity, and timing protocol; it returns a `Report`
and does not close your backend. Only text and the question go to `decide`, never
gold labels or example IDs. Quality uses the first measured attempt per case;
repeats measure timing and do not create more labeled observations. Failures stay
in the denominator and make the run ineligible for paired comparison.

`compare` reads saved runs without inference and returns a typed `Comparison`
dictionary. `cases` contains every example in suite order, paired by ID:

| Field | Value |
| --- | --- |
| `id`, `text`, `expected` | Original example and gold label |
| `baseline_choice`, `candidate_choice` | First measured choices |
| `outcome` | `both_correct`, `fixed`, `regressed`, or `both_wrong` |

The four outcome counts sum to the number of examples. Fixes minus regressions
equals the change in the number of correct predictions. The comparison requires
identical examples, labels, provenance, and candidate order. Changed wording or
request transformations require `allow_recipe_change=True`; missing or failed
runs raise `ValueError`. Inspect the report's changed factors before attributing
a difference to your intended intervention. HTTP timing cannot establish a local
inference speedup because the server environment is unknown.

`render_comparison` returns Markdown with regressions first, then fixes and
remaining errors. It displays up to ten cases per group and shortens cells to
240 characters. The Python result retains every case and full text. Save it with
`json.dumps(result, ensure_ascii=False, indent=2)` if you want a JSON artifact.
These reports contain your input text; choose their storage and audience accordingly.

For pinned customer-support data and rejection diagnostics, see
[HINT3 v2 and routing metrics](hint3-evaluation.md).

## Compare another classifier or change a metric

Pass caller-supplied `PredictionSet` values or saved run paths to `benchmark`.
The first mapping entry is the baseline. The typed `BenchmarkResult` has metric
values and the same case records; ineligible pairs have reasons instead of cases:

```python
from kayak.eval import benchmark

analysis = benchmark({"before": ".benchmarks/support-before",
                      "after": ".benchmarks/support-after"}, allow_recipe_change=True)
accuracy = analysis["runs"]["after"]["metrics"]["accuracy"]["value"]
for pair in analysis["comparisons"]:
    if pair["eligible"]:
        regressions = [case for case in pair["cases"] if case["outcome"] == "regressed"]
    else:
        print(pair["exclusions"])
```

The [classifier example](../examples/benchmark_classifiers.py) contains the
complete path from text and descriptions through predictions to a custom error
cost. Run its Python function with the [editable support suite](../examples/suites/support.json):

```python
from pathlib import Path
from examples.benchmark_classifiers import run
from kayak.eval import load_suite

result = run(Path(".benchmarks/support-demo"), suite=load_suite("examples/suites/support.json"))
cost = result["runs"]["word_overlap"]["metrics"]["example_error_cost"]["value"]
```

This example runs from a checkout or source archive with base dependencies and
no model or service. It compares a constant predictor with word overlap on eight
fictional cases; its scores demonstrate data handling, not model quality.
Replace the predictor or supply a `Metric` containing your own pure function over
immutable `MetricInput`. Neither change requires editing Kayak or registering a
plugin. See [prediction exchange and custom metrics](classification-benchmarks.md)
for the full contracts and statistical assumptions.

For a complete application evaluation with provisional quality, latency, and
failure criteria, use the [support-routing pilot](support-routing.md).

For useful quality evidence, use reviewed cases representative of your task,
keep development and final test data separate, and freeze the chosen recipe and
metrics before the final test. A successful integration check or a better score
on these constructed examples does not establish generalization.
