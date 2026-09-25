# Classify and evaluate text with Kayak

Kayak connects text and named questions to typed decisions. This walkthrough
uses support routing to show the complete path: define candidate descriptions,
call a backend, read its chosen ID, then compare predictions with separate labels.

Start with two checks that need no model, service, API key, or accelerator. Then
connect real inference when you have suitable hardware. Python 3.11+ and
[uv](https://docs.astral.sh/uv/) are required.

## Get the examples

```sh
git clone --branch main --depth 1 https://github.com/teilomillet/kayak.git
cd kayak
uv sync
```

This is the 0.5.0 development checkout; that version is not published yet.
To use its library in another project, run `uv add /path/to/kayak`.
The previous [0.4.0 source distribution](https://pypi.org/project/kayak/0.4.0/#files)
remains available with the guides and examples included in that release.

## 1. Exercise the typed client

```sh
uv run -m examples.mock_integration
```

Expected output:

```text
Simulated integration passed; no model was loaded.
```

Open [the example](../examples/mock_integration.py). The application defines
`billing` and `technical`, submits a request through the real `kayak.Client`,
and reads `result.answers["department"].choice`. An HTTP mock supplies a fixed
answer. Response validation and the application assertion run; language
understanding is not being tested.

The selected ID belongs to your application. Map it to a queue or handler in
your code; permissions and execution remain application decisions.

## 2. Produce a report before loading a model

The [support suite](../examples/suites/support.json) has eight fictional requests,
three candidate departments, and separate expected labels. Compare a constant
predictor with a classifier that counts words shared with candidate descriptions:

```sh
uv run -m examples.benchmark_classifiers \
  --suite examples/suites/support.json --output .benchmarks/first-report
cat .benchmarks/first-report/benchmark.md
```

Use a new output directory if you rerun it; reports are not overwritten.
The folder contains a Markdown report, JSON, a metric CSV, and both classifiers'
predictions. These simple baselines do not use CLM or any downloaded model.
Inspect the per-case errors as well as aggregate accuracy. Eight constructed
examples teach the workflow; they do not estimate real-world performance.

The [classifier source](../examples/benchmark_classifiers.py) shows where inputs
become predictions. It does not read expected labels while choosing candidates.
Labels enter only when the evaluator scores those predictions.

## 3. Connect a real model

The default is `Contrastive-LM/CLM-v0.1-8B`. Initial loading downloads about 16 GB
of encoder weights plus 76 MB of heads. Runtime memory exceeds the weight size.
Read the [hardware guide](validation.md) before running this step.

Start one service on your inference machine:

```sh
uv run --extra serve kayak serve --device auto
```

Automatic device selection chooses CUDA, then MPS, then CPU. It does not certify
that the selected machine has sufficient memory. The recorded Mac smoke test
used an M4 Pro with 24 GB unified memory, MPS and **BF16**, with limited headroom;
it did not validate the default FP16 configuration. The hardware guide records
the exact command, memory controls, timings, and limits for that run.

In another terminal, from the checkout root, call the service:

```sh
uv run -m examples.http_client
```

For an existing service, set `KAYAK_BASE_URL` and, when needed, `KAYAK_API_KEY` in
your environment. The [HTTP example](../examples/http_client.py) prints the
chosen department and handles connection, authentication, and overload failures.
Its output is a model prediction, not a guaranteed correct answer.

To keep inference in your Python process instead, follow
[local decisions](../examples/local_decisions.py). Reuse the model context across
requests so the weights load once. For existing Laya or Jev objects, use the
[provider adapters](provider-adapters.md).

## 4. Evaluate your actual routing question

With the service running, save the following as `evaluate_support.py` at the
checkout root. It uses the same eight-case suite from step 2:

```python
import os

from kayak import Client
from kayak.eval import evaluate, load_suite

suite = load_suite("examples/suites/support.json")
with Client(
    base_url=os.environ.get("KAYAK_BASE_URL", "http://127.0.0.1:8000"),
    api_key=os.environ.get("KAYAK_API_KEY"),
) as client:
    report = evaluate(client, suite, output=".benchmarks/support-model")

print(report.summary["accuracy"])
```

```sh
uv run evaluate_support.py
uv run -m examples.benchmark_classifiers \
  --run .benchmarks/support-model --output .benchmarks/support-comparison
```

The comparison uses that run's exact inputs, candidate descriptions and labels.
Failed or missing predictions remain in the quality denominator. Start with
these fictional cases, then replace them with independently reviewed examples
from your application, including ambiguous and out-of-scope inputs. Freeze the
question and candidates before evaluating held-out cases.

## What the existing measurements say

The [recorded BANKING77 development experiment](records/clm-development-results.md)
provides a concrete reason to keep evaluation close to inference:

| System | Correct / 770 | Accuracy |
| --- | ---: | ---: |
| Always first candidate | 10 | 1.30% |
| Description word overlap | 276 | 35.84% |
| CLM, unchanged input recipe | 58 | 7.53% |
| CLM, exploratory question-first recipe | 77 | 10.00% |

These are 77-way, training-derived development results from an earlier measured
implementation on MPS/BF16, not the official test split or a current-checkout run. The
question-first experiment is not the production recipe. The report retains
method, source and dataset identities, timing and memory observations, and errors.
Raw inference artifacts are retained locally and are not bundled with the release.

CLM's normalized score shares are uncalibrated. A high share does not establish
correctness, and a candidate is selected even when every option is unsuitable.
Choose application fallbacks and thresholds using evidence from your own task.

For the full workflow, continue with the [Python evaluation guide](evaluation-python.md).
If you try Kayak on another task, a useful [issue](https://github.com/teilomillet/kayak/issues)
includes the task, a synthetic reproduction, package/model versions, hardware,
and the observed result.
