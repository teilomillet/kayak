# Classify and evaluate text with Kayak

Kayak connects text and named questions to typed decisions. This walkthrough
uses support routing to show the complete path: define candidate descriptions,
call a backend, read its chosen ID, then compare predictions with separate labels.

Start with two checks that need no model, service, API key, or accelerator. Then
classify a file with a local Laya model. Python 3.11+ and
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

## 3. Classify your own file

Use the [file classifier](../examples/classify_file.py) to assign categories to
messages, support requests, or other short text. Copy
[department.json](../examples/department.json) and edit its instructions and
candidate descriptions for your task:

```json
{
  "type": "choice",
  "instructions": "Which team should handle this request?",
  "criteria": {
    "billing": "Charges, invoices, and refunds",
    "technical": "Bugs and service outages"
  }
}
```

Each line of the input file contains only an ID and text, like
[tickets.jsonl](../examples/tickets.jsonl):

```json
{"id":"ticket-001","text":"I was charged twice for my subscription."}
{"id":"ticket-002","text":"Our entire team cannot log in and work has stopped."}
```

Check the complete file before loading a model:

```sh
uv run -m examples.classify_file examples/tickets.jsonl \
  --question examples/department.json --validate
```

This checks JSON structure, required fields, and Kayak's request limits. Invalid
records report a line number. Empty files are rejected. It does not check model
availability, tokenizer limits, or whether the categories suit the task.

Run with an existing local Laya checkpoint:

```sh
uv run --with 'laya==0.3.20' --with 'transformers<5' -m examples.classify_file \
  examples/tickets.jsonl --question examples/department.json \
  --model /path/to/laya --output decisions.jsonl
```

If you do not have weights yet, replace `/path/to/laya` with
`convaiinnovations/laya`; the provider downloads its checkpoint on first use.
Inference is local and defaults to CPU. No API key or hosted inference is needed.
The [provider guide](provider-adapters.md#laya) explains loading and pinning weights.
Laya owns tokenization and truncation, so use short text and concise descriptions
and check its settings when handling longer inputs.

Each output line contains `id`, `choice`, and `result`, the full
[`ProviderResult`](provider-adapters.md#what-stays-the-same-and-what-the-result-means).
Read the selected IDs in Python:

```python
import json

with open("decisions.jsonl", encoding="utf-8") as records:
    for line in records:
        record = json.loads(line)
        print(record["id"], record["choice"])
```

The command validates all inputs before opening the output or loading the model.
It loads weights once, processes records in input order, and flushes each result.
Choose a new output path for each run; existing files are never overwritten.
If inference fails or you press Ctrl-C, it exits nonzero and keeps completed
records. The output may be empty if loading failed. There is no retry, resume, or
ID deduplication. Keep the input unchanged while the command runs. An abrupt
termination or storage failure can leave the last line incomplete.

Use `choice` as a suggestion for your application to review. The example does not
move tickets or execute actions. Every request gets one of your candidates, even
when none fits; provider confidence is not established correctness. Results
include raw provider data, so treat output files like the original input data.

## 4. Evaluate your actual categories

Keep expected labels in a separate evaluation suite. The
[provider evaluation example](provider-adapters.md#compare-on-the-same-cases)
compares Laya with a word-overlap baseline and saves per-case results:

```sh
uv run --with 'laya==0.3.20' --with 'transformers<5' -m examples.evaluate_provider \
  --provider laya --model /path/to/laya \
  --suite examples/suites/support.json --output .benchmarks/laya-support
```

The bundled suite contains fictional requests and its own question. For your
application, copy the [Suite format](evaluation-python.md), use the same
instructions and candidates as your classifier, and add independently reviewed
examples, including ambiguous and out-of-scope inputs. Freeze the question and
candidates before evaluating held-out cases. Inspect per-case errors as well as
accuracy; failed or missing predictions remain in the quality denominator.

## Optional: run native CLM as a service

Kayak also loads `Contrastive-LM/CLM-v0.1-8B`. Initial loading downloads about
16 GB of encoder weights plus 76 MB of heads. Runtime memory exceeds the weight
size. Read the [hardware guide](validation.md) before running:

```sh
uv run --extra serve kayak serve --device auto
# In another terminal:
uv run -m examples.http_client
```

Automatic device selection chooses CUDA, then MPS, then CPU. It does not certify
sufficient memory. The recorded Mac smoke test used MPS and BF16 on an M4 Pro
with 24 GB memory and limited headroom; it did not validate default FP16.
For an existing service, set `KAYAK_BASE_URL` and optionally `KAYAK_API_KEY`.
For inference in your Python process, follow
[local decisions](../examples/local_decisions.py). For a caller-owned Laya or Jev
object, use the [provider adapters](provider-adapters.md).

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
