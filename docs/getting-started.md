# Classify and evaluate text with Kayak

Choose categories, validate a file, classify it locally, and read the results.
This walkthrough uses Laya on CPU through Kayak's provider adapter. You need
Python 3.11+ and [uv](https://docs.astral.sh/uv/); no API key or hosted service
is required. Validation needs no model. The classification step loads Laya
and may download its weights on first use.

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

## 1. Choose categories and input

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

The commands below use those bundled files. To classify your own data, replace
`examples/tickets.jsonl` with your input path and `examples/department.json`
with your edited question file. Keep expected labels in a separate evaluation
dataset; input records contain only `id` and `text`.

## 2. Check the file

Check the complete file before loading a model:

```sh
uv run -m examples.classify_file examples/tickets.jsonl \
  --question examples/department.json --validate
```

Expected output: `Validated 2 records and the question. No model loaded.`

This checks JSON structure, required fields, and Kayak's request limits. Invalid
records identify the line and field to fix. Empty files are rejected. It does not
check model availability, tokenizer limits, or whether the categories suit the task.

## 3. Classify your own file

Run the same input and question through Laya:

```sh
uv run --with 'laya==0.3.20' --with 'transformers<5' -m examples.classify_file \
  examples/tickets.jsonl --question examples/department.json \
  --model convaiinnovations/laya --output decisions.jsonl
```

The provider downloads its checkpoint on first use. If you already have weights,
replace `convaiinnovations/laya` with their local directory. Inference is local
and defaults to CPU. Loading status and the current record number appear on
stderr before each operation, so you can see which step is running.
The [provider guide](provider-adapters.md#laya) explains loading and pinning weights.
Laya owns tokenization and truncation, so use short text and concise descriptions
and check its settings when handling longer inputs.

## 4. Read the results

The terminal previews up to five saved predictions and prints the output path.
For example, the display has this shape; the actual choices depend on the model:

```text
Preview (first 2 predictions):
  "ticket-001" -> "billing"
  "ticket-002" -> "technical"
Classified 2 records -> decisions.jsonl
```

Each output line contains `id`, `choice`, and `result`, the full
[`ProviderResult`](provider-adapters.md#what-stays-the-same-and-what-the-result-means).
The preview shortens long IDs and escapes line breaks and terminal control
characters. The saved JSONL retains the complete IDs and provider responses.

The command validates all inputs before opening the output or loading the model.
Missing dependencies produce a copyable setup/run command and leave no output file.
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

To change the routing, edit your question's instructions or category descriptions,
validate again, then choose a fresh output such as `decisions-v2.jsonl`.

## Fix a failed run

| Message | Next step |
| --- | --- |
| `path not found` | Check the named input/question path, or create the output's parent directory. |
| `file.jsonl:2: invalid ticket: text: Field required` | Add the missing `text` string on line 2, then run `--validate` again. Each line must be its own JSON object. |
| `invalid Choice: criteria.…` | Fix the named category in your question file. Each category needs a nonblank text description. |
| `Cannot import Laya or its dependencies` | Copy the full command printed beneath the error; it includes the pinned dependencies and your arguments. |
| `already exists` | Keep the existing results and choose a new `--output` path. |
| `Stopped during model loading` | Check the checkpoint path/model ID and device. The new output may be empty. |
| `Stopped during classification` | Inspect the last record shown in progress and the saved partial results. There is no automatic retry or resume. |

Progress goes to stderr; the preview and final summary go to stdout. The JSONL
file contains only predictions, so progress messages never need to be removed
before another program reads it.

## 5. Evaluate your actual categories

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

## Optional: check integration without weights

These commands exercise the client and produce an example evaluation report
using controlled responses and simple classifiers:

```sh
uv run -m examples.mock_integration
uv run -m examples.benchmark_classifiers \
  --suite examples/suites/support.json --output .benchmarks/first-report
cat .benchmarks/first-report/benchmark.md
```

Use a fresh output directory for each report. The eight fictional cases check
the workflow; these commands do not establish learned-model quality.

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
