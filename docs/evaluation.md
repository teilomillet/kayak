# Benchmark models with BANKING77

`kayak.eval` measures classification quality and sequential decision latency
through the existing `Model` or `Client`. It ships in the wheel, needs no new
dependencies, and saves inputs, predictions, failures, and measurements together.
Full-model runs are manual; no dataset download or model benchmark is added to CI.
For your own dataset and code, start with the [Python evaluation guide](evaluation-python.md).
Start with the [mock audit](records/benchmark-audit.md) to exercise the tooling without
model weights. The [benchmark protocol](benchmark-protocol.md) separates that
check from the later Mac/GPU run and defines what its results could support.

The [2026-09-25 development report](records/clm-development-results.md) records the first
full CLM baseline, question-first experiment, and verified candidate reuse.

To compare multiple systems, import another classifier's predictions, measure
probability quality, or add custom metrics, see
[classification benchmark reports](classification-benchmarks.md). The read-only
`kayak eval benchmark` command writes JSON, Markdown, and CSV without inference.

For varying Choice requests, external rankers, late interaction, or recorded RAG
stages, see [case and RAG evaluation](rag-evaluation.md). These additive APIs
preserve the fixed-classification formats used in this guide.
To execute your own RAG application, configure repeats and gates, use an external
judge, or score JSON from another language, start with [RAG experiments](rag-experiments.md).

## Prepare and run

From this checkout, use uv to install the source with local inference support:

```sh
uv sync --extra local
```

Use `uv sync` without the extra to evaluate an existing service. Dataset
preparation is an explicit network operation:

```sh
uv run --no-sync kayak eval prepare banking77

# A development subset; every request still includes all 77 intents.
uv run --no-sync kayak eval run banking77 --split dev --limit 77 \
  --device mps --dtype bfloat16 --batch-size 1 \
  --output .benchmarks/eval/dev-baseline

# Reuse a running service, without allocating another model.
uv run --no-sync kayak eval run banking77 --split dev --limit 77 \
  --base-url http://127.0.0.1:8000 \
  --output .benchmarks/eval/dev-http
```

`uv run --no-sync -m kayak.eval` exposes the same commands. A standalone tool
installed with `uv tool install '.[local]'` exposes `kayak eval` directly.
Use `--help` for all options. Choose the device and memory budget for your
machine; MPS/BF16 above is a Mac example. The default model is CLM-v0.1-8B.
`--model PATH` selects a compatible [local bundle](model-contract.md).
`--model-cache-dir PATH --local-files-only` uses cached model files offline.
`--data-cache-dir PATH` changes the dataset cache for both preparation and runs.

The default split is `dev`, with one warmup and one measured call per example.
Omit `--limit` to evaluate all 770 development examples. After freezing a model
and input recipe, run the full official test split explicitly:

```sh
uv run --no-sync kayak eval run banking77 --split test \
  --device mps --dtype bfloat16 --batch-size 1 \
  --output .benchmarks/eval/test-frozen
```

This makes 3,080 measured decisions, each with 77 candidates. At batch size 1,
the resident model reuses the last ordered candidate embedding block when its
token rows match; the first call and changed candidate blocks still encode all
candidates. Larger batch sizes encode candidates on every request. Model loading,
warmups, and repeated calls add to the run time. `--batch-size` controls encoder
batching inside a single request, not concurrent requests.

## Dataset and task contract

The source is [PolyAI's BANKING77 dataset](https://github.com/PolyAI-LDN/task-specific-datasets),
pinned at commit `57ec275d8078af65b7731c2a98be812d844a6d6b`. Preparation verifies
the byte length and SHA-256 of `train.csv`, `test.csv`, `categories.json`, and
`LICENSE`. A corrupt cached file is rejected rather than silently replaced.
Evaluation rechecks the cached bytes and never downloads data implicitly.

| Split | Examples | Selection |
| --- | ---: | --- |
| `dev` | 770 | Ten original training examples per intent, selected by fixed SHA-256 order |
| `train` | 9,233 | Remaining original training examples, excluding `dev` |
| `test` | 3,080 | All official test examples, 40 per intent |

This partition preserves all 10,003 original training rows. Example IDs identify
the original split and zero-based CSV row. Within each intent, selection sorts
SHA-256 of `banking77-partition-v1:42:{id}`. Round-robin selection across the
original category order balances limited runs: 77 examples cover every intent
once. The separate `--seed` shuffles execution order without changing membership.
Limits below a full split are explicitly marked `subset` in the report.

Each request contains the customer text, the question “Which banking intent
best matches this customer request?”, and all 77 category descriptions. IDs
and candidate order come from the original `categories.json`; descriptions
replace underscores with spaces and otherwise preserve the original text.
Gold labels and example IDs are never passed to `decide`. The model's existing
input preparation remains unchanged and is identified in its metadata.

This is **zero-shot label-description classification**, with no training examples
or demonstrations supplied by the evaluator. It is a different protocol from
supervised or few-shot results in the original paper. Whether public examples
appeared in any encoder or head training stage is unknown. Use development data for model,
prompt, and runtime experiments; repeated tuning against test results turns that
test set into development data.

The dataset is CC BY 4.0. Retain attribution to Iñigo Casanueva, Tadas Temčinas,
Daniela Gerz, Matthew Henderson, and Ivan Vulić, *Efficient Intent Detection with
Dual Sentence Encoders* (2020), [paper](https://aclanthology.org/2020.nlp4convai-1.5/),
[license](https://creativecommons.org/licenses/by/4.0/). Saved manifests include
source, revision, license, citation, and the partition and description changes.

## Python API and ownership

```python
import kayak
from kayak.eval import banking77, evaluate

suite = banking77(split="dev", limit=77)  # Prepared cache required.
with kayak.load(device="mps", dtype="bfloat16") as model:
    report = evaluate(
        model, suite, output=".benchmarks/eval/python-baseline",
        warmups=1, repeats=2,
    )
print(report.summary["accuracy"])
```

Pass a `kayak.Client` in the same place to measure HTTP. The caller owns the
backend's lifetime; `evaluate` never closes it. `Suite` and `Example` also accept
an explicitly labeled classification corpus with one shared `Choice` question.
A custom backend must implement the typed `decide` interface; if it dispatches
asynchronous accelerator work, supply `sync` to include completion in timing.

The data path is pinned bytes → validated suite → ordinary decision requests →
validated results → pure metric calculation → saved report. The runner owns
ordering, clocks, synchronization, and checkpoint files. There is no alternative
inference implementation, backend registry, background scheduler, or promotion
service. Existing serving defaults and resource ownership remain authoritative.

## Interpret the evidence

Quality uses the **first measured attempt** for each distinct example. Missing
or failed first attempts count as incorrect even if a later repeat succeeds.
The denominator remains the entire selected suite when execution stops early.
Use **accuracy and macro F1** as the main quality measures, then inspect missed
intents and paired regressions before accepting a change. Accuracy is the headline
measure in the [original BANKING77 paper](https://aclanthology.org/2020.nlp4convai-1.5/).
Sharing a metric does not make our zero-shot development protocol comparable to
that paper's supervised/few-shot test results. A subset score is not a full
BANKING77 benchmark score.

| Metric | Definition and purpose |
| --- | --- |
| `accuracy` | Correct first attempts divided by all selected examples. Headline task success. |
| `macro_f1` | Mean per-intent F1 over **all candidate labels**, including zero-support labels. Exposes weak intents. |
| `balanced_accuracy` | Mean recall over labels with gold examples. Gives rare intents equal weight on imbalanced corpora. |
| `weighted_f1` | Per-intent F1 weighted by gold support. Describes performance at the observed class mix. |
| `matthews_correlation` | Multiclass MCC: association between gold and predicted labels; 1 is perfect, 0 indicates no association. A constant predictor scores 0. |
| `top5_accuracy` | Gold label in the first five scores (or all candidates if fewer than five); useful for a shortlist, not automatic routing success. Ties retain candidate order. |
| `zero_recall_labels` | Supported intents with no correct prediction. These intents are completely missed in this run. |
| `predicted_label_count`, `largest_prediction_share` | Number of intents ever predicted and largest intent prediction count divided by all selected examples. Expose concentration in a few labels. Failures are not an intent. |
| `uniform_chance_accuracy`, `majority_class_accuracy` | Expected uniform-guess accuracy, and the accuracy of always choosing the most frequent gold label in this suite. Contextual references, not measured model runs or trained baselines. |

Per-intent precision/recall/F1 and the confusion matrix remain available. Class
scores with zero denominators are 0. Balanced accuracy excludes zero-support
labels; macro F1 retains them. On full, balanced BANKING77 dev/test, balanced
accuracy equals accuracy and weighted F1 equals macro F1. They become distinct
on imbalanced custom suites. Definitions follow the standard
[balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html),
[F1 averaging](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html), and
[MCC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)
conventions without adding a scikit-learn runtime dependency.

The `__failed__` confusion column retains failed/missing first attempts as false
negatives for their true intent. MCC treats that column as an additional predicted
category with zero gold support, using the complete selected-example denominator;
MCC with zero variance is defined as 0. Failed/incomplete runs remain ineligible
for comparisons. These metrics do not measure calibrated confidence, abstention,
or out-of-scope detection; this corpus supplies only in-scope intent labels.

`--repeats N` measures variation on the same examples; it does not add independent
quality observations. The report counts failed calls,
failed warmups, changed choices across repeats, and maximum repeated-score drift.
Drift means the largest absolute score change **from the first successful attempt**,
not the largest difference between any two repeats. Completed attempts are saved;
an interrupted in-flight call can lack a result and duration. No hidden retries
occur. A model identity change during a run becomes a failure.

Latency covers each synchronous `decide` call, including request validation and
response parsing. Local CUDA/MPS work is synchronized before and after timing,
including after a failed decision. The evaluator's defensive question copy occurs
before the timer. A caller-supplied `sync` hook is recorded as custom and does not
establish completion for timing comparisons. Loading, warmups, memory sampling,
reporting, and extra result
checks sit outside measured calls. CLI setup timing distinguishes local model
loading from HTTP client construction. Successful-call and all-attempt timings
are separate. Mean, median, range, and p95 are descriptive over this dataset's
heterogeneous requests; p95 is omitted below 20 calls. Serial service rate is
not concurrent server capacity or a production latency guarantee.

Memory counters are process peak RSS, CUDA peak tensor allocation, or the largest
post-call MPS tensor/driver snapshot. They overlap and must not be added. RSS and
CUDA counters include earlier allocations in the same process. HTTP reports
client memory; server memory and hardware are unknown. `--max-memory-gib` is a
local post-call guard using MPS driver, CUDA tensor, or CPU RSS measurements.
It cannot prevent a transient allocation or OOM. Only a known local `Model` accepts
the guard; HTTP clients and custom adapters do not establish model memory ownership.

Each output directory must be new. `report.json` stores the suite, protocol,
source hash, package/environment details, model identity, counters, and summary.
Local runs also record CPU/accelerator identity, encoder batch size, PyTorch build,
thread counts, numeric settings, and selected runtime environment variables.
These are snapshots before execution, not continuous monitoring of source,
hardware, or settings during a run.
`predictions.jsonl` stores ordered attempts, scores, durations, and bounded error
type/request ID fields. The CLI does not persist API keys or raw backend exception
messages; Python callers' explicit `config` metadata must contain finite JSON
values and is copied before execution so later caller mutations cannot rewrite it.
These files contain the evaluation texts and predictions; choose storage suitable
for your data. Predictions flush per example; metadata checkpoints every 25
examples and at exit. Interruptions retain completed attempts. A hard kill can
leave a partial final line, which the loader rejects rather than guessing.

New reports use schema version 3, which adds the classification metrics above.
`load_report(path)` checks the suite hash,
execution order, candidate contract, model identity, saved summary, and the exact
byte length and SHA-256 of `predictions.jsonl`. Terminal reports bind the entire
file. A `running` checkpoint binds only the prediction prefix it committed; later
readable rows can be recovered but are not covered by that checkpoint's digest.
Schema version 1 remains readable with structural checks and an explicit legacy
notice in rendered reports; it has no prediction-byte guarantee, and its timing
boundary predates these fixes. Hashes detect inconsistent artifacts, not fabricated
execution or someone rewriting both files together. Schema 2 keeps its prediction
digest and original summary contract. Both earlier versions are verified against
their exact historical summaries; missing, altered, or extra summary fields are
rejected. `load_report` retains their schema and summary, while `report`, `history`,
and `compare` derive the current metrics after verification without changing files.
The same read-only calculation is available in Python:

```python
from kayak.eval import load_report, summarize

metrics = summarize(load_report(".benchmarks/eval/dev-baseline"))
print(metrics["matthews_correlation"], metrics["zero_recall_labels"])
```

A `running` checkpoint remains incomplete even if later prediction rows are
readable. `complete` means execution completed
without failures; it does not mean every prediction was correct. CLI exit codes
are 0 for completion, 1 for execution failure or a missed `--min-accuracy` target,
2 for setup/validation errors, and 130 for a keyboard interrupt.

Generate a Markdown report directly from verified artifacts, without inference:

```sh
uv run --no-sync kayak eval report .benchmarks/eval/dev-baseline \
  > .benchmarks/eval/dev-baseline/report.md
```

Keep both JSON artifacts alongside the Markdown. Mock runs marked with
`config.evidence_kind = "mock"` receive a prominent fixture notice. The renderer
also exposes incomplete runs, unknown custom-backend timing, dataset and execution
identity, failures, and memory limitations. It does not certify caller metadata
or infer whether an undisclosed backend actually uses a model.

## Compare and iterate

```sh
uv run --no-sync kayak eval history .benchmarks/eval
uv run --no-sync kayak eval compare .benchmarks/eval/dev-baseline .benchmarks/eval/dev-candidate
```

Comparison reports deltas for accuracy, top-5 accuracy, macro F1, balanced accuracy,
weighted F1, and MCC. `paired_outcomes` counts `fixed`, `regressed`, `both_correct`,
and `both_wrong` examples by ID, using only each first attempt. Fixes minus
regressions equals the change in correct answers; a positive net change can still
hide regressions on important intents. These are descriptive counts, not a
significance test.

Comparison requires two complete, verified runs on identical examples, labels,
and candidate order. Different models or precision are allowed. A changed
question or candidate description requires `--allow-recipe-change`. An adapter
that transforms requests must also record a stable identifier with
`evaluate(..., config={"request_transform": "question-first-v1"})`. The default
is `"identity"`; differing identifiers require the same explicit flag and
suppress latency speedup. This metadata declares the transformation; retain
its source with the experiment to establish what actually ran. Quality
deltas remain descriptive; no statistical significance or automatic promotion
is claimed. Comparisons expose both configurations and environments, plus changed
source, model, precision, and batch size. Those may be deliberate experimental
factors; changing several at once does not identify which caused the difference.
Latency speedup requires complete local provenance and matching hardware, execution
protocol, recipe, dependency versions, and other runtime settings. HTTP and custom
backends, legacy reports, and missing provenance withhold a speedup. Raw timings
remain available. Host load and thermal state can still confound a comparison.

From a source checkout, test the question-first hypothesis against a complete
local BANKING77 development baseline:

```sh
uv run --no-sync -m benchmarks.question_first \
  --baseline .benchmarks/eval/dev-baseline \
  --output .benchmarks/eval/dev-question-first \
  --model-cache-dir validation/model-cache --local-files-only
uv run --no-sync kayak eval compare .benchmarks/eval/dev-baseline \
  .benchmarks/eval/dev-question-first --allow-recipe-change
```

This experiment reuses the saved examples, candidate order, model identity,
device, precision, batch size, and execution protocol. Exchanging the state and
instruction fields of this single-question task produces question-first text
through the unchanged model preparation. The report records
`request_transform="question-first-v1"` and a copy/hash of the experiment source;
the model metadata still describes the released model's own preparation. The
script rejects non-development and incomplete baselines. It does not change
the production recipe or establish a performance improvement.

Use this runner for dataset quality and broad latency measurements. The existing
[inference loop](inference-evals.md) serves a different present purpose: repeated
process trials on fixed interactive cases, numerical-preservation gates, and
bounded batch-size searches. For an optimization, freeze a development suite,
save the baseline, change one factor, retain the candidate's raw results, and
inspect both quality and timing. Confirm improvements with fresh repeated runs
before selecting a configuration, then evaluate the frozen choice on test data.
