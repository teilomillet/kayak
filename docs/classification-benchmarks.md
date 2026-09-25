# Build and compare classification benchmarks

`kayak eval benchmark` analyzes existing predictions from Kayak or another
classifier. It writes a shareable Markdown comparison, machine-readable JSON,
and a metric CSV. It never loads a model, downloads data, or calls a service.
Use the [evaluation workflow](evaluation.md) to collect native predictions first.

## Generate a report

```sh
# One input produces a report; additional inputs are compared to the first.
uv run --no-sync kayak eval benchmark .benchmarks/eval/baseline \
  .benchmarks/eval/candidate --output .benchmarks/comparison

# Optional confidence policy and calibration resolution, recorded in the report.
uv run --no-sync kayak eval benchmark .benchmarks/eval/baseline \
  --ece-bins 15 --confidence-threshold 0.8 --output .benchmarks/confidence
```

The output directory must be new. It contains `benchmark.md`, `benchmark.json`,
and `metrics.csv`. CLI inputs are numbered in order; the report also shows system,
method, dataset, split, evidence type, and answered/selected counts. Python callers
can give runs descriptive names. JSON retains metric versions, parameters,
directions, coverage, per-class metrics, confusion matrices, input hashes, Python
version, and analysis source hashes. CSV has one row per run and metric. Keep
the source predictions alongside the derived report for reproducibility.

Native artifacts pass `load_report` before analysis. Their original schema,
summary, and prediction bytes remain unchanged. Legacy runs retain their limited
integrity guarantees. Mock evidence remains labeled. Imported predictions receive
structural validation and input hashes, not certification that inference ran.

## Measures and conventions

Accuracy, macro/weighted F1, balanced accuracy, MCC, and classification diagnostics
share arithmetic with the original evaluator. Additional measures are:

| Measure | Meaning | Convention |
| --- | --- | --- |
| Top-k accuracy | Whether the correct intent appears in a shortlist | Defaults k=3 and 5; Python accepts any positive k, capped by candidate count. Explicit rankings preserve order; probability ties follow suite candidate order. |
| Log loss | Penalizes assigning little probability to the correct intent | Mean `-ln(max(epsilon, p_gold))`; epsilon defaults to `1e-15`. Floor only, without renormalization. Lower is better. |
| Brier score | Measures error across the whole probability distribution | Mean sum of squared errors against a one-hot target; **[0, 2] for binary and multiclass**, without halving. Lower is better. |
| ECE | Compares chosen-label confidence with correctness within bins | Default ten equal-width bins, `[lower, upper)`, with 1 in the last bin. Weighted absolute confidence/accuracy gap. Lower is better; bin choices can hide errors. |
| Selective accuracy | Accuracy after applying a confidence threshold | Include chosen-label probabilities `>= threshold`; unavailable if none qualify. |
| Coverage | Fraction of selected examples meeting that threshold | Always inspect with selective accuracy. A low-coverage result is not full-task accuracy. |
| Accuracy interval | Uncertainty estimate for accuracy | Wilson 95%; shown only with complete predictions and a complete native run, if applicable. |
| Paired comparison | What changed on identical examples | Fixes, regressions, both correct/wrong, metric deltas, exact two-sided McNemar p, and Bonferroni adjustment across eligible comparisons in this report. |

Missing/failed predictions remain incorrect in label-based measures. Probability
and ranking measures require coverage of **every selected example**; otherwise
they return `null` with coverage and a reason. The evaluator does not discard
hard cases or turn hard labels into probabilities. Top-k needs full rankings or
probabilities; it cannot be inferred from a single chosen label.

CLM's normalized score shares are not established as calibrated confidence.
Log loss and Brier assess probability quality; neither isolates calibration from
discrimination. ECE is a descriptive binning estimate, not proof of calibration.
Choose thresholds on development data, then freeze them before testing. This
in-scope dataset does not establish out-of-scope rejection performance.

Statistical calculations assume independent, representative examples. Repeats
are not new labeled samples. McNemar concerns correctness, not changes in F1 or
probability quality. Bonferroni includes only this report's eligible baseline
comparisons; it cannot correct for earlier prompt/metric selection or development
tuning. The tooling does not assign a winner or claim statistical significance.

## Inspect fixes and regressions in code

`benchmark(...)` returns a typed `BenchmarkResult` dictionary. Eligible entries
in `result["comparisons"]` include `cases`: every example in suite order, with
its ID, original text, expected label, baseline/candidate choices, and outcome
(`both_correct`, `fixed`, `regressed`, or `both_wrong`). Ineligible comparisons
have `exclusions` and omit cases. JSON keeps full text and all examples; Markdown
shows up to ten regressions, fixes, and remaining errors each, with shortened cells.
Reports therefore contain evaluation input text as well as predictions.

For task construction, loading your own data, native `compare` results, and
direct Python examples, see the [Python evaluation guide](evaluation-python.md).

## Bring another classifier

`PredictionSet` contains the exact `Suite`, a system identity, a declared `method`,
and predictions keyed by example ID. Describe training data, demonstrations,
prompt, and label-selection procedure in the method and metadata. A different
model does not need to fabricate Kayak logits, model fingerprints, or call times.

```python
from pathlib import Path
from kayak.eval import Prediction, PredictionSet, load_report, benchmark, write_benchmark

suite = load_report(".benchmarks/eval/baseline").suite

# Illustrative constant classifier: replace this with your own predictor.
# Pass customer text and candidate descriptions to it, never gold labels.
first_label = next(iter(suite.question.criteria))
other = PredictionSet(
    system="constant baseline",
    method="No training; always predict the first candidate",
    suite=suite,
    predictions=[Prediction(id=case.id, choice=first_label) for case in suite.examples],
)
Path("other.predictions.json").write_text(other.model_dump_json(indent=2))
result = benchmark(
    {"kayak": ".benchmarks/eval/baseline", "other": other},
    allow_recipe_change=True,
)
write_benchmark(result, ".benchmarks/other-comparison")
```

If your model has `predict_proba`, map its class order explicitly into
`Prediction(probabilities={label: probability, ...})` for **all** suite labels.
Probability key order does not define class order. Values must be finite, in
[0, 1], and sum to 1 within `1e-6`; no silent normalization occurs. You may supply
a complete `ranking` with the choice first. Explicit native rankings come from
scores and take precedence over rounded probability ties. A choice may differ
from the probability argmax, for example under a cost policy; calibration and
threshold measures then use the chosen label's probability. If ranking is omitted,
top-k uses descending probabilities, independently of that decision policy.

Duplicate/unknown IDs, unknown labels, incomplete probability vectors, invalid
rankings, and nonfinite data are rejected. Missing IDs or `choice=None` remain
unanswered. Failed/unanswered rows cannot supply probabilities or rankings.

Export a verified native run to the same interchange format:

```sh
uv run --no-sync kayak eval export .benchmarks/eval/baseline \
  --system CLM --method 'Zero-shot label descriptions; unchanged request recipe' \
  --output clm.predictions.json
uv run --no-sync kayak eval benchmark clm.predictions.json other.predictions.json \
  --allow-recipe-change --output .benchmarks/external-comparison
```

Compare only the same dataset provenance, examples, gold labels, and candidate
order. Mismatches produce explicit comparison exclusions, while individual
metrics remain visible. Changed questions or declared methods require
`allow_recipe_change=True` / `--allow-recipe-change`. This permits a disclosed
experiment; it does not make supervised, few-shot, and zero-shot protocols equivalent.
Incomplete native runs and unanswered examples are ineligible for paired analysis.
Export/import retains the native `original_status` restriction, including failures
in warmups or later repeats when every first-attempt prediction is present.
Such runs also remain ineligible for accuracy intervals.
Native latency comparisons retain the existing hardware/protocol gates. External
prediction files do not establish latency, throughput, or memory consumption.

A published leaderboard number alone lacks the predictions and protocol needed
for paired evaluation. Run that system on this suite or import its predictions
with matching provenance. Do not label this development subset as an official
BANKING77 or MTEB result.

See [independent reference checks](evaluation-reference-checks.md) for numerical
agreement with scikit-learn/SciPy, recipe identity, and the conditions required
for a comparable external benchmark.

## Extend metrics in Python

A metric is a pure function over immutable `MetricInput` samples. Each sample has
an ID, gold label, choice (possibly `None`), optional probabilities in `data.labels`
order, and an optional ranking. No registry, downloads, or automatic module imports
are involved. Custom code runs only when a Python caller explicitly supplies it.

```python
from kayak.eval import Metric, MetricInput, benchmark, default_metrics, write_benchmark

def missed_fraction(data: MetricInput) -> float:
    return sum(row.choice != row.label for row in data.samples) / len(data.samples)

metric = Metric(
    "missed_fraction", missed_fraction, "Selected examples not classified correctly.",
    direction="lower", version="1",
)
result = benchmark(
    {"baseline": ".benchmarks/eval/baseline"},
    metrics=(*default_metrics(), metric),
)
write_benchmark(result, ".benchmarks/custom-report")
```

Names must be unique. Return a finite number, or `None` for an undefined measure.
Use `requires="probabilities"` or `requires="ranking"` to require complete coverage.
Metric versions and parameter tuples are recorded; preserve your implementation
and update its version when semantics change. They are caller declarations, not
authenticated code identities. Exceptions abort analysis before report writing.
Inputs are immutable snapshots; changes to file inputs during analysis are rejected.

Factories include `top_k_accuracy(k)`, `log_loss(epsilon=...)`,
`expected_calibration_error(bins=...)`, and `confidence_metrics(threshold)`.
`score_metrics(metric_input(predictions), metrics)` exposes the calculation without
report I/O. `reliability_bins(...)` supplies counts, mean confidence, and accuracy
for a custom calibration plot. Empty bins retain zero counts and null values.

The runnable [classifier example](../examples/benchmark_classifiers.py) compares
a constant predictor and word overlap, and adds an illustrative error-cost metric:

```sh
uv run --no-sync -m examples.benchmark_classifiers --output .benchmarks/demo
# Compare against an existing run on its exact suite:
uv run --no-sync -m examples.benchmark_classifiers \
  --run .benchmarks/eval/clm-dev770-local-001 --output .benchmarks/lexical-comparison
```

## Sources behind the design

The separation of reusable metric functions from prediction collection follows
the useful pattern in [Hugging Face Evaluate](https://huggingface.co/docs/evaluate/a_quick_tour)
and [LightEval](https://huggingface.co/docs/lighteval/adding-a-new-metric), without
adding their runtimes as dependencies. Metric conventions are explicit because
names alone do not ensure comparable numbers: see [log loss](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html),
[Brier scaling](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html),
and [probability calibration](https://scikit-learn.org/1.8/modules/calibration.html).
ECE and reliability bins follow the confidence/correctness approach in
[Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html), with bin edges
declared above. Statistical references are the [NIST Wilson interval](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm),
[exact McNemar test](https://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html),
and the [NLP significance-testing guide](https://aclanthology.org/P18-1128/).
