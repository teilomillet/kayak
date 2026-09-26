# Customer-support routing with HINT3 v2

Use HINT3 to evaluate intent routing and requests that do not match any supported
intent. Kayak provides pinned data preparation, offline suites for any classifier,
and separate routing and out-of-scope (OOS) metrics. This complements the
[reviewed-ticket workflow](support-review.md): a public benchmark cannot establish
performance on your own customers, policies, languages, or changing intent list.

## Data and split contract

[HINT3](https://github.com/hellohaptik/HINT3) contains curated training examples and
real user test queries from three deployed chatbots. Kayak pins the corrected
**v2** files at revision `99965c892ea9a6801a87083d3861b0b04c91a4e7`, checking
both byte counts and SHA-256. These files differ from the original paper's v1
release; scores must not be compared as though the versions were identical.

| Domain | Official train | Kayak train / dev | Official test | In scope / OOS | Intents + reject |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sofmattress` | 328 | 267 / 61 | 397 | 253 / 144 | 21 + 1 |
| `curekart` | 599 | 488 / 111 | 991 | 459 / 532 | 28 + 1 |
| `powerplay11` | 471 | 389 / 82 | 983 | 309 / 674 | 57 + 1 |

There is no official development split. Kayak reserves 20% of normalized training
text groups per intent, rounded down, with at least one group when an intent has
more than one. Singleton groups stay in training. Group ordering uses SHA-256 of
`hint3-dev-v1:42:` followed by normalized text. Normalization is Unicode NFKC,
case folding, and whitespace collapse; punctuation stays intact. Conflicting
training labels for the same normalized text are rejected. Train and dev never
read test files. This development split contains **no OOS examples**, so it is
insufficient for selecting a rejection threshold.

Official test rows and order remain unchanged. Eight Curekart test rows overlap
normalized official training text. `exclude_train_overlap=True` produces a
separately identified `test-no-train-overlap` sensitivity split of 983 rows;
it does not replace the official result. SOFMattress and Powerplay11 have no such
overlap. Repeated texts within a split remain present: observations are not all
independent. Overlap checks do not detect semantic paraphrases or pretraining
contamination. Some training intents have no test examples.

Candidates are sorted training label IDs, with underscores changed to spaces in
their descriptions. `NO_NODES_DETECTED` is the final explicit reject option.
This simple fixed recipe uses no test labels to design descriptions. Upstream
Powerplay11 training annotation notes are discarded; only `sentence` reaches
inference. A `limit` selects a source-order prefix, **not a balanced sample**.
Suite provenance records selection, recipe, source hashes, license, and overlap.

Data is downloaded only when explicitly requested and is not bundled with Kayak.
The [pinned license](https://github.com/hellohaptik/HINT3/blob/99965c892ea9a6801a87083d3861b0b04c91a4e7/LICENSE.md)
is ODbL 1.0 for the database and DbCL 1.0 for individual contents. Cite
[Arora et al. (2020)](https://aclanthology.org/2020.insights-1.16/); retain
attribution and the applicable data license when redistributing data or derivatives.

## Prepare and export

```sh
# Explicit download: training data and license only.
uv run kayak eval prepare hint3 --domain sofmattress

# Offline export; no model load or implicit network request.
uv run kayak eval suite hint3 --domain sofmattress --split dev --output sof-dev.json

# Freeze your input recipe and model before opening the test split.
uv run kayak eval prepare hint3 --domain sofmattress --include-test
uv run kayak eval suite hint3 --domain sofmattress --split test --output sof-test.json
```

Use `curekart` or `powerplay11` for the other domains. Preparation rejects a
modified cache instead of overwriting it. Output files must be new. From Python:

```python
from kayak.eval import hint3, prepare_hint3

prepare_hint3(domain="sofmattress")
suite = hint3(domain="sofmattress", split="dev")
```

The exported `Suite` works with your own classifier's `PredictionSet`, the
[provider example](../examples/evaluate_provider.py), or the native evaluator:
`kayak eval run hint3 --domain sofmattress --split dev --output .benchmarks/sof-dev`
loads the configured CLM or uses `--base-url` for an existing service. Native
execution requires its documented model resources; exporting/scoring does not.
Keep expected labels outside classifier inputs.

## Score routing decisions

```python
from kayak.eval import PredictionSet, routing_summary
from pathlib import Path

predictions = PredictionSet.model_validate_json(Path("predictions.json").read_bytes())
report = routing_summary(predictions, reject_label="NO_NODES_DETECTED")
print(report["counts"])
print(report["rates"])
```

```sh
uv run kayak eval routing predictions.json --reject-label NO_NODES_DETECTED \
    --output routing.json
# Optional JSON object {"case-id": finite_score, ...}, covering every case:
uv run kayak eval routing predictions.json --reject-label NO_NODES_DETECTED \
    --scores rejection-scores.json --output detection.json
```

An explicit reject answer is distinct from a failed/missing answer. Reports count
correct and wrong routes, false rejections, correct rejections, false accepts,
and failures separately. Every selected example remains in the denominator.

| Metric | Definition |
| --- | --- |
| In-scope accuracy | Correct intent routes / all in-scope cases |
| OOS recall | Correct rejections / all OOS cases |
| OOS precision | Correct rejections / all explicit rejections |
| OOS F1 | `2 TP / (2 TP + FP + FN)`; missing OOS answers count as FN |
| OOS false-accept rate | OOS cases routed to an intent / all OOS cases |
| In-scope false-reject rate | In-scope cases explicitly rejected / all in-scope cases |
| Routing precision | Correct intent routes / all non-reject answers |
| Routing coverage | Non-reject answers / all selected cases |
| Failure rate | Missing answers / all selected cases |

A zero denominator produces `null`, with a reason. Harm rates (false accepts and
false rejections) are also unavailable when missing decisions in that population
could hide harm. Precision is conditional on answers actually given; always read
it alongside coverage, failures, and the `complete` flag. OOS F1 has no interval.
Rates include counts and Wilson 95% intervals, conditional on independent,
representative observations; duplicated texts weaken that assumption. General
classification metrics remain available. Macro F1 includes all candidates;
`unobserved_gold_labels` makes absent test classes visible.

Optional rejection scores are independent scalars where higher means OOS. They
need not be probabilities or calibrated; Kayak never renormalizes provider
scores. They must cover every selected ID with a finite number, and detection
metrics require complete predictions and both gold populations. Missing rows are
never dropped to make a curve look better.

The report includes AUROC, non-interpolated average precision (AP), and false
positive rate at 95% OOS recall. Equal scores enter together. FPR95 uses the first
attainable threshold reaching 95% recall, without interpolation; its negative
population is in-scope cases. The curve's initial threshold is `null` (no positive
decisions), avoiding non-finite JSON. These are **test diagnostics**, not permission
to choose a deployment threshold on the test set. AP depends on OOS prevalence.

## Evidence and limits

The [2026-09-26 experiment](records/hint3-20260926.md) records a bounded cached-model
run on 1,388 test cases, including substantial false rejection rates.

The arithmetic is checked against 43 independently generated scikit-learn cases,
including perfect, reversed, constant, tied, and imbalanced scores, plus analytic
failure cases. The generator imports no Kayak code. Run the pinned command in
[the reference generator](../benchmarks/routing_references.py) to reproduce the
checked-in fixture. Existing paired comparisons retain exact McNemar tests and
within-report Bonferroni correction through `benchmark`; these require the same
cases and complete predictions. Neither test is proof of generalization.

The metrics follow established OOS evaluation practice, including
[recent dialogue-system OOS research](https://aclanthology.org/2025.acl-industry.25/),
and the published definitions of
[AUROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
and [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
This strengthens evaluation coverage and numerical evidence. It does not certify
that the evaluator is universally best or that any evaluated model reaches SOTA.

For Laya 0.3.20, audit tokenization before inference: the default `head_max_len=192`
shortens these candidate lists. The observed lossless budgets with the cached
checkpoint tokenizer are 250/334/734 for SOFMattress/Curekart/Powerplay11;
Powerplay11 also needs more than the default total token budget. Verify both
instructions and every option/message against your actual tokenizer, set the
caller-owned agent configuration, and record it in `method` and metadata. A
successful call alone does not establish that inputs survived truncation. The
provider adapter deliberately preserves the provider's input policy.
