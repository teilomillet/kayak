# Evaluation reference checks

Kayak's evaluator is checked against established numerical implementations and
documented benchmark protocols. This establishes agreement within the checked
cases. It is not a certification, a model-quality result, or a claim that Kayak
has state-of-the-art accuracy.

## Independent arithmetic

[The reference generator](../benchmarks/eval_references.py) imports scikit-learn
and SciPy, without importing Kayak. Its fixed seed generates 80 cases, including
all-missing and perfect predictions, zero-support candidates, binary and
multiclass probabilities, graded relevance, and extreme binomial counts.
[The saved fixture](../tests/fixtures/eval-reference.json) contains the inputs,
reference outputs, seed, and library versions. Normal tests need neither library:

```sh
uv run --no-sync pytest -q tests/test_eval_references.py
```

Regenerate independently in a separate environment, preserving the existing file:

```sh
uv run --no-project --with scikit-learn==1.9.1 --with scipy==1.18.0 \
    --with numpy==2.5.3 \
    python benchmarks/eval_references.py --output /tmp/eval-reference.json
cmp /tmp/eval-reference.json tests/fixtures/eval-reference.json
```

The output must be a new file. A mismatch requires investigating library versions,
conventions, and arithmetic; do not replace the fixture just to make a test pass.

| Kayak calculation | Independent reference | Matched convention |
| --- | --- | --- |
| Accuracy, macro/weighted F1, balanced accuracy, MCC | scikit-learn classification metrics | Missing predictions are incorrect; MCC retains a separate failure category. Macro F1 includes every candidate; balanced accuracy includes supported gold labels. |
| Log loss | scikit-learn `log_loss` | Natural log, interior probabilities. At zero, Kayak's explicit probability floor differs from sklearn's dtype-based clipping; existing boundary tests check Kayak's documented rule. |
| Brier | scikit-learn `brier_score_loss` | Explicit `scale_by_half=False`, including binary inputs. |
| nDCG | scikit-learn `ndcg_score` | Linear gain, fully judged lists, strict ranks without score ties. Unknown judgments and incomplete prefixes retain Kayak's documented unavailable values. |
| Exact McNemar | SciPy `binomtest` | Two-sided p=0.5 on discordant pairs. |
| Wilson interval | SciPy `BinomTestResult.proportion_ci` | 95%, Wilson method. |

The 80 cases compare 219 scalar results, with absolute/relative tolerances at
`1e-12` (McNemar uses only relative tolerance so tiny p-values cannot pass as
zero). They complement the existing
hand-calculated, property-based, integrity, incomplete-run, and leakage checks.
They do not independently validate every evaluator feature or every numerical
input. References: [scikit-learn metrics](https://scikit-learn.org/stable/api/sklearn.metrics.html),
[Brier scaling](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html),
[nDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html),
[SciPy binomial test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

## Recipe identity and comparison

`PredictionSet.method` is the declared recipe compared by `benchmark`. Free-form
metadata remains descriptive; a changed metadata field alone does not trigger
the recipe-change gate. The provider example accepts `run(..., method=...)` and
`--method` so callers can record token budgets, retrieval, prompt changes, and
provider settings before collecting predictions. Its live CLI defaults record
the SDK version, requested model, and local configuration or hosted call settings.
An explicit method replaces that default, so the caller must describe the full
recipe. A model name/configuration does not verify checkpoint bytes or a hosted
provider's internal implementation.

Changed recipes require `allow_recipe_change=True` to receive paired comparisons.
This is a disclosed intervention, not evidence that the systems had identical
inputs. Record a checkpoint revision/hash separately, retain all attempts, and
freeze the recipe before final testing. See [provider comparison](provider-adapters.md#compare-on-the-same-cases)
and [classification conventions](classification-benchmarks.md).

## What a SOTA comparison would require

The references consulted on September 25, 2026 define tasks and evaluation
conditions, not one universal score for an evaluator:

- [MTEB benchmark selection](https://docs.mteb.org/get_started/usage/selecting_tasks/)
  fixes tasks, languages, and splits. Its
  [classification tasks](https://docs.mteb.org/overview/available_tasks/classification/)
  include probes trained on embeddings. A zero-shot decision among descriptions
  is a different protocol.
- [MTEB's two-stage example](https://docs.mteb.org/get_started/advanced_usage/two_stage_reranking/)
  retains first-stage predictions and evaluates a second stage over that shortlist.
  For Kayak intent routing, measure retrieval-only top1 and recall@k, then full-task
  decision accuracy. Gold missing from the shortlist is an end-to-end error;
  accuracy conditional on retrieval success is only a diagnostic. Include online
  retrieval in pipeline timing and disclose index/model setup separately.
- [BANKING77's source](https://github.com/PolyAI-LDN/task-specific-datasets)
  defines 10,003 training and 3,080 test examples across 77 intents. Kayak's 770-case
  development subset comes from the official training set. It is not the official
  test set, and unknown pretraining overlap prevents an uncontaminated evaluation
  claim.

For a release-quality comparison, choose the target protocol and baseline first,
audit train/development/test overlap, retain every tested configuration, then
freeze the system and evaluate an untouched test set once. A public dataset
alone cannot establish absence of pretraining contamination. A claim of broader
generalization also needs independent tasks or private labeled cases. Report
per-class errors, missing outputs, resources, and paired changes on identical
examples. The existing Wilson/McNemar calculations assume representative,
independent samples; repeated calls and development tuning do not supply those
assumptions. Repeated training seeds belong to a future training study.

The current development experiments keep the official test split sealed. This
audit strengthens the evaluator; it does not establish a leaderboard position.
