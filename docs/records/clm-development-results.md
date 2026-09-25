# CLM development results — 2026-09-25

The full CLM development runs completed on the existing Apple M4 Pro using MPS,
BF16 and encoder batch size 1. Quality deserves attention first: the unchanged
BANKING77 development accuracy was 7.53%, and question-first reached 10.00%.
Candidate embedding reuse preserved the unchanged results and substantially
reduced warm request latency. The official test split remains reserved.

These measurements used frozen experiment sources based on Kayak commit
`57efcad7bdb0f40fe852e0be0fd2b2def9d4700c`, before subsequent SDK changes.
They describe that measured implementation and workload, not a benchmark of
every later commit or a claim of equivalence to upstream's full vLLM encoder.
The historical encoder revision used in CLM training remains unestablished;
see the [model contract](../model-contract.md).

## Quality

| Experiment | Correct | Accuracy | Macro F1 | Top-5 accuracy | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged, balanced 77 | 6/77 | 7.79% | 0.036079 | 24.68% | 71 |
| Unchanged, all 770 | 58/770 | 7.53% | 0.044403 | 22.21% | 712 |
| Question-first, all 770 | 77/770 | 10.00% | 0.073031 | 32.21% | 693 |
| Corrected candidate reuse, all 770 | 58/770 | 7.53% | 0.044403 | 22.21% | 712 |

The unchanged model assigned 516/770 predictions to two fee-related labels,
`card_payment_fee_charged` and `top_up_by_bank_transfer_charge`. Of 77 intents,
63 had zero recall. Every pending-card-payment and declined-card-payment example
was assigned to card-payment fees. All calls returned valid results; these are
classification errors, not failed executions.

Question-first fixed 53 baseline errors and introduced 34 new errors; 24 examples
remained correct and 659 remained wrong. It leaves 49 intents with zero recall
and 693 total errors. It remains an experiment rather than a production recipe.
The [evaluation workflow](../evaluation.md#compare-and-iterate) records the request
transformation explicitly and suppresses speedup for an input-order change.

All 77 examples shared by the two unchanged runs returned exactly equal complete
results. The corrected reuse candidate also matched all 770 complete baseline
results and the warmup, including scores, probabilities, token counts and model
metadata. Score preservation establishes unchanged behavior, not improved quality.

### Additional classification metrics

The expanded evaluator recalculated these metrics from the same saved predictions,
without loading the model, changing raw artifacts, or using the official test split.
The existing legacy integrity and timing limitations still apply.

| Full development run | Balanced accuracy | Weighted F1 | MCC | Intents predicted / 77 | Intents with zero recall / 77 | Largest prediction share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged | 7.53% | 0.044403 | 0.072315 | 23 | 63 | 38.44% |
| Question-first | 10.00% | 0.073031 | 0.093149 | 38 | 49 | 27.14% |
| Corrected candidate reuse | 7.53% | 0.044403 | 0.072315 | 23 | 63 | 38.44% |

Each intent has ten examples, so balanced accuracy equals accuracy and weighted
F1 equals macro F1. Uniform random guessing has expected accuracy 1/77 (1.30%);
always choosing any one intent also scores 1.30% on this balanced suite. These
references are calculated, not measured model runs. Prediction diversity and MCC
support the diagnosis of concentrated, weak classification despite valid outputs.
Question-first improves those descriptive measures while still missing 49 intents
entirely. Paired comparison reproduces the 53 fixes and 34 regressions above.

Derived JSON metrics, rendered reports, input SHA-256 receipts, and comparison
output are retained in `validation/clm-development/classification-metrics/`.
To regenerate a report with the current definitions:

```sh
uv run --no-sync kayak eval report .benchmarks/eval/clm-dev770-local-001
uv run --no-sync kayak eval compare .benchmarks/eval/clm-dev770-local-001 \
  .benchmarks/eval/clm-dev770-question-first-001 --allow-recipe-change
```

### Probability quality and an external classifier

The modular benchmark analysis adds another useful check on the same saved 770
development predictions. No CLM inference was rerun for this analysis.

| Measure | Unchanged CLM | Question-first CLM | Better |
| --- | ---: | ---: | --- |
| Log loss, natural log, floor `1e-15` | 5.117465 | 4.680816 | Lower |
| Brier score, unscaled multiclass | 1.036669 | 1.069825 | Lower |
| Chosen-label ECE, ten equal-width bins | 0.204947 | 0.259463 | Lower |
| Top-3 accuracy | 14.42% | 23.51% | Higher |

Question-first improves accuracy and log loss while worsening Brier score and
ECE. Accuracy alone therefore misses part of the probability-quality tradeoff.
The exact two-sided McNemar calculation on 53 fixes and 34 regressions gives
`p = 0.053003`. This is exploratory development analysis with the assumptions
and selection limits described in the [benchmark guide](../classification-benchmarks.md).
It is not held-out confirmation or an automatic promotion criterion.

The [classifier example](../../examples/benchmark_classifiers.py) also ran a
model-free baseline on these exact examples and candidate descriptions:

| System | Correct / 770 | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Always first candidate | 10 | 1.30% | 0.000333 |
| Description word overlap | 276 | 35.84% | 0.347674 |
| Saved unchanged CLM | 58 | 7.53% | 0.044403 |

Word overlap ranks labels by the count of distinct lowercase `\w+` tokens shared
between the customer text and each label description. It uses no gold labels,
training examples, stopword filtering, or stemming. Ties preserve candidate order.
These are different declared methods on the same development task; the comparison
explicitly allows that method change. The stronger lexical baseline reinforces
the need to investigate CLM's task recipe and quality before optimizing further.
It does not establish production quality or a published leaderboard result.

The lexical system supplies rankings, not probabilities, so probability measures
remain unavailable. No inference speed comparison is made from imported
predictions. JSON, Markdown, CSV, and the external prediction files are retained
locally under `validation/eval-extension/final/clm-comparison/` and
`validation/eval-extension/final/external-comparison/`.

## Profiling and repeated timing

Profiling three fixed development requests with 13, 20 and 91 state tokens found
78 singleton encoder forwards per request: one state and 77 candidate descriptions.
Candidate encoding accounted for 92.89–97.41% of instrumented elapsed time. All
results before, during and after instrumentation exactly matched the saved
baseline. This is elapsed-time attribution, not kernel profiling.

The corrected cache passed two independent process groups:

| Partition | Geometric speedup | Process-bootstrap 95% interval | MPS driver growth |
| --- | ---: | ---: | ---: |
| Initial three cases | 33.14x | 32.08–33.53x | 4 MiB |
| Three confirmation cases | 42.30x | 41.66–43.01x | 2 MiB |

Each group used three fresh processes per variant, two excluded warmups and ten
measured rounds per case, fixed process order, and scoped sleep prevention.
All 12 processes completed within the existing memory/time guards. All 432
measured and warmup results exactly matched the original development baseline.
Bootstrap units were whole processes; three trials per group provide limited
uncertainty evidence, not a production guarantee.

| State tokens | Partition | Baseline median | Reuse median |
| ---: | --- | ---: | ---: |
| 13 | Initial | 5.709 s | 0.103 s |
| 20 | Initial | 5.743 s | 0.139 s |
| 91 | Initial | 5.983 s | 0.377 s |
| 18 | Confirmation | 5.761 s | 0.140 s |
| 23 | Confirmation | 5.773 s | 0.135 s |
| 30 | Confirmation | 5.762 s | 0.133 s |

Each latency is the median of three process medians. The gain concerns warm
requests sharing the same 77-candidate block at batch size 1. First calls and
changed candidate blocks still encode candidates; larger batches do not reuse
them. The six cases were selected by state length before timing. The inference
harness's `holdout` tag denotes additional already-seen development examples,
not official test data. All six were classification errors in both variants.

The first cache prototype was rejected after a one-shot nonfinite candidate
output poisoned later calls. The corrected implementation retains a new block
only after a valid complete result. It passed the failure-recovery regression,
then a fresh full development run and all twelve fresh timing trials. No timing
from the rejected prototype was used to adopt the correction.

## Memory, validation and limits

Observed MPS driver allocation was approximately 15.2 GiB. Driver, tensor and
RSS counters overlap and must not be added. Loading headroom was limited on the
available Mac: the unchanged full run reached 1.575 GiB available and 3.821 GiB
system swap. The corrected cache quality run recorded one brief critical-pressure
sample before evaluation began; later samples were normal. Sampling and guards
cannot prevent or exclude every transient allocation/OOM.

The original 770-example baseline experienced system sleep. Its descriptive
5.338-second median uses an active clock, so ratios against that run are not
adoption evidence. The separate repeated timing processes had no sleep gap,
no sampled critical pressure, a minimum 3.029 GiB available, and peak system swap
of 3.831 GiB. Brief warning samples occurred during loading/warmup. macOS psutil
Pageins/Pageouts are not swap-only I/O.

Before integration with later upstream work, all 273 local tests passed without
skips, including native MPS, live HTTP, released-head references, eviction,
question boundaries, and recovery after rejected candidate output. Lint,
formatting, strict types and diff checks passed. One existing Starlette test-client
deprecation warning remained. CUDA, other precisions, changing-candidate workloads
and cold-start gains were not established by these measurements.

## Provenance and local evidence

- CLM: `Contrastive-LM/CLM-v0.1-8B`, revision
  `87655cb835bd76fd66c2da78e1e3709f7fa11a94`.
- Encoder: `Qwen/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- BANKING77 dataset revision: `57ec275d8078af65b7731c2a98be812d844a6d6b`.
  Ten training-derived development examples per intent, unchanged category text
  and order; see the [dataset contract and attribution](../evaluation.md#dataset-and-task-contract).
- Development suite SHA-256:
  `9126dcbf6ba37e3d16afa2eb4867a8fe27ceba12be3488584b12973b3933627c`.
- Timing corpus SHA-256:
  `2c2f5024e2a197c3c7359e0a45c16dfdf2920d4cab03494b8e30f21cbe855171`.
- Corrected measured runtime `_model.py` SHA-256:
  `37d49ac40135c0cbbddb35eabfa15b67306471b7ccc441024273901b46b97ee5`.
- Quality protocol: one warmup, one attempt per example, seed 42.
  Timing seed: 20260924. Python 3.13.5, PyTorch 2.14.0, Transformers 4.57.6.

Raw data is retained locally in ignored directories: `.benchmarks/eval/` contains
`clm-dev77-local-001`, `clm-dev770-local-001`, `clm-dev770-question-first-001`,
and `clm-dev770-reuse-004`; `.benchmarks/inference/clm-reuse-004/` contains the
fresh initial and confirmation trials. `validation/clm-development/` contains
individual errors, paired fixes/regressions, source snapshots, independent audits,
execution history and links to the monitor logs under `validation/`. These large
artifacts and model files are
not included in this commit; this document is the portable result summary.

The saved CLM evaluation reports use schema version 1. The current loader retains
their quality results with a legacy notice; current dataset comparisons withhold
their latency speedups because they predate schema-2 timing provenance. The raw
reports have not been retrofitted with new guarantees. The repeated inference
comparisons above use their separately recorded harness and process evidence.

Use the existing commands with those local artifacts:

```sh
uv run --no-sync kayak eval history .benchmarks/eval
uv run --no-sync kayak eval compare .benchmarks/eval/clm-dev770-local-001 \
  .benchmarks/eval/clm-dev770-question-first-001 --allow-recipe-change
uv run --no-sync -m benchmarks.inference history --root .benchmarks/inference/clm-reuse-004
uv run --no-sync -m benchmarks.inference compare \
  .benchmarks/inference/clm-reuse-004/holdout/baseline \
  .benchmarks/inference/clm-reuse-004/holdout/candidate
```

New runs need fresh output paths and explicitly recorded source revisions.
The current runtime includes reuse and cannot stand in for the original
unchanged baseline. Keep the official test split reserved until a model and
input recipe are frozen.
