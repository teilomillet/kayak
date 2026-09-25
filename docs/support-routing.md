# Evaluate support-ticket routing

The first application target is a suggestion for the team that should review a
support ticket: billing, shipping, account, or general manual review. A person
confirms every suggestion. The pilot never moves a ticket, sends a reply,
executes a refund, or changes an account.

Use the [runnable evaluator](../examples/evaluate_support.py) and its
[starter suite](../examples/suites/support_pilot.json). The suite owns the exact
question, ordered candidates, ticket text, labels, and provenance. Keep that
question identical when integrating the pilot into an application. Only the
ticket text and question enter inference; IDs and expected labels stay in the
evaluation process.

## Provisional acceptance criteria

These are proposed targets for the human-reviewed pilot, authorized as a
starting point rather than established business requirements. The evaluator
records the targets and their observed values in `acceptance.json`. Review and
freeze any changes on development data before measuring the final test set.

| Boundary | Provisional target |
| --- | --- |
| Overall agreement | At least 95% of distinct tickets match their reviewed label; failures remain incorrect |
| Each outcome | At least 90% recall for each of billing, shipping, account, and review |
| Manual-review cases | All labeled unclear, unrelated, and multi-team cases select review |
| Added value | At least 5 percentage points higher accuracy than description word overlap on the same tickets |
| Coverage | At least 200 distinct held-out tickets, with at least 40 per outcome |
| Reliability | No failed warmups or measured calls, and a complete run |
| Response time | At most 2 seconds p95 of measured HTTP attempts, including failed attempts; sequential calls, one excluded warmup |
| Application fallback | Every result requires human confirmation; missing/failed results enter general manual review without a model suggestion |

The evaluator owns these numeric targets. Its p95 uses the evaluation library's
interpolated percentile and is unavailable below 20 measured attempts. Keep all
raw timings and report the sample size. Sequential timing does not establish
capacity under load; the service has one active inference slot and no queue.
The 5-second client network timeout is a per-operation HTTP timeout, not an
end-to-end deadline. A timeout leaves remote completion uncertain.

The coverage gate counts distinct exact ticket texts; all rows remain in quality
metrics. The evaluator checks text overlap before inference and rejects datasets
with unresolved findings. Use the [review workflow](support-review.md) to retain
two annotations, adjudications, group/split checks, and verified compiled suites.
These checks cannot certify independence,
label quality, or representativeness. Passing the numeric gates with clear data checks produces
`provisional_gates_passed=true`, while `deployment_accepted` remains false.
The application owner must review the evidence and accept the targets, hardware,
capacity, and recovery procedure before using the pilot with real traffic.

## Prepare independent evidence

The included 20 tickets are fictional development fixtures, five per outcome.
Their labels are examples of the intended policy, not independent review.
They deliberately cannot meet the coverage gate. Use them to check data handling
and inspect failure reports before collecting private application data.

For the real evaluation:

1. Sample permissioned, de-identified support tickets from the intended workload.
   Keep an additional natural-frequency sample when oversampling rare outcomes.
2. Have support reviewers label each ticket before showing model predictions.
   Resolve disagreements explicitly; record the label policy, reviewers,
   collection window, and adjudication in suite provenance. Label insufficient
   information, unrelated requests, and multiple-team issues as `review`.
3. Split by conversation/customer and time where appropriate, keeping related
   tickets together. Use development data for wording and target changes;
   reserve the final test set until those choices are frozen.
4. Include paraphrases, rare teams, long and malformed inputs, multi-issue tickets,
   and operational failures. Empty/oversized inputs and HTTP errors are separate
   integration challenges; a labeled suite alone cannot exercise all of them.
5. Retain the suite hash, source commit, model fingerprint/revisions, server
   precision/device/settings, package versions, every prediction/failure, raw
   timings, and independent review. The reports contain ticket text; retain
   them in the application's approved storage.

The [offline review walkthrough](support-review.md#complete-offline-walkthrough)
exercises these file boundaries with fictional annotations. It prepares the
workflow; reviewers still supply the actual application judgments.

`review` is an ordinary model candidate. A high relative share does not certify
correctness, and adding that candidate does not guarantee abstention. Human
confirmation is the pilot's fallback policy; there is no invented confidence
cutoff. HTTP/input errors retain their failure identity in the quality report,
even when the application sends the ticket to manual review.

## Run the checks

The pilot files are new checkout examples using the published 0.4.0 API. They are
not in the already published 0.4.0 source archive. From this checkout:

```sh
uv sync
uv run -m examples.evaluate_support --validate
uv run -m examples.evaluate_support --simulate --output .benchmarks/support-demo
```

Use a fresh output path for each run. Simulation uses the real client with a
controlled response that always picks the first candidate, independently of the
labels. It should record 5/20 matches and failed quality/coverage gates. It makes
no model-quality or latency claim. Exit 0 means the simulated integration ran;
`provisional_gates_passed` stays false.

With reviewed data and an existing service on validated hardware:

```sh
KAYAK_BASE_URL=http://127.0.0.1:8000 \
uv run -m examples.evaluate_support --suite /path/to/reviewed-support.json \
  --output .benchmarks/support-test
```

Set `KAYAK_API_KEY` when required by the service. Calls are sequential, without
retries. The command validates the complete suite before sending requests,
retains all outcomes, and exits 1 if a provisional gate fails. Invalid inputs or
an output-directory error return 2. Existing output is preserved.

Inspect `report.json`, `predictions.jsonl`, `acceptance.json`, and
`baselines/benchmark.md` together. Baseline comparison uses the exact same
suite; its word-overlap classifier sees only text and candidate descriptions.
Quality uses one measured attempt per ticket. HTTP timing includes the client
and network, while server memory is not measured by this client-side evaluator.

## Current conclusion and next decision

The runnable fixtures establish evaluation and fallback behavior. Support-ticket
model usefulness remains unmeasured until independently reviewed application
data is run against the pinned full model on the intended deployment.
Historical [BANKING77 development results](records/clm-development-results.md)
reported 7.53% unchanged-model accuracy versus 35.84% for word overlap on that
separate 77-intent task. Those results motivate the baseline check; they do not
estimate accuracy on this four-outcome support task.

Before application traffic, retain the hardware, load, integration, and recovery
evidence required by [deployment acceptance](release.md#deployment-acceptance).
Independent full-encoder equivalence, CUDA, default FP16, and calibrated
confidence remain separate open claims in the [release scope](release.md).
