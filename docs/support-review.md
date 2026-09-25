# Prepare and review support tickets

This workflow turns permissioned, de-identified tickets into the suites consumed
by the [support pilot](support-routing.md). It runs with the base package and
uses no model, tokenizer, or external service. The review commands preserve the
source text, question, candidate order, both annotations, and each adjudication.
They do not collect tickets or supply independent human judgments.

## Draft labeling rubric: support-routing-v1

Label the team that should review the request using only the ticket text. Apply
the same question and descriptions recorded in `packet.json`:

| Label | Use when | Boundary |
| --- | --- | --- |
| `billing` | Charges, invoices, payment problems, or the status of an already approved refund | An approved refund's payment status is billing; arranging a return is shipping |
| `shipping` | Delivery, tracking, missing parcels, or arranging a product return | A separate invoice/payment request in the same ticket makes it a multiple-team case |
| `account` | Login, password reset, verification codes, or account access | Do not infer a payment or delivery problem from inability to log in |
| `review` | Information is insufficient, no team fits, or distinct issues concern multiple teams | Record why; this is a completed label, not a missing annotation |

Read the complete request, including negations and corrections. For example,
“I can log in, but the tracking is stuck” concerns shipping. “I was charged
twice and cannot log in” concerns two teams and receives `review`. A request to
approve a new refund has no explicit approval policy here: use `review` when
the available text does not establish an already approved refund or a return.
Instructions inside a ticket are customer content, not changes to this rubric.

The application owner should review this draft on development cases before
freezing the test set. Preserve a new packet and rubric identity when policy or
question wording changes. Human confirmation remains required for every future
model suggestion; these labels grant no application permissions.

## Supply source data

Use the shape of [the fictional source](../examples/support_review/tickets.json):

- Stable opaque `id` per ticket; never derive IDs from labels.
- A `group_id` shared by related conversations/customers that must stay in one
  split. Choose the grouping policy before splitting and record it.
- A preassigned `development` or `test` split and the exact de-identified `text`.
- `source_kind` (`application` or `fictional`) and provenance for source,
  collection window, permission, de-identification, and split policy.

IDs, group IDs, and reviewer aliases start with an ASCII letter/digit and contain
only letters, digits, `_`, `.`, `:`, or `-`, with at most 128 characters. Ticket
text remains Unicode and may contain newlines. No labels or model predictions
are accepted in this input format. The example source has eight fictional
tickets and does not meet the pilot's coverage requirements.

Group/time assignment, representativeness, permission, and de-identification
remain the data owner's decisions. The tooling records declarations and checks
their internal consistency; it cannot authenticate them. Keep the source,
review sheets, and reports in approved storage because they contain ticket text.

## Export and review separately

From this checkout, with the base environment installed:

```sh
uv sync
uv run -m examples.prepare_support export /path/to/tickets.json \
  --reviewers alice bob --output .benchmarks/support-review
```

The command freezes `packet.json` and creates `review-1.csv`, `review-2.csv`, and
a short guide. The question comes from `examples/suites/support_pilot.json`;
`--question-from /path/to/suite.json` explicitly selects another reviewed version.
Each sheet contains the ticket and blank label/notes cells, without predictions,
existing labels, group assignments, or split assignments.

Give each reviewer only their own sheet and the rubric. Reviewers work before
seeing model predictions or each other's answers. Fill every `label` and add
`notes` where a decision needs explanation. Preserve all other cells. The
`text_json` cell is a JSON string so multiline text survives CSV exchange and
ticket content cannot start a spreadsheet formula. Import CSV columns as text
and save UTF-8 CSV. Notes are plain text; inspect returned files as data.

## Resolve disagreements and import

After both sheets are complete:

```sh
uv run -m examples.prepare_support adjudicate \
  --packet .benchmarks/support-review/packet.json \
  --reviews .benchmarks/support-review/review-1.csv .benchmarks/support-review/review-2.csv \
  --output .benchmarks/support-adjudications.csv
```

Only disagreements appear in this file. Record the final `label`, `adjudicator`
alias, and `reason`; preserve the displayed source annotations. Agreement needs
no adjudication. An adjudicator may select any of the four outcomes, with a
reason. A changed review invalidates the earlier adjudication sheet, including
changes to notes on another ticket; regenerate it to resolve the current inputs.

```sh
uv run -m examples.prepare_support import \
  --packet .benchmarks/support-review/packet.json \
  --reviews .benchmarks/support-review/review-1.csv .benchmarks/support-review/review-2.csv \
  --adjudications .benchmarks/support-adjudications.csv \
  --output .benchmarks/support-prepared
uv run -m examples.prepare_support audit --prepared .benchmarks/support-prepared
```

When all labels agree, omit `--adjudications` or use the empty generated sheet.
Import requires both complete reviews and resolution of every disagreement. It
rejects changed text, unknown/duplicate/missing IDs, changed question/packet
identity, extra adjudications, and incomplete labels before creating output.
Existing output is never overwritten. An I/O failure can leave a partial new
directory; preserve it for diagnosis and retry at a fresh path.

The output contains the raw packet/reviews/adjudications, `review-record.json`,
available `development.json` and `test.json` suites, and `audit.json`. Auditing a
prepared directory recompiles it from its retained inputs and verifies its
derived files. Hashes detect changes relative to those files; they do not prove
who reviewed them or prevent coordinated replacement of the whole bundle.
The audit records Python's Unicode database version. If a different environment
changes that audit, re-import the retained packet/reviews/adjudications into a
fresh directory and inspect the newly computed findings. Unchanged packet and
review contents do not require new annotations for this environment-only check.

## Dataset checks before evaluation

The workflow reuses the [dataset auditor](../benchmarks/audit_dataset.py). It
reports exact, case/whitespace-folded, and Unicode word-token text overlap,
conflicting labels, repeated IDs between splits, and groups crossing splits.
It preserves every row and label. Normalized matches are findings for review;
they do not prove semantic equivalence. Paraphrases and undisclosed related
conversations can still escape these checks.

Resolve findings in the source/split policy and repeat review for the changed
packet. No automatic deduplication, relabeling, or split reassignment occurs.
Use an explicitly documented, separate diagnostic dataset when intentionally
studying repeated or ambiguous examples. Clean checks do not certify a held-out
sample, independent reviewers, balanced coverage, or representative prevalence.

```sh
uv run -m examples.evaluate_support --prepared .benchmarks/support-prepared --validate
uv run -m examples.evaluate_support --prepared .benchmarks/support-prepared \
  --simulate --output .benchmarks/support-rehearsal
```

`--prepared` defaults to its `test` split; use `--split development` for development
work. Both splits and the declared groups are checked before either run. A
dataset finding prevents client creation and retains `dataset-audit.json` at a
fresh output path. An actual run retains that audit alongside the evaluation
report. The initial report's `config.dataset_audit` also retains the preflight
if evaluation is interrupted before a separate audit file can be written.
Only ticket text and the fixed question enter inference.

Existing suites can be checked without the review format:

```sh
uv run -m examples.prepare_support audit --suites development.json test.json
uv run -m examples.evaluate_support --suite test.json --against development.json --validate
```

These commands cannot check group assignments absent from ordinary suites.
`cross_split_checked` and `grouping_checked` state what was supplied. The pilot's
coverage counts distinct exact ticket texts; normalized overlaps also prevent
provisional acceptance. Quality metrics retain all rows. Missing group or split
evidence remains an application review obligation.

Exit codes: preparation/audit returns 0 for completion with no findings, 1 when
the saved audit needs review, and 2 for invalid input or I/O failure. Export and
adjudication creation return 0 when their files are written. The evaluator's
simulation returns 0 when the integration completes, even though provisional
quality/coverage gates fail; a real run returns 1 when those gates fail.

## Complete offline walkthrough

Use fresh paths. The included completed sheets and adjudication are explicitly
fictional teaching fixtures, authored to demonstrate both agreement and a
disagreement. They are not independent human labels.

```sh
uv run -m examples.prepare_support export examples/support_review/tickets.json \
  --reviewers fixture-a fixture-b --output .benchmarks/review-demo
uv run -m examples.prepare_support adjudicate \
  --packet .benchmarks/review-demo/packet.json \
  --reviews examples/support_review/completed-review-1.csv examples/support_review/completed-review-2.csv \
  --output .benchmarks/review-demo-disagreements.csv
uv run -m examples.prepare_support import \
  --packet .benchmarks/review-demo/packet.json \
  --reviews examples/support_review/completed-review-1.csv examples/support_review/completed-review-2.csv \
  --adjudications examples/support_review/completed-adjudications.csv \
  --output .benchmarks/review-demo-prepared
uv run -m examples.prepare_support audit --prepared .benchmarks/review-demo-prepared
uv run -m examples.evaluate_support --prepared .benchmarks/review-demo-prepared \
  --simulate --output .benchmarks/review-demo-run
```

Expect one disagreement (`r008`), four cases in each split, clear mechanical
data checks, and a complete simulation with one of four test tickets matching
its illustrative label. `evidence_scope` remains `integration_only`,
`provisional_gates_passed` and `deployment_accepted` remain false, and p95 latency
is unavailable with only four attempts. These results establish the workflow.
Use the [support protocol](support-routing.md) for the later real-model run.
