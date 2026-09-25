# Offline evaluation handoff — 2026-09-26

The support review workflow and saved-output encoder diagnostic are implemented
and verified without model execution, model/tokenizer downloads, paid services,
or external compute. Local file operations, scalar calculations, packaging, and
contract tests supplied the evidence below. The work builds on the single
0.5.0 initial commit `b68efc0`; the raw receipt identifies the tested working files.

## Delivered workflow

| Requested outcome | Result and entry point |
| --- | --- |
| Ticket review preparation | [Draft rubric and review commands](../support-review.md): unlabeled source format, two prediction-free sheets, explicit disagreement resolution, and validated suite import |
| Dataset checks before evaluation | Existing text auditor reused for duplicates, conflicting labels, and split overlap; ticket group and ID overlap checks added; `--prepared` rechecks retained review inputs before client creation |
| Reproducible encoder diagnostic | [Probe and comparison commands](../encoder-comparison.md): pinned text specification, supplied-token binding, declared identity checks, saved raw/normalized vectors, scores, rankings, and margins |
| Verified handoff | [Support walkthrough](../support-review.md#complete-offline-walkthrough) and [synthetic encoder rehearsal](../encoder-comparison.md#complete-synthetic-rehearsal), exercised from the built source archive using the installed base-only wheel |

The source archive includes the review CSV fixtures and all diagnostic JSON
fixtures. The public `kayak` implementation, model manifest, dependencies, and
`/v1` contract are unchanged. The support example now counts distinct exact texts
for coverage, retains every row in quality metrics, and prevents unresolved
dataset findings from reaching inference. Its initial report retains the
preflight audit even if execution is interrupted.

## Verification

| Observation | Evidence and limit |
| --- | --- |
| Ruff, formatting, strict mypy | Passed across first-party source; 154 formatted files and 149 checked source files |
| Non-inference pytest/Hypothesis suite | 970 passed, 39 inference tests deliberately deselected; one existing Starlette deprecation warning; Linux/Python 3.13.15 |
| Distributions | Wheel and source archive built with offline dependency resolution; strict Twine and isolated wheel checks passed |
| Packaged examples | Fifteen base-only command checks passed, including their expected nonzero outcomes; model, tokenizer, vLLM, and NumPy packages absent |
| Source archive | Included source files compared with the checkout; documentation links and source structure checked from the extracted archive |
| Support fixture | One explicit disagreement, four development and four test cases, 1/4 simulated matches; all deployment/quality acceptance remains false |
| Encoder fixture | Equal snapshots passed; a hand-authored vector rotation changed one choice, with maximum vector/score error 0.8 and cosine similarity 0.6; a wrong token row was rejected |

The new tests challenge incomplete or changed reviews, stale adjudications,
unknown/duplicate IDs, corrupted derived files, split/group leakage, repeated
texts inflating coverage, timeouts, interrupted runs, and preservation of prior
output. Encoder challenges include wrong identities/token rows, missing or
duplicate records, nonfinite values, wrong dimensions/normalization, scoring
identity/scale changes, ties, numeric boundaries, and choice changes below the
declared numeric tolerance. A fresh-process check rejects optional model-runtime
imports and network attempts while exercising the offline tools.

A boundary check found that JSON quoting a valid maximum-length ticket could
exceed Python's default CSV field limit. The sequential reader now permits a
field up to the size of its already-read file and restores the prior parser
limit afterward. The request's existing text/token validation remains intact.

Local raw logs, command arguments, JUnit results, distributions, and reports are
retained in `validation/offline-workflow-20260926/`, which is ignored by Git. The
receipt records hashes for the working source and packaged artifacts. These
hashes identify files; they do not authenticate human review or model execution.

Before pushing, this work was integrated with the newer `main` commit `0f4cd75`,
preserving its independent metric fixtures and provider recipes. The combined
checkout passed 1,053 non-inference tests, with 39 inference tests deselected,
plus Ruff, formatting (156 files), and strict mypy (151 source files). The new
workflow code and fixtures remained identical to the packaged checks above.
The combined test log and JUnit results are retained as `integrated-pytest.log`
and `integrated-pytest.xml` in the same local evidence directory.

## What remains external

1. Supply permissioned application tickets and actual independent reviews using
   the [review workflow](../support-review.md). The included completed sheets
   are fictional teaching data. They cannot establish routing quality.
2. On a suitable machine, capture genuine tokenizer rows and both encoder paths
   under the [snapshot contract](../encoder-comparison.md#capture-the-two-encoder-paths-later).
   Backend-specific capture adapters and their execution are outside this offline
   change. The historical 83-sequence probe is absent here; it was not fabricated.
3. Run the [support acceptance exercise](../support-routing.md) against a real
   service after freezing the data and targets. Review quality, latency, hardware,
   capacity, and recovery evidence before accepting application traffic.

There is no new model-quality, encoder-equivalence, calibration, or hardware
performance claim. Real encoder execution and representative application
measurement remain unperformed, as required by this work's resource boundary.
