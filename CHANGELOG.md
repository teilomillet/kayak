# Changelog

## 0.5.0 — Unreleased

- A direct first-run path for classifying JSONL files with local Laya and editable
  categories, including input/setup fixes, loading and record progress, a bounded
  result preview, and preservation of completed predictions on failure.
- Offline support-ticket review sheets, explicit adjudication, verified suite
  import, and duplicate/group/split checks before evaluation. Coverage counts
  distinct exact texts and retains all observations in quality metrics.
- Frozen encoder diagnostic specifications, saved-token binding, and offline
  embedding/score/ranking comparison with explicit synthetic rehearsals.
- Provider evaluations can declare their exact method; live example commands
  record SDK/model settings for recipe comparisons. Independent scikit-learn and
  SciPy fixtures check classification, probability, ranking, and paired statistics.
- Start `main` from a single initial commit while preserving the `/v1` API and
  pinned model recipe. Compatibility checks install the hash-pinned published
  0.4.0 wheel and no longer require earlier Git history.
- Support-routing pilot with fictional starter cases, provisional quality and
  latency targets, lexical comparison, and explicit human-review outcomes.
- Separate current guides from dated validation records; verify documentation
  file/heading links and record installation, package, and model-evidence limits.

- Runnable Laya/Jev Choice evaluation against a word-overlap baseline, with
  offline fixtures, retained provider responses, and partial results on failure.
  Provider quickstarts are prominent in the README and documentation catalog.

## 0.4.0 — 2026-09-25

The first CLM-focused Kayak release, replacing the earlier project direction.
This is not a drop-in upgrade of the earlier API. The release includes the
tested SDK, service, and evaluation contracts; full-model validation remains
limited to the configurations and observations recorded in the release guide.

- A typed `Choice` API shared by local inference and the synchronous HTTP client.
- `Model.rank`, `Client.rank`, and `kayak rank` for supplied candidate sets, with
  stable ties, full-set score/share preservation, and offline ranking validation.
- `AsyncClient` with awaited decisions, ranking, and model inspection, using the
  frozen `/v1` bodies and the synchronous client's validation/error policy.
- Typed `Noul` and `Score` questions through `Model.judge`, `Client.judge`, and
  `AsyncClient.judge`, using the pinned CLM recipes over unchanged Choice requests.
  Results retain full distributions and model identity; conformance probes do
  not establish task quality or calibration.
- Optional Python adapters for caller-owned Laya and synchronous/asynchronous Jev
  clients. They preserve native question types and provider-reported evidence,
  without changing CLM results, `/v1`, or the base dependencies.
- Public request validation, remote model inspection, and `validate`, `info`,
  and `decide` CLI commands with file/stdin input and JSON output.
- Pinned CLM-v0.1-8B artifacts, compatible local bundles, and offline cache use.
- Automatic CUDA/MPS/CPU selection, explicit precision, and owned model lifetime.
- A local HTTP runtime with bounded admission, bearer authentication, and
  explicit timeout and shutdown behavior.
- Response request IDs, matching SDK error context, and optional JSON diagnostics
  with bounded CLI buffering; enabled through `serve --diagnostics json`.
- Executable examples, API and service references, and a portable hardware
  validation procedure with timing and memory reports.
- Labeled JSONL evaluation through the HTTP client, with per-case outcomes,
  timings, model identity, and an accuracy summary that includes failed calls.
- A Jev/Laya migration guide and executable Choice request migration example.
- Portable released-head reference checks and memory reporting for indexed MPS devices.
- Plain SDK/CLI input errors for API keys that cannot be encoded in HTTP headers.
- A documented `/v1` compatibility contract and a release-gating CI check of
  independently installed baseline/current clients and servers, with failure logs.
- Manual inference evals with labeled cases, latency and memory history, strict
  baseline comparisons, and bounded batch-size searches with holdout confirmation.
- Packaged `kayak.eval` API and CLI with pinned BANKING77 data, separate development
  and test splits, local/HTTP runs, quality and latency reports, and run comparisons.
- Verified Markdown benchmark reports, exact prediction-byte hashes, explicit
  local timing provenance, and mock-only adversarial regression checks; model
  quality and accelerator performance still require the documented real runs.
- Public case callbacks for Choice and external rankers, partial-judgment ranking
  metrics, and recorded RAG stage assessments with bound answer reviews and
  diagnostic replay provenance. Model-free examples show MaxSim and evidence loss;
  applications retain execution, model/index ownership, and independent judging.
- Configurable sync/async RAG experiments with ordinary callable adapters, bounded
  workers, retained repetitions and failures, optional progress hooks, custom
  independent reviews, coverage-aware gates, and verified JSON/Markdown reports.
  Versioned JSON exchange and `kayak eval rag` support external applications and
  reviewers. Reranker prefixes preserve unknown tails; model-free callable,
  async, and HTTP examples exercise settings and evaluation boundaries.
- `RAGTrace.evaluate(...)` checks reference answers and required sources without
  executing a pipeline or judge. A standalone example owns lexical retrieval and
  Ollama generation, then records the actual context and answer for evaluation.
- A 48-line retrieval-augmented decision example feeds source context to native
  Choice/Noul/Score questions and returns their typed answers and model identity.
- Ruff, strict typing, pytest/Hypothesis, code benchmarks, and clean-wheel
  checks that gate the publishing workflow.
- Documentation organized by integration, operation, and evaluation, with current
  architecture and contribution guides, package project links, and code checks
  on the release branch before promotion to `main`.
- Measured reductions in response validation and SDK serialization overhead,
  with unchanged contracts and benchmark workers that verify their imported source.

See [release readiness](docs/release.md) for completed checks and remaining
full-model evidence. No task accuracy, calibrated confidence, or accelerator
parity claim is made by this changelog.
