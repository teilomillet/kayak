# Kayak evaluation report

**MOCK EVIDENCE — model quality and inference performance are not measured.**
The choices are controlled fixtures. Durations measure evaluator control flow.

**Run status:** `interrupted`.
This is not an accepted completed benchmark; retain its failures and missing cases.

The complete prediction file passed its recorded byte-length and SHA-256 checks.
These checks detect inconsistent or altered artifacts; they do not authenticate execution
or prevent someone from rewriting observations and their hashes together.

## Outcomes against the supplied labels

| Metric | Value |
| --- | ---: |
| Selected examples (quality denominator) | 4 |
| Attempted examples | 1 |
| Correct first attempts | 0 |
| Failed or missing first attempts | 3 |
| Accuracy | 0.000000 |
| Top-2 accuracy | 0.250000 |
| Macro F1 over all candidate labels | 0.000000 |
| Failed measured calls, including repeats | 0 |
| Failed warmups | 0 |
| Examples with changed choices across repeats | 0 |
| Maximum score change from first successful attempt | 0.000000 |

Quality uses the first measured attempt for each selected example. Failed and missing
first attempts stay in the denominator. Repeats do not add independent labeled examples.
Zero-support candidate labels remain in macro F1. Top-k ties preserve candidate order.

## Observed call durations

| Calls | Count | Mean seconds | Median seconds | p95 seconds | Min seconds | Max seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Successful measured calls | 1 | 0.000071 | 0.000071 | unavailable | 0.000071 | 0.000071 |
| All recorded measured attempts | 1 | 0.000071 | 0.000071 | unavailable | 0.000071 | 0.000071 |
| Warmups (excluded from measured calls) | 0 | unavailable | unavailable | unavailable | unavailable | unavailable |

Durations cover the decision call and any caller-supplied synchronization hook. Completion of backend work is not independently verified.
Model loading, evaluator input preparation, output verification, and report writing are excluded from these durations.
Mock durations are not inference timings.
p95 uses linear interpolation and is unavailable below 20 calls. These are descriptive
statistics of this run, not confidence intervals, independent repetitions, or concurrent throughput.
An interrupted in-flight call may have no saved result or duration. The tables cover retained attempts.

## Dataset and protocol

```json
{
  "name": "mock-arithmetic",
  "split": "synthetic",
  "examples": 4,
  "candidates": 2,
  "suite_sha256": "19041bdc3970762c954ee2dcf4b430d65c4abe4ce787f6000534f1336efc3d6d",
  "provenance": {
    "coverage": "synthetic_fixture",
    "purpose": "evaluator checks, no model quality"
  },
  "question": {
    "type": "choice",
    "instructions": "Select intent",
    "criteria": {
      "a": "Alpha",
      "b": "Beta"
    }
  },
  "protocol": {
    "warmups": 0,
    "repeats": 2,
    "seed": 42
  }
}
```

## Execution and artifact identity

```json
{
  "created_at": "2026-09-25T10:27:04.544895+00:00",
  "transport": "custom",
  "model": {
    "id": "mock/first-candidate",
    "revision": "fixture-v1",
    "fingerprint": "mock-no-weights",
    "encoder": "none",
    "encoder_revision": "none",
    "input_recipe": "clm-choice-v1",
    "device": "none",
    "dtype": "none"
  },
  "environment": {
    "python": "3.13.15 (main, Aug 25 2026, 14:01:07) [Clang 22.1.3 ]",
    "platform": "Linux-7.0.0-31-generic-x86_64-with-glibc2.43",
    "machine": "x86_64",
    "host": "vps-8fdf5ee2",
    "versions": {
      "kayak": "0.4.0",
      "torch": "2.14.0+cpu",
      "transformers": "4.57.6",
      "tokenizers": "0.22.2",
      "pydantic": "2.13.5",
      "httpx": "0.28.1"
    },
    "kayak_source_sha256": "c43f7efa436bbcea23c2292c430bcee078d3597214306f6a97505854f44026aa"
  },
  "config": {
    "evidence_kind": "mock",
    "scenario": "interrupted",
    "max_memory_gib": null
  },
  "schema_version": 2,
  "prediction_artifact": {
    "sha256": "9015e5617dc16199923c25f90f0fb9ad0bc7f6af47a49139668c7152f22951ac",
    "bytes": 433
  },
  "run_error": "MockInterruption"
}
```

## Memory observations

```json
{
  "process_peak_rss_bytes": 45039616
}
```

RSS and CUDA peaks cover the process lifetime. MPS values are sampled snapshots,
not guaranteed peaks. Counters overlap and must not be added. For HTTP/custom
backends, caller memory does not establish model-server memory.

## Interpretation and reproduction

Keep `report.json` and `predictions.jsonl` together. The JSON artifacts retain the
selected texts, supplied labels, every retained completed attempt, per-class metrics, and confusion matrix.
Regenerate this document with `kayak eval report PATH_TO_RUN` using the recorded source.
Source and runtime metadata are snapshots before execution, not continuous monitoring of changes during a run.

A subset score applies only to that subset. A full-split score describes this fixed
dataset under this question and candidate recipe. It does not establish production
generalization, training-data independence, calibrated confidence, statistical significance,
or superiority over another system. Record dataset overlaps and any prior test exposure
in the study protocol; this renderer cannot determine them from predictions.
