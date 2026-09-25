# Released Laya and live Jev validation

Observed on 2026-09-25 against Kayak commit
`595fc68da006467c9d6c1041ac4963f0598825e6`.
The tested adapter paths passed without production changes. These bounded
observations support integration correctness, not general model quality or
probability calibration.

## Protocol and observations

Twelve requests were frozen before inference: eight fictional support cases
from `examples/suites/support.json`, plus Unicode question IDs with custom Noul
descriptions, a single candidate, equal candidate descriptions, and a longer
input exercising native provider input policy. Choice, Noul, and Score were
all exercised through Laya, synchronous Jev, and asynchronous Jev.

| Boundary | Observed result |
| --- | --- |
| Released Laya | 48 inference calls: each request called directly and through Kayak twice, reversing call order for the second repetition. All 24 direct/wrapped response pairs matched exactly, with no errors. Both repetitions were identical. |
| Live Jev through OpenRouter | 24 successful HTTP responses: eight synchronous pairs and four asynchronous pairs, alternating direct/wrapped order. No errors or retries. |
| Jev requests | All 12 paired request bodies matched exactly. State, question descriptions and order, and candidate order matched the frozen inputs. |
| Wrapped results | All 60 Laya and 30 Jev typed answers preserved provider values, supplied probabilities or their absence, and Score legends. Every wrapped raw response was preserved. |
| Evidence metadata | All wrapped results retained `calibration="unknown"`; model identity and usage remained provider reports. |
| Laya resource bounds | CPU FP32, two threads, seed 7. Sampled peak process RSS was 2.10 GiB; machine swap stayed at 1.22 GiB. No 600-second, 6-GiB RSS, or 4-GiB swap guard triggered. |
| Jev accounting | Responses reported 12,458 input tokens and **$0.000523236** total cost. The run finished after its planned 24 calls, below the $0.01 reported-cost stop. Billing was not independently audited. |

Three Jev pairs differed in their HTTP answer bodies despite equal outgoing
payloads: two Noul values varied by 0.01, and the equal-description case varied
its probabilities by 0.01 while retaining the selected ID. Kayak preserved
each received value. Cross-call answer identity is therefore not a supported
requirement for this live service. Equal candidate descriptions also do not
establish a numerical tie: providers can consume candidate IDs.

## Illustrative task results

The eight support Choice labels came from the existing fictional suite. Noul
and Score labels were constructed before inference for this experiment and
have not received independent review. Each provider contributes one wrapped
observation per case below; repetitions are not extra independent examples.

| Metric on eight cases | Laya | Jev |
| --- | ---: | ---: |
| Correct Choice routing | 8/8 | 8/8 |
| Noul mean squared error against binary labels | 0.02169607 | 0.0007125 |
| Score mean absolute error on the 0–2 rubric | 0.574925 | 0 |

These cases do not establish a provider ranking or representative accuracy.
For example, Laya returned an access-impact Score of **1.1331** for a duplicate
charge complaint whose constructed target was **0**. Its direct call returned
exactly the same value. Kayak preserves that output; it does not certify its
meaning. No Noul decision threshold was tuned, and calibration remains unknown.

## Versions and retained evidence

- Laya `0.3.20`, using `convaiinnovations/laya` revision
  `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. Five pinned files totaling
  846,195,574 bytes were downloaded and their sizes and hashes checked before
  loading. The weights' SHA-256 was
  `891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c`.
  Inference used local files with Hugging Face/Transformers offline mode,
  without compile or autocast.
- Local environment: macOS ARM64, Python `3.13.5`, PyTorch `2.14.0`,
  Transformers `4.57.6`, and Pydantic `2.13.5`.
- Jev used TypeSafe SDK `0.7.1` and httpx2 `2.13.1`. The requested model was
  `jev-1.13`; all responses reported `typesafe/jev-1.13-20260917`.
  Calls went exclusively to `https://openrouter.ai/api/v1/systemone` with
  redirects and retries disabled and a 30-second per-call timeout.
  Only synthetic inputs were submitted; no credential was saved in artifacts.

The full inputs, outputs, experiment scripts, and offline audit remain local in
the Git-ignored directory `validation/provider-adapters/live/`. This committed
report is a summary, not a self-contained reproducibility archive. The following
SHA-256 hashes identify the retained observations; they do not independently
establish the observations' correctness:

| Local artifact | SHA-256 |
| --- | --- |
| `cases.json` | `603c1ae1268ef96fd4ae86efb383a1b59d32dc699b521707fac22b8b7ecd979e` |
| `laya-report.json` | `507ec50b6e8647f91b1b2057552762917e8dcf975e69887f8893b6bd27c9d647` |
| `jev-report.json` | `bda225564b5c24c4bc681d1c034a1506011732da0fa0664391e770b022063978` |
| `resource-receipt.json` | `9c0d8b035c96d892ac62c315a794340d5e578051cb2d6b72946251a914451273` |

For a repeatable check included in the repository, see the
[model-free provider conformance command](../provider-adapters.md#evaluate-and-reproduce).
It validates different boundaries and does not reproduce these live observations.

This run did not evaluate Laya GPU/Router configurations, the direct TypeSafe
host, service load or cancellation under live traffic, CLM's full 8B model, or
general task accuracy. Existing controlled tests cover additional error and
cancellation paths; the live run adds evidence for the successful real-provider
paths described above.
