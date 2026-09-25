# Kayak input boundary investigation — 2026-09-25

The evaluator was first checked at `53e14b4` (rebased as `3a32418`). It separates
prediction collection from analysis and lets a classifier or metric change
without modifying the encoder.
The review found and fixed one export/import bug: a failed native run must retain
its comparison and accuracy-interval restrictions even when all first attempts
succeeded. Failed warmups and later repeats now retain that restriction through
both prediction objects and files.

Before integration with the incoming examples, the checkpoint passed independent
review, 535 tests without skips, Ruff, formatting, strict mypy, source/wheel builds,
Twine, and a built-wheel prediction and custom-metric smoke check. The full tests
needed local socket/MPS access;
the sandbox-only attempt was not a pass. Builds used modern setuptools with
`--no-isolation --skip-dependency-check` because the environment lacked the
separately declared `wheel` package. One existing Starlette deprecation warning
remains. No full encoder inference ran during this investigation.

After rebasing onto `c6ba62d` and linking the classifier benchmark in the incoming
task catalog, the combined tree passed 730 tests without skips, Ruff, formatting,
strict mypy, source/wheel builds, and Twine. This integration validation is
separate from the original offline audit receipt below.

## What is easy to change

| Requested change | Existing boundary | Obligation |
| --- | --- | --- |
| Another classifier | `PredictionSet` on the same `Suite` | Declare method and provenance; supply labels, rankings, or actual probabilities without inventing execution metadata. |
| Another metric | `Metric` over immutable `MetricInput` | Declare requirements, direction, version, and parameters; retain every selected example. |
| Another classification dataset | `Suite` and `Example` | Fix candidate order, labels, and provenance; data acquisition stays outside scoring. |
| Another report presentation | `render_benchmark` / `write_benchmark` | Render the same analysis result; do not recompute decision policy. |
| Question wording or candidate descriptions | `Suite.question` | Treat the change as a declared recipe experiment on development data. |
| Native preparation/token policy | `runtime/_preparation.py` | Preserve `clm-choice-v1` and overflow behavior; a changed production recipe needs an explicit contract and reference evidence. |

The first four changes do not require model weights or a new inference backend.
The [benchmark guide](../classification-benchmarks.md) includes working examples.
There is no present need for a plugin registry, alternate production encoder, or
automatic recipe-selection layer. Collection still uses the existing
`DecisionBackend`; external classifiers can instead import predictions directly.

## Offline observations

The audit used saved development errors first, then the pinned upstream
[`build_pairs` implementation](https://github.com/Contrastive-LM/CLM/blob/7956937c58ed5839c06ddc4dc6b6b61c3a3e4094/src/clm/schema.py)
and the cached tokenizer named in the [model manifest](../../kayak/models/clm-v0.1-8b.json).
Upstream specifies state followed by instructions and verbatim Choice candidate
text. Its training helper's chat handling applies to structured message states;
that does not establish a chat-template requirement for Kayak's string inputs.

| Check | Observed result | What it establishes |
| --- | --- | --- |
| Saved BANKING77 error rows against verified reports | 712 errors agree on ID, text, gold, prediction, and gold rank | The diagnosis follows retained predictions. |
| Current preparation against pinned upstream | All 770 development requests match exactly | No transcription mismatch at this text boundary for these requests. |
| Special-token setting | All 847 distinct prepared texts have identical token IDs with it on or off | This flag does not explain these failures. |
| Token accounting and limits | All 770 saved token counts match; sequences are 2–91 tokens, requests total 283–361 | Neither sequence truncation nor total-budget overflow occurred here. |
| Candidate reuse | All 770 complete saved results match the baseline | The saved reuse experiment did not change these predictions. |
| Current heads versus pinned upstream on saved synthetic embeddings | 13 questions; maximum same-CPU score error 0 | Projection/scoring agrees on these fixed representations. |
| Prior diagnostic artifact hashes | All 21 match the retained manifest | The historical evidence inspected here is unchanged. |

The saved synthetic diagnostics also show that MPS/BF16 SDPA, eager attention,
and streamed CPU/FP32 retained the same 13 original choices (6 correct). That
is historical evidence for that small corpus, not a fresh encoder run or a
BANKING77 backend-parity result.

The existing BANKING77 question-first experiment changes 58/770 correct to
77/770: 53 fixes, 34 regressions, 24 both correct, and 659 both wrong. This
supports input-order sensitivity. It does not establish that Kayak implemented
upstream incorrectly, and it does not justify changing the production recipe.
The [development results](clm-development-results.md) retain the probability
quality tradeoffs and the lexical baseline.

## Remaining uncertainty and the next discriminating check

**Hypothesis:** Transformers and the reference vLLM pooling path produce
materially different representations for identical token IDs and pinned weights,
enough to change candidate rankings after the same heads.

This is narrower than another quality run. The current environment has neither
CUDA nor vLLM. The pinned upstream package specifies
[`vllm>=0.6`](https://github.com/Contrastive-LM/CLM/blob/7956937c58ed5839c06ddc4dc6b6b61c3a3e4094/pyproject.toml),
not an exact historical runtime. The released head checkpoint's configuration
names `Qwen/Qwen3-8B` without an encoder revision. Kayak's manifest pins its own
reproducible snapshot, but it cannot recover those historical facts.

An offline probe input file is frozen under
`validation/evaluator-checkpoint/encoder-parity-inputs.json`. Selection is the
first suite-ordered error in each of the five most common directed confusions,
plus the first baseline-correct example:

`train:5131`, `train:5968`, `train:8744`, `train:687`, `train:5768`, `train:1847`.

The file records exact state/candidate text, token IDs, model manifest, gold
annotations, and saved choices. It requires only **83 distinct sequences**:
six states and the unchanged 77 candidates. SHA-256:
`27286d17b511237d96ad64b8bbc7436e00c02932db847dab66b129530cd302af`.

When a suitable reference runtime is available:

1. Pin its exact vLLM version, encoder/tokenizer revisions, dtype, and pooling
   settings. Feed the frozen token IDs directly, with no text or label tuning.
2. Compare last-token embeddings, normalization, and scores through the same
   heads. Record absolute errors, cosine similarity, all candidate ranks, and
   top-two/gold margins; small numerical drift alone is not a quality explanation.
3. Separately verify vLLM's text endpoint yields the same token IDs. This
   distinguishes token preparation from encoder execution.
4. If representations change choices, isolate that boundary before broadening.
   If the two paths retain the same choices and gold-label ranks, the backend
   explanation is weakened for these cases; report representation and score
   differences separately. A larger quality run still needs a specific
   remaining hypothesis.

These are inspected development examples, selected to diagnose errors. They are
not an acceptance set or an estimate of general accuracy. Historical training
revision equivalence remains unknown even if two current pinned backends agree.
The official test split remains reserved.

## Reproduce the offline audit

With the existing local artifacts and cached tokenizer/head checkpoint:

```sh
.venv/bin/python validation/evaluator-checkpoint/audit_recipe.py
```

The script loads only tokenizer artifacts, projection heads, and saved
embeddings. It produces `recipe-audit.json` and the probe input file. The audit
records source hashes, tokenizer hashes, artifact identity, counts, and limits.
These local diagnostic files are ignored, like the existing raw evaluation
artifacts. The native saved reports remain schema 1: structural checks and new
hashes do not retroactively authenticate their original execution.

The production recipe, model manifest, runtime, `/v1`, and evaluation labels
were unchanged. No model-quality improvement is claimed by this checkpoint.
