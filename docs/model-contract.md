# Model contract: CLM Qwen3

The default model is `Contrastive-LM/CLM-v0.1-8B`. Its [Kayak manifest](../kayak/models/clm-v0.1-8b.json) is the
canonical source for the pinned head/encoder revisions, head integrity hash,
representation width, context limit, and recipe identifier.

The upstream reference is
[Contrastive-LM/CLM at 7956937](https://github.com/Contrastive-LM/CLM/tree/7956937c58ed5839c06ddc4dc6b6b61c3a3e4094).
The [model card](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B/blob/87655cb835bd76fd66c2da78e1e3709f7fa11a94/README.md)
identifies the heads as dependent on Qwen3-8B with last-token pooling. Kayak
implements the documented Choice path with a direct Transformers encoder.
Upstream's service uses vLLM; numerical equivalence of the full encoders remains
a separate validation requirement.

`rank()` compiles to the same Choice request and only orders its returned
scores. Async HTTP uses the same runtime. Neither changes this numerical recipe.
The [Noul/Score investigation](records/sdk-validation.md#noulscore-investigation-and-reproducible-probes)
records upstream's additional transformations and reproducible probes; native
typed-judgment quality remains unverified on the pinned checkpoint.
The public Python [`judge()` adapters](typed-judgments.md) now expose those
transformations through the same Choice computation. Their supported contract
is recipe conformance and typed decoding, not established task accuracy or calibration.

The upstream card names Qwen3-8B without identifying the encoder's historical
training revision. Kayak pins the available encoder snapshot inspected on
2026-09-24; matching that snapshot to the one used during training remains an
assumption to check against the independent reference.

The [offline input-boundary investigation](records/input-recipe-investigation.md) checks
the saved development inputs and identifies the bounded encoder-parity probe
needed to narrow that uncertainty without another full evaluation run.
The [saved-output comparison workflow](encoder-comparison.md) now provides a
frozen probe format and offline analysis, with synthetic integrity tests.
Actual encoder captures and equivalence evidence remain pending.

## Data path

1. Validate string state and named Choice questions. Snapshot input mappings.
2. For each question, concatenate stripped state and stripped instructions,
   separated by two newlines. Candidate descriptions are passed verbatim.
   Question IDs and candidate IDs are not appended to model input.
3. Tokenize plain text with the pinned Qwen tokenizer, `add_special_tokens=True`,
   no chat template, no truncation. Reject empty or excessive token sequences
   before any encoder forward pass. Batch with right padding and attention masks.
4. Run the causal Qwen3 base model, with no generation and no KV cache. Select
   the final non-padding hidden vector for each sequence. Cast it to FP32 and
   divide by its L2 norm plus `1e-12`, matching upstream's embedder normalization.
5. Apply the state or action projection in FP32, then L2-normalize with
   PyTorch's `normalize` (`eps=1e-12`). The published checkpoint uses
   `4096 → 1536 → 1536 → 512`, GELU, LayerNorm on its hidden layer, and no residual.
6. Score each candidate using its dot product with the projected state,
   multiplied by `exp(logit_scale).clamp(max=100)`. Preserve the reference
   matrix-vector operation. Return scores and their softmax, selecting the first
   maximum in input order. No additional temperature or calibrator is applied.

The context cap is 2,048 tokens per prepared sequence for this initial recipe.
Upstream's service can request truncation; Kayak intentionally rejects overflow
instead. This is a native Kayak contract, not a claim of identical wire behavior.

At batch size 1, a resident model retains one ordered block of normalized
candidate encoder vectors, keyed by their exact token rows. Every call still
tokenizes and validates all inputs, counts all input tokens, encodes each question
state, and applies both projection heads and scoring. Only singleton encoder rows
are reused, preserving their padding and batch context. A cache miss replaces the
previous block; new vectors are retained only after a valid complete result, so a
failed computation can be retried. The existing model lock serializes cache
access, and `close()` releases it with the model. Larger batch sizes do not reuse
candidate vectors. This changes execution cost without changing the input recipe.

## Artifact boundary

The manifest is strict and versioned (`format_version=1`, `family=clm-qwen3`,
`input_recipe=clm-choice-v1`). A local directory supplies `kayak.json` and the
named checkpoint; its encoder remains a pinned Hub artifact that can be cached
offline. The checkpoint filename cannot traverse directories. Its SHA-256 is
checked before parsing; `torch.load(weights_only=True)` is used. There is no
fallback to unrestricted pickle loading or remote Python code.

The checkpoint contains `state_head`, `action_head`, scalar `logit_scale`, and
`cfg` with projection architecture settings. Optional upstream metadata is
allowed in the checkpoint, but missing keys, incompatible dimensions, unexpected
state-dict keys, non-finite tensors, and a conflicting encoder name are errors.
The encoder configuration must be Qwen3 with the declared hidden width and
sufficient context. Encoder files use the pinned Hugging Face revision and
Safetensors. A local filesystem path shadowing the encoder Hub ID is rejected.

The result fingerprint hashes the normalized manifest, which includes the
checkpoint integrity hash and encoder revision. Device and precision are
reported separately because they can affect numerical results. A loaded model
never hot-reloads or silently switches revisions.

For a later checkpoint with this same family and recipe, create a local manifest
using its actual artifact identity and encoder revision. A checkpoint whose
training recipe differs needs an explicit new recipe/implementation and
reference fixtures; changing the filename is insufficient.
