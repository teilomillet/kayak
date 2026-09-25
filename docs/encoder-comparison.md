# Compare saved encoder outputs

The [input-boundary investigation](records/input-recipe-investigation.md) leaves
a specific question open: do the Transformers and reference encoder paths
produce different representations for the same pinned weights and token IDs,
enough to change scores or rankings through the same heads?

The [probe preparer](../benchmarks/encoder_probe.py) freezes the experiment; the
[comparison tool](../benchmarks/compare_encoders.py) analyzes supplied saved
outputs. Both run with the base package. They do not load models, tokenize text,
download artifacts, contact a service, or execute a reference backend. Actual
token and encoder captures must come from a separately available runtime.

## Freeze a bounded development probe

Use an already reviewed development suite. The following tolerance values are
illustrative diagnostic limits, not established tolerances for Qwen3. Choose and
justify limits for the intended precision before observing the comparison.

```sh
mkdir -p .benchmarks/encoder-diagnostic
uv run -m benchmarks.encoder_probe prepare \
  --suite .benchmarks/support-prepared/development.json \
  --embedding-atol 0.00001 --score-atol 0.00001 \
  --output .benchmarks/encoder-diagnostic/texts.json
```

By default this selects the first six available cases in suite order. `--limit`
changes that bound; `--ids id-one id-two` selects explicit development cases in
their original order. The tool refuses a suite named `test`. Record any prior
inspection or selection in the suite's provenance; this remains a diagnostic
selection, not a held-out quality estimate.

The specification records the suite hash, selection, exact model manifest,
tolerances, preparation recipe, deduplicated prepared texts, candidate order,
and case-to-sequence mapping. Candidate IDs and gold annotations remain in the
analysis mapping and never become encoder text. The default manifest is the
pinned [CLM manifest](../kayak/models/clm-v0.1-8b.json); `--manifest` explicitly
selects another compatible bundle identity.

The initial status is `awaiting_saved_token_rows`. This is deliberate: text
alone does not establish what either tokenizer or encoder consumed. The earlier
83-sequence BANKING77 probe described in the historical record is not present
in this checkout and is not silently reconstructed. Preserve that original
artifact and identity if it becomes available, or record a new selection.

## Bind saved tokenizer output

A token collector must read the frozen specification and produce a JSON file
matching `TokenCapture` in the probe module. The supplied
[synthetic token file](../benchmarks/data/encoder-comparison/tokens.json) shows
the complete shape, with deliberately fictional identities and token IDs.

| Field | Required meaning |
| --- | --- |
| `text_spec_sha256` | `Probe.text_spec_sha256` from the exact text specification |
| `tokenizer_id`, `tokenizer_revision` | Same repository and exact revision as the pinned encoder snapshot |
| `tokenizer_files_sha256` | Hash the actual local tokenizer files, including `tokenizer.json` and `tokenizer_config.json` |
| `tokenizer_library`, `tokenizer_library_version` | Library and exact installed version used for this capture |
| `collector_sha256` | SHA-256 of the collector implementation used |
| Tokenization flags | Special tokens on, no chat template, no truncation |
| `sequences` | Every frozen sequence exactly once: its ID, UTF-8 text SHA-256, and actual unpadded integer token IDs |

On the capture machine, use the pinned tokenizer with the recorded settings.
Do not guess token IDs or substitute a different tokenizer. Check each prepared
sequence and each complete request against the recipe's token budgets.

```sh
uv run -m benchmarks.encoder_probe bind \
  --probe .benchmarks/encoder-diagnostic/texts.json \
  --tokens /path/to/captured-tokens.json \
  --output .benchmarks/encoder-diagnostic/probe.json
```

Binding verifies identities, text hashes, exact sequence coverage, nonnegative
token IDs, and per-sequence/request token limits. It creates a new file and
prints the bound `probe_sha256`. Changing the selected texts, model, candidate
order, labels, or tolerances invalidates the earlier capture binding. Hashes
verify supplied-file consistency; they cannot certify tokenizer execution.

## Capture the two encoder paths later

For each backend, retain its collector code and environment and produce a
`Snapshot` matching [the reference shape](../benchmarks/data/encoder-comparison/reference.json).
The example values are synthetic and must not be presented as model output.

1. Feed the bound unpadded token rows directly. Record the actual token IDs
   consumed, not a copy of the intended inputs without checking execution.
   Use the same pinned encoder weights and the specified last non-padding-token
   pooling. Preserve attention masks and batch/padding context.
2. For every sequence, export the raw pooled vector cast to FP32 and the vector
   after division by its L2 norm plus `1e-12`. Export all components, without
   rounding for display. The comparator checks normalization with its separately
   recorded `normalization_atol` (default `1e-6`).
3. Run both sets of normalized vectors through the same checked projection and
   scoring implementation and the same hash-verified head checkpoint. Retain
   the scoring implementation hash and actual score multiplier
   `exp(logit_scale).clamp(max=100)`. Export a complete candidate-score mapping
   for every case; do not export only the winner.
4. Record the bound probe hash, model manifest fingerprint, evidence kind,
   exact backend name/version, collector hash, device, dtype, attention
   implementation, batch size, and dependency versions. Keep runtime settings
   and logs with the snapshots. These metadata are declarations, not proof that
   the specified checkpoint executed.
5. Separately capture each text endpoint with `input_mode="text"`, recording its
   actual token rows. Comparing that capture to the token-input capture can
   expose tokenization changes before attributing them to encoder numerics.

The two snapshots may use different backends, devices, precisions, and batch
sizes; those differences are retained and must be considered when interpreting
the result. The probe/model/token identity, pooling, normalization recipe,
scoring implementation hash, and score multiplier must agree. Missing raw
vectors or tokens are missing evidence: do not manufacture them from normalized
vectors or expected inputs. This workflow defines a file exchange contract;
it does not include or validate backend-specific capture adapters.

## Compare without model execution

```sh
uv run -m benchmarks.compare_encoders \
  --probe .benchmarks/encoder-diagnostic/probe.json \
  --reference /path/to/reference.json --candidate /path/to/candidate.json \
  --output .benchmarks/encoder-diagnostic/comparison.json
```

The report retains:

- Declared model/runtime/scoring identities, tolerances, input modes, and hashes
  of the normalized snapshot JSON actually compared.
- Raw and normalized vector maximum absolute error, RMS error, cosine similarity,
  and each vector's norm, per sequence.
- Every candidate score and rank, chosen candidate, top-two margin, gold rank,
  and gold-minus-best-other margin, per case and backend. Ties use the original
  candidate order. Gold fields remain null if no gold annotation was supplied.
- Maximum embedding/score errors and all changed choices. A choice change stays
  visible even when numeric differences fall within the frozen tolerances.

`comparison_passed` means both raw/normalized embedding errors and score errors
meet the declared absolute limits and no top choice changes. It does not assert
that every lower rank is unchanged: inspect the full rankings and gold margins.
Different rankings can matter even when the same candidate remains first.

Exit 0 means the supplied comparison meets those conditions. Exit 1 retains a
valid comparison that exceeds a limit or changes a choice. Exit 2 rejects
invalid/incomparable inputs or an I/O failure before a report is written.
No output file is overwritten. For an I/O error during writing, inspect any
partial new file and retry at a fresh path.

A match weakens the backend-difference hypothesis for these cases and settings.
A mismatch identifies something to investigate; it does not by itself explain
task accuracy. Saved scores are compared, not recomputed through the heads here.
This tool cannot authenticate the collector, weights, or review provenance, and
does not establish calibration, historical training-revision equivalence,
hardware capacity, or support-task usefulness.

## Complete synthetic rehearsal

The fixtures use two-dimensional hand-authored vectors, fictional token IDs,
fake artifact identities, and identity-like scoring. No encoder ran. The changed
case rotates `[1, 0]` to `[0.6, 0.8]`, changes scores from `[1, 0]` to `[0.6, 0.8]`,
and moves the gold candidate from rank one to rank two.

```sh
mkdir -p .benchmarks/encoder-demo
uv run -m benchmarks.encoder_probe prepare \
  --suite benchmarks/data/encoder-comparison/suite.json \
  --manifest benchmarks/data/encoder-comparison/manifest.json --synthetic-fixture \
  --embedding-atol 0.001 --score-atol 0.001 \
  --output .benchmarks/encoder-demo/texts.json
uv run -m benchmarks.encoder_probe bind \
  --probe .benchmarks/encoder-demo/texts.json \
  --tokens benchmarks/data/encoder-comparison/tokens.json \
  --output .benchmarks/encoder-demo/probe.json
uv run -m benchmarks.compare_encoders \
  --probe .benchmarks/encoder-demo/probe.json \
  --reference benchmarks/data/encoder-comparison/reference.json \
  --candidate benchmarks/data/encoder-comparison/reference.json \
  --output .benchmarks/encoder-demo/equal.json
# Expected exit 1: a retained, valid comparison with one changed choice.
uv run -m benchmarks.compare_encoders \
  --probe .benchmarks/encoder-demo/probe.json \
  --reference benchmarks/data/encoder-comparison/reference.json \
  --candidate benchmarks/data/encoder-comparison/candidate.json \
  --output .benchmarks/encoder-demo/changed.json
```

The preparation/binding commands reproduce the checked-in text/probe files
byte-for-byte. The changed report should show maximum embedding and score
errors of `0.8`, raw-vector cosine similarity `0.6`, and `changed_choices` of
`["case-a"]`. Corruption tests additionally challenge wrong hashes/token rows,
missing and duplicate records, non-finite values, vector widths, normalization,
scoring identity/scale, stale tolerances, and preservation of existing files.
