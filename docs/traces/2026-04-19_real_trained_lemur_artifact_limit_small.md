# Real Trained LEMUR Artifact On LIMIT-small

## Goal

Run the new trained-checkpoint and latent-proxy artifact benchmark path on:

- one **real task JSON** already present in the workspace
- one **real trained upstream LEMUR checkpoint pair**

The concrete question was:

- does the exported latent-proxy artifact preserve the trained checkpoint's
  behavior on a non-synthetic task?
- if it does, what quality/latency tradeoff does that trained checkpoint show
  on the chosen real slice?

## Why this task

I used:

- task: `.cache/kayak/limit_small_real_subset/python_task.json`

Reason:

- it is a real task slice already present locally
- it is the smallest available real slice in the workspace (`46` docs, `32`
  queries, `dim=128`)
- that makes it the lowest-risk way to verify the whole training -> checkpoint
  -> export -> artifact benchmark loop end to end before trying larger slices

## Upstream source

I cloned the official LEMUR repository:

- `https://github.com/ejaasaari/lemur`

This was the primary source for the checkpoint format and training loop.

Verified from the upstream code:

- `lemur/lemur.py` writes `mlp.pt` as `{state_dict, config, output_mean, output_std}`
- `lemur/lemur.py` writes `w.pt` as `{W: tensor}`
- `lemur/model.py` uses the same MLP / ELM block structure that the Kayak
  loader expects

That means Kayak's trained-checkpoint loader is not relying on a guessed format
 here; it matches the upstream implementation directly.

## Training setup

I trained an upstream LEMUR checkpoint on the real task corpus using:

- corpus token vectors as `train`
- task query vectors as `test` for validation
- `epochs = 10`
- device `cpu`

I kept the upstream defaults otherwise. The reason was to stay close to the
official upstream training path and avoid introducing extra tuning assumptions.

Output root:

- `.cache/kayak/lemur_upstream_limit_small_real/checkpoint`

Saved artifacts:

- `.cache/kayak/lemur_upstream_limit_small_real/checkpoint/mlp.pt`
- `.cache/kayak/lemur_upstream_limit_small_real/checkpoint/w.pt`

Measured training result:

- training time: `2.1363914169999703 s`
- latent dim: `2048`

## Benchmark outputs

Generated files:

- `.cache/kayak/lemur_upstream_limit_small_real/training_summary.json`
- `.cache/kayak/lemur_upstream_limit_small_real/trained_benchmark.json`
- `.cache/kayak/lemur_upstream_limit_small_real/artifact_equivalence.json`
- `.cache/kayak/lemur_upstream_limit_small_real/artifact_benchmark.json`
- `.cache/kayak/lemur_upstream_limit_small_real/exact_benchmark.json`
- `.cache/kayak/lemur_upstream_limit_small_real/artifact_benchmark_k20.json`
- `.cache/kayak/lemur_upstream_limit_small_real/artifact_benchmark_k30.json`
- `.cache/kayak/lemur_upstream_limit_small_real/artifact_benchmark_k46.json`

## Main results

### 1. Artifact fidelity against the trained checkpoint

From `artifact_equivalence.json`:

- `max_feature_abs_error = 0.0`
- `mean_feature_abs_error = 0.0`
- `max_score_abs_error = 1.0728836059570312e-06`
- `mean_score_abs_error = 2.6448548684498974e-07`
- `exact_topk_match_rate = 1.0` at `topk = 10`
- `mean_topk_overlap = 1.0`

Conclusion:

- on this real task, the exported latent-proxy artifact reproduces the trained
  checkpoint's stage-1 behavior to numerical precision

### 2. Trained checkpoint vs artifact benchmark at `candidate_k = 10`

From `trained_benchmark.json`:

- `nDCG@10 = 0.8095367981913646`
- `mean_search_seconds = 0.0006158332537517453`

From `artifact_benchmark.json`:

- `nDCG@10 = 0.8095367981913646`
- `mean_search_seconds = 0.0006201090250005592`

Measured delta:

- primary metric delta: `0.0`
- mean search delta: `4.275771248813927e-06 s`

Conclusion:

- the artifact-backed benchmark path matches the trained-checkpoint benchmark
  result on the real task

### 3. Comparison to exact Kayak on the same task

From `exact_benchmark.json`:

- `nDCG@10 = 0.9674217039434331`
- `mean_search_seconds = 0.0007026150500001194`

At `candidate_k = 10` the artifact/trained path reaches:

- quality ratio vs exact: `0.8367982596333188`
- time ratio vs exact: `0.8825729323623994`

Interpretation:

- on this small corpus, the trained LEMUR proxy is only about `12%` faster than
  exact search, while losing about `0.158` absolute `nDCG@10`
- so for this slice, `candidate_k = 10` is not a good operating point

## Candidate-window sweep

Because the artifact reproduced checkpoint scores within `1.1e-6`, I used the
artifact path as a faithful stand-in for sweeping larger candidate windows.

Results:

| candidate_k | nDCG@10 | quality ratio vs exact | mean search s | time ratio vs exact |
| --- | ---: | ---: | ---: | ---: |
| 10 | 0.8095367981913646 | 0.8367982596333188 | 0.0006201090250005592 | 0.8825729323623994 |
| 20 | 0.8827976523608774 | 0.912526201099677 | 0.0010250218162504154 | 1.4588668663590985 |
| 30 | 0.9553325537173536 | 0.9875037430142395 | 0.0013175345387472959 | 1.8751869017708518 |
| 46 | 0.9674217039434331 | 1.0 | 0.0017341610875001834 | 2.468152493317484 |

Conclusion:

- the quality gap is mostly a shortlist-budget issue, not an artifact export
  issue
- however, on this tiny corpus, increasing `candidate_k` quickly erases the
  latency advantage and then becomes slower than exact search
- that means `LIMIT-small` is a valid real-path verification slice, but not a
  persuasive speed showcase for trained LEMUR

## What this verifies

Verified:

- Kayak can load a **real upstream-trained LEMUR checkpoint**
- Kayak can export that checkpoint into the native latent-proxy artifact format
- the exported artifact reproduces checkpoint scores on a **real task**
- the artifact-backed benchmark path produces the same judged result as the
  trained-checkpoint path on that real task

Not verified:

- that this operating point is competitive on larger corpora
- that trained LEMUR is a better whole-search choice than exact Kayak on this
  task family

## Next justified step

The strongest next step is not more work on artifact fidelity. That question is
already settled on this slice.

The next justified step is one of:

1. repeat the same real trained-checkpoint loop on a larger local task slice
   such as `legal_rag_bench_real_subset` or `bright_stackoverflow_real_subset`
   where exact search has a better chance to become materially expensive
2. add a proper trained-checkpoint candidate-window sweep contract so the
   quality/latency frontier can be compared mechanically across tasks
