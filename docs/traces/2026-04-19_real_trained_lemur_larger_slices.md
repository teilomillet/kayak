# Real Trained LEMUR On Larger Local Slices

## Goal

After the first real run on `LIMIT-small`, the next justified question was:

- does the same real upstream-trained LEMUR -> Kayak artifact loop hold on
  larger local task slices where exact search is materially more expensive?
- if it does, is the quality/latency frontier persuasive there?

This trace records the larger-slice follow-up.

## Candidate task screening

I first measured exact Kayak search on the larger real slices already present in
the workspace:

| task | docs | queries | total doc vectors | exact mean search s |
| --- | ---: | ---: | ---: | ---: |
| `bright_stackoverflow_real_subset` | 150 | 8 | 23035 | `0.0019531137449996549` |
| `legal_rag_bench_real_subset` | 136 | 8 | 21336 | `0.0023289854250003825` |
| `r2med_biology_real_subset` | 178 | 8 | 23068 | `0.0025931843699999037` |
| `lemb_narrativeqa_real_subset` | 355 | 8 | 63900 | `0.005918869165001297` |

Reason for task selection:

- `LEMB` was the strongest next target by measured exact cost
- `R2MED` was the next strongest slice while also keeping a different task
  profile from `LEMB`

I therefore ran the full real checkpoint loop on:

- `.cache/kayak/lemb_narrativeqa_real_subset/python_task.json`
- `.cache/kayak/r2med_biology_real_subset/python_task.json`

## Upstream source and training setup

Primary source:

- upstream LEMUR repo: `https://github.com/ejaasaari/lemur`

Verified from upstream code before training:

- `lemur/lemur.py` saves `mlp.pt` as `{state_dict, config, output_mean, output_std}`
- `lemur/lemur.py` saves `w.pt` as `{W: tensor}`
- `lemur/model.py` uses the same `MLP` / `ELM` block structure that Kayak's
  trained loader already expects

Training setup used for both slices:

- device: `cpu`
- epochs: `50`
- corpus token vectors as `train`
- task query vectors as `test` for upstream validation
- otherwise keep the upstream `Lemur.fit()` defaults

Reason:

- `50` epochs materially reduce the "undertrained checkpoint" confound while
  staying cheap enough to run repeatedly on these local slices

## Output roots

LEMB:

- `.cache/kayak/lemur_upstream_lemb_narrativeqa_real/`

R2MED:

- `.cache/kayak/lemur_upstream_r2med_biology_real/`

Each root contains:

- `training_summary.json`
- `exact_benchmark.json`
- `trained_benchmark.json`
- `artifact_equivalence.json`
- `artifact_benchmark_k*.json`
- `bundle.json`

## Training results

LEMB:

- training time: `38.42690533299992 s`
- latent dim: `2048`

R2MED:

- training time: `11.933644125 s`
- latent dim: `2048`

## Artifact fidelity on larger slices

LEMB:

- `max_score_abs_error = 4.172325134277344e-07`
- `exact_topk_match_rate = 1.0` at `topk = 10`
- `mean_topk_overlap = 1.0`

R2MED:

- `max_score_abs_error = 3.5762786865234375e-07`
- `exact_topk_match_rate = 1.0` at `topk = 10`
- `mean_topk_overlap = 1.0`

Conclusion:

- the exported latent-proxy artifact reproduces the real upstream checkpoint on
  both larger slices too

That means the artifact contract is no longer only validated on the tiny
`LIMIT-small` smoke path. It now matches real trained checkpoints on larger
local tasks as well.

## Main benchmark results

### LEMB

25-iteration benchmark outputs:

- exact: `nDCG@10 = 0.375`, `mean_search_seconds = 0.005206917709999743`
- trained checkpoint, `candidate_k = 10`:
  `nDCG@10 = 0.375`, `mean_search_seconds = 0.0012355562449977242`
- artifact, `candidate_k = 10`:
  `nDCG@10 = 0.375`, `mean_search_seconds = 0.0020628773000078127`

Observed quality behavior:

- `candidate_k = 10, 20, 40, 80, 355` all reached the same `nDCG@10 = 0.375`

Interpretation:

- on this slice, candidate budget is not the limiting factor
- the trained stage-1 proxy already preserves the exact final metric at
  `candidate_k = 10`

Because the trained vs artifact timing differed more than expected, I reran the
`candidate_k = 10` comparison at a heavier budget (`warmup = 10`,
`measurement = 100`):

- exact mean search: `0.005026021033749543`
- trained mean search: `0.0007579301300000906`
- artifact mean search: `0.0020693198087531075`

Derived ratios:

- trained / exact time ratio: `0.15080122524569994`
- artifact / exact time ratio: `0.4117212790908956`
- trained / exact quality ratio: `1.0`

Conclusion:

- the **real trained checkpoint** on `LEMB` is compelling:
  exact quality at about `15%` of exact-search time
- however, the current **Python artifact evaluator is slower than the trained
  torch path** on this slice
- therefore the artifact sweep on `LEMB` is trustworthy for quality, but not as
  an honest proxy for the trained path's latency

### R2MED

25-iteration benchmark outputs:

- exact: `nDCG@10 = 0.8646255554317486`,
  `mean_search_seconds = 0.0029160614699992493`
- trained checkpoint, `candidate_k = 10`:
  `nDCG@10 = 0.8642963760248189`,
  `mean_search_seconds = 0.0013825458399963964`
- artifact, `candidate_k = 10`:
  `nDCG@10 = 0.8642963760248189`,
  `mean_search_seconds = 0.0005846815250049531`

Artifact candidate sweep:

| candidate_k | nDCG@10 | quality ratio vs exact | mean search s | time ratio vs exact |
| --- | ---: | ---: | ---: | ---: |
| 10 | 0.8642963760248189 | 0.9996192809651974 | 0.0005846815250049531 | 0.20050384088957618 |
| 20 | 0.8646255554317486 | 1.0 | 0.0008323960300037924 | 0.2854521547531049 |
| 40 | 0.8646255554317486 | 1.0 | 0.0013030984899995701 | 0.446869348745212 |
| 80 | 0.8646255554317486 | 1.0 | 0.0022209418800048296 | 0.7616238213268534 |
| 178 | 0.8646255554317486 | 1.0 | 0.004057507359997885 | 1.391434097580573 |

I again reran the `candidate_k = 10` comparison at a heavier budget
(`warmup = 10`, `measurement = 100`) to check the trained vs artifact timing:

- exact mean search: `0.0025413080212491933`
- trained mean search: `0.0006206305199992812`
- artifact mean search: `0.0006350479150023603`

Derived ratios:

- trained / exact time ratio: `0.24421696024640374`
- artifact / exact time ratio: `0.24989017847989914`
- trained / exact quality ratio: `0.9996192809651974`

Conclusion:

- on `R2MED`, the artifact timing is effectively aligned with the trained
  checkpoint timing at `candidate_k = 10`
- `candidate_k = 20` already restores exact quality while remaining at about
  `28.5%` of exact-search time
- this is the clearest end-to-end quality/latency story from the larger-slice
  runs

## What changed relative to LIMIT-small

`LIMIT-small` showed:

- correct artifact export
- but weak end-to-end operating points on that tiny corpus

The larger-slice runs show a more useful split:

1. Artifact fidelity is robust.

   It held on `LIMIT-small`, `LEMB`, and `R2MED`, always with negligible score
   error and exact top-k agreement.

2. Whether trained LEMUR is compelling is task-dependent.

   - `LEMB`: yes, strongly, for the **trained checkpoint**
   - `R2MED`: yes, with a clean frontier and near-exact quality at much lower
     cost
   - `LIMIT-small`: no, not at the tested operating point

3. The current Python artifact evaluator is not a universally faithful timing
   stand-in for the trained checkpoint.

   - on `R2MED`, it is close enough to use for timing
   - on `LEMB`, it materially overstates stage-1 time

## Strongest measured conclusion

The real story is now:

- Kayak's latent-proxy artifact contract is **semantically correct** against the
  upstream trained checkpoint
- trained LEMUR can already produce a strong real-task retrieval story on local
  larger slices
- but the current Python artifact evaluator can distort the latency story on at
  least some tasks

## Next justified step

The next step should not be more artifact-fidelity work. That is settled.

The strongest next implementation target is:

1. add a **native or collection-backed latent-proxy benchmark path** so the
   artifact is measured through Kayak's intended runtime boundary rather than
   through the slower Python reference evaluator

Why:

- `LEMB` already shows that the trained checkpoint is much faster than exact
  while preserving quality
- the artifact contract reproduces that checkpoint exactly
- the remaining distortion is the evaluation path, not the export or scoring
  semantics
