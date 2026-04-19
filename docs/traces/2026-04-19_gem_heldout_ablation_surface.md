# GEM Held-Out Ablation Surface

Date: `2026-04-19`
Status: `implemented and locally validated`

## Objective

Add a paper-shaped benchmark surface for supervised GEM features that is
epistemically cleaner than self-training on the same judged queries later used
for evaluation.

This is the right next step because:

- the active GEM builder already supports adaptive cutoff and shortcuts
- mirror/setup plumbing now accepts full `GemGraphBuildConfig`
- the missing benchmark question was how to exercise those supervised features
  without leaking evaluation labels into the build step

## Design

New benchmark module:

- [`kayak/benchmarks/gem_heldout_ablation_json.mojo`](../../kayak/benchmarks/gem_heldout_ablation_json.mojo)
- [`kayak/benchmarks/gem_heldout_adaptive_diagnostics.mojo`](../../kayak/benchmarks/gem_heldout_adaptive_diagnostics.mojo)
- the adaptive policy is now explicit in the core GEM build config and stored
  metadata:
  - paper-faithful `first_relevant_cluster_rank`
  - experimental `relevant_cluster_coverage`

New benchmark entrypoint:

- [`benchmarks/synthetic_hard_recall_gem_heldout_ablation.mojo`](../../benchmarks/synthetic_hard_recall_gem_heldout_ablation.mojo)
- [`benchmarks/synthetic_hard_recall_gem_heldout_ablation_focused.mojo`](../../benchmarks/synthetic_hard_recall_gem_heldout_ablation_focused.mojo)
- [`benchmarks/synthetic_hard_recall_gem_heldout_ablation_smoke.mojo`](../../benchmarks/synthetic_hard_recall_gem_heldout_ablation_smoke.mojo)
- [`benchmarks/synthetic_hard_recall_gem_heldout_ablation_adaptive_probe.mojo`](../../benchmarks/synthetic_hard_recall_gem_heldout_ablation_adaptive_probe.mojo)
- [`benchmarks/synthetic_hard_recall_gem_heldout_ablation_adaptive_probe_smoke.mojo`](../../benchmarks/synthetic_hard_recall_gem_heldout_ablation_adaptive_probe_smoke.mojo)

New focused tests:

- [`tests/test_gem_heldout_ablation_json.mojo`](../../tests/test_gem_heldout_ablation_json.mojo)

### Supervision Boundary

The new surface uses a deterministic held-out split:

- training queries are selected by an evenly spaced query-index schedule
- evaluation queries are the disjoint complement in original query order
- each supervised training pair uses the first relevant doc id for that query

Reason:

- evenly spaced selection covers more of the synthetic query range than a plain
  prefix split while remaining deterministic
- first-relevant-doc selection is explicit, reproducible, and sufficient for
  the current synthetic fixtures where relevant duplicates are interchangeable

### Variant Surface

The current benchmark surface enumerates four build variants over the same
frontier-style base config:

- `baseline`
- `adaptive_cutoff`
- `shortcuts`
- `adaptive_cutoff_shortcuts`

The adaptive probe surface now adds one explicit experimental variant without
changing the default paper-shaped quartet:

- `adaptive_cutoff_relevant_coverage`

Query-time GEM search parameters remain explicit and constant in the summary so
the measured change is primarily in graph construction, not hidden search-plan
drift.

## What Changed

The new module adds:

- `GemHeldoutQuerySplit`
- `GemHeldoutAblationVariantSpec`
- `GemHeldoutAblationSummary`
- deterministic split helpers
- standard supervised GEM variant enumeration
- JSON helpers for held-out ablation summaries

The new runner layer adds:

- [`kayak/benchmarks/gem_heldout_ablation_runner.mojo`](../../kayak/benchmarks/gem_heldout_ablation_runner.mojo)
- explicit default versus smoke run options
- a focused first-profile run option between smoke and full
- bounded candidate-window selection for default runs
- a truly small synthetic smoke profile instead of reusing the first paper-sized
  default profile

The summary records both:

- held-out quality/latency metrics via the existing
  `build_faithfulness_frontier_summary_for_plan(...)` seam
- built GEM artifact metadata such as cluster count, edge count, shortcut count,
  and entry-point count
- adaptive-cutoff diagnostics such as training-label statistics and predicted
  profile-limit statistics

That boundary is important because the next GEM question is not just whether the
graph retrieves the right candidates, but also what graph structure was built to
achieve that result.

## Validation

Focused held-out benchmark tests:

```bash
pixi run mojo -I . tests/test_gem_heldout_ablation_json.mojo
```

Observed result:

- `5` tests passed locally
- these cover:
  - disjoint held-out splitting
  - supervised variant enumeration
  - smoke versus full candidate-window selection
  - adaptive smoke diagnostics for label and predicted-limit collapse
  - summary/JSON construction over a real configured GEM mirror

Regression checks on adjacent seams:

```bash
pixi run mojo -I . tests/test_collection_mirror.mojo
pixi run mojo -I . tests/test_faithfulness_frontier_json.mojo
```

Observed result:

- both test modules passed locally

Benchmark-facing compile check:

```bash
pixi run mojo build benchmarks/synthetic_hard_recall_gem_heldout_ablation.mojo -I . -o /tmp/kayak-synthetic-hard-recall-gem-heldout-ablation
pixi run mojo build benchmarks/synthetic_hard_recall_gem_heldout_ablation_focused.mojo -I . -o /tmp/kayak-synthetic-hard-recall-gem-heldout-ablation-focused
pixi run mojo build benchmarks/synthetic_hard_recall_gem_heldout_ablation_smoke.mojo -I . -o /tmp/kayak-synthetic-hard-recall-gem-heldout-ablation-smoke
pixi run mojo build benchmarks/synthetic_hard_recall_gem_heldout_ablation_adaptive_probe.mojo -I . -o /tmp/kayak-synthetic-hard-recall-gem-heldout-ablation-adaptive-probe
```

Observed result:

- all four entrypoints compiled locally

Smoke run check:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_ablation_smoke.mojo
```

Observed result:

- the smoke runner completed locally and wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_ablation_smoke_small.json`
- this produced `8` summaries over one tiny held-out synthetic slice
- first observed signal on that smoke slice:
  - `baseline` recall stayed at `1.0`
  - `adaptive_cutoff` recall dropped to `0.5`
  - `shortcuts` injected `0` shortcut edges on that tiny slice and therefore
    behaved like baseline
- the new adaptive diagnostics show why that happened:
  - adaptive training labels had mean `1.0`
  - adaptive training-label one-share was `1.0`
  - adaptive predicted profile-limit mean was `1.0`
  - adaptive predicted profile-limit one-share was `1.0`

That matters because it rules out one tempting but weaker explanation. On the
smoke slice, the width-1 collapse is already present in the supervision target,
so the current failure is not just a bad split learned by the tree.

Paper check:

- the GEM paper's adaptive target matches the current
  `first_relevant_cluster_rank` policy
- the repo therefore keeps that policy as the default paper-faithful behavior
  and introduces `relevant_cluster_coverage` as an explicit experimental
  alternative instead of silently changing the default primitive

That is not a claim about the larger paper-shaped profiles. It is a useful
sanity check that the new benchmark surface can both confirm and falsify a GEM
variant on held-out data rather than only reporting success.

Focused run spot check:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_ablation_focused.mojo
```

Observed result:

- the run was interrupted after the first emitted line because the focused slice
  is intentionally much heavier than smoke
- the first emitted line already showed a qualitatively different regime:
  - on `slots6_values3_docs1530__gem_heldout_eval`
  - `baseline`
  - `candidate_k=2`
  - held-out candidate recall was `0.0`

That is still only a partial run, but it confirms the focused entrypoint is
probing a materially harder stage-1 setting than the tiny smoke slice.

Adaptive probe run:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_ablation_adaptive_probe.mojo
```

Observed result:

- the adaptive probe completed locally and wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_ablation_adaptive_probe.json`
- on the first default real synthetic profile:
  - `baseline` recall was `0.0` at `candidate_k=2` and `candidate_k=4`
  - `adaptive_cutoff` recall was also `0.0` at `candidate_k=2` and
    `candidate_k=4`
  - `adaptive_cutoff_relevant_coverage` recall also stayed `0.0` at
    `candidate_k=2` and `candidate_k=4`
  - baseline mean document-profile limit was about `5.82`
  - adaptive mean document-profile limit collapsed to `1.0`
  - coverage-policy adaptive mean document-profile limit rose to about `3.76`
  - adaptive training-label mean was `1.0`
  - coverage-policy adaptive training-label mean rose to about `5.17`
  - adaptive predicted profile-limit mean was `1.0`
  - coverage-policy adaptive predicted profile-limit mean rose to about `3.76`
  - evaluation positive-profile hit rate still stayed at `1.0` for both
    baseline, paper-faithful adaptive, and coverage adaptive

This narrows the interpretation:

- the adaptive supervision target still collapses to width `1` on the first
  real synthetic profile
- the coverage policy measurably changes the primitive and restores wider
  profiles, so the benchmark is sensitive to the label-policy choice
- but that slice's dominant failure is broader than profile filtering alone,
  because baseline and both adaptive policies already sit at `0.0` recall even
  with much wider document profiles under the coverage variant

Smoke adaptive-policy probe:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_ablation_adaptive_probe_smoke.mojo
```

Observed result:

- the smoke adaptive probe completed locally and wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_ablation_adaptive_probe_smoke.json`
- on the tiny smoke slice:
  - `baseline` recall stayed `1.0`
  - paper-faithful `adaptive_cutoff` stayed at `0.5`
  - `adaptive_cutoff_relevant_coverage` recovered recall back to `1.0`
  - coverage-policy mean document-profile limit rose to about `2.14`
  - coverage-policy training-label mean was `3.0`
  - coverage-policy evaluation positive-profile hit rate stayed `1.0`

This is the cleanest current evidence for the new primitive:

- explicit label policy matters
- the coverage policy can fix the tiny conjunction-heavy smoke regression
- the harder real slice still needs traversal/construction work beyond label
  policy alone

Reachability interpretation on the first real profile:

- `baseline` now shows:
  - evaluation positive entry rate `0.0`
  - evaluation positive reachable rate `1.0`
  - evaluation positive profile-hit rate `1.0`
  - held-out recall `0.0` at low candidate windows (`candidate_k=2,4`)
- paper-faithful adaptive shows:
  - reachable rate reduced to about `0.833`
  - recall still `0.0`
- coverage adaptive restores reachable rate back to `1.0`
  - but recall still stays `0.0`

This is a stronger diagnosis than the earlier notes:

- the hard-slice bottleneck is not simply missing profile coverage
- the hard-slice bottleneck is not simply graph disconnection
- relevant positives are reachable in the gated graph under both baseline and
  coverage adaptive
- the remaining failure is therefore in query-time search order or query-time
  candidate budgeting

Beam-width probe:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_beam_probe.mojo
```

Observed result:

- the beam probe wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_beam_probe.json`
- it fixed:
  - profile: first real synthetic slice only
  - variants: `baseline`, paper-faithful `adaptive_cutoff`,
    `adaptive_cutoff_relevant_coverage`
  - candidate window: `candidate_k=32`
  - beam widths: `32`, `64`, `128`, `256`
- observed behavior:
  - baseline recall improved from the earlier low-window `0.0` regime to `0.5`
    at beam `32` and `64`
  - baseline recall then dropped to about `0.333` at beam `128` and `256`
  - paper-faithful adaptive improved from about `0.166` at beam `32` to about
    `0.333` at beam `64+`
  - coverage adaptive stayed at about `0.166` across all tested beam widths
  - larger beam widths increased visited vertices substantially but did not
    monotonically improve recall

Interpretation:

- widening beam can matter on the hard slice, so query-time search breadth is a
  real lever
- beam alone does not cleanly solve the problem
- the earlier jump from `candidate_k=4` to `candidate_k=32` already changed the
  regime for baseline, so candidate budgeting is now at least as important a
  question as beam width

Query-time candidate-window probe:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_querytime_probe.mojo
```

Observed result:

- the query-time probe wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_querytime_probe.json`
- it fixed:
  - profile: first real synthetic slice only
  - variants: `baseline`, paper-faithful `adaptive_cutoff`,
    `adaptive_cutoff_relevant_coverage`
  - beam width: `32`
  - candidate windows: `4`, `8`, `16`, `32`, `64`
- observed behavior:
  - baseline recall stayed `0.0` through `candidate_k=16`, improved to `0.5`
    at `candidate_k=32`, and reached `1.0` at `candidate_k=64`
  - paper-faithful adaptive stayed `0.0` through `candidate_k=16`, improved to
    about `0.166` at `candidate_k=32`, and reached about `0.333` at
    `candidate_k=64`
  - coverage adaptive stayed `0.0` through `candidate_k=16`, improved to about
    `0.166` at `candidate_k=32`, and reached `0.5` at `candidate_k=64`
  - evaluation positive entry rate stayed `0.0` across all tested variants and
    candidate windows
  - graph visitation stayed flat across `candidate_k=4,8,16,32` and only
    jumped at `candidate_k=64`:
    - baseline visited vertices: about `167.5` then `395.0`
    - paper-faithful adaptive: about `180.5` then `292.0`
    - coverage adaptive: about `196.0` then `350.0`

Interpretation:

- candidate budget is a threshold effect on this hard slice, not a smooth gain
- on this slice, widening the final candidate window helps only once the search
  actually explores more of the graph
- because entry rate stays `0.0` while reachable rate stays high, the recovery
  at larger candidate windows comes from deeper traversal rather than better
  seeded entry points
- baseline remains the strongest current setting on this first real synthetic
  slice; coverage adaptive helps relative to paper-faithful adaptive at
  `candidate_k=64` but still trails baseline

Query-time cluster-gate probe:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_cluster_topk_probe.mojo
```

Observed result:

- the cluster-gate probe wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_cluster_topk_probe.json`
- it fixed:
  - profile: first real synthetic slice only
  - variants: `baseline`, paper-faithful `adaptive_cutoff`,
    `adaptive_cutoff_relevant_coverage`
  - beam width: `32`
  - candidate windows: `32`, `64`
  - search cluster gates: `1`, `2`, `4`, `8`
- observed behavior:
  - baseline at `candidate_k=32` improved from `0.5` at the default
    `cluster_top_k=2` to about `0.583` at `4`, then degraded to `0.25` at `8`
  - baseline at `candidate_k=64` was best at the default `cluster_top_k=2`
    with recall `1.0`; widening to `4` and `8` reduced recall to about `0.833`
    and `0.75`
  - paper-faithful adaptive stayed flat across the whole sweep:
    about `0.166` at `candidate_k=32` and `0.333` at `64`
  - coverage adaptive improved from about `0.166` to `0.333` at
    `candidate_k=32` only when widened all the way to `cluster_top_k=8`, but
    stayed flat at `0.5` for every tested gate at `candidate_k=64`
  - narrowing to `cluster_top_k=1` reduced baseline reachable rate from `1.0`
    to `0.5`
  - entry rate still stayed `0.0` across the whole sweep

Interpretation:

- widening the cluster gate is not a clean rescue on this slice
- the default `cluster_top_k=2` remains the best high-budget baseline setting
- `cluster_top_k=4` gives only a modest mid-budget baseline gain and does not
  remove the candidate-budget threshold
- because entry rate remains `0.0`, gate widening changes traversal breadth
  rather than solving seeding directly

Query-time hop diagnostics:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_querytime_probe.mojo
```

Observed result:

- the query-time probe now also records positive hop-distance diagnostics inside
  the relevant-cluster-filtered graph:
  - mean reachable positive hop count
  - within-`1`, within-`2`, and within-`4` hop rates
- on the first real synthetic slice at the default search gate `2`:
  - baseline:
    - reachable rate `1.0`
    - mean reachable positive hop count about `4.83`
    - within-`2` hop rate `0.0`
    - within-`4` hop rate about `0.333`
  - paper-faithful adaptive:
    - reachable rate about `0.833`
    - mean reachable positive hop count about `9.6`
    - within-`2` hop rate about `0.166`
    - within-`4` hop rate about `0.166`
  - coverage adaptive:
    - reachable rate `1.0`
    - mean reachable positive hop count about `8.0`
    - within-`2` hop rate about `0.166`
    - within-`4` hop rate `0.5`

Interpretation:

- the hard-slice positives are not shallow relative to the current seeded entry
  docs
- baseline already needs roughly `5` gated graph hops on average to reach the
  held-out positives
- adaptive variants make that problem materially worse by pushing positives
  deeper in hop space even when they remain reachable
- this strengthens the earlier candidate-window result: the current threshold
  behavior is not just a ranking artifact among shallow neighbors, it is also a
  multi-hop traversal problem

Representative-depth probe:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_representative_depth_probe.mojo
```

Observed result:

- the representative-depth probe wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_representative_depth_probe.json`
- it fixed:
  - profile: first real synthetic slice only
  - variants: `baseline`, paper-faithful `adaptive_cutoff`,
    `adaptive_cutoff_relevant_coverage`
  - cluster gates: `2`, `4`
  - representative depths: `1`, `2`, `4`, `8`
- observed behavior:
  - evaluation positive representative rate stayed `0.0` for every tested
    variant, cluster gate, and representative depth
  - this remained true even when reachable rate stayed high:
    - baseline reachable rate `1.0`
    - paper-faithful adaptive reachable rate about `0.833`
    - coverage adaptive reachable rate `1.0`

Interpretation:

- the hard-slice positives are not merely "a few representatives deeper" inside
  the selected clusters
- a simple multi-entry-per-cluster primitive is therefore not the highest-value
  next bet on this slice
- the remaining problem is deeper graph traversal or graph-construction quality,
  not just representative seeding depth

Entry-hop probe:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_entry_hop_probe.mojo
```

Observed result:

- the entry-hop probe wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_entry_hop_probe.json`
- it compared the current `cluster_entry` seeding against representative-based
  entry sets with depths `1`, `2`, `4`, and `8`
- observed behavior:
  - baseline:
    - `cluster_entry` mean hop count stayed about `4.83`
    - representative depth `4` improved mean hop count slightly to about `4.67`
    - representative depth `8` improved mean hop count to about `4.17`
    - only representative depth `8` raised within-`2` hop rate at all, and only
      to about `0.166`
  - paper-faithful adaptive:
    - `cluster_entry` mean hop count stayed about `9.6`
    - representative depth `2` or higher improved mean hop count only to about
      `8.4`
    - within-`4` hop rate stayed flat at about `0.166`
  - coverage adaptive:
    - `cluster_entry` mean hop count stayed about `8.0`
    - representative depth `2` or higher improved mean hop count to about `7.0`
    - within-`4` hop rate stayed flat at `0.5`

Interpretation:

- representative-based seeding is not useless, but its effect is modest on this
  slice
- current entry selection is therefore not the dominant failure mode
- deeper or broader seed sets do not pull the positives close enough to explain
  the large candidate-budget threshold by themselves
- that leaves graph construction quality and query-time frontier ordering as the
  stronger remaining suspects

Construction-density probe:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_construction_probe.mojo
```

Observed result:

- the construction probe wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_construction_probe.json`
- it fixed:
  - profile: first real synthetic slice only
  - variant: `baseline`
  - search settings:
    - `cluster_top_k=2`
    - `beam=32`
    - `candidate_k=32,64`
  - construction sweeps:
    - `construction_neighbor_count`: `4`, `6`, `12`
    - `degree_limit`: `4`, `8`, `16`
- observed behavior:
  - the best mid-budget setting was `neighbors=4`, `degree_limit=8`:
    - recall about `0.583` at `candidate_k=32`
    - recall about `0.833` at `candidate_k=64`
  - the best high-budget setting was the current default
    `neighbors=6`, `degree_limit=8`:
    - recall `0.5` at `candidate_k=32`
    - recall `1.0` at `candidate_k=64`
  - low degree caused a consistent reachability collapse:
    - `4/4` reachable rate about `0.333`
    - `6/4` reachable rate about `0.333`
    - `12/4` reachable rate about `0.333`
  - increasing neighbors while keeping degree low made the graph worse:
    - `4/4` recall about `0.333` at both candidate windows
    - `6/4` fell to about `0.166` at `candidate_k=32`
    - `12/4` fell to `0.0` at `candidate_k=32`
  - increasing degree above `8` did not help the best settings:
    - `6/16` matched `6/8` at `candidate_k=32` but dropped from `1.0` to
      about `0.833` at `candidate_k=64`
    - `12/16` matched `12/8` at `candidate_k=32` but dropped from about
      `0.833` to about `0.666` at `candidate_k=64`
  - denser settings also expanded the stored graph materially:
    - `4/8` graph edges about `11713`
    - `6/8` graph edges about `11893`
    - `12/8` graph edges about `12011`
    - `12/16` graph edges about `20899`
  - entry rate still stayed `0.0` across the whole sweep

Interpretation:

- low degree is a real failure mode on this slice because it collapses
  reachable-rate before query-time budgeting can help
- higher neighbor sampling is not a free win; when degree is constrained it is
  actively harmful
- increasing degree beyond the default `8` mainly adds graph density and build
  cost without improving the best observed recall
- the current default `6/8` remains the best high-budget baseline setting on
  this first real synthetic slice
- scalar density sweeps therefore constrain the problem but do not solve it:
  the next construction change should be structural rather than simply "more
  neighbors" or "more degree"
- the new hop diagnostics make that tradeoff sharper:
  - denser high-degree graphs do shorten the path to positives
  - for example `6/16` lowered mean hop count to about `3.33` and raised the
    within-`4` hop rate to about `0.833`
  - but `6/16` still lost to the default `6/8` at `candidate_k=64`
    (`0.833` versus `1.0` recall)
- so shorter hop distance alone is not sufficient on this slice; query-time
  frontier ordering remains a live bottleneck even when the positives become
  structurally closer

Frontier-policy probe:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_frontier_policy_probe.mojo
```

Observed result:

- the frontier-policy probe wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_frontier_policy_probe.json`
- it compared:
  - the current `local_per_entry` policy used by the serving path
  - a benchmark-only `global_best_first` frontier over the same cached GEM
    artifacts
  - a benchmark-only `hybrid_best_head_per_entry` policy that keeps one local
    queue per entry root but always expands the globally best current queue
    head
  - a benchmark-only `hybrid_best_head_fair_round` policy that still chooses
    the best current root head but limits each root to one expansion per round
  - a benchmark-only `hybrid_best_head_quota2_round` policy that relaxes that
    fairness limit to two expansions per root per round
- all five policies used:
  - the same held-out graph artifacts
  - the same cluster gate `2`
  - the same beam width `32`
  - the same exact stage-2 rerank and exact oracle candidate-recall check
- observed behavior:
  - baseline:
    - at `candidate_k=32`, `global_best_first` dropped recall from `0.5` to
      about `0.333`
    - the new hybrid recovered part of that loss:
      - at `candidate_k=32`, recall rose to about `0.417`
      - but it still stayed below the current local policy
    - at `candidate_k=64`, `global_best_first` dropped recall from `1.0` to
      `0.5`
    - the hybrid still failed to recover that strongest local point:
      - at `candidate_k=64`, it also stayed at `0.5`
    - it did reduce search work:
      - mean visited vertices fell from about `392.5` to `342.0` at
        `candidate_k=64`
      - mean max frontier size fell from about `92.5` to `47.0`
    - the hybrid was cheaper than the local policy too:
      - mean visited vertices fell to about `145.5` at `candidate_k=32`
      - mean visited vertices fell to about `360.5` at `candidate_k=64`
      - mean max frontier size stayed around `30.5` at `candidate_k=32` and
        `46.5` at `candidate_k=64`
    - the fair-round policy then recovered the current local behavior almost
      exactly:
      - at `candidate_k=32`, recall returned to `0.5`
      - at `candidate_k=64`, recall returned to `1.0`
      - visited vertices and frontier size also matched local within noise
    - the quota-`2` policy behaved the same on this slice:
      - it also matched local recall at both `candidate_k=32` and `64`
      - and it did not keep the cheaper medium-budget behavior from the looser
        hybrid
  - paper-faithful adaptive:
    - recall stayed unchanged at every tested candidate window
    - `global_best_first` still reduced visited vertices and frontier size
    - the hybrid matched `global_best_first` exactly on this slice, so
      preserving one queue per root without round-robin fairness did not buy
      anything for paper-faithful adaptive
    - both fairer variants collapsed back to local behavior:
      - `hybrid_best_head_fair_round` and `hybrid_best_head_quota2_round`
        matched local recall and cost at every tested candidate window
  - coverage adaptive:
    - `global_best_first` improved one low-budget point:
      - at `candidate_k=16`, recall rose from `0.0` to about `0.166`
    - but it stayed flat at `candidate_k=32` and `64`
    - the hybrid helped much more at medium budget:
      - at `candidate_k=32`, recall rose to `0.5` versus about `0.167` for
        both local and pure global
      - at `candidate_k=64`, all three policies reached `0.5`, but the hybrid
        used the fewest visited vertices at about `293.5`
    - the fair-round policy then lost that medium-budget gain and returned to
      local behavior:
      - at `candidate_k=16`, recall fell back to `0.0`
      - at `candidate_k=32`, recall fell back to about `0.167`
    - the quota-`2` policy also failed to recover the looser-hybrid gain:
      - it matched local recall at `candidate_k=16`, `32`, and `64`
      - it shaved only a small amount of visited work at lower budgets
    - all non-local policies still reduced frontier size substantially relative
      to the current local path, but only the quota-free hybrid improved this
      coverage-adaptive medium-budget recall point

Interpretation:

- pure global best-first ordering is not the missing query-time rescue
- on the strongest current baseline, it is clearly harmful despite lowering
  search cost
- that means the current per-entry local frontier is not just incidental
  implementation detail; it is acting as a useful diversity mechanism across
  entry roots
- the follow-up fair-round and quota-`2` results narrow that story further:
  - the beneficial part of the local policy on the strongest baseline really is
    round-style root fairness
  - simple scalar relaxations of that fairness rule do not produce a useful
    middle point on this slice:
    - quota `1` behaves like local
    - quota `2` also behaves like local
    - quota-free best-head ordering recovers the coverage-adaptive medium-budget
      gain but breaks the strongest baseline badly
- the remaining query-time question is therefore not "should we replace local
  queues with one global queue?"
- the stronger next question is no longer a simple scheduling-scalar tweak
- if frontier-policy work continues, it likely needs a more stateful root
  diversity primitive than just per-round quota tuning
- otherwise the more promising next GEM move is probably structural graph
  improvement rather than another simple frontier-ordering variant

Shortcut probe:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_shortcut_probe.mojo
```

Observed result:

- the shortcut probe wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_shortcut_probe.json`
- it fixed:
  - profile: first real synthetic slice only
  - variants: `baseline`, `shortcuts`, paper-faithful `adaptive_cutoff`,
    `adaptive_cutoff_shortcuts`
  - search settings:
    - `cluster_top_k=2`
    - `beam=32`
    - `candidate_k=32,64`
- observed behavior:
  - the plain `shortcuts` variant injected `0` shortcut edges and exactly
    matched `baseline`:
    - recall `0.5` at `candidate_k=32`
    - recall `1.0` at `candidate_k=64`
    - reachable rate `1.0`
  - paper-faithful `adaptive_cutoff` stayed weaker:
    - recall about `0.166` at `candidate_k=32`
    - recall about `0.333` at `candidate_k=64`
    - reachable rate about `0.833`
  - `adaptive_cutoff_shortcuts` injected only `2` shortcut edges and still
    matched paper-faithful adaptive exactly at both candidate windows

Interpretation:

- the plain shortcut path is inactive on this slice because it produces no
  additional shortcut edges at the default shortcut mining budget
- the adaptive shortcut path is active but weak; injecting only `2` shortcut
  edges does not move recall or reachability relative to paper-faithful
  adaptive without shortcuts
- the next shortcut question is therefore not whether shortcuts help in the
  abstract, but whether the current shortcut-mining primitive is too weak to
  create useful bridges on this slice
- the implementation was then tightened so shortcut insertion can rewire
  saturated degree lists instead of only appending into spare slots, and a new
  saturated-degree unit test confirms that primitive on a minimal graph
- re-running this held-out shortcut probe after that change produced the same
  edge counts and recall as above, which falsifies the simpler
  "degree saturation is blocking useful shortcuts" explanation on this slice

Shortcut-budget probe:

```bash
pixi run mojo -I . benchmarks/synthetic_hard_recall_gem_heldout_shortcut_budget_probe.mojo
```

Observed result:

- the shortcut-budget probe wrote:
  `.cache/kayak/synthetic_hard_recall_gem_heldout_shortcut_budget_probe.json`
- it fixed:
  - profile: first real synthetic slice only
  - variants: `shortcuts`, `adaptive_cutoff_shortcuts`
  - shortcut candidate budgets: `6`, `16`, `32`, `64`
  - search settings:
    - `cluster_top_k=2`
    - `beam=32`
    - `candidate_k=32,64`
- observed behavior:
  - the plain `shortcuts` variant stayed completely flat across all shortcut
    budgets:
    - `shortcut_edge_count` stayed `0`
    - recall stayed `0.5` at `candidate_k=32`
    - recall stayed `1.0` at `candidate_k=64`
    - reachable rate stayed `1.0`
  - `adaptive_cutoff_shortcuts` also stayed completely flat across all tested
    shortcut budgets:
    - `shortcut_edge_count` stayed `2`
    - recall stayed about `0.166` at `candidate_k=32`
    - recall stayed about `0.333` at `candidate_k=64`
    - reachable rate stayed about `0.833`

Interpretation:

- the shortcut failure is not a hidden scalar-budget issue on this slice
- the shortcut failure is also not explained by the original spare-capacity-only
  insertion rule; after switching shortcut insertion to the same
  degree-limited rewiring heuristic used by the base graph builder, the full
  shortcut-budget surface stayed identical
- widening `shortcut_candidate_k` by more than `10x` does not create new
  shortcut edges for the plain shortcut variant and does not improve the weak
  adaptive shortcut variant
- this rules out the simplest "just mine more shortcut candidates" explanation
- the next sound shortcut-related change would need to be structural:
  a different shortcut-mining rule, bridge criterion, or insertion policy
  rather than a larger shortcut budget
- the probe now stores and validates `shortcut_candidate_k` explicitly in GEM
  artifacts and query-time summaries so future sweeps cannot silently collapse
  distinct budgets into the same cached metadata

## Current Boundary

What this step establishes:

- Kayak can now benchmark adaptive-cutoff and shortcut-aware GEM variants on a
  held-out synthetic slice without using the same judged queries for both
  supervision and evaluation
- the benchmark output stays within the existing planner/runtime/evaluator
  contracts instead of introducing a GEM-only evaluation path
- the benchmark family now has both:
  - a paper-shaped full runner
  - a fast smoke runner that actually terminates on a genuinely small fixture
- the benchmark artifact now explains adaptive behavior instead of only
  reporting recall, because it records both supervision-label and predicted
  profile-limit statistics
- the query-time probe cache now validates stored GEM build metadata before
  reuse, so held-out summaries do not silently inherit stale shortcut-budget
  or construction metadata from earlier artifact schemas
- the adaptive label policy is now an explicit stored primitive rather than an
  implicit hidden assumption inside the builder
- the benchmark artifact now also separates:
  - profile-hit failures
  - entry-point coverage
  - gated graph reachability
  so hard-slice misses can be attributed to connectivity versus query-time
  search behavior
- the query-time surfaces now also separate:
  - candidate-budget thresholds
  - cluster-gate sensitivity
  - representative-depth coverage
  so the current hard-slice miss is no longer plausibly explained by a shallow
  representative-seeding failure
- the baseline construction surface now also separates:
  - degree-starvation failures
  - neighbor-overgeneration failures
  - over-dense graph bloat above the current default
  so the remaining construction question is no longer "should the graph just be
  bigger?"

What this step does not establish:

- it does not yet prove a frontier win on public real datasets
- it does not claim the synthetic split is a substitute for external
  train/eval corpora
- it does not claim the current adaptive-max heuristic is paper-optimal
- it does not yet prove whether adaptive cutoff should keep the current label
  definition, because the new evidence suggests the supervised target itself may
  be too weak for conjunction-heavy workloads
- it does not yet prove the coverage policy is broadly better, because it fixes
  the smoke regression but not the first real synthetic slice
- it does not yet identify the best query-time setting beyond the first real
  synthetic slice; the current best observed setting here is baseline with
  `beam=32` and `candidate_k=64`, but that is only one slice
- it does not yet show how to lower that threshold through a new primitive;
  it only rules out several weaker explanations
- it does not yet show a better graph-construction primitive than the current
  baseline default; it only shows that the nearby scalar density sweeps do not
  beat that default on this slice

## Next Honest Question

Now that the supervision boundary exists, the next GEM work should be empirical:

- decide whether the adaptive-cutoff supervision target should remain
  "first relevant cluster rank" or be replaced by a stricter label that better
  preserves conjunction-style coverage
- compare the paper-faithful and coverage policies on more hard synthetic and
  public slices now that the label policy is explicit in both config and stored
  metadata
- test graph-construction changes that are structural rather than scalar-density
  sweeps, because the current `construction_neighbor_count` and `degree_limit`
  sweep did not beat the default high-budget baseline
- extend the fixed-beam candidate-window sweep to more slices and public tasks
  before treating `beam=32`, `candidate_k=64` as a robust search default
- keep treating graph traversal quality and graph construction quality as the
  main problem, because the current probes now rule out:
  - missing profile coverage
  - graph disconnection under baseline and coverage adaptive
  - shallow representative seeding depth
- use the recorded label/prediction diagnostics to avoid blaming the decision
  tree for behavior that is already present in the labels
