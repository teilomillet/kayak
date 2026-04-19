# GEM Paper-Faithfulness Gap Check

Date: `2026-04-20`
Status: `verified on current host and current worktree`

## Objective

Answer the narrower question left open after the frontier-policy promotion:

- does the current in-tree GEM path merely run
- or is it already working as well as the paper story would justify

This note separates three claims that should not be conflated:

1. mechanical correctness of the in-tree implementation
2. algorithmic alignment with the public GEM design
3. executable side-by-side parity with the public GEM reference

## What Was Re-Verified

Paper-facing live-path checks:

```bash
pixi run mojo -I . tests/test_gem_transport.mojo
pixi run mojo -I . tests/test_gem_graph_full.mojo
pixi run mojo -I . tests/test_collection_search_plan.mojo
```

Observed result:

- `tests/test_gem_transport.mojo`: `1/1` passed
- `tests/test_gem_graph_full.mojo`: `7/7` passed
- the current GEM search-plan execution path was already rechecked in the
  current worktree through `tests/test_collection_search_plan.mojo`, including:
  - `test_gem_graph_search_plan_executes_with_exact_rerank`
  - `test_gem_graph_search_plan_executes_with_configured_frontier_policy`

These checks justify the narrower statement that the in-tree GEM path works
mechanically on its own contract.

## Public Reference Executable Parity On This Host

### Fresh Host Check

Attempted reference smoke compile:

```bash
clang++ -std=c++17 -I/tmp/sigmod26gem-ref/hnswlib/hnswlib /tmp/gem_ref_smoke.cpp -c -o /tmp/gem_ref_smoke.o
```

Observed result:

- the expected public reference checkout is not currently present at
  `/tmp/sigmod26gem-ref`
- the compile therefore fails immediately with `hnswlib.h` not found

### What This Means

On the current machine state, executable parity with the public GEM reference
is not available at all.

That is a stronger practical blocker than the earlier `2026-04-13` note, which
already established that an unmodified public reference build is not runnable on
this Apple Silicon host because the reference hardcodes x86 intrinsics.

So the current host reality is:

- no local public reference checkout is staged
- even if it were restored unmodified, the earlier fidelity note still says the
  public reference is not directly runnable on this host architecture

The sound conclusion is therefore:

- this host can support algorithmic and benchmark-gap checks
- this host cannot currently support an executable side-by-side "Kayak GEM vs
  public GEM binary" result

## Strongest Same-Surface Local Gap Check

The strongest benchmark surface available on this host is the shared
faithfulness frontier on the public BrowseComp-plus gold subset:

```bash
pixi run mojo -I . benchmarks/browsecomp_plus_gold_faithfulness_frontier.mojo
```

Observed artifact:

- `.cache/kayak/browsecomp_plus_gold_faithfulness_frontier.json`

Key rows on the shared `posting_cap = 0` surface:

- `document_proxy`, `candidate_k = 40`
  - candidate recall `1.0`
  - mean search seconds `0.0010415`
- `gem_graph`, `candidate_k = 40`
  - candidate recall `0.925`
  - mean search seconds `0.00160275`
- `document_proxy`, `candidate_k = 80`
  - candidate recall `1.0`
  - mean search seconds `0.0020175`
- `gem_graph`, `candidate_k = 80`
  - candidate recall `1.0`
  - mean search seconds `0.002531`

Graph counters for `gem_graph` at the first full-recall row (`candidate_k = 80`):

- document count `90`
- mean visited vertex count `89.25`
- mean expanded edge count `549.5`
- mean visited cluster count `7.75`
- mean entry point count `7.75`
- mean max frontier size `49.25`

## Verified Gap

The current gap is now explicit.

The in-tree GEM path:

- does work mechanically
- does include paper-facing features such as exact transport, adaptive cutoff,
  and shortcut injection on the live builder path
- does not yet earn the pruning story that would make it "working as well as
  the paper said" on this checked public slice

Reason:

- at `candidate_k = 40`, GEM is still behind the simpler `document_proxy`
  baseline on both recall and latency
- at `candidate_k = 80`, GEM finally matches full recall but is still slower
- to reach that full-recall point, GEM is visiting about `89.25` of `90`
  documents on average, which means it is effectively touching almost the whole
  graph

That is the practical gap:

- the current Kayak GEM path is not yet earning a real graph-pruning advantage
  on this public same-surface benchmark

## Alignment Boundary

The current implementation remains paper-shaped rather than full reference
parity.

The code states that boundary explicitly in
[`kayak/index/gem_graph.mojo`](../../kayak/index/gem_graph.mojo):

- qCH and qEMD use quantized similarity with `1 - dot`
- the graph is stored explicitly
- the public GEM reference instead uses HNSW internals

That means it is still sound to say:

- the design is aligned enough for meaningful GEM-shaped research inside Kayak

It is not yet sound to say:

- Kayak has matched the public GEM executable
- Kayak has reproduced the paper's intended efficiency frontier

## Conclusion

The answer to "does it work as well as the paper said?" is currently:

- mechanically: yes
- as a paper-faithful executable parity claim: not verified on this host
- as a practical frontier claim on the checked public slice: no

The remaining high-value gap is no longer basic feature presence.
It is graph pruning quality.

The next sound GEM step should therefore target one of:

- a stronger structural graph primitive
- a more stateful diversity primitive
- an x86 or patched-reference environment for true executable parity testing
