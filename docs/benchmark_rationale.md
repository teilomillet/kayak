# Benchmark Rationale

`kayak` now ships small proxy workloads and judged proxy tasks for five benchmark families.
These are intentionally lightweight and fast. They are not official benchmark reproductions.

## Why These Families

- `LoTTE`
  Source: https://github.com/stanford-futuredata/ColBERT/blob/main/LoTTE.md
  Why it matters: LoTTE is part of the ColBERT ecosystem and is a direct fit for late interaction.
  Proxy focus in `kayak`: domain-specific forum retrieval with partial distractors.

- `BEIR`
  Source: https://github.com/beir-cellar/beir/wiki/Datasets-available
  Why it matters: BEIR is the broad cross-domain IR benchmark most public retrieval systems still report.
  Proxy focus in `kayak`: factual retrieval across scientific and finance-like slices.

- `MS MARCO`
  Source: https://arxiv.org/abs/1611.09268
  Why it matters: MS MARCO remains the standard short-passage ranking anchor and is explicitly based on real user search queries.
  Proxy focus in `kayak`: compact passage retrieval for web-style QA.

- `BRIGHT`
  Source: https://github.com/xlang-ai/BRIGHT
  Why it matters: BRIGHT is explicitly positioned as a reasoning-intensive retrieval benchmark.
  Proxy focus in `kayak`: conjunction-heavy queries where shallow overlap should lose.

- `MIRACL`
  Source: https://github.com/project-miracl/miracl
  Why it matters: MIRACL is the official multilingual retrieval benchmark spanning many languages.
  Proxy focus in `kayak`: cross-lingual relevance with multiple language variants.

## Epistemic Boundary

- The proxy tasks are correctness and regression slices, not leaderboard submissions.
- The workload profiles are systems-shape approximations, not measured corpus statistics from the public datasets.
- Official claims still require running the real public datasets, their qrels, and their evaluation scripts.
- The current suite is meant to answer:
  - does the search core behave correctly on several retrieval styles?
  - how does exact late interaction latency move as query/document vector budgets change?
  - can we scale one family later without redesigning the code?

## First Real Slices

- The current real public end-to-end slices in the repo are `BEIR/SciFact`, `BEIR/FIQA`, `LIMIT-small`, and a light `BrowseComp-Plus` evidence slice.
- These were chosen after checking the practical loader costs and official dataset shapes on April 11-12, 2026:
  - `beir/scifact/test` loaded through `ir_datasets` with `5,183` docs, `300` queries, and `339` qrels, and downloaded a `2.82 MB` archive in our environment.
  - `beir/fiqa/test` loaded through `ir_datasets` with `57,638` docs, `648` queries, and `1,706` qrels, and downloaded a `17.9 MB` archive in our environment.
  - `LIMIT-small` exposes `46` docs, `1,000` queries, and `2,000` qrels in the official Google DeepMind release, and the official repo explicitly positions it as the small MTEB-style retrieval task.
  - `BrowseComp-Plus` exposes `830` benchmark queries and a fixed `100,195`-document corpus in the official Tevatron release, which is exactly why the repo keeps only a small evidence-plus-hard-negative slice in the smoke path.
  - the straightforward `LoTTE` loader path through `ir_datasets` immediately requested the full `3.58 GB` `lotte.tar.gz` archive.
- `LIMIT-small` is the sound next retrieval slice because:
  - it is officially small
  - it is explicitly adversarial to single-vector retrieval
  - it aligns with adding `nDCG@10`, which is the standard MTEB retrieval metric
- `BrowseComp-Plus` is the sound reasoning-heavy slice because:
  - it is a fixed-corpus benchmark rather than a live-web benchmark
  - it ships human-verified evidence documents and curated hard negatives
  - it is much harder than BEIR-style factual smoke tests without forcing the full deep-research agent stack into `kayak`
- That makes `SciFact` the sound first real smoke path for CPU ColBERT-to-Mojo integration, `FIQA` the next good slice for broader domain/scale coverage, `LIMIT-small` the next good adversarial retrieval slice, and `BrowseComp-Plus` the next good reasoning-heavy retrieval slice, while `LoTTE` remains the preferred later late-interaction benchmark once we add a lighter data-access path or accept the heavier download.
