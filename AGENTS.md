# AGENTS.md

This repository is for a late-interaction retrieval engine with a strong systems focus.
The codebase should stay easy to browse, easy to profile, and easy to validate.

## Core Working Rules

1. Justify decisions explicitly.
Every meaningful design or implementation choice must include a short reason.
If a tradeoff is unclear, state the competing options and why one was chosen.

2. Do not assume; verify.
Before making claims about behavior, performance, correctness, or architecture:
- inspect the local code
- run the relevant tests or checks when possible
- measure performance instead of inferring it
- distinguish facts from hypotheses

3. Validate or debunk.
If an optimization, bottleneck, or design claim is proposed, try to confirm or falsify it with:
- a focused benchmark
- a profiler trace
- a minimal reproduction
- a direct code inspection with cited evidence

4. Surface uncertainty.
If something is not verified, say so plainly and describe what would verify it.

5. Treat vector count as a first-class design axis.
Number of vectors in search representations must always be explicit in APIs,
benchmarks, and design notes. This includes query vectors, document vectors, and
related sparse-attention budgets or pruning policies. Relevance, memory,
latency, and index cost all depend on this choice.

6. Keep scalar types centralized.
Numeric representation choices such as vector, score, and metric scalars must be
defined in one central place and threaded through storage and APIs explicitly.
Do not scatter hard-coded `Float32` or `Float64` decisions across the codebase.

## Code Organization

The default should be many small files with narrow responsibilities, not large multi-purpose modules.

- Prefer folders with small focused modules over a single large file.
- Keep one main concept per file.
- Use descriptive names that reveal responsibility.
- Keep orchestration separate from kernels, math, storage, and I/O.
- Avoid "utils" catch-all modules unless the functions are truly generic and shared.
- Prefer composition over deep inheritance or hidden control flow.

Soft limits, unless a clear reason justifies otherwise:
- files should usually stay below roughly 200-300 lines
- functions should usually stay below roughly 40-60 lines
- complex logic should be split into named helpers instead of long branches

When a file grows, split it by responsibility, not arbitrarily.

## Readability Standards

This codebase is intended to be read by humans doing systems work.

- Favor straightforward control flow over cleverness.
- Keep data shapes, memory layout, and invariants explicit.
- Name hot-path functions and data structures clearly.
- Add brief comments where intent, invariants, or performance constraints are not obvious.
- Do not add decorative comments or redundant narration.

For public or central modules, include short top-level docstrings or comments that answer:
- what this module owns
- what it does not own
- what assumptions it relies on

## Profiling and Performance

Performance work must be measurable and reproducible.

- Design hot paths so they can be benchmarked in isolation.
- Separate setup from compute so profiling is not polluted by unrelated work.
- Make data movement, allocation, and conversion boundaries explicit.
- Prefer deterministic benchmark inputs where practical.
- Record enough context to make a timing meaningful: input sizes, dtypes, device, batch sizes.
- Avoid hidden caches, implicit global state, or silent fallbacks in performance-critical code.
- If caching is needed, make it explicit and measurable.

Every performance-sensitive subsystem should ideally support:
- a microbenchmark entry point
- a correctness check against a simpler reference implementation
- profiler-friendly boundaries with small, named functions

For retrieval code specifically:
- keep scoring kernels separate from index construction and serving logic
- keep storage layout explicit
- keep CPU and future GPU backends behind a narrow interface
- avoid mixing algorithmic choices with incidental framework code

## Validation Workflow

When changing behavior or performance-sensitive code:

1. State the claim.
Example: "packed token storage reduces scoring overhead by lowering pointer chasing."

2. Identify the evidence needed.
Example: benchmark exact scorer before and after on the same shapes.

3. Run the smallest check that can validate or debunk the claim.

4. Report the result precisely.
If the result is ambiguous, say that it is ambiguous.

## Testing Expectations

- Add unit tests for logic with clear correctness conditions.
- Keep a simple reference path for mathematically sensitive code when practical.
- For optimized kernels, compare against the reference path on representative inputs.
- Do not merge performance claims without numbers.
- Do not rely on a benchmark result from a single noisy run when variance is material.

## Communication

When working in this repository:

- explain what you are doing before large edits
- explain why the change is justified
- cite files, measurements, or sources when making claims
- clearly label assumptions, verified facts, and open questions

Short, direct explanations are preferred over long narratives, but they must contain the reasoning.

## What To Avoid

- large opaque files
- speculative optimizations without measurement
- architecture claims without checking the code or sources
- hidden behavior that makes profiling difficult
- premature abstraction that obscures data flow
- mixing prototype code, benchmark code, and production code in the same module
