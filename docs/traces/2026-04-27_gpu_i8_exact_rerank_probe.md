# 2026-04-27: GPU I8 Exact Rerank Kernel Probe

## Claim

The current exact address scorer may be slowed by materializing one partial
score per `(query, candidate, query vector, document vector)` before reducing.

Reason: under the current `coverage_safety_v1` candidate-window policy,
non-full rows are exact-rerank dominated. The address scorer writes and reads a
large partial-score buffer, so a fused query-vector implementation was worth
testing before deeper kernel work.

## Probe 1: Single-Thread Query-Vector Fusion

Tested an additive local implementation that computed one best score per
`(query, candidate, query vector)` directly, then reduced those query-vector
scores to final candidate scores.

Validation smoke:

```bash
pixi run env PYTHONPATH=python:python/scripts python -c 'from bench_fastplaid_speed_track import SpeedTrackShape, build_synthetic_inputs; from kayak_bridge.gpu_device_capability import probe_mojo_gpu; from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex; from kayak_bridge.mojo_gpu_i8_rerank import prepare_i8_address_session_handle; shape=SpeedTrackShape(document_count=128, document_vector_count=16, query_count=2, query_vector_count=8, vector_dim=128, top_k=10, update_document_count=0); inputs=build_synthetic_inputs(shape, seed=31, normalize_vectors=False); index=KayakPlaidApproxIndex.build(doc_ids=inputs.doc_ids, documents=inputs.documents, config=KayakPlaidApproxConfig(centroid_count=64, centroids_per_query_vector=8, candidate_k=64, payload="i8"), final_k=10); cap=probe_mojo_gpu("gpu-query"); handle=prepare_i8_address_session_handle(target_accelerator=cap.target_accelerator or "", shape=shape, candidate_k=64, payload=index.i8_payload_snapshot()); candidates=[list(range(64)) for _ in range(shape.query_count)]; old=handle.score_topk_without_reference(queries=inputs.queries, candidate_positions_by_query=candidates, top_k=10); fused=handle.score_topk_without_reference_query_vector_fused(queries=inputs.queries, candidate_positions_by_query=candidates, top_k=10); print({"old_s": old.extension_call_seconds, "fused_s": fused.extension_call_seconds, "positions_equal": old.positions==fused.positions, "max_score_delta": max(abs(a-b) for a,b in zip(old.scores, fused.scores))}); handle.close()'
```

Result:

| scorer | extension seconds |
| --- | ---: |
| existing partial-score scorer | `0.00026196900580544025` |
| fused query-vector scorer | `0.03170162300375523` |

Correctness matched for this smoke:

- `positions_equal`: `True`
- `max_score_delta`: `0.0`

## Decision

Reject this fused query-vector implementation.

Reason: it removed partial-score memory traffic but collapsed too much
parallelism into each thread. On the smoke case it was roughly `121x` slower
than the existing scorer, so the experimental code was removed instead of
leaving an unused internal path.

## Probe 2: Block-Level Query-Vector Reduction

Tested a second additive local implementation with one block per
`(query, candidate, query vector)`. Threads in the block covered document
vectors and reduced the max score in shared memory before reducing query-vector
scores to final candidate scores.

Reason: this kept parallelism across document vectors while still avoiding the
full partial-score write/read path.

Validation smoke used the same synthetic shape, seed, payload, candidate
window, and top-k as Probe 1.

Result:

| scorer | extension seconds |
| --- | ---: |
| existing partial-score scorer | `0.00023619100102223456` |
| block-level query-vector scorer | `0.024296386996866204` |

Correctness matched for this smoke:

- `positions_equal`: `True`
- `max_score_delta`: `0.0`

Decision: reject this block-level implementation too.

Reason: preserving document-vector parallelism was not enough to offset the
cost of launching thousands of small blocks and doing per-block shared-memory
reductions. It was roughly `103x` slower than the existing scorer on the smoke
case, so the experimental code was removed.

## Next

The two negative probes suggest the existing partial-score kernel is fast
because it exposes much finer-grained parallelism than the fused alternatives.
The next exact-rerank work should use profiler data from the existing kernels
instead of reducing launch granularity further. Plausible directions are:

- improve memory locality or vectorized loads in the existing token-dot kernel
- specialize fixed dim128/document-vector-count kernels if Mojo supports useful
  compile-time specialization
- avoid exact rerank work algorithmically through a measured shortlist policy
  rather than by making each exact score thread coarser
