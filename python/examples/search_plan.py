from __future__ import annotations

import numpy as np

import kayak


def dim128(index: int) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    vector[index] = 1.0
    return vector


query = kayak.query(np.stack([dim128(0), dim128(1)]))
index = kayak.documents(
    ["doc-b", "doc-a", "doc-c"],
    [
        np.stack([dim128(0), dim128(0)]),
        np.stack([dim128(0), dim128(1)]),
        np.stack([dim128(1), dim128(1)]),
    ],
).pack()

plan = kayak.document_proxy_search_plan(final_k=1, candidate_k=2)
result = kayak.search_with_plan(query, index, plan)

print("candidate hits:", result.candidate_stage.hits)
print("final hits:", result.hits)
print("candidate stage:", result.candidate_stage.profile)
print("exact stage:", result.exact_stage)
