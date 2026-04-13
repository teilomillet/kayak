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
print("stage2:", result.stage2)

text_query = kayak.query(
    np.stack([dim128(0), dim128(0)]),
    text=(
        "Gugulethu township logo. founded in 1984 in a church "
        "longest serving employee artistic director"
    ),
)
text_index = kayak.documents(
    ["doc-context", "doc-answer"],
    [
        np.stack([dim128(0), dim128(0)]),
        np.stack([dim128(0), dim128(0) * np.float32(0.8) + dim128(1) * np.float32(0.2)]),
    ],
    texts=[
        "Gugulethu township logo emblem heritage schools history",
        (
            "Zama Dance School was founded in 1984 in a church and "
            "the longest serving employee is the artistic director."
        ),
    ],
).pack()
text_plan = kayak.exact_full_scan_search_plan(
    final_k=1,
    candidate_k=2,
    stage3_verifier=kayak.clause_text_stage3_verifier_operator(),
)
text_result = kayak.search_with_plan(text_query, text_index, text_plan)

print("text candidate hits:", text_result.candidate_stage.hits)
print("text stage2:", text_result.stage2)
print("text stage3 verifier:", text_result.stage3_verifier)
print("text final hits:", text_result.hits)
