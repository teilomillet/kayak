from __future__ import annotations

import os
from functools import lru_cache
from typing import Final

import numpy as np

import kayak
from kayak_bridge.mojo_exact_cpu import _detect_mojo_command
from ordeal import ChaosTest, always, invariant, rule


def _dim128_vector(*entries: tuple[int, float]) -> np.ndarray:
    vector = np.zeros(128, dtype=np.float32)
    for index, value in entries:
        vector[index] = np.float32(value)
    return vector


FIXTURE_SPECS: Final[dict[str, dict[str, object]]] = {
    "primary": {
        "doc_ids": ("doc-a", "doc-b", "doc-c"),
        "query": (((0, 1.0),), ((1, 1.0),)),
        "documents": (
            (((0, 1.0),), ((1, 1.0),)),
            (((0, 1.0),), ((0, 0.5), (1, 0.5))),
            (((1, 1.0),), ((0, 1.0),)),
        ),
        "expected_scores": (2.0, 1.5, 2.0),
        "expected_hits": ("doc-a", "doc-c"),
        "expected_vector_counts": (2, 2, 2),
    },
    "secondary": {
        "doc_ids": ("doc-x", "doc-y", "doc-z"),
        "query": (((0, 1.0),), ((1, 1.0),), ((2, 1.0),)),
        "documents": (
            (((0, 1.0),), ((1, 1.0),), ((2, 1.0),)),
            (((0, 1.0),), ((1, 1.0),)),
            (((2, 1.0),),),
        ),
        "expected_scores": (3.0, 2.0, 1.0),
        "expected_hits": ("doc-x", "doc-y"),
        "expected_vector_counts": (3, 2, 1),
    },
}


def _vector_matrix(spec: tuple[tuple[tuple[int, float], ...], ...]) -> np.ndarray:
    return np.stack([_dim128_vector(*entries) for entries in spec])


@lru_cache(maxsize=1)
def _mojo_backend_enabled() -> bool:
    # The default Ordeal run should stay lightweight and require only Python.
    if os.environ.get("KAYAK_ORDEAL_ENABLE_MOJO") != "1":
        return False

    try:
        _detect_mojo_command()
    except RuntimeError:
        return False
    return True


class _FixtureCase:
    def __init__(self, fixture_name: str) -> None:
        spec = FIXTURE_SPECS[fixture_name]
        query = kayak.query(_vector_matrix(spec["query"]))
        prefix_query = kayak.query(query.as_vector_matrix()[:-1])
        documents = kayak.documents(
            spec["doc_ids"],
            [_vector_matrix(document) for document in spec["documents"]],
        )
        index = documents.pack()

        self.fixture_name = fixture_name
        self.query = query
        self.prefix_query = prefix_query
        self.query_batch = kayak.query_batch(
            [query.as_vector_matrix(), prefix_query.as_vector_matrix()]
        )
        self.index = index
        self.flat_query = query.to_layout("flat_dim128")
        self.flat_query_batch = self.query_batch.to_layout("flat_dim128")
        self.hybrid_index = index.to_layout("hybrid_flat_dim128")
        self.expected_scores = np.array(spec["expected_scores"], dtype=np.float32)
        self.expected_prefix_scores = kayak.maxsim(
            self.prefix_query,
            self.index,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        ).numpy()
        self.expected_hits = tuple(spec["expected_hits"])
        self.expected_vector_counts = tuple(spec["expected_vector_counts"])


def _fixture_case(fixture_name: str) -> _FixtureCase:
    return _FixtureCase(fixture_name)


class KayakPythonSdkChaos(ChaosTest):
    """Stateful Ordeal exploration for Python SDK layout and scoring invariants."""

    def __init__(self) -> None:
        super().__init__()
        self.fixture_name = "primary"

    @rule()
    def use_primary_fixture(self) -> None:
        self.fixture_name = "primary"

    @rule()
    def use_secondary_fixture(self) -> None:
        self.fixture_name = "secondary"

    @rule()
    def score_packed_numpy(self) -> None:
        case = _fixture_case(self.fixture_name)
        scores = kayak.maxsim(
            case.query,
            case.index,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        ).numpy()
        always(bool(np.all(np.isfinite(scores))), "numpy packed scores finite")
        always(
            bool(np.allclose(scores, case.expected_scores)),
            f"{case.fixture_name} packed numpy scores match fixture",
        )

    @rule()
    def score_hybrid_numpy(self) -> None:
        case = _fixture_case(self.fixture_name)
        scores = kayak.maxsim(
            case.flat_query,
            case.hybrid_index,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        ).numpy()
        always(bool(np.all(np.isfinite(scores))), "numpy hybrid scores finite")
        always(
            bool(np.allclose(scores, case.expected_scores)),
            f"{case.fixture_name} hybrid numpy scores match fixture",
        )

    @rule()
    def search_numpy(self) -> None:
        case = _fixture_case(self.fixture_name)
        hits = kayak.search(
            case.query,
            case.index,
            k=2,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        always(
            tuple(hit.doc_id for hit in hits) == case.expected_hits,
            f"{case.fixture_name} numpy top-k stays stable",
        )

    @rule()
    def score_batch_numpy(self) -> None:
        case = _fixture_case(self.fixture_name)
        scores_batch = kayak.maxsim_batch(
            case.query_batch,
            case.index,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        always(
            bool(
                np.allclose(scores_batch[0].numpy(), case.expected_scores)
            ),
            f"{case.fixture_name} batch numpy primary scores match fixture",
        )
        always(
            bool(
                np.allclose(
                    scores_batch[1].numpy(), case.expected_prefix_scores
                )
            ),
            f"{case.fixture_name} batch numpy prefix scores match fixture",
        )

    @rule()
    def score_mojo_exact_cpu(self) -> None:
        if not _mojo_backend_enabled():
            return

        case = _fixture_case(self.fixture_name)
        scores = kayak.maxsim(
            case.flat_query,
            case.hybrid_index,
            backend=kayak.MOJO_EXACT_CPU_BACKEND,
        ).numpy()
        always(bool(np.all(np.isfinite(scores))), "mojo hybrid scores finite")
        always(
            bool(np.allclose(scores, case.expected_scores)),
            f"{case.fixture_name} hybrid mojo scores match fixture",
        )

    @rule()
    def score_batch_mojo_exact_cpu(self) -> None:
        if not _mojo_backend_enabled():
            return

        case = _fixture_case(self.fixture_name)
        scores_batch = kayak.maxsim_batch(
            case.flat_query_batch,
            case.hybrid_index,
            backend=kayak.MOJO_EXACT_CPU_BACKEND,
        )
        always(
            bool(
                np.allclose(scores_batch[0].numpy(), case.expected_scores)
            ),
            f"{case.fixture_name} batch mojo primary scores match fixture",
        )
        always(
            bool(
                np.allclose(
                    scores_batch[1].numpy(), case.expected_prefix_scores
                )
            ),
            f"{case.fixture_name} batch mojo prefix scores match fixture",
        )

    @invariant()
    def shape_invariants_hold(self) -> None:
        case = _fixture_case(self.fixture_name)
        assert case.query.vector_dim == 128
        assert case.index.vector_dim == 128
        assert tuple(case.index.vector_counts) == case.expected_vector_counts
        assert case.index.total_vector_count == sum(case.expected_vector_counts)

    @invariant()
    def backend_contract_holds(self) -> None:
        numpy_info = kayak.backend_info(kayak.NUMPY_REFERENCE_BACKEND)
        assert numpy_info.available
        assert not numpy_info.requires_mojo
        assert "nested" in numpy_info.query_layouts
        assert "packed" in numpy_info.index_layouts


TestKayakPythonSdkChaos = KayakPythonSdkChaos.TestCase
