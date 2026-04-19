from __future__ import annotations

"""Smoke coverage for the main internal course sequence.

This module owns the deterministic teaching spine behind the integrated
"debug your broken RAG" notebook. It does not try to prove broad benchmark
claims; those remain in lesson-specific traces and judged-slice notes.
"""

import json
import unittest
from pathlib import Path

import numpy as np

import kayak
from kayak_bridge.judged_metrics import summarize_ranked_task


class CourseSequenceSmokeTests(unittest.TestCase):
    DIM = 64
    LIMIT_SMALL_TASK_PATH = Path(".cache/kayak/limit_small_real_subset/python_task.json")

    def setUp(self) -> None:
        self._token_to_index: dict[str, int] = {}

    def _token_vector(self, token: str) -> np.ndarray:
        index = self._token_to_index.setdefault(token, len(self._token_to_index))
        if index >= self.DIM:
            raise ValueError("Increase DIM for this smoke test.")
        vector = np.zeros(self.DIM, dtype=np.float32)
        vector[index] = np.float32(1.0)
        return vector

    def _encode_tokens(self, tokens: list[str]) -> np.ndarray:
        return np.stack([self._token_vector(token) for token in tokens])

    def _mean_vec(self, matrix: np.ndarray) -> np.ndarray:
        return np.mean(
            matrix,
            axis=0,
            dtype=np.float32,
            keepdims=True,
        ).astype(np.float32)

    def _build_chunked_index(
        self,
        documents: dict[str, list[str]],
        *,
        chunk_size: int,
    ) -> tuple[kayak.LateIndex, dict[str, str]]:
        chunk_ids: list[str] = []
        chunk_vectors: list[np.ndarray] = []
        parent_by_chunk: dict[str, str] = {}
        for doc_id, tokens in documents.items():
            matrix = self._encode_tokens(tokens)
            for chunk_index, start in enumerate(range(0, len(matrix), chunk_size)):
                chunk_id = f"{doc_id}::chunk{chunk_index}"
                chunk_ids.append(chunk_id)
                chunk_vectors.append(self._mean_vec(matrix[start : start + chunk_size]))
                parent_by_chunk[chunk_id] = doc_id
        return kayak.documents(chunk_ids, chunk_vectors).pack(), parent_by_chunk

    def _build_chunked_task_index(
        self,
        task: dict[str, object],
        *,
        chunk_size: int,
    ) -> tuple[kayak.LateIndex, dict[str, str]]:
        chunk_ids: list[str] = []
        chunk_vectors: list[np.ndarray] = []
        parent_by_chunk: dict[str, str] = {}
        for row in task["documents"]:
            matrix = np.asarray(row["vectors"], dtype=np.float32)
            for chunk_index, start in enumerate(range(0, len(matrix), chunk_size)):
                chunk_id = f"{row['doc_id']}::chunk{chunk_index}"
                chunk_ids.append(chunk_id)
                chunk_vectors.append(self._mean_vec(matrix[start : start + chunk_size]))
                parent_by_chunk[chunk_id] = row["doc_id"]
        return kayak.documents(
            chunk_ids,
            chunk_vectors,
        ).pack(), parent_by_chunk

    def _dedup_parent_docs(
        self,
        chunk_hits,
        parent_by_chunk: dict[str, str],
        *,
        k: int,
    ) -> tuple[str, ...]:
        ranked_doc_ids: list[str] = []
        seen: set[str] = set()
        for hit in chunk_hits:
            doc_id = parent_by_chunk[hit.doc_id]
            if doc_id in seen:
                continue
            seen.add(doc_id)
            ranked_doc_ids.append(doc_id)
            if len(ranked_doc_ids) >= k:
                break
        return tuple(ranked_doc_ids)

    def _load_limit_small_task(self) -> dict[str, object]:
        if not self.LIMIT_SMALL_TASK_PATH.exists():
            self.skipTest(
                "cached LIMIT-small task JSON is missing; rebuild it with "
                "python/scripts/build_task_json.py --dataset-key limit_small"
            )
        return json.loads(self.LIMIT_SMALL_TASK_PATH.read_text())

    def test_section1_exact_anchor_beats_one_vector_baseline(self) -> None:
        query_tokens = ["cancel", "subscription"]
        documents = {
            "doc-relevant": query_tokens + [f"noise-{index}" for index in range(20)],
            "doc-partial": ["cancel", "cancel", "cancel", "cancel"],
            "doc-other": ["billing", "invoice"],
        }

        exact_index = kayak.documents(
            list(documents),
            [self._encode_tokens(tokens) for tokens in documents.values()],
        ).pack()
        onevec_index = kayak.documents(
            list(documents),
            [self._mean_vec(self._encode_tokens(tokens)) for tokens in documents.values()],
        ).pack()

        exact_hits = kayak.search(
            kayak.query(self._encode_tokens(query_tokens)),
            exact_index,
            k=3,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        onevec_hits = kayak.search(
            kayak.query(self._mean_vec(self._encode_tokens(query_tokens))),
            onevec_index,
            k=3,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(exact_hits[0].doc_id, "doc-relevant")
        self.assertEqual(onevec_hits[0].doc_id, "doc-partial")
        self.assertGreater(exact_hits[0].score, exact_hits[1].score)

    def test_section2_chunking_can_help_when_evidence_is_local(self) -> None:
        documents = {
            "relevant": [
                "noise1",
                "noise2",
                "noise3",
                "noise4",
                "cancel",
                "subscription",
                "noise5",
                "noise6",
            ],
            "partial": ["cancel", "noise7", "cancel", "noise8"],
            "other": ["invoice", "billing"],
        }
        query = kayak.query(self._mean_vec(self._encode_tokens(["cancel", "subscription"])))
        onevec_index = kayak.documents(
            list(documents),
            [
                self._mean_vec(self._encode_tokens(tokens))
                for tokens in documents.values()
            ],
        ).pack()
        chunk_index, parent_by_chunk = self._build_chunked_index(documents, chunk_size=2)

        onevec_hits = kayak.search(
            query,
            onevec_index,
            k=3,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        chunk_hits = kayak.search(
            query,
            chunk_index,
            k=chunk_index.document_count,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(onevec_hits[0].doc_id, "partial")
        self.assertEqual(
            self._dedup_parent_docs(chunk_hits, parent_by_chunk, k=3)[0],
            "relevant",
        )

    def test_section2_chunking_can_hurt_when_evidence_spans_boundaries(self) -> None:
        documents = {
            "relevant": ["alpha", "noise1", "beta", "noise2", "gamma", "noise3"],
            "partial": ["alpha", "beta", "noise4", "noise5"],
            "other": ["invoice", "billing"],
        }
        exact_index = kayak.documents(
            list(documents),
            [self._encode_tokens(tokens) for tokens in documents.values()],
        ).pack()
        chunk_index, parent_by_chunk = self._build_chunked_index(documents, chunk_size=2)

        exact_hits = kayak.search(
            kayak.query(self._encode_tokens(["alpha", "beta", "gamma"])),
            exact_index,
            k=3,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        chunk_hits = kayak.search(
            kayak.query(self._mean_vec(self._encode_tokens(["alpha", "beta", "gamma"]))),
            chunk_index,
            k=chunk_index.document_count,
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(exact_hits[0].doc_id, "relevant")
        self.assertEqual(
            self._dedup_parent_docs(chunk_hits, parent_by_chunk, k=3)[0],
            "partial",
        )

    def test_section3_limit_small_exact_stays_above_compressed_baselines(self) -> None:
        task = self._load_limit_small_task()
        exact_index = kayak.documents(
            [row["doc_id"] for row in task["documents"]],
            [np.asarray(row["vectors"], dtype=np.float32) for row in task["documents"]],
            texts=[row["text"] for row in task["documents"]],
        ).pack()
        onevec_index = kayak.documents(
            [row["doc_id"] for row in task["documents"]],
            [
                self._mean_vec(np.asarray(row["vectors"], dtype=np.float32))
                for row in task["documents"]
            ],
            texts=[row["text"] for row in task["documents"]],
        ).pack()
        chunk16_index, parent_by_chunk = self._build_chunked_task_index(
            task,
            chunk_size=16,
        )

        exact_ranked = []
        onevec_ranked = []
        chunk16_ranked = []
        proxy_ranked = []

        for row in task["queries"]:
            query_matrix = np.asarray(row["vectors"], dtype=np.float32)
            exact_query = kayak.query(query_matrix, text=row["text"])
            onevec_query = kayak.query(self._mean_vec(query_matrix), text=row["text"])

            exact_hits = kayak.search(
                exact_query,
                exact_index,
                k=task["k"],
                backend=kayak.NUMPY_REFERENCE_BACKEND,
            )
            onevec_hits = kayak.search(
                onevec_query,
                onevec_index,
                k=task["k"],
                backend=kayak.NUMPY_REFERENCE_BACKEND,
            )
            chunk16_hits = kayak.search(
                onevec_query,
                chunk16_index,
                k=chunk16_index.document_count,
                backend=kayak.NUMPY_REFERENCE_BACKEND,
            )
            proxy_hits = kayak.search_with_plan(
                exact_query,
                exact_index,
                kayak.document_proxy_search_plan(
                    final_k=task["k"],
                    candidate_k=task["k"],
                ),
                backend=kayak.NUMPY_REFERENCE_BACKEND,
            ).hits

            exact_ranked.append(tuple(hit.doc_id for hit in exact_hits))
            onevec_ranked.append(tuple(hit.doc_id for hit in onevec_hits))
            chunk16_ranked.append(
                self._dedup_parent_docs(chunk16_hits, parent_by_chunk, k=task["k"])
            )
            proxy_ranked.append(tuple(hit.doc_id for hit in proxy_hits))

        exact_summary = summarize_ranked_task(
            task=task,
            ranked_doc_ids_by_query=exact_ranked,
        )
        onevec_summary = summarize_ranked_task(
            task=task,
            ranked_doc_ids_by_query=onevec_ranked,
        )
        chunk16_summary = summarize_ranked_task(
            task=task,
            ranked_doc_ids_by_query=chunk16_ranked,
        )
        proxy_summary = summarize_ranked_task(
            task=task,
            ranked_doc_ids_by_query=proxy_ranked,
        )

        # The course claim is qualitative: exact late interaction is the
        # stronger diagnosis surface on this cached slice. Exact decimals are
        # recorded in the lesson note; the smoke path keeps only the stable gap.
        self.assertEqual(task["slice_name"], "limit_small_real_subset")
        self.assertGreater(exact_summary.primary_value, onevec_summary.primary_value + 0.1)
        self.assertGreater(exact_summary.primary_value, chunk16_summary.primary_value + 0.1)
        self.assertGreater(exact_summary.primary_value, proxy_summary.primary_value + 0.1)
        self.assertGreater(
            exact_summary.mean_recall_at_k,
            proxy_summary.mean_recall_at_k + 0.1,
        )

    def test_section4_shortlist_can_drop_oracle_before_rerank(self) -> None:
        query = kayak.query(self._encode_tokens(["cancel", "subscription"]))
        index = kayak.documents(
            ["doc-partial", "doc-relevant", "doc-other"],
            [
                self._encode_tokens(["cancel", "cancel", "cancel"]),
                self._encode_tokens(["cancel", "subscription"]),
                self._encode_tokens(["invoice"]),
            ],
        ).pack()

        exact_result = kayak.search_with_plan(
            query,
            index,
            kayak.exact_full_scan_search_plan(final_k=1, candidate_k=3),
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        narrow_result = kayak.search_with_plan(
            query,
            index,
            kayak.document_proxy_search_plan(
                final_k=1,
                candidate_k=1,
                query_vector_budget=1,
                document_vector_budget=1,
            ),
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        wider_result = kayak.search_with_plan(
            query,
            index,
            kayak.document_proxy_search_plan(
                final_k=1,
                candidate_k=2,
                query_vector_budget=1,
                document_vector_budget=1,
            ),
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual([hit.doc_id for hit in exact_result.hits], ["doc-relevant"])
        self.assertEqual(narrow_result.candidate_stage.candidate_doc_ids, ("doc-partial",))
        self.assertEqual([hit.doc_id for hit in narrow_result.hits], ["doc-partial"])
        self.assertEqual(
            wider_result.candidate_stage.candidate_doc_ids,
            ("doc-partial", "doc-relevant"),
        )
        self.assertEqual([hit.doc_id for hit in wider_result.hits], ["doc-relevant"])
        self.assertEqual(wider_result.stage2.stage_name, "exact_late_interaction")


if __name__ == "__main__":
    unittest.main()
