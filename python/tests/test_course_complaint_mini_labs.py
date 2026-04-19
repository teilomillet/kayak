from __future__ import annotations

"""Smoke checks for the judged-slice complaint-first mini labs."""

import json
import unittest
from pathlib import Path

import numpy as np

import kayak
from kayak_bridge.judged_metrics import summarize_ranked_query
from kayak_bridge.judged_metrics import summarize_ranked_task


class CourseComplaintMiniLabTests(unittest.TestCase):
    CACHE_ROOT = Path(".cache/kayak")

    def _load_task(self, task_name: str) -> dict[str, object]:
        path = self.CACHE_ROOT / task_name / "python_task.json"
        if not path.exists():
            self.skipTest(
                f"cached task JSON is missing for {task_name}; rebuild the local cache first"
            )
        return json.loads(path.read_text())

    def _mean_vec(self, matrix: np.ndarray) -> np.ndarray:
        return np.mean(
            matrix,
            axis=0,
            dtype=np.float32,
            keepdims=True,
        ).astype(np.float32)

    def _build_exact_index(self, task: dict[str, object]) -> kayak.LateIndex:
        return kayak.documents(
            [row["doc_id"] for row in task["documents"]],
            [np.asarray(row["vectors"], dtype=np.float32) for row in task["documents"]],
            texts=[row["text"] for row in task["documents"]],
        ).pack()

    def _build_onevec_index(self, task: dict[str, object]) -> kayak.LateIndex:
        return kayak.documents(
            [row["doc_id"] for row in task["documents"]],
            [
                self._mean_vec(np.asarray(row["vectors"], dtype=np.float32))
                for row in task["documents"]
            ],
            texts=[row["text"] for row in task["documents"]],
        ).pack()

    def test_bright_stackoverflow_answer_bearing_page_beats_one_vector(self) -> None:
        task = self._load_task("bright_stackoverflow_real_subset")
        query_row = task["queries"][3]
        relevant_doc_ids = tuple(query_row["relevant_doc_ids"])
        query_matrix = np.asarray(query_row["vectors"], dtype=np.float32)

        exact_hits = kayak.search(
            kayak.query(query_matrix, text=query_row["text"]),
            self._build_exact_index(task),
            k=task["k"],
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        onevec_hits = kayak.search(
            kayak.query(self._mean_vec(query_matrix), text=query_row["text"]),
            self._build_onevec_index(task),
            k=task["k"],
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        exact_rank = tuple(hit.doc_id for hit in exact_hits)
        onevec_rank = tuple(hit.doc_id for hit in onevec_hits)
        exact_summary = summarize_ranked_query(
            ranked_doc_ids=exact_rank,
            relevant_doc_ids=relevant_doc_ids,
            k=task["k"],
        )
        onevec_summary = summarize_ranked_query(
            ranked_doc_ids=onevec_rank,
            relevant_doc_ids=relevant_doc_ids,
            k=task["k"],
        )

        self.assertEqual(
            exact_rank[0],
            "Python_pandas_functions_with_style/General_Function_5_2.txt",
        )
        self.assertEqual(exact_summary.recall_at_k, 1.0)
        self.assertAlmostEqual(exact_summary.ndcg_at_k, 0.7904, places=4)
        self.assertEqual(onevec_summary.recall_at_k, 0.0)
        self.assertEqual(onevec_summary.ndcg_at_k, 0.0)
        self.assertTrue(
            all(doc_id not in relevant_doc_ids for doc_id in onevec_rank[:5])
        )

    def test_bright_stackoverflow_wider_candidate_window_recovers_answer_page(
        self,
    ) -> None:
        task = self._load_task("bright_stackoverflow_real_subset")
        query_row = task["queries"][3]
        relevant_doc_ids = tuple(query_row["relevant_doc_ids"])
        exact_index = self._build_exact_index(task)
        query = kayak.query(
            np.asarray(query_row["vectors"], dtype=np.float32),
            text=query_row["text"],
        )

        narrow_result = kayak.search_with_plan(
            query,
            exact_index,
            kayak.document_proxy_search_plan(
                final_k=task["k"],
                candidate_k=10,
                query_vector_budget=8,
                document_vector_budget=8,
            ),
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )
        wide_result = kayak.search_with_plan(
            query,
            exact_index,
            kayak.document_proxy_search_plan(
                final_k=task["k"],
                candidate_k=20,
                query_vector_budget=8,
                document_vector_budget=8,
            ),
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        narrow_rank = tuple(hit.doc_id for hit in narrow_result.hits)
        wide_rank = tuple(hit.doc_id for hit in wide_result.hits)
        narrow_summary = summarize_ranked_query(
            ranked_doc_ids=narrow_rank,
            relevant_doc_ids=relevant_doc_ids,
            k=task["k"],
        )
        wide_summary = summarize_ranked_query(
            ranked_doc_ids=wide_rank,
            relevant_doc_ids=relevant_doc_ids,
            k=task["k"],
        )

        for doc_id in relevant_doc_ids:
            self.assertNotIn(doc_id, narrow_result.candidate_stage.candidate_doc_ids)
        self.assertIn(
            "Python_pandas_functions_with_style/General_Function_5_2.txt",
            wide_result.candidate_stage.candidate_doc_ids,
        )
        self.assertEqual(
            wide_rank[0],
            "Python_pandas_functions_with_style/General_Function_5_2.txt",
        )
        self.assertEqual(narrow_summary.recall_at_k, 0.0)
        self.assertEqual(narrow_summary.ndcg_at_k, 0.0)
        self.assertEqual(wide_summary.recall_at_k, 0.5)
        self.assertAlmostEqual(wide_summary.ndcg_at_k, 0.6131, places=4)

    def test_long_document_family_surface_is_harsher_but_not_uniform(self) -> None:
        task_names = [
            "lemb_narrativeqa_real_subset",
            "browsecomp_plus_real_subset",
            "r2med_biology_real_subset",
        ]
        summaries: dict[str, tuple[float, float, float, float]] = {}

        for task_name in task_names:
            task = self._load_task(task_name)
            exact_index = self._build_exact_index(task)
            onevec_index = self._build_onevec_index(task)
            exact_ranked = []
            onevec_ranked = []
            proxy_ranked = []

            for row in task["queries"]:
                query_matrix = np.asarray(row["vectors"], dtype=np.float32)
                exact_query = kayak.query(query_matrix, text=row["text"])
                onevec_query = kayak.query(
                    self._mean_vec(query_matrix),
                    text=row["text"],
                )
                exact_ranked.append(
                    tuple(
                        hit.doc_id
                        for hit in kayak.search(
                            exact_query,
                            exact_index,
                            k=task["k"],
                            backend=kayak.NUMPY_REFERENCE_BACKEND,
                        )
                    )
                )
                onevec_ranked.append(
                    tuple(
                        hit.doc_id
                        for hit in kayak.search(
                            onevec_query,
                            onevec_index,
                            k=task["k"],
                            backend=kayak.NUMPY_REFERENCE_BACKEND,
                        )
                    )
                )
                proxy_ranked.append(
                    tuple(
                        hit.doc_id
                        for hit in kayak.search_with_plan(
                            exact_query,
                            exact_index,
                            kayak.document_proxy_search_plan(
                                final_k=task["k"],
                                candidate_k=task["k"],
                                query_vector_budget=8,
                                document_vector_budget=8,
                            ),
                            backend=kayak.NUMPY_REFERENCE_BACKEND,
                        ).hits
                    )
                )

            exact_summary = summarize_ranked_task(
                task=task,
                ranked_doc_ids_by_query=exact_ranked,
            )
            onevec_summary = summarize_ranked_task(
                task=task,
                ranked_doc_ids_by_query=onevec_ranked,
            )
            proxy_summary = summarize_ranked_task(
                task=task,
                ranked_doc_ids_by_query=proxy_ranked,
            )
            doc_lengths = [
                len(row["vectors"])
                for row in task["documents"]
            ]
            doc_mean = sum(doc_lengths) / float(len(doc_lengths))
            summaries[task_name] = (
                exact_summary.primary_value,
                onevec_summary.primary_value,
                proxy_summary.primary_value,
                doc_mean,
            )

        narrativeqa_exact, narrativeqa_onevec, narrativeqa_proxy, narrativeqa_mean = (
            summaries["lemb_narrativeqa_real_subset"]
        )
        browsecomp_exact, browsecomp_onevec, browsecomp_proxy, browsecomp_mean = (
            summaries["browsecomp_plus_real_subset"]
        )
        r2med_exact, r2med_onevec, r2med_proxy, r2med_mean = (
            summaries["r2med_biology_real_subset"]
        )

        self.assertAlmostEqual(narrativeqa_mean, 180.0, places=2)
        self.assertGreater(narrativeqa_exact, narrativeqa_onevec)
        self.assertGreater(narrativeqa_onevec, narrativeqa_proxy)
        self.assertGreater(browsecomp_mean, 170.0)
        self.assertGreater(browsecomp_exact, browsecomp_onevec)
        self.assertGreater(browsecomp_exact, browsecomp_proxy)
        self.assertLess(r2med_mean, narrativeqa_mean)
        self.assertGreater(
            narrativeqa_exact - narrativeqa_proxy,
            r2med_exact - r2med_proxy,
        )

        task = self._load_task("lemb_narrativeqa_real_subset")
        query_row = task["queries"][2]
        exact_index = self._build_exact_index(task)
        onevec_index = self._build_onevec_index(task)
        query_matrix = np.asarray(query_row["vectors"], dtype=np.float32)
        relevant_doc_id = str(query_row["relevant_doc_ids"][0])

        exact_rank = tuple(
            hit.doc_id
            for hit in kayak.search(
                kayak.query(query_matrix, text=query_row["text"]),
                exact_index,
                k=task["k"],
                backend=kayak.NUMPY_REFERENCE_BACKEND,
            )
        )
        onevec_rank = tuple(
            hit.doc_id
            for hit in kayak.search(
                kayak.query(self._mean_vec(query_matrix), text=query_row["text"]),
                onevec_index,
                k=task["k"],
                backend=kayak.NUMPY_REFERENCE_BACKEND,
            )
        )
        proxy_result = kayak.search_with_plan(
            kayak.query(query_matrix, text=query_row["text"]),
            exact_index,
            kayak.document_proxy_search_plan(
                final_k=task["k"],
                candidate_k=task["k"],
                query_vector_budget=8,
                document_vector_budget=8,
            ),
            backend=kayak.NUMPY_REFERENCE_BACKEND,
        )

        self.assertEqual(relevant_doc_id, "doc_157")
        self.assertEqual(exact_rank[0], relevant_doc_id)
        self.assertEqual(onevec_rank.index(relevant_doc_id) + 1, 5)
        self.assertNotIn(
            relevant_doc_id,
            proxy_result.candidate_stage.candidate_doc_ids,
        )


if __name__ == "__main__":
    unittest.main()
