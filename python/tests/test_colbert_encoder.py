from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from kayak_bridge.colbert_encoder import (
    _trim_encoded_document_with_token_ids,
    _trim_zero_padded_rows,
)
from kayak_bridge.retrieval_task_builder import build_retrieval_subset_task


class ColbertEncoderTests(unittest.TestCase):
    def test_trim_zero_padded_rows_drops_trailing_zeros(self) -> None:
        tensor = torch.tensor(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        trimmed = _trim_zero_padded_rows(tensor)
        self.assertEqual(tuple(trimmed.shape), (2, 2))
        self.assertTrue(torch.equal(trimmed, tensor[:2]))

    def test_trim_encoded_document_token_ids_with_same_boundary(self) -> None:
        encoded = torch.tensor(
            [
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 0.0],
            ]
        )
        token_ids = torch.tensor([101, 202, 0])

        trimmed, trimmed_ids = _trim_encoded_document_with_token_ids(
            encoded,
            token_ids,
        )

        self.assertEqual(tuple(trimmed.shape), (2, 2))
        self.assertEqual(trimmed_ids, [101, 202])

    def test_retrieval_task_builder_can_include_document_token_ids(self) -> None:
        with (
            patch(
                "kayak_bridge.retrieval_task_builder."
                "encode_document_texts_with_token_ids",
                return_value=(
                    ([[1.0, 0.0], [0.0, 1.0]], [101, 202]),
                ),
            ),
            patch(
                "kayak_bridge.retrieval_task_builder.encode_query_text",
                return_value=[[1.0, 0.0]],
            ),
        ):
            task = build_retrieval_subset_task(
                family="unit",
                slice_name="tiny",
                why="unit test",
                primary_metric="ndcg",
                k=1,
                documents=[{"doc_id": "d1", "text": "alpha"}],
                queries=[
                    {
                        "query_id": "q1",
                        "text": "alpha?",
                        "relevant_doc_ids": ["d1"],
                    }
                ],
                model_name="unit-model",
                dataset_id="dataset://unit",
                include_document_token_ids=True,
            )

        self.assertEqual(task["document_token_ids"], "colbert_doc_tokenizer_input_ids")
        self.assertEqual(task["documents"][0]["token_ids"], [101, 202])


if __name__ == "__main__":
    unittest.main()
