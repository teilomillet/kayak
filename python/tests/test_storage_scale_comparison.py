from __future__ import annotations

import unittest

from kayak_bridge.storage_scale_comparison import (
    StorageScaleComparisonRow,
    StorageScaleComparisonSummary,
    _safe_ratio,
)


class StorageScaleComparisonTests(unittest.TestCase):
    def test_safe_ratio_handles_zero_denominator(self) -> None:
        self.assertIsNone(_safe_ratio(1.0, 0.0))
        self.assertEqual(_safe_ratio(6.0, 3.0), 2.0)

    def test_summary_serializes_rows(self) -> None:
        summary = StorageScaleComparisonSummary(
            dataset_id="dataset://tiny",
            model_name="tiny-model",
            family="tiny_family",
            slice_name="tiny_slice",
            primary_metric="ndcg",
            k=10,
            vector_dim=4,
            base_document_count=2,
            base_nominal_document_vector_count=3,
            base_zero_document_vector_count_filtered=1,
            base_zero_query_vector_count_filtered=0,
            protected_document_count=1,
            repeatable_distractor_source_count=1,
            inflation_policy="repeat_nonprotected_documents_with_unique_doc_ids",
            kayak_storage_engine="kayak",
            kayak_storage_format="packed_index_binary_le",
            lancedb_storage_engine="lancedb",
            lancedb_storage_engine_version="0.0.0",
            rows=(
                StorageScaleComparisonRow(
                    target_document_count=2,
                    actual_document_count=2,
                    scale_factor_vs_base=1.0,
                    duplicated_document_count=0,
                    stored_document_vector_count_total=6,
                    kayak_storage_byte_size=100,
                    kayak_bytes_per_document=50.0,
                    kayak_bytes_per_vector=16.6666666667,
                    kayak_build_seconds=0.1,
                    kayak_load_seconds=0.01,
                    kayak_search_seconds=0.001,
                    kayak_primary_value=1.0,
                    kayak_vector_payload_encoding="binary_le",
                    lancedb_storage_byte_size=120,
                    lancedb_bytes_per_document=60.0,
                    lancedb_bytes_per_vector=20.0,
                    lancedb_build_seconds=0.2,
                    lancedb_open_table_seconds=0.02,
                    lancedb_search_seconds=0.003,
                    lancedb_primary_value=1.0,
                    lancedb_storage_byte_ratio_vs_kayak=1.2,
                    lancedb_build_seconds_ratio_vs_kayak=2.0,
                    lancedb_search_seconds_ratio_vs_kayak=3.0,
                    lancedb_primary_ratio_vs_kayak=1.0,
                ),
            ),
        )

        payload = summary.to_json_ready()

        self.assertEqual(payload["dataset_id"], "dataset://tiny")
        self.assertEqual(payload["kayak_storage_format"], "packed_index_binary_le")
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["rows"][0]["lancedb_storage_byte_size"], 120)


if __name__ == "__main__":
    unittest.main()
