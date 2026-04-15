from __future__ import annotations

import unittest
from unittest.mock import patch

from kayak_bridge.r2med_biology_query_variants import (
    R2MEDQueryVariantSpec,
    _load_variant_texts,
)


class R2MEDBiologyQueryVariantTests(unittest.TestCase):
    def test_empty_variant_text_falls_back_to_base_query(self) -> None:
        with patch(
            "kayak_bridge.r2med_biology_query_variants._base_query_texts",
            return_value={"1": "base query"},
        ), patch(
            "kayak_bridge.r2med_biology_query_variants.load_dataset",
            return_value=[{"id": "1", "hy_doc": "   "}],
        ):
            texts = _load_variant_texts(
                dataset_id="R2MED/Biology",
                spec=R2MEDQueryVariantSpec(
                    "lamer_gpt4",
                    "lamer",
                    "gpt4",
                    "hy_doc",
                ),
            )

        self.assertEqual(texts, {"1": "base query"})


if __name__ == "__main__":
    unittest.main()
