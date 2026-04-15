from __future__ import annotations

import unittest

import numpy as np
import torch

import kayak


class _FakeCheckpoint:
    def queryFromText(self, texts: list[str], *, to_cpu: bool) -> list[torch.Tensor]:
        del to_cpu
        assert len(texts) == 1
        text = texts[0]
        return [
            torch.tensor(
                [
                    [float(len(text)), 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            )
        ]

    def docFromText(self, texts: list[str], *, to_cpu: bool) -> list[torch.Tensor]:
        del to_cpu
        assert len(texts) == 1
        text = texts[0]
        return [
            torch.tensor(
                [
                    [1.0, float(len(text)), 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )
        ]


class EncoderApiTests(unittest.TestCase):
    def test_callable_text_encoder_emits_public_late_objects(self) -> None:
        encoder = kayak.CallableLateTextEncoder(
            query_encoder=lambda text: [[len(text), 0.0], [0.0, 1.0]],
            document_encoder=lambda text: [[1.0, len(text)], [0.5, 0.5]],
        )

        query = encoder.encode_query("hello")
        documents = encoder.encode_documents(
            ["doc-a", "doc-b"],
            ["hello", "world"],
        )

        self.assertEqual(query.text, "hello")
        self.assertEqual(query.vector_count, 2)
        np.testing.assert_allclose(
            query.as_vector_matrix(),
            np.array([[5.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        self.assertEqual(documents.doc_ids, ("doc-a", "doc-b"))
        self.assertEqual(documents.texts, ("hello", "world"))
        self.assertEqual(documents.vector_counts, (2, 2))

    def test_colbert_text_encoder_accepts_injected_checkpoint(self) -> None:
        encoder = kayak.ColBERTTextEncoder(checkpoint=_FakeCheckpoint())

        query = encoder.encode_query("kayak")
        documents = encoder.encode_documents(
            ["doc-a"],
            ["late interaction"],
        )

        self.assertEqual(query.text, "kayak")
        self.assertEqual(query.vector_dim, 3)
        self.assertEqual(query.vector_count, 2)
        np.testing.assert_allclose(
            query.as_vector_matrix(),
            np.array([[5.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        )
        self.assertEqual(documents.doc_ids, ("doc-a",))
        self.assertEqual(documents.texts, ("late interaction",))
        np.testing.assert_allclose(
            documents.token_matrices[0],
            np.array([[1.0, 16.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        )

    def test_encoder_registry_supports_builtin_and_custom_factories(self) -> None:
        builtin = kayak.open_encoder(
            "callable",
            query_encoder=lambda text: [[len(text), 0.0]],
            document_encoder=lambda text: [[0.0, len(text)]],
        )
        self.assertIsInstance(builtin, kayak.CallableLateTextEncoder)

        class _CustomEncoder(kayak.CallableLateTextEncoder):
            pass

        kayak.register_encoder("custom-callable", _CustomEncoder, replace=True)
        custom = kayak.open_encoder(
            "custom-callable",
            query_encoder=lambda text: [[len(text), 1.0]],
            document_encoder=lambda text: [[1.0, len(text)]],
        )
        self.assertIsInstance(custom, _CustomEncoder)


if __name__ == "__main__":
    unittest.main()
