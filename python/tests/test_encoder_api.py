from __future__ import annotations

import unittest

import numpy as np
import torch

import kayak
from kayak.encoders.registry import EncoderFactory, _ENCODER_FACTORIES


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


class _FakeDocTokenizer:
    def tensorize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(texts) == 1
        ids = torch.tensor(
            [[101, len(texts[0]), 102]],
            dtype=torch.long,
        )
        return ids, torch.ones_like(ids)


class _FakeCheckpointWithDocTokenizer:
    doc_tokenizer = _FakeDocTokenizer()

    def queryFromText(self, texts: list[str], *, to_cpu: bool) -> list[torch.Tensor]:
        return _FakeCheckpoint().queryFromText(texts, to_cpu=to_cpu)

    def doc(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        keep_dims: bool,
        to_cpu: bool,
    ) -> torch.Tensor:
        del attention_mask, to_cpu
        assert keep_dims is True
        rows = []
        for token_id in input_ids[0].tolist():
            rows.append([float(token_id), 0.0, 1.0])
        return torch.tensor([rows], dtype=torch.float32)


class EncoderApiTests(unittest.TestCase):
    def _restore_encoder_registry(
        self,
        factories: dict[str, EncoderFactory],
    ) -> None:
        _ENCODER_FACTORIES.clear()
        _ENCODER_FACTORIES.update(factories)

    def test_callable_text_encoder_can_bind_model_methods_directly(self) -> None:
        class _Model:
            def encode_query_tokens(self, text: str) -> list[list[float]]:
                return [[float(len(text)), 0.0], [0.0, 1.0]]

            def encode_document_tokens(self, text: str) -> list[list[float]]:
                return [[1.0, float(len(text))], [0.5, 0.5]]

        encoder = kayak.CallableLateTextEncoder.from_model(_Model())

        query = encoder.encode_query("hello")
        documents = encoder.encode_documents(["doc-a"], ["world"])

        np.testing.assert_allclose(
            query.as_vector_matrix(),
            np.array([[5.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            documents.token_matrices[0],
            np.array([[1.0, 5.0], [0.5, 0.5]], dtype=np.float32),
        )

    def test_callable_text_encoder_from_model_supports_custom_method_names(
        self,
    ) -> None:
        class _Model:
            def query_tokens(self, text: str) -> list[list[float]]:
                return [[float(len(text)), 1.0]]

            def document_tokens(self, text: str) -> list[list[float]]:
                return [[1.0, float(len(text))]]

        encoder = kayak.CallableLateTextEncoder.from_model(
            _Model(),
            query_method="query_tokens",
            document_method="document_tokens",
        )

        query = encoder.encode_query("hello")
        documents = encoder.encode_documents(["doc-a"], ["world"])

        np.testing.assert_allclose(
            query.as_vector_matrix(),
            np.array([[5.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(
            documents.token_matrices[0],
            np.array([[1.0, 5.0]], dtype=np.float32),
        )

    def test_callable_text_encoder_from_model_reports_missing_methods(self) -> None:
        class _Model:
            def encode_document_tokens(self, text: str) -> list[list[float]]:
                return [[1.0, float(len(text))]]

        with self.assertRaisesRegex(ValueError, "query method"):
            kayak.CallableLateTextEncoder.from_model(_Model())

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
        self.assertIsNone(documents.token_ids)

    def test_colbert_text_encoder_preserves_aligned_document_token_ids(self) -> None:
        encoder = kayak.ColBERTTextEncoder(
            checkpoint=_FakeCheckpointWithDocTokenizer(),
        )

        documents = encoder.encode_documents(["doc-a"], ["late interaction"])

        self.assertEqual(documents.vector_counts, (3,))
        np.testing.assert_array_equal(
            documents.token_ids[0],
            np.array([101, 16, 102], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            documents.pack().token_ids,
            np.array([101, 16, 102], dtype=np.int64),
        )

    def test_encoder_registry_supports_builtin_and_custom_factories(self) -> None:
        self.addCleanup(
            self._restore_encoder_registry,
            dict(_ENCODER_FACTORIES),
        )
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

    def test_encoder_registry_supports_model_backed_callable_variant(self) -> None:
        class _Model:
            def encode_query_tokens(self, text: str) -> list[list[float]]:
                return [[float(len(text)), 0.0]]

            def encode_document_tokens(self, text: str) -> list[list[float]]:
                return [[0.0, float(len(text))]]

        builtin = kayak.open_encoder("callable", model=_Model())

        self.assertIsInstance(builtin, kayak.CallableLateTextEncoder)
        np.testing.assert_allclose(
            builtin.encode_query("hello").as_vector_matrix(),
            np.array([[5.0, 0.0]], dtype=np.float32),
        )

    def test_open_encoder_unknown_kind_reports_available_kinds(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Available kinds: callable, colbert",
        ):
            kayak.open_encoder("missing-encoder")

    def test_open_encoder_non_string_kind_reports_type_error(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            'kayak.help\\("Encoders"\\)',
        ):
            kayak.open_encoder(object())  # type: ignore[arg-type]

    def test_encoder_registry_rejects_mixed_callable_and_model_arguments(
        self,
    ) -> None:
        class _Model:
            def encode_query_tokens(self, text: str) -> list[list[float]]:
                return [[float(len(text)), 0.0]]

            def encode_document_tokens(self, text: str) -> list[list[float]]:
                return [[0.0, float(len(text))]]

        with self.assertRaisesRegex(
            ValueError,
            "cannot be combined",
        ):
            kayak.open_encoder(
                "callable",
                model=_Model(),
                query_encoder=lambda text: [[len(text), 0.0]],
            )


if __name__ == "__main__":
    unittest.main()
