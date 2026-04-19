from __future__ import annotations

import tempfile
import unittest

import torch

from kayak_bridge.hf_dense_baseline import HFDenseTextBatcher


class _FakeTokenizer:
    def __init__(self) -> None:
        self.model_max_length = 16
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        texts,
        *,
        padding,
        truncation,
        max_length,
        return_tensors,
    ):
        self.calls.append(
            {
                "texts": list(texts),
                "padding": padding,
                "truncation": truncation,
                "max_length": max_length,
                "return_tensors": return_tensors,
            }
        )
        rows = []
        for text in texts:
            token_values = [float(len(token)) for token in text.split()]
            rows.append(token_values[: max_length or len(token_values)])
        width = max((len(row) for row in rows), default=0)
        padded_rows = [row + [0.0] * (width - len(row)) for row in rows]
        attention_rows = [
            [1] * len(row) + [0] * (width - len(row))
            for row in rows
        ]
        return {
            "input_ids": torch.tensor(padded_rows, dtype=torch.float32),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.float32),
        }


class _FakeModelOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **tokenized):
        self.calls += 1
        input_ids = tokenized["input_ids"]
        last_hidden_state = torch.stack(
            (
                input_ids,
                input_ids + 1.0,
                input_ids + 2.0,
            ),
            dim=-1,
        )
        return _FakeModelOutput(last_hidden_state)


class HFDenseBaselineTests(unittest.TestCase):
    def test_embed_texts_caches_and_mean_pools(self) -> None:
        tokenizer = _FakeTokenizer()
        model = _FakeModel()
        with tempfile.TemporaryDirectory() as temp_dir:
            batcher = HFDenseTextBatcher(
                "fake-model",
                tokenizer=tokenizer,
                model=model,
                cache_dir=temp_dir,
                max_length=8,
            )

            first = batcher.embed_texts(["aa bbbb", "c"], "fake-model", 8)
            second = batcher.embed_texts(["aa bbbb", "c"], "fake-model", 8)

            self.assertEqual(model.calls, 1)
            self.assertEqual(len(tokenizer.calls), 1)
            self.assertEqual(first, second)
            self.assertEqual(first[0], [3.0, 4.0, 5.0])
            self.assertEqual(first[1], [1.0, 2.0, 3.0])

    def test_embed_texts_rejects_mismatched_model_name(self) -> None:
        batcher = HFDenseTextBatcher(
            "expected-model",
            tokenizer=_FakeTokenizer(),
            model=_FakeModel(),
            cache_dir=None,
        )

        with self.assertRaisesRegex(ValueError, "mismatched model_name"):
            batcher.embed_texts(["alpha"], "other-model", 4)


if __name__ == "__main__":
    unittest.main()
