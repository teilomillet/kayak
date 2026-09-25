"""Check installed Laya/Jev integration without downloads or hosted inference.

Install Kayak, laya==0.3.20, typesafe-sdk==0.7.1, and transformers<5 in a test
environment. Run: python scripts/check_provider_adapters.py

Uses a tiny random Bert/Laya checkpoint and actual SDK clients over a controlled
transport. This establishes input/output conformance, not learned task quality.
"""

import asyncio
import importlib
import importlib.metadata
import json
import tempfile
from pathlib import Path
from typing import Protocol

from kayak import Choice, JudgmentQuestion, Noul, Score
from kayak.adapters import AsyncJev, Jev, Laya


class HTTPRequest(Protocol):
    @property
    def content(self) -> bytes: ...


def questions() -> dict[str, JudgmentQuestion]:
    return {
        "team": Choice(instructions="Which team?", criteria={"a": "Billing", "b": "Support"}),
        "duplicate": Noul(instructions="Was the customer charged twice?"),
        "impact": Score(instructions="Assess impact", criteria=["Low", "Medium", "High"]),
    }


def check_laya() -> None:
    # Dynamic imports keep these optional packages outside Kayak's normal type/install surface.
    laya = importlib.import_module("laya")
    common = importlib.import_module("laya.common")
    torch = importlib.import_module("torch")
    transformers = importlib.import_module("transformers")
    tokenizers = importlib.import_module("tokenizers")
    models = importlib.import_module("tokenizers.models")
    safetensors = importlib.import_module("safetensors.torch")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)
        config = transformers.BertConfig(
            vocab_size=6,
            hidden_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=128,
        )
        config.save_pretrained(path / "encoder")
        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=tokenizers.Tokenizer(
                models.WordLevel(
                    {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4, "hello": 5},
                    unk_token="[UNK]",
                )
            ),
            pad_token="[PAD]",
            unk_token="[UNK]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        tokenizer.save_pretrained(path / "tokenizer")
        model = common.DecisionModel(transformers.BertModel(config), head_layers=0, n_act=1)
        safetensors.save_file(model.state_dict(), path / "model.safetensors")
        (path / "rl_agent_config.json").write_text(
            json.dumps(
                {
                    "encoder": "unused/offline",
                    "head_layers": 0,
                    "max_len": 64,
                    "head_max_len": 32,
                }
            )
        )
        with laya.load(str(path), device="cpu") as agent:
            typed = questions()
            native = {
                name: question.model_dump(exclude_none=True) for name, question in typed.items()
            }
            for state in ("hello", "charged twice", "café"):
                expected = agent.predict(state, native)
                result = Laya(agent).judge(state=state, questions=typed)
                assert result.raw == expected
                assert result.input_tokens == expected["usage"]["input_tokens"]
                assert result.answers["duplicate"].probabilities is None
                assert agent.model is not None
    print("Laya: 3 mixed requests match direct tiny-model outputs; task quality not evaluated")


def check_jev() -> None:
    sdk = importlib.import_module("typesafe_sdk")
    http = importlib.import_module("httpx2")
    typed = questions()
    expected_questions = {name: item.model_dump(exclude_none=True) for name, item in typed.items()}
    body = {
        "model": "jev-controlled-fixture",
        "usage": {"input_tokens": 10, "output_tokens": 0},
        "answers": {
            "team": {
                "type": "choice",
                "choice": "a",
                "confidence": 0.4,
                "probabilities": {"a": 0.7, "b": 0.3},
            },
            "duplicate": {"type": "noul", "noul": 0.5},
            "impact": {
                "type": "score",
                "score": 1.0,
                "confidence": 0.2,
                "legend": {"0": "Low", "1": "Medium", "2": "High"},
                "probabilities": {"0": 0.3333, "1": 0.3333, "2": 0.3333},
            },
        },
        "future_metadata": {"preserved": True},
    }
    observed: list[object] = []

    def respond(request: HTTPRequest) -> object:
        payload = json.loads(request.content)
        assert payload["state"] == "charged twice"
        assert payload["questions"] == expected_questions
        assert payload["model"] == "jev-configured-model"
        observed.append(payload)
        return http.Response(200, json=body)

    with sdk.TypeSafeClient(
        api_key="fixture-key",
        base_url="https://provider.invalid",
        model="jev-configured-model",
        retry=sdk.RetryPolicy(max_retries=0),
        transport=http.MockTransport(respond),
    ) as client:
        result = Jev(client).judge(state="charged twice", questions=typed)
        assert result.raw == body
        assert result.answers["impact"].type == "score"
        assert result.answers["impact"].score == 1.0
        client.system_one(state="charged twice", questions=expected_questions)

    async def run() -> None:
        async with sdk.AsyncTypeSafeClient(
            api_key="fixture-key",
            base_url="https://provider.invalid",
            model="jev-configured-model",
            retry=sdk.RetryPolicy(max_retries=0),
            transport=http.MockTransport(respond),
        ) as client:
            actual = await AsyncJev(client).judge(state="charged twice", questions=typed)
            assert actual == result
            await client.system_one(state="charged twice", questions=expected_questions)

    asyncio.run(run())
    assert len(observed) == 4
    print("Jev: installed sync/async SDKs preserve requests and raw evidence; no hosted calls")


if __name__ == "__main__":
    print(
        {
            name: importlib.metadata.version(name)
            for name in (
                "laya",
                "typesafe-sdk",
                "torch",
                "transformers",
                "pydantic",
            )
        }
    )
    check_laya()
    check_jev()
