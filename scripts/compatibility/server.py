"""Run an installed Kayak server with controlled decisions and bounded test blocking."""

import argparse
import json
import socket
import sys
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from time import monotonic, sleep

import uvicorn

import kayak
from kayak import Choice, ChoiceAnswer, DecisionResult, InferenceError, InputError, ModelClosedError
from kayak.server import create_app


class Arguments(argparse.Namespace):
    fixtures: Path
    work: Path


class FixtureModel:
    def __init__(self, result: DecisionResult, work: Path) -> None:
        self.result = result
        self.info = result.model
        self.work = work

    def decide(self, *, state: str, questions: Mapping[str, Choice]) -> DecisionResult:
        if state == "test:invalid":
            raise InputError("controlled input failure")
        if state == "test:failed":
            raise InferenceError("private fixture detail")
        if state == "test:closed":
            raise ModelClosedError("controlled closed model")
        if state == "test:block":
            deadline = monotonic() + 15
            while not (self.work / "release").exists():
                if monotonic() > deadline:
                    raise InferenceError("test controller did not release inference")
                sleep(0.01)
        if list(questions) == ["noul", "score"]:
            return DecisionResult(
                model=self.info,
                answers={
                    name: ChoiceAnswer(
                        choice=next(iter(question.criteria)),
                        scores=dict.fromkeys(question.criteria, 0.0),
                        probabilities=dict.fromkeys(question.criteria, 1 / len(question.criteria)),
                    )
                    for name, question in questions.items()
                },
                input_tokens=self.result.input_tokens,
            )
        if list(questions) == ["rank"]:
            return DecisionResult(
                model=self.info,
                answers={"rank": self.result.answers["department"]},
                input_tokens=self.result.input_tokens,
            )
        return self.result

    def close(self) -> None:
        # Uvicorn re-raises SIGTERM after cleanup; checks after run() may never execute.
        assert not {"torch", "transformers", "huggingface_hub"} & sys.modules.keys()
        (self.work / "closed").touch()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args(namespace=Arguments())
    assert sys.flags.isolated and not sys.flags.optimize, "run with -I, without -O"
    assert kayak.__file__ is not None
    package = Path(kayak.__file__).resolve().parent
    assert package.is_relative_to(Path(sys.prefix).resolve()), "install Kayak without -e"
    result = DecisionResult.model_validate_json((args.fixtures / "response.json").read_bytes())
    model = FixtureModel(result, args.work)
    app = create_app(lambda: model, api_key="compatibility-fixture", timeout=0.2)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        receipt = {
            "url": f"http://127.0.0.1:{sock.getsockname()[1]}",
            "package": str(package),
            "python": sys.version,
            "versions": {
                name: version(name) for name in ("kayak", "pydantic", "fastapi", "uvicorn")
            },
        }
        pending = args.work / "address.pending"
        pending.write_text(json.dumps(receipt))
        pending.replace(args.work / "address.json")
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", access_log=False))
        server.run(sockets=[sock])


if __name__ == "__main__":
    main()
