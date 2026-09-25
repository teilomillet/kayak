"""Discover, validate, and call the same decision contract from the terminal."""

import argparse
import math
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import version

from pydantic import ValidationError

from . import decisions
from .bundle import DEFAULT_MODEL
from .client import Client
from .decisions import DecisionRequest, DecisionResult, ModelInfo
from .errors import InputError, RemoteError, TransportError
from .ranking import RankingRequest, RankingResult


class Arguments(argparse.Namespace):
    command: str | None
    model: str
    host: str
    port: int
    device: str
    dtype: str
    batch_size: int
    cache_dir: str | None
    local_files_only: bool
    timeout: float
    api_key_env: str
    base_url: str
    input: str
    pretty: bool
    diagnostics: str
    ranking: bool


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kayak",
        description="Typed CLM decisions in Python and over HTTP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Start here:\n"
            "  kayak serve                       Load and serve a model\n"
            "  kayak info --pretty               Inspect the running model\n"
            "  kayak validate request.json       Check a request without a model\n"
            "  kayak decide request.json         Ask the running model\n"
            "  kayak rank candidates.json        Rank a supplied candidate set\n"
            "  kayak eval run banking77 --help    Benchmark decision quality and latency\n\n"
            "Use '-' as the input filename to read JSON from stdin.\n"
            "info, validate, decide, and rank need only the base package.\n"
            "Run 'kayak COMMAND --help' for options and defaults."
        ),
    )
    parser.add_argument("--version", action="version", version=f"kayak {version('kayak')}")
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.add_parser("eval", help="prepare datasets, benchmark models and compare saved runs")
    serve = commands.add_parser(
        "serve",
        help="load and serve one CLM model",
        description="Own one resident model; serve its decisions over HTTP. Requires kayak[serve].",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    serve.add_argument(
        "model", nargs="?", default=DEFAULT_MODEL, help="model ID or bundle directory"
    )
    serve.add_argument(
        "--host", default="127.0.0.1", help="bind address; network binding needs a key"
    )
    serve.add_argument("--port", type=int, default=8000, help="listening port (1..65535)")
    serve.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="auto selects available CUDA, then MPS, then CPU",
    )
    serve.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
        help="encoder precision; auto uses device-dependent precision",
    )
    serve.add_argument(
        "--batch-size", type=int, default=1, help="sequences per encoder forward pass"
    )
    serve.add_argument(
        "--cache-dir", help="artifact cache directory; otherwise use the Hub default"
    )
    serve.add_argument("--local-files-only", action="store_true", help="require cached artifacts")
    serve.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="response timeout in seconds; timed-out computation may continue",
    )
    serve.add_argument(
        "--api-key-env", default="KAYAK_API_KEY", help="environment variable for auth"
    )
    serve.add_argument(
        "--diagnostics",
        choices=["off", "json"],
        default="off",
        help="optional structured Kayak events to stderr; request IDs remain available when off",
    )

    for command, description in (
        ("info", "inspect the running model without requesting inference"),
        ("validate", "validate and re-emit request JSON without a model or network"),
        ("decide", "submit request JSON to a running Kayak service"),
        ("rank", "rank supplied candidates using a running Kayak service"),
    ):
        subcommand = commands.add_parser(
            command,
            help=description,
            description=description.capitalize() + ".",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        subcommand.add_argument("--pretty", action="store_true", help="indent JSON output")
        if command in {"validate", "decide", "rank"}:
            subcommand.add_argument("input", help="UTF-8 request JSON file, or '-' for stdin")
        if command == "validate":
            subcommand.add_argument(
                "--ranking", action="store_true", help="validate a ranking request instead"
            )
        if command != "validate":
            subcommand.add_argument(
                "--base-url",
                default="http://127.0.0.1:8000",
                help="Kayak service URL",
            )
            subcommand.add_argument(
                "--api-key-env",
                default="KAYAK_API_KEY",
                help="environment variable for auth",
            )
            subcommand.add_argument(
                "--timeout",
                type=float,
                default=120.0,
                help="network timeout in seconds",
            )
    return parser


def read_input(filename: str) -> bytes:
    """Bound file and stdin input before parsing or opening a network client."""
    limit = decisions.MAX_REQUEST_BYTES
    if filename == "-":
        raw = sys.stdin.buffer.read(limit + 1)
    else:
        with open(filename, "rb") as stream:
            raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise InputError("request body exceeds 1 MiB")
    return raw


def read_request(filename: str) -> DecisionRequest:
    """Read bounded JSON bytes and validate the unchanged decision contract."""
    try:
        return DecisionRequest.model_validate_json(read_input(filename))
    except ValidationError as exc:
        raise InputError(decisions.validation_message(exc)) from exc


def read_ranking_request(filename: str) -> RankingRequest:
    """Read bounded JSON bytes and apply the underlying Choice input limits."""
    try:
        return RankingRequest.model_validate_json(read_input(filename))
    except ValidationError as exc:
        raise InputError(decisions.validation_message(exc)) from exc


def serve_model(args: Arguments) -> None:
    # Configuration errors must be useful even before optional libraries are installed.
    if not 1 <= args.port <= 65535:
        raise InputError("--port must be 1..65535")
    if args.batch_size < 1:
        raise InputError("--batch-size must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise InputError("--timeout must be finite and positive")
    api_key = os.environ.get(args.api_key_env)
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not api_key:
        raise InputError(f"set {args.api_key_env} before binding a network interface")
    try:
        import uvicorn
        from fastapi import FastAPI

        from ._logging import json_logging
        from .runtime import load
        from .server import create_app
    except ImportError as exc:
        raise InputError("serving requires: pip install 'kayak[serve]'") from exc
    app = create_app(
        lambda: load(
            args.model,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        ),
        api_key=api_key,
        timeout=args.timeout,
        diagnostics=args.diagnostics == "json",
    )
    model_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        # Uvicorn configures logging before startup and restores signals after
        # shutdown. Own the writer between those boundaries so it can drain.
        with json_logging(args.diagnostics == "json"):
            async with model_lifespan(application):
                yield

    app.router.lifespan_context = lifespan
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=1,
        limit_concurrency=16,
        access_log=args.diagnostics != "json",
    )


def main() -> None:
    if sys.argv[1:2] == ["eval"]:
        from .eval._cli import main as evaluation_main

        raise SystemExit(evaluation_main(sys.argv[2:]))
    parser = argument_parser()
    args = parser.parse_args(namespace=Arguments())
    if args.command is None:
        parser.print_help()
        return
    try:
        if args.command == "serve":
            serve_model(args)
            return
        output: DecisionRequest | DecisionResult | ModelInfo | RankingRequest | RankingResult
        if args.command == "validate":
            output = read_ranking_request(args.input) if args.ranking else read_request(args.input)
        else:
            # Reject bad input before opening a network client.
            request = read_request(args.input) if args.command == "decide" else None
            rank_request = read_ranking_request(args.input) if args.command == "rank" else None
            with Client(
                base_url=args.base_url,
                api_key=os.environ.get(args.api_key_env),
                timeout=args.timeout,
            ) as client:
                if rank_request is not None:
                    output = client.rank(
                        state=rank_request.state,
                        instructions=rank_request.instructions,
                        candidates=rank_request.candidates,
                    )
                elif request is not None:
                    output = client.decide(state=request.state, questions=request.questions)
                else:
                    output = client.model_info()
        print(output.model_dump_json(indent=2 if args.pretty else None))
    except InputError as exc:
        suffix = f" [request_id={exc.request_id}]" if exc.request_id else ""
        parser.error(str(exc) + suffix)
    except RemoteError as exc:
        suffix = f" [request_id={exc.request_id}]" if exc.request_id else ""
        raise SystemExit(f"kayak: HTTP {exc.status_code} ({exc.code}): {exc}{suffix}") from None
    except TransportError as exc:
        suffix = f" [request_id={exc.request_id}]" if exc.request_id else ""
        raise SystemExit(
            f"kayak: {exc}{suffix}\n"
            "Check --base-url and service readiness. No automatic retry was made."
        ) from None
    except OSError as exc:
        raise SystemExit(f"kayak: {exc}") from None


if __name__ == "__main__":
    main()
