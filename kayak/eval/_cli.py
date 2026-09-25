"""Prepare datasets, evaluate classifiers, compare runs, and score RAG observations."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Literal

import httpx

from ..bundle import DEFAULT_MODEL
from ..client import Client
from ..errors import KayakError
from ..runtime import Model, load
from . import _rag_cli
from ._banking77 import DEFAULT_CACHE, banking77, prepare_banking77
from ._benchmark import benchmark, write_benchmark
from ._compare import compare
from ._metrics import summarize
from ._predictions import export_predictions
from ._report import render_report
from ._runner import evaluate, load_report
from ._schema import Protocol
from ._scoring import confidence_metrics, default_metrics, expected_calibration_error


class Arguments(argparse.Namespace):
    command: str | None
    data_cache_dir: Path
    split: Literal["train", "dev", "test"]
    limit: int | None
    output: Path
    model: str
    base_url: str | None
    device: str
    dtype: str
    batch_size: int
    model_cache_dir: Path | None
    local_files_only: bool
    api_key_env: str
    timeout: float
    warmups: int
    repeats: int
    seed: int
    max_memory_gib: float | None
    min_accuracy: float | None
    baseline: Path
    candidate: Path
    allow_recipe_change: bool
    root: Path
    run_path: Path
    runs: list[Path]
    system: str
    method: str
    ece_bins: int
    confidence_threshold: float | None


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="kayak eval", description=__doc__)
    commands = root.add_subparsers(dest="command")
    commands.add_parser("rag", help="validate, score, and inspect portable RAG experiments")
    prepare = commands.add_parser("prepare", help="download and verify pinned dataset files")
    prepare.add_argument("dataset", choices=["banking77"])
    prepare.add_argument("--data-cache-dir", type=Path, default=DEFAULT_CACHE)
    run = commands.add_parser(
        "run", help="measure a model on a fixed dataset; defaults to the development split"
    )
    run.add_argument("dataset", choices=["banking77"])
    run.add_argument("--data-cache-dir", type=Path, default=DEFAULT_CACHE)
    run.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    run.add_argument("--limit", type=int, help="deterministic balanced subset; not a full score")
    run.add_argument("--output", type=Path, required=True, help="new directory for raw results")
    target = run.add_mutually_exclusive_group()
    target.add_argument("--model", default=DEFAULT_MODEL, help="local model ID or bundle")
    target.add_argument("--base-url", help="evaluate a running Kayak service instead")
    run.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    run.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--model-cache-dir", type=Path)
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--api-key-env", default="KAYAK_API_KEY")
    run.add_argument("--timeout", type=float, default=120.0)
    run.add_argument("--warmups", type=int, default=1, help="excluded calls before measurement")
    run.add_argument("--repeats", type=int, default=1, help="measured calls per distinct example")
    run.add_argument("--seed", type=int, default=42, help="execution order; does not alter splits")
    run.add_argument("--max-memory-gib", type=float, help="local post-call memory guard")
    run.add_argument("--min-accuracy", type=float, help="optional accuracy target for exit status")
    comparison = commands.add_parser("compare", help="compare two complete saved runs")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--allow-recipe-change", action="store_true")
    history = commands.add_parser("history", help="summarize saved runs, without inference")
    history.add_argument("root", type=Path)
    report = commands.add_parser("report", help="verify a saved run and print a Markdown report")
    report.add_argument("run_path", type=Path)
    export = commands.add_parser(
        "export", help="export first attempts to model-neutral prediction JSON"
    )
    export.add_argument("run_path", type=Path)
    export.add_argument("--output", type=Path, required=True, help="new prediction JSON file")
    export.add_argument("--system", required=True, help="classifier/model identity")
    export.add_argument("--method", required=True, help="training, prompting, and decision recipe")
    analysis = commands.add_parser(
        "benchmark", help="analyze saved runs or external prediction JSON"
    )
    analysis.add_argument(
        "runs", type=Path, nargs="+", help="first input is the comparison baseline"
    )
    analysis.add_argument(
        "--output", type=Path, required=True, help="new JSON/Markdown/CSV directory"
    )
    analysis.add_argument("--allow-recipe-change", action="store_true")
    analysis.add_argument("--ece-bins", type=int, default=10)
    analysis.add_argument("--confidence-threshold", type=float)
    return root


def _run(args: Arguments) -> int:
    Protocol(warmups=args.warmups, repeats=args.repeats, seed=args.seed)
    if args.batch_size < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("batch size and finite timeout must be positive")
    if args.max_memory_gib is not None and (
        not math.isfinite(args.max_memory_gib) or args.max_memory_gib <= 0
    ):
        raise ValueError("memory budget must be finite and positive")
    if args.min_accuracy is not None and not 0 <= args.min_accuracy <= 1:
        raise ValueError("min accuracy must be between 0 and 1")
    if args.output.exists():
        raise ValueError("output already exists; choose a fresh run directory")
    if args.base_url and (
        args.device != "auto"
        or args.dtype != "auto"
        or args.batch_size != 1
        or args.model_cache_dir is not None
        or args.local_files_only
        or args.max_memory_gib is not None
    ):
        raise ValueError("local model/memory options do not configure a remote HTTP service")
    suite = banking77(split=args.split, cache_dir=args.data_cache_dir, limit=args.limit)
    started = perf_counter()
    backend: Client | Model
    if args.base_url:
        backend = Client(
            base_url=args.base_url, api_key=os.environ.get(args.api_key_env), timeout=args.timeout
        )
        setup_kind = "client construction"
    else:
        backend = load(
            args.model,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            cache_dir=args.model_cache_dir,
            local_files_only=args.local_files_only,
        )
        setup_kind = "model loading"
    setup_seconds = perf_counter() - started

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"{done}/{total} examples", file=sys.stderr, flush=True)

    with backend:
        report = evaluate(
            backend,
            suite,
            output=args.output,
            warmups=args.warmups,
            repeats=args.repeats,
            seed=args.seed,
            max_memory_gib=args.max_memory_gib,
            progress=progress,
            config={
                "setup_seconds": setup_seconds,
                "setup_kind": setup_kind,
                "batch_size": None if args.base_url else args.batch_size,
                "min_accuracy": args.min_accuracy,
            },
        )
    summary = {
        key: value
        for key, value in report.summary.items()
        if key not in {"per_class", "confusion_matrix"}
    }
    print(
        json.dumps(
            {
                "status": report.status,
                "coverage": suite.provenance["coverage"],
                "report": str(args.output / "report.json"),
                **summary,
            },
            indent=2,
        )
    )
    accuracy = report.summary["accuracy"]
    assert isinstance(accuracy, float)
    return int(
        report.status != "complete"
        or (args.min_accuracy is not None and accuracy < args.min_accuracy)
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["rag"]:
        return _rag_cli.main(arguments[1:])
    cli = parser()
    args = cli.parse_args(arguments, namespace=Arguments())
    try:
        if args.command == "prepare":
            print(prepare_banking77(args.data_cache_dir))
        elif args.command == "run":
            return _run(args)
        elif args.command == "compare":
            print(
                json.dumps(
                    compare(
                        args.baseline, args.candidate, allow_recipe_change=args.allow_recipe_change
                    ),
                    indent=2,
                )
            )
        elif args.command == "report":
            print(render_report(args.run_path), end="")
        elif args.command == "export":
            export_predictions(args.run_path, args.output, system=args.system, method=args.method)
            print(args.output)
        elif args.command == "benchmark":
            if args.output.exists():
                raise ValueError("output already exists; choose a fresh benchmark directory")
            named = {str(index): path for index, path in enumerate(args.runs, 1)}
            selected = [metric for metric in default_metrics() if metric.name != "ece"]
            selected.append(expected_calibration_error(bins=args.ece_bins))
            if args.confidence_threshold is not None:
                selected.extend(confidence_metrics(args.confidence_threshold))
            result = benchmark(
                named, metrics=selected, allow_recipe_change=args.allow_recipe_change
            )
            write_benchmark(result, args.output)
            print(args.output / "benchmark.md")
        elif args.command == "history":
            for path in sorted(args.root.rglob("report.json")):
                report = load_report(path)
                summary = summarize(report)
                print(
                    json.dumps(
                        {
                            "path": str(path),
                            "created_at": report.created_at,
                            "status": report.status,
                            "dataset": report.suite.name,
                            "split": report.suite.split,
                            "examples": len(report.suite.examples),
                            "accuracy": summary["accuracy"],
                            "macro_f1": summary["macro_f1"],
                            "balanced_accuracy": summary["balanced_accuracy"],
                            "weighted_f1": summary["weighted_f1"],
                            "matthews_correlation": summary["matthews_correlation"],
                            "zero_recall_labels": summary["zero_recall_labels"],
                            "latency": report.summary["latency"],
                        }
                    )
                )
        else:
            cli.print_help()
    except KeyboardInterrupt:
        print("Evaluation interrupted; inspect the saved partial report.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KayakError, httpx.HTTPError) as exc:
        # Validation and file errors are actionable; model/transport errors may carry input.
        message = type(exc).__name__ if isinstance(exc, (KayakError, httpx.HTTPError)) else str(exc)
        print(f"kayak eval: {message}", file=sys.stderr)
        return 2
    return 0
