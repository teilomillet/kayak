"""Local inference experiments: run, compare, bounded batch search, and history.

Run `uv run --extra serve --extra bench -m benchmarks.inference --help`.
This is deliberately outside CI.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import subprocess
import sys
from pathlib import Path

from kayak.bundle import DEFAULT_MODEL

from .evaluation import Protocol, Run, compare, memory_high_water, read_cases, summarize

DEFAULT_SUITE = Path(__file__).with_name("data") / "interactive-v1.jsonl"


class Arguments(argparse.Namespace):
    command: str
    output: Path
    suite: Path
    model: str
    device: str
    dtype: str
    batch_size: int
    batch_sizes: list[int]
    cache_dir: str | None
    local_files_only: bool
    transport: str
    split: str
    warmups: int
    repeats: int
    trials: int
    seed: int
    max_memory_gib: float | None
    score_atol: float
    min_speedup: float
    max_case_regression: float
    max_memory_regression: float
    trial_timeout: float
    baseline: Path
    candidate: Path
    root: Path


def thresholds(args: Arguments) -> dict[str, float]:
    return {
        "score_atol": args.score_atol,
        "min_speedup": args.min_speedup,
        "max_case_regression": args.max_case_regression,
        "max_memory_regression": args.max_memory_regression,
    }


def read_runs(path: Path) -> list[Run]:
    if path.is_dir():
        for attempt in path.glob("*.attempt.json"):
            value = json.loads(attempt.read_text())
            worker = path / attempt.name.removesuffix(".attempt.json") / "run.json"
            if value.get("exit_code") != 0 or not worker.exists():
                raise ValueError(f"failed or missing worker cannot be omitted: {attempt}")
    files = [path] if path.is_file() else sorted(path.glob("*/run.json"))
    if (path / "run.json").is_file():
        files = [path / "run.json"]
    if not files:
        raise ValueError(f"no run.json reports found in {path}")
    return [Run.model_validate_json(file.read_text()) for file in files]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def launch(args: Arguments, output: Path, batch: int, split: str) -> Run | None:
    command = [
        sys.executable,
        "-m",
        "benchmarks.inference",
        "run",
        "--output",
        str(output),
        "--suite",
        str(args.suite.resolve()),
        "--model",
        args.model,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--batch-size",
        str(batch),
        "--transport",
        args.transport,
        "--split",
        split,
        "--warmups",
        str(args.warmups),
        "--repeats",
        str(args.repeats),
        "--seed",
        str(args.seed),
    ]
    if args.cache_dir:
        command.extend(["--cache-dir", str(Path(args.cache_dir).resolve())])
    if args.local_files_only:
        command.append("--local-files-only")
    if args.max_memory_gib is not None:
        command.extend(["--max-memory-gib", str(args.max_memory_gib)])
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting {split} batch={batch} {output.name}", flush=True)
    with output.with_suffix(".log").open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=args.trial_timeout)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            code = -1
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    # Keep the attempt even if a hard crash happened before run.json existed.
    write_json(output.with_suffix(".attempt.json"), {"command": command, "exit_code": code})
    report_path = output / "run.json"
    run = Run.model_validate_json(report_path.read_text()) if report_path.exists() else None
    if code != 0 and run is not None:
        run.status = "failed"
        run.error = run.error or f"worker exited {code}"
        report_path.write_text(run.model_dump_json(indent=2) + "\n")
    print(f"Finished {split} batch={batch}: {run.status if run else 'no report'}", flush=True)
    return run


def sweep(args: Arguments) -> dict[str, object]:
    # Validate both partitions before allocating a model or starting a search.
    read_cases(args.suite, "dev")
    read_cases(args.suite, "holdout")
    args.output.mkdir(parents=True, exist_ok=False)
    batches = list(dict.fromkeys([1, *args.batch_sizes]))
    groups: dict[int, list[Run]] = {batch: [] for batch in batches}
    report: dict[str, object] = {
        "status": "running",
        "batch_sizes": batches,
        "trials": args.trials,
        "thresholds": thresholds(args),
        "search": {},
        "confirmation": None,
        "note": "Local search requires confirmation; runtime defaults are unchanged.",
    }
    write_json(args.output / "sweep.json", report)
    rng = random.Random(args.seed)
    for trial in range(args.trials):
        order = batches.copy()
        rng.shuffle(order)
        for batch in order:
            run = launch(
                args, args.output / "search" / f"batch-{batch}" / f"trial-{trial}", batch, "dev"
            )
            if run is not None:
                groups[batch].append(run)
    comparisons = {}
    finalists: list[tuple[float, int]] = []
    for batch in batches:
        if batch == 1:
            continue
        if len(groups[batch]) != args.trials or len(groups[1]) != args.trials:
            comparison: dict[str, object] = {
                "status": "rejected",
                "reasons": ["missing trial reports"],
            }
        else:
            comparison = compare(groups[1], groups[batch], **thresholds(args))
        comparisons[str(batch)] = comparison
        speedup = comparison.get("geomean_speedup")
        if comparison["status"] == "passed" and isinstance(speedup, float):
            finalists.append((speedup, batch))
    report["search"] = comparisons
    report["status"] = "no_winner"
    write_json(args.output / "sweep.json", report)
    if finalists:
        _, winner = max(finalists)
        # Freeze the selected configuration; never try the runner-up on holdout.
        confirm: dict[int, list[Run]] = {1: [], winner: []}
        for trial in range(args.trials):
            order = [1, winner]
            rng.shuffle(order)
            for batch in order:
                run = launch(
                    args,
                    args.output / "confirmation" / f"batch-{batch}" / f"trial-{trial}",
                    batch,
                    "holdout",
                )
                if run is not None:
                    confirm[batch].append(run)
        if any(len(runs) != args.trials for runs in confirm.values()):
            result: dict[str, object] = {
                "status": "rejected",
                "reasons": ["missing confirmation trials"],
            }
        else:
            result = compare(confirm[1], confirm[winner], **thresholds(args))
        report["confirmation"] = result
        report["selected_batch_size"] = winner
        report["status"] = "confirmed" if result["status"] == "passed" else "confirmation_failed"
        write_json(args.output / "sweep.json", report)
    print(json.dumps(report, indent=2))
    return report


def history(root: Path) -> None:
    print(
        "created_at\tstatus\tbatch\ttransport\tsplit\taccuracy\tcase_geomean_ms\tmemory_GiB\tpath"
    )
    for path in sorted(root.rglob("run.json")):
        run = Run.model_validate_json(path.read_text())
        summary = summarize(run)
        latencies = [
            statistics.median(sample.seconds for sample in case.samples if sample.result)
            for case in run.cases
            if any(sample.result for sample in case.samples)
        ]
        latency = f"{1000 * statistics.geometric_mean(latencies):.2f}" if latencies else "-"
        try:
            memory = f"{memory_high_water(run) / 1024**3:.2f}"
        except ValueError:
            memory = "-"
        print(
            f"{run.created_at}\t{run.status}\t{run.config['batch_size']}\t{run.protocol.transport}"
            f"\t{run.protocol.split}\t{summary['accuracy']}\t{latency}\t{memory}\t{path}"
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("run", "sweep"):
        command = commands.add_parser(name)
        command.add_argument(
            "--output", type=Path, required=True, help="new directory; never overwritten"
        )
        command.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
        command.add_argument("--model", default=DEFAULT_MODEL)
        command.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
        command.add_argument(
            "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
        )
        command.add_argument("--cache-dir")
        command.add_argument("--local-files-only", action="store_true")
        command.add_argument("--transport", choices=["local", "http"], default="local")
        command.add_argument("--warmups", type=int, default=2)
        command.add_argument("--repeats", type=int, default=10)
        command.add_argument("--seed", type=int, default=20260924)
        command.add_argument("--max-memory-gib", type=float)
        if name == "run":
            command.add_argument("--batch-size", type=int, default=1)
            command.add_argument("--split", choices=["dev", "holdout", "all"], default="all")
        else:
            command.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
            command.add_argument("--trials", type=int, default=3)
            command.add_argument("--trial-timeout", type=float, default=900)
    comparison = commands.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("candidate", type=Path)
    comparison.add_argument("--output", type=Path, help="optional new JSON file")
    for command in (comparison, commands.choices["sweep"]):
        command.add_argument("--score-atol", type=float, default=0.0)
        command.add_argument("--min-speedup", type=float, default=0.05)
        command.add_argument("--max-case-regression", type=float, default=0.10)
        command.add_argument("--max-memory-regression", type=float, default=0.10)
    history_parser = commands.add_parser("history")
    history_parser.add_argument("--root", type=Path, default=Path(".benchmarks/inference"))
    return result


def main() -> None:
    argument_parser = parser()
    args = argument_parser.parse_args(namespace=Arguments())
    try:
        if args.command in {"run", "sweep"}:
            if args.warmups < 1 or args.repeats < 2:
                raise ValueError("use at least one warmup and two measured repeats")
            if args.max_memory_gib is not None and (
                not math.isfinite(args.max_memory_gib) or args.max_memory_gib <= 0
            ):
                raise ValueError("memory budget must be finite and positive")
        if args.command in {"compare", "sweep"}:
            if any(not math.isfinite(value) or value < 0 for value in thresholds(args).values()):
                raise ValueError("comparison thresholds must be finite and nonnegative")
        if args.command == "run":
            from .measure import run_model

            if args.batch_size < 1:
                raise ValueError("batch size must be positive")
            protocol = Protocol.model_validate(
                {
                    key: getattr(args, key)
                    for key in ("transport", "split", "warmups", "repeats", "seed")
                }
            )
            run = run_model(
                suite=args.suite,
                output=args.output,
                protocol=protocol,
                model_id=args.model,
                device=args.device,
                dtype=args.dtype,
                batch_size=args.batch_size,
                cache_dir=args.cache_dir,
                local_files_only=args.local_files_only,
                max_memory_gib=args.max_memory_gib,
            )
            raise SystemExit(0 if run.status == "passed" else 1)
        if args.command == "sweep":
            if args.trials < 3 or any(batch < 1 for batch in args.batch_sizes):
                raise ValueError("sweep needs at least three trials and positive batch sizes")
            if not math.isfinite(args.trial_timeout) or args.trial_timeout <= 0:
                raise ValueError("trial timeout must be finite and positive")
            if not any(batch != 1 for batch in args.batch_sizes):
                raise ValueError("include at least one candidate batch size other than 1")
            sweep(args)
        elif args.command == "compare":
            report = compare(
                read_runs(args.baseline), read_runs(args.candidate), **thresholds(args)
            )
            if args.output is not None:
                with args.output.open("x") as stream:
                    json.dump(report, stream, indent=2, allow_nan=False)
                    stream.write("\n")
            print(json.dumps(report, indent=2))
            raise SystemExit(0 if report["status"] == "passed" else 1)
        else:
            history(args.root)
    except (ValueError, OSError) as exc:
        argument_parser.error(str(exc))


if __name__ == "__main__":
    main()
