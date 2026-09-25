"""Own one real model and record local or loopback HTTP measurements."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from kayak import Client, Model, load
from kayak.decisions import DecisionRequest, DecisionResult, check_result

from .evaluation import (
    CaseRun,
    Protocol,
    Run,
    Sample,
    observations,
    read_cases,
    result_drift,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]
Call = Callable[[DecisionRequest], DecisionResult]


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def tree_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        digest.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def environment() -> dict[str, object]:
    import torch

    packages = {}
    for name in ("kayak", "torch", "transformers", "tokenizers", "pydantic", "httpx", "uvicorn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    accelerator = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
        if platform.system() == "Darwin"
        else platform.processor()
    )
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "python": platform.python_version(),
        "accelerator": accelerator,
        "versions": packages,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "cuda_version": torch.version.cuda,
        "allocator_environment": {
            key: os.environ.get(key)
            for key in (
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "PYTORCH_MPS_LOW_WATERMARK_RATIO",
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_FAST_MATH",
                "PYTORCH_MPS_PREFER_METAL",
                "PYTORCH_CUDA_ALLOC_CONF",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
    }


def system_snapshot() -> dict[str, object]:
    if platform.system() == "Darwin":
        return {
            "vm_stat": command_output(["vm_stat"]),
            "swap": command_output(["sysctl", "-n", "vm.swapusage"]),
            "pressure": command_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]),
        }
    if Path("/proc/meminfo").exists():
        return {"meminfo": Path("/proc/meminfo").read_text()}
    return {"memory": "unavailable"}


def synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device.startswith("mps"):
        torch.mps.synchronize()


def memory_snapshot(device: str) -> dict[str, int | None]:
    import torch

    values: dict[str, int | None] = {"process_peak_rss_bytes": None}
    if platform.system() in {"Linux", "Darwin"}:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        values["process_peak_rss_bytes"] = rss if platform.system() == "Darwin" else rss * 1024
    if device.startswith("cuda"):
        values["cuda_peak_tensor_bytes"] = torch.cuda.max_memory_allocated()
    elif device.startswith("mps"):
        values["mps_current_tensor_bytes"] = torch.mps.current_allocated_memory()
        values["mps_current_driver_bytes"] = torch.mps.driver_allocated_memory()
    return values


def observe_memory(entry: CaseRun, device: str, max_memory_gib: float | None) -> None:
    for key, value in memory_snapshot(device).items():
        previous = entry.memory.get(key)
        entry.memory[key] = max(value, previous or 0) if value is not None else previous
    counter = {"mps": "mps_current_driver_bytes", "cuda": "cuda_peak_tensor_bytes"}.get(
        device.split(":")[0], "process_peak_rss_bytes"
    )
    used = entry.memory.get(counter)
    if max_memory_gib is not None and used is not None and used > max_memory_gib * 1024**3:
        raise RuntimeError(
            f"observed {counter} exceeds --max-memory-gib; stopping further requests"
        )
    if platform.system() == "Darwin":
        pressure = command_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"])
        if pressure == "4":
            raise RuntimeError("critical system memory pressure; stopping further requests")


@contextmanager
def caller(model: Model, transport: str) -> Iterator[Call]:
    if transport == "local":
        yield lambda request: model.decide(state=request.state, questions=request.questions)
        return
    import uvicorn

    from kayak.server import create_app

    # A real loopback socket and public Client, with one resident encoder.
    app = create_app(lambda: model, api_key="benchmark-local-fixture")
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", limit_concurrency=16))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 180
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("loopback service did not become ready")
                time.sleep(0.05)
            with Client(
                base_url=f"http://127.0.0.1:{sock.getsockname()[1]}",
                api_key="benchmark-local-fixture",
            ) as client:
                yield lambda request: client.decide(
                    state=request.state, questions=request.questions
                )
        finally:
            server.should_exit = True
            thread.join(timeout=180)
            if thread.is_alive():
                raise RuntimeError("loopback service failed to drain")


def measure(call: Call, request: DecisionRequest, device: str) -> Sample:
    synchronize(device)
    started = time.perf_counter()
    try:
        result = call(request)
        synchronize(device)
        seconds = time.perf_counter() - started
        check_result(request, result)
        return Sample(seconds=seconds, result=result)
    except Exception as exc:
        return Sample(seconds=time.perf_counter() - started, error=f"{type(exc).__name__}: {exc}")


def save(run: Run, output: Path) -> None:
    temporary = output / "run.json.tmp"
    temporary.write_text(run.model_dump_json(indent=2) + "\n")
    temporary.replace(output / "run.json")
    summary = summarize(run)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def run_model(
    *,
    suite: Path,
    output: Path,
    protocol: Protocol,
    model_id: str,
    device: str,
    dtype: str,
    batch_size: int,
    cache_dir: str | None,
    local_files_only: bool,
    max_memory_gib: float | None = None,
) -> Run:
    cases, suite_hash = read_cases(suite, protocol.split)
    output.mkdir(parents=True, exist_ok=False)
    run = Run(
        run_id=str(uuid.uuid4()),
        created_at=datetime.now(UTC).isoformat(),
        protocol=protocol,
        suite_sha256=suite_hash,
        harness_sha256=tree_hash(ROOT / "benchmarks"),
        source={
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "git_status": command_output(["git", "status", "--porcelain"]),
            "kayak_sha256": tree_hash(ROOT / "kayak"),
        },
        environment=environment(),
        config={
            "model": model_id,
            "device": device,
            "dtype": dtype,
            "batch_size": batch_size,
            "cache_dir": cache_dir,
            "local_files_only": local_files_only,
            "max_memory_gib": max_memory_gib,
        },
        cases=[CaseRun(case=case) for case in cases],
        system_before=system_snapshot(),
    )
    save(run, output)
    try:
        started = time.perf_counter()
        with load(
            model_id,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        ) as model:
            synchronize(model.info.device)
            run.load_seconds = time.perf_counter() - started
            run.model = model.info.model_dump()
            run.memory_after_load = memory_snapshot(model.info.device)
            save(run, output)
            with caller(model, protocol.transport) as call:
                rng = random.Random(protocol.seed)
                for phase, count in (("warmups", protocol.warmups), ("samples", protocol.repeats)):
                    for index in range(count):
                        order = list(run.cases)
                        rng.shuffle(order)
                        for entry in order:
                            sample = measure(call, entry.case.request, model.info.device)
                            getattr(entry, phase).append(sample)
                            observe_memory(entry, model.info.device, max_memory_gib)
                            if sample.error:
                                raise RuntimeError(f"{entry.case.id}: {sample.error}")
                        save(run, output)
                        print(f"{output.name}: {phase} {index + 1}/{count}", flush=True)
            run.status = "passed"
            # Repeated calls in a fixed configuration must agree before comparison.
            for key, results in observations(run).items():
                if any(result_drift(results[0], result) != (0, 0.0) for result in results[1:]):
                    raise ValueError(f"nonrepeatable inference: {key}")
    except Exception as exc:
        run.status = "failed"
        run.error = f"{type(exc).__name__}: {exc}"
    finally:
        run.system_after = system_snapshot()
        save(run, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": run.status,
                "error": run.error,
                "quality": {k: v for k, v in summarize(run).items() if k != "cases"},
            }
        ),
        flush=True,
    )
    return run
