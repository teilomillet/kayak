"""Small, optional backend measurements; importing eval never imports PyTorch."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..runtime import Model

RUNTIME_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "RAYON_NUM_THREADS",
    "CUBLAS_WORKSPACE_CONFIG",
    "NVIDIA_TF32_OVERRIDE",
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_ENABLE_MPS_FALLBACK",
    "PYTORCH_MPS_FAST_MATH",
    "PYTORCH_MPS_PREFER_METAL",
    "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
    "PYTORCH_MPS_LOW_WATERMARK_RATIO",
)


def _command_output(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=5, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _cpu_model() -> str | None:
    if platform.system() == "Darwin":
        return _command_output(["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"])
    if platform.system() == "Linux":
        try:
            lines = Path("/proc/cpuinfo").read_text().splitlines()
        except OSError:
            return None
        names = set()
        for line in lines:
            key, separator, value = line.partition(":")
            if separator and key.strip() in {"model name", "Hardware"} and value.strip():
                names.add(value.strip())
        return "; ".join(sorted(names)) or None
    return platform.processor() or None


def _cpu_affinity() -> dict[str, object]:
    if not hasattr(os, "sched_getaffinity"):
        return {"status": "unsupported", "cpus": None}
    try:
        cpus = sorted(os.sched_getaffinity(0))
    except OSError:
        return {"status": "unavailable", "cpus": None}
    return {"status": "recorded", "cpus": cpus}


def _mps_devices() -> list[dict[str, object]] | None:
    raw = _command_output(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"])
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("SPDisplaysDataType"), list):
        return None
    devices = []
    for item in payload["SPDisplaysDataType"]:
        if not isinstance(item, dict):
            return None
        raw_cores = item.get("spdisplays_cores")
        cores = None
        if type(raw_cores) is int and raw_cores > 0:
            cores = raw_cores
        elif isinstance(raw_cores, str) and raw_cores.isdecimal() and int(raw_cores) > 0:
            cores = int(raw_cores)
        # Omit connected displays and serial numbers; only identify the accelerators.
        devices.append(
            {
                "name": item.get("sppci_model"),
                "cores": cores,
                "memory": item.get("spdisplays_vram") or item.get("spdisplays_vram_shared"),
            }
        )
    return devices


def _cuda_driver() -> str | None:
    try:
        return Path("/proc/driver/nvidia/version").read_text().strip() or None
    except OSError:
        return _command_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        )


def _local_environment(backend: Model) -> tuple[dict[str, object], dict[str, object]]:
    import torch

    device = backend.info.device
    hardware: dict[str, object] = {
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "cpu_affinity": _cpu_affinity(),
        "accelerator": None,
    }
    runtime: dict[str, object] = {
        "device": device,
        "batch_size": backend._batch_size,
        "torch_build": torch.__config__.show(),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "environment_variables": {
            name: os.environ.get(name) for name in RUNTIME_ENVIRONMENT_VARIABLES
        },
    }
    if device.split(":", 1)[0] == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware["accelerator"] = {
            "kind": "cuda",
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "multiprocessors": properties.multi_processor_count,
            "uuid": str(getattr(properties, "uuid", "")) or None,
        }
        runtime["cuda"] = {
            "driver": _cuda_driver(),
            "cuda_version": torch.version.cuda,
            "hip_version": torch.version.hip,
            "cudnn_version": cast(Callable[[], int | None], torch.backends.cudnn.version)(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "matmul_allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "matmul_allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "flash_sdp_enabled": cast(Callable[[], bool], torch.backends.cuda.flash_sdp_enabled)(),
            "mem_efficient_sdp_enabled": cast(
                Callable[[], bool], torch.backends.cuda.mem_efficient_sdp_enabled
            )(),
            "math_sdp_enabled": cast(Callable[[], bool], torch.backends.cuda.math_sdp_enabled)(),
            "cudnn_sdp_enabled": cast(Callable[[], bool], torch.backends.cuda.cudnn_sdp_enabled)(),
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
    elif device.split(":", 1)[0] == "mps":
        hardware["accelerator"] = {
            "kind": "mps",
            "devices": _mps_devices(),
            "recommended_max_memory_bytes": torch.mps.recommended_max_memory(),
        }
    return hardware, runtime


def environment(backend: Model | None = None) -> dict[str, object]:
    versions: dict[str, str | None] = {}
    for name in ("kayak", "torch", "transformers", "tokenizers", "pydantic", "httpx"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".json"}:
            digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
    values: dict[str, object] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "versions": versions,
        "kayak_source_sha256": digest.hexdigest(),
    }
    if backend is not None:
        values["hardware"], values["runtime"] = _local_environment(backend)
        device = backend.info.device.split(":", 1)[0]
        values["synchronization"] = (
            f"torch.{device}.synchronize" if device in {"cuda", "mps"} else "synchronous_cpu"
        )
    return values


def synchronize(device: str | None) -> None:
    if device is None or device.split(":", 1)[0] not in {"mps", "cuda"}:
        return
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    else:
        torch.mps.synchronize()


def memory_snapshot(device: str | None) -> dict[str, int | None]:
    values: dict[str, int | None] = {}
    if platform.system() in {"Darwin", "Linux"}:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        values["process_peak_rss_bytes"] = rss if platform.system() == "Darwin" else rss * 1024
    if device is not None and device.split(":", 1)[0] in {"mps", "cuda"}:
        import torch

        if device.startswith("cuda"):
            values["cuda_peak_tensor_bytes"] = torch.cuda.max_memory_allocated(device)
        else:
            values["mps_tensor_snapshot_bytes"] = torch.mps.current_allocated_memory()
            values["mps_driver_snapshot_bytes"] = torch.mps.driver_allocated_memory()
    return values
