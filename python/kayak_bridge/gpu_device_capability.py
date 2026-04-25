from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Sequence


GPU_STATUS_AVAILABLE = "available"
GPU_STATUS_TOOL_MISSING = "tool_missing"
GPU_STATUS_UNAVAILABLE = "unavailable"
GPU_STATUS_ERROR = "error"

GPU_DEVICE_NODE_PATHS = (
    "/dev/nvidia0",
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/dri/renderD128",
    "/dev/dri/renderD129",
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    def to_json_ready(self) -> dict[str, object]:
        return {
            "command": list(self.command),
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
        }


@dataclass(frozen=True, slots=True)
class MojoGpuCapability:
    status: str
    probe: CommandResult
    device_name: str | None = None
    driver_version: str | None = None
    compute_capability: str | None = None
    api: str | None = None
    vendor: str | None = None
    arch: str | None = None
    arch_name: str | None = None
    target_accelerator: str | None = None
    device_memory_bytes: int | None = None
    supported_dtypes: tuple[str, ...] = ()
    unverified_fields: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.status == GPU_STATUS_AVAILABLE

    def to_json_ready(self) -> dict[str, object]:
        return {
            "status": self.status,
            "probe": self.probe.to_json_ready(),
            "device_name": self.device_name,
            "driver_version": self.driver_version,
            "compute_capability": self.compute_capability,
            "api": self.api,
            "vendor": self.vendor,
            "arch": self.arch,
            "arch_name": self.arch_name,
            "target_accelerator": self.target_accelerator,
            "device_memory_bytes": self.device_memory_bytes,
            "supported_dtypes": list(self.supported_dtypes),
            "unverified_fields": list(self.unverified_fields),
        }


def run_command(command: Sequence[str], *, timeout_seconds: float = 10.0) -> CommandResult:
    command_tuple = tuple(command)
    try:
        completed = subprocess.run(
            command_tuple,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return CommandResult(command_tuple, None, "", str(exc))
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            command_tuple,
            None,
            (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            (exc.stderr or "") if isinstance(exc.stderr, str) else "",
            timed_out=True,
        )
    return CommandResult(
        command_tuple,
        completed.returncode,
        completed.stdout.strip(),
        completed.stderr.strip(),
    )


def probe_mojo_gpu(gpu_query_command: str) -> MojoGpuCapability:
    probe = run_command((gpu_query_command,))
    if probe.timed_out:
        return MojoGpuCapability(
            status=GPU_STATUS_ERROR,
            probe=probe,
            unverified_fields=("backend", "memory", "supported_dtypes"),
        )
    if probe.returncode is None:
        return MojoGpuCapability(
            status=GPU_STATUS_TOOL_MISSING,
            probe=probe,
            unverified_fields=("backend", "memory", "supported_dtypes"),
        )
    if probe.returncode != 0:
        return MojoGpuCapability(
            status=GPU_STATUS_UNAVAILABLE,
            probe=probe,
            unverified_fields=("memory", "supported_dtypes"),
        )

    return MojoGpuCapability(
        status=GPU_STATUS_AVAILABLE,
        probe=probe,
        device_name=gpu_query_value(probe.stdout, "name"),
        driver_version=gpu_query_value(probe.stdout, "driver_version"),
        compute_capability=gpu_query_value(probe.stdout, "compute_capability"),
        api=_optional_gpu_query_field(gpu_query_command, "--api"),
        vendor=_optional_gpu_query_field(gpu_query_command, "--vendor"),
        arch=_optional_gpu_query_field(gpu_query_command, "--arch"),
        arch_name=_optional_gpu_query_field(gpu_query_command, "--arch-name"),
        target_accelerator=_optional_gpu_query_field(
            gpu_query_command,
            "--target-accelerator",
        ),
        device_memory_bytes=gpu_query_memory_bytes(probe.stdout),
        unverified_fields=("supported_dtypes",),
    )


def gpu_query_value(stdout: str, key: str) -> str | None:
    prefix = key + ":"
    for line in stdout.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :].strip().replace("\x00", "")
            return value or None
    return None


def gpu_query_memory_bytes(stdout: str) -> int | None:
    value = gpu_query_value(stdout, "memory")
    if value is None:
        return None
    normalized = value.strip().upper()
    units = (
        ("GIB", 1024**3),
        ("GB", 1000**3),
        ("MIB", 1024**2),
        ("MB", 1000**2),
    )
    for suffix, multiplier in units:
        if normalized.endswith(suffix):
            number = normalized[: -len(suffix)].strip()
            try:
                return int(float(number) * multiplier)
            except ValueError:
                return None
    return None


def host_gpu_inventory() -> dict[str, object]:
    lspci = run_command(("lspci",))
    nvidia_smi = run_command(("nvidia-smi",))
    display_devices = _display_devices_from_lspci(lspci.stdout)
    proc_info = _read_optional_path(
        Path("/proc/driver/nvidia/gpus/0000:01:00.0/information")
    )
    proc_version = _read_optional_path(Path("/proc/driver/nvidia/version"))
    device_nodes = {path: Path(path).exists() for path in GPU_DEVICE_NODE_PATHS}
    hardware_present = bool(display_devices) or proc_info is not None
    return {
        "hardware_present": hardware_present,
        "display_devices": display_devices,
        "device_nodes": device_nodes,
        "nvidia_smi": nvidia_smi.to_json_ready(),
        "nvidia_proc_information": proc_info,
        "nvidia_proc_version": proc_version,
        "lspci_probe": lspci.to_json_ready(),
    }


def _display_devices_from_lspci(stdout: str) -> list[str]:
    keywords = ("vga", "3d", "display", "nvidia", "amd/ati")
    devices: list[str] = []
    for line in stdout.splitlines():
        normalized = line.lower()
        if any(keyword in normalized for keyword in keywords):
            devices.append(line)
    return devices


def _read_optional_path(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _optional_gpu_query_field(gpu_query_command: str, flag: str) -> str | None:
    result = run_command((gpu_query_command, flag))
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value if value else None
