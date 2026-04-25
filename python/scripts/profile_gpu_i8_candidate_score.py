from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.gpu_device_capability import (  # noqa: E402
    MojoGpuCapability,
    probe_mojo_gpu,
)
from kayak_bridge.gpu_i8_candidate_score import (  # noqa: E402
    run_gpu_i8_candidate_score_probe,
)
from kayak_bridge.gpu_i8_rerank_contract import write_report  # noqa: E402


@dataclass(frozen=True, slots=True)
class CandidateScoreProfileShape:
    name: str
    query_count: int
    query_vector_count: int
    document_count: int
    document_vector_count: int
    candidate_k: int
    vector_dim: int = 128

    def to_json_ready(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "query_count": self.query_count,
            "query_vector_count": self.query_vector_count,
            "document_count": self.document_count,
            "document_vector_count": self.document_vector_count,
            "candidate_k": self.candidate_k,
            "candidate_score_count": self.query_count * self.candidate_k,
            "total_document_vector_count": (
                self.document_count * self.document_vector_count
            ),
            "vector_dim": self.vector_dim,
        }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile benchmark-only GPU i8 candidate score kernels across "
            "explicit vector-count shapes."
        )
    )
    parser.add_argument(
        "--shape-set",
        choices=("smoke", "profile"),
        default="smoke",
    )
    parser.add_argument(
        "--gpu-query-command",
        default="gpu-query",
        help="Executable used to probe Mojo-visible GPU devices.",
    )
    parser.add_argument(
        "--allow-missing-gpu",
        action="store_true",
        help="Exit successfully when no Mojo GPU is visible; the report stays blocked.",
    )
    parser.add_argument(
        "--emit-quiet-mean",
        action="store_true",
        help="Print run_bench_quiet-compatible Mean sections for each profile row.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/gpu_i8_candidate_score_profile/summary.json"),
    )
    return parser.parse_args(argv)


def shapes_for_set(shape_set: str) -> tuple[CandidateScoreProfileShape, ...]:
    smoke = (
        CandidateScoreProfileShape(
            name="default_2q_8qv_64d_16dv_32k",
            query_count=2,
            query_vector_count=8,
            document_count=64,
            document_vector_count=16,
            candidate_k=32,
        ),
        CandidateScoreProfileShape(
            name="candidate_window_2q_8qv_256d_16dv_128k",
            query_count=2,
            query_vector_count=8,
            document_count=256,
            document_vector_count=16,
            candidate_k=128,
        ),
    )
    if shape_set == "smoke":
        return smoke
    return smoke + (
        CandidateScoreProfileShape(
            name="query_batch_4q_8qv_256d_16dv_128k",
            query_count=4,
            query_vector_count=8,
            document_count=256,
            document_vector_count=16,
            candidate_k=128,
        ),
        CandidateScoreProfileShape(
            name="more_vectors_2q_16qv_256d_32dv_128k",
            query_count=2,
            query_vector_count=16,
            document_count=256,
            document_vector_count=32,
            candidate_k=128,
        ),
    )


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], MojoGpuCapability]:
    capability = probe_mojo_gpu(args.gpu_query_command)
    rows: list[dict[str, Any]] = []
    if capability.available:
        for shape in shapes_for_set(args.shape_set):
            probe = run_gpu_i8_candidate_score_probe(
                shape,
                candidate_k=shape.candidate_k,
                target_accelerator=capability.target_accelerator,
            )
            rows.append(profile_row_from_probe(shape, probe))

    status = "ok" if capability.available and _all_rows_ok(rows) else "blocked"
    return (
        {
            "benchmark": "gpu_i8_candidate_score_profile",
            "created_at": datetime.now(UTC).isoformat(),
            "status": status,
            "shape_set": args.shape_set,
            "mojo_gpu_capability": capability.to_json_ready(),
            "rows": rows,
            "measurement_note": (
                "Rows are benchmark-only deterministic GPU probes. They are "
                "useful for profiling kernel structure, not production backend "
                "speedup claims."
            ),
        },
        capability,
    )


def profile_row_from_probe(
    shape: CandidateScoreProfileShape,
    probe: dict[str, object],
) -> dict[str, Any]:
    parsed = probe.get("parsed")
    parsed_payload = parsed if isinstance(parsed, dict) else {}
    row = {
        "shape": shape.to_json_ready(),
        "status": probe.get("status"),
        "target_accelerator": probe.get("target_accelerator"),
        "counts": probe.get("counts"),
        "parsed": parsed_payload,
        "derived": derive_candidate_score_metrics(parsed_payload),
    }
    return row


def derive_candidate_score_metrics(parsed: dict[str, object]) -> dict[str, object]:
    candidate_score_count = _optional_float(parsed.get("candidate_score_count"))
    serial_kernel = _optional_float(parsed.get("serial_kernel_mean_seconds"))
    twopass_kernel = _optional_float(parsed.get("twopass_kernel_mean_seconds"))
    cpu_reference = _optional_float(parsed.get("cpu_reference_mean_seconds"))
    h2d = _optional_float(parsed.get("host_to_device_mean_seconds"))
    d2h = _optional_float(parsed.get("device_to_host_mean_seconds"))

    return {
        "serial_kernel_candidate_scores_per_second": _rate(
            candidate_score_count,
            serial_kernel,
        ),
        "twopass_kernel_candidate_scores_per_second": _rate(
            candidate_score_count,
            twopass_kernel,
        ),
        "cpu_scalar_reference_candidate_scores_per_second": _rate(
            candidate_score_count,
            cpu_reference,
        ),
        "twopass_kernel_speedup_vs_serial_kernel": _ratio(
            serial_kernel,
            twopass_kernel,
        ),
        "twopass_kernel_speedup_vs_cpu_scalar_reference": _ratio(
            cpu_reference,
            twopass_kernel,
        ),
        "twopass_copy_to_kernel_ratio": _ratio(
            _sum_optional(h2d, d2h),
            twopass_kernel,
        ),
    }


def print_quiet_sections(report: dict[str, Any]) -> None:
    rows = report.get("rows")
    if not isinstance(rows, list):
        return
    for row in rows:
        if not isinstance(row, dict):
            continue
        shape = row.get("shape")
        parsed = row.get("parsed")
        if not isinstance(shape, dict) or not isinstance(parsed, dict):
            continue
        mean_seconds = parsed.get("twopass_kernel_mean_seconds")
        if not isinstance(mean_seconds, (float, int)):
            continue
        print("== gpu_i8_candidate_twopass_kernel:", shape.get("name"), "==")
        print("Mean:", mean_seconds)


def exit_code(report: dict[str, Any], capability: MojoGpuCapability, args: argparse.Namespace) -> int:
    if not capability.available and not args.allow_missing_gpu:
        return 2
    if not _all_rows_ok(report.get("rows")):
        return 3
    return 0


def _all_rows_ok(rows: object) -> bool:
    if not isinstance(rows, list) or not rows:
        return False
    return all(isinstance(row, dict) and row.get("status") == "ok" for row in rows)


def _optional_float(value: object) -> float | None:
    if isinstance(value, (float, int)):
        return float(value)
    return None


def _sum_optional(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left + right


def _rate(count: float | None, seconds: float | None) -> float | None:
    if count is None or seconds is None or seconds <= 0.0:
        return None
    return count / seconds


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report, capability = build_report(args)
    write_report(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.emit_quiet_mean:
        print_quiet_sections(report)
    return exit_code(report, capability, args)


if __name__ == "__main__":
    raise SystemExit(main())
