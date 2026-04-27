from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stdout

from kayak_bridge.cache_paths import REPO_ROOT

import numpy as np  # noqa: E402


SCRIPT_ROOT = REPO_ROOT / "python" / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.append(str(SCRIPT_ROOT))

import bench_gpu_i8_rerank_contract as bench_contract  # noqa: E402
import compare_gpu_i8_fastplaid as fastplaid_compare  # noqa: E402
import gpu_i8_cpu_reference as cpu_reference  # noqa: E402
import profile_gpu_i8_candidate_score as candidate_profile  # noqa: E402
import profile_gpu_i8_address_serve_sweep as address_sweep_script  # noqa: E402
import profile_gpu_i8_real_payload_rerank as real_payload_profile  # noqa: E402
from kayak_bridge.gpu_device_capability import (  # noqa: E402
    CommandResult,
    GPU_STATUS_AVAILABLE,
    GPU_STATUS_ERROR,
    GPU_STATUS_TOOL_MISSING,
    GPU_STATUS_UNAVAILABLE,
    MojoGpuCapability,
    gpu_query_memory_bytes,
    gpu_query_value,
)
from kayak_bridge.gpu_copy_roundtrip import (  # noqa: E402
    gpu_copy_probe_counts,
    parse_gpu_copy_probe_output,
)
from kayak_bridge.gpu_i8_candidate_score import (  # noqa: E402
    gpu_candidate_score_probe_counts,
)
from kayak_bridge import gpu_i8_centroid_budget_sweep as centroid_budget_sweep  # noqa: E402
from kayak_bridge.gpu_i8_real_payload_score import (  # noqa: E402
    GPU_REAL_PAYLOAD_STATUS_OK,
    gpu_real_payload_probe_counts,
    write_gpu_i8_real_payload_probe_inputs,
)
from kayak_bridge import gpu_i8_address_serve_sweep as address_serve_sweep  # noqa: E402
from kayak_bridge.mojo_gpu_i8_rerank import (  # noqa: E402
    MojoGpuI8AddressMultiWindowSessionResult,
    MojoGpuI8AddressResidentSessionResult,
    MojoGpuI8AddressSessionHandle,
    MojoGpuI8AddressServeResult,
    MojoGpuI8AddressTopKNoReferenceResult,
    MojoGpuI8AddressTopKResult,
    MojoGpuI8PreparedSessionResult,
    MojoGpuI8RerankBridgeResult,
)
from kayak_bridge.gpu_i8_rerank_contract import (  # noqa: E402
    GPU_RERANK_STATUS_BACKEND_INTEGRATION_MISSING,
    GPU_RERANK_STATUS_BATCH_KERNEL_MISSING,
    GPU_RERANK_STATUS_CANDIDATE_SCORE_FAILED,
    GPU_RERANK_STATUS_COPY_PROBE_FAILED,
    GPU_RERANK_STATUS_KERNEL_MISSING,
    GPU_RERANK_STATUS_NO_HARDWARE,
    GPU_RERANK_STATUS_RUNTIME_UNAVAILABLE,
    GPU_RERANK_STATUS_SINGLE_SCORE_FAILED,
    exit_code_for_report,
    print_quiet_wrapper_section,
    report_status,
    shape_contract,
    tensor_contract,
)
from kayak_bridge.gpu_i8_score_agreement import (  # noqa: E402
    gpu_i8_score_agreement_fields,
    gpu_i8_score_delta_tolerance,
)
from kayak_bridge.gpu_i8_single_score import (  # noqa: E402
    gpu_single_score_probe_counts,
)
from kayak_bridge.gpu_i8_profile_contract import (  # noqa: E402
    GPU_TIMING_FIELDS,
    PROFILE_STATUS_DECISION_READY,
    PROFILE_STATUS_EXPLORATORY,
    PROFILE_STATUS_FALSIFIED,
    gpu_profile_contract,
    gpu_profile_decision_status,
    gpu_profile_row_template,
)
from kayak_bridge.plaid_approx import KayakPlaidI8PayloadSnapshot  # noqa: E402


class GpuI8RerankContractTests(unittest.TestCase):
    def test_shape_contract_keeps_vector_counts_explicit(self) -> None:
        args = bench_contract.parse_args(
            [
                "--document-count",
                "8",
                "--document-vector-count",
                "4",
                "--query-count",
                "2",
                "--query-vector-count",
                "3",
                "--candidate-k",
                "5",
                "--allow-missing-gpu",
                "--allow-missing-kernel",
            ]
        )
        shape = bench_contract.build_shape(args)

        payload = shape_contract(shape, candidate_k=args.candidate_k)

        self.assertEqual(payload["document_vector_count_total"], 32)
        self.assertEqual(payload["query_vector_count_total"], 6)
        self.assertEqual(payload["candidate_position_count_total"], 10)
        self.assertEqual(payload["vector_dim"], 128)

    def test_tensor_contract_names_gpu_boundary_dtypes(self) -> None:
        shape = bench_contract.SpeedTrackShape(
            document_count=8,
            document_vector_count=4,
            query_count=2,
            query_vector_count=3,
            vector_dim=128,
            top_k=5,
            update_document_count=0,
        )

        payload = tensor_contract(shape, candidate_k=5)

        self.assertEqual(payload["token_codes"]["dtype"], "Int8")
        self.assertEqual(payload["token_scales"]["dtype"], "ScoreScalar/Float32")
        self.assertEqual(payload["candidate_scores"]["shape"], [2, 5])

    def test_missing_gpu_fails_unless_explicitly_allowed(self) -> None:
        capability = MojoGpuCapability(
            status=GPU_STATUS_UNAVAILABLE,
            probe=CommandResult(("gpu-query",), 2, "", "no gpu"),
        )

        self.assertEqual(
            exit_code_for_report(
                capability=capability,
                allow_missing_gpu=False,
                allow_missing_kernel=True,
            ),
            2,
        )
        self.assertEqual(
            exit_code_for_report(
                capability=capability,
                allow_missing_gpu=True,
                allow_missing_kernel=True,
            ),
            0,
        )

    def test_kernel_missing_fails_unless_explicitly_allowed(self) -> None:
        capability = MojoGpuCapability(
            status=GPU_STATUS_AVAILABLE,
            probe=CommandResult(("gpu-query",), 0, "ok", ""),
        )

        self.assertEqual(
            exit_code_for_report(
                capability=capability,
                allow_missing_gpu=True,
                allow_missing_kernel=False,
            ),
            3,
        )

    def test_probe_errors_are_not_treated_as_missing_gpu(self) -> None:
        tool_missing = MojoGpuCapability(
            status=GPU_STATUS_TOOL_MISSING,
            probe=CommandResult(("gpu-query",), None, "", "missing"),
        )
        timed_out = MojoGpuCapability(
            status=GPU_STATUS_ERROR,
            probe=CommandResult(("gpu-query",), None, "", "", True),
        )

        self.assertEqual(
            exit_code_for_report(
                capability=tool_missing,
                allow_missing_gpu=True,
                allow_missing_kernel=True,
            ),
            5,
        )
        self.assertEqual(
            exit_code_for_report(
                capability=timed_out,
                allow_missing_gpu=True,
                allow_missing_kernel=True,
            ),
            4,
        )

    def test_copy_probe_failure_blocks_gpu_labelled_report(self) -> None:
        capability = MojoGpuCapability(
            status=GPU_STATUS_AVAILABLE,
            probe=CommandResult(("gpu-query",), 0, "ok", ""),
        )

        self.assertEqual(
            report_status(
                capability,
                {"hardware_present": True},
                copy_probe_status="roundtrip_failed",
            ),
            GPU_RERANK_STATUS_COPY_PROBE_FAILED,
        )
        self.assertEqual(
            exit_code_for_report(
                capability=capability,
                allow_missing_gpu=True,
                allow_missing_kernel=True,
                copy_probe_status="roundtrip_failed",
            ),
            6,
        )

    def test_single_score_probe_failure_blocks_gpu_labelled_report(self) -> None:
        capability = MojoGpuCapability(
            status=GPU_STATUS_AVAILABLE,
            probe=CommandResult(("gpu-query",), 0, "ok", ""),
        )

        self.assertEqual(
            report_status(
                capability,
                {"hardware_present": True},
                copy_probe_status="ok",
                single_score_probe_status="score_agreement_failed",
            ),
            GPU_RERANK_STATUS_SINGLE_SCORE_FAILED,
        )
        self.assertEqual(
            exit_code_for_report(
                capability=capability,
                allow_missing_gpu=True,
                allow_missing_kernel=True,
                copy_probe_status="ok",
                single_score_probe_status="score_agreement_failed",
            ),
            7,
        )

    def test_candidate_score_probe_failure_blocks_gpu_labelled_report(self) -> None:
        capability = MojoGpuCapability(
            status=GPU_STATUS_AVAILABLE,
            probe=CommandResult(("gpu-query",), 0, "ok", ""),
        )

        self.assertEqual(
            report_status(
                capability,
                {"hardware_present": True},
                copy_probe_status="ok",
                single_score_probe_status="ok",
                candidate_score_probe_status="score_agreement_failed",
            ),
            GPU_RERANK_STATUS_CANDIDATE_SCORE_FAILED,
        )
        self.assertEqual(
            exit_code_for_report(
                capability=capability,
                allow_missing_gpu=True,
                allow_missing_kernel=True,
                copy_probe_status="ok",
                single_score_probe_status="ok",
                candidate_score_probe_status="score_agreement_failed",
            ),
            8,
        )

    def test_quiet_wrapper_section_prefers_same_candidate_reference(self) -> None:
        stdout = io.StringIO()
        report = {
            "cpu_reference": {
                "cpu_i8_same_candidate_reference": {
                    "query_batch_mean_seconds": 0.03125,
                },
                "cpu_i8_reference": {
                    "query_batch_mean_seconds": 0.125,
                },
                "exact_reference": {
                    "query_batch_mean_seconds": 9.0,
                },
            }
        }

        with redirect_stdout(stdout):
            print_quiet_wrapper_section(report)

        self.assertEqual(
            stdout.getvalue(),
            "== cpu_i8_same_candidate_reference_query_batch ==\nMean: 0.03125\n",
        )

    def test_same_candidate_score_ranking_preserves_candidate_ids(self) -> None:
        ranked = cpu_reference._rank_candidate_positions_by_score(
            ((7, 8, 9), (1, 2, 3)),
            ((0.1, 0.9, 0.2), (5.0, 5.0, 4.0)),
            final_k=2,
        )

        self.assertEqual(
            ranked,
            ((8, 9), (1, 2)),
        )

    def test_report_status_distinguishes_hardware_from_runtime(self) -> None:
        unavailable = MojoGpuCapability(
            status=GPU_STATUS_UNAVAILABLE,
            probe=CommandResult(("gpu-query",), 2, "", "no runtime"),
        )
        available = MojoGpuCapability(
            status=GPU_STATUS_AVAILABLE,
            probe=CommandResult(("gpu-query",), 0, "ok", ""),
        )

        self.assertEqual(
            report_status(unavailable, {"hardware_present": True}),
            GPU_RERANK_STATUS_RUNTIME_UNAVAILABLE,
        )
        self.assertEqual(
            report_status(unavailable, {"hardware_present": False}),
            GPU_RERANK_STATUS_NO_HARDWARE,
        )
        self.assertEqual(
            report_status(available, {"hardware_present": True}),
            GPU_RERANK_STATUS_BACKEND_INTEGRATION_MISSING,
        )
        self.assertEqual(
            GPU_RERANK_STATUS_KERNEL_MISSING,
            GPU_RERANK_STATUS_BATCH_KERNEL_MISSING,
        )
        self.assertEqual(
            GPU_RERANK_STATUS_BATCH_KERNEL_MISSING,
            GPU_RERANK_STATUS_BACKEND_INTEGRATION_MISSING,
        )

    def test_gpu_query_output_parser_records_device_identity(self) -> None:
        output = (
            "name: NVIDIA GeForce RTX 4070 Ti\n"
            "driver_version: 595.58.03\x00\x00\n"
            "memory: 10.15GB\n"
            "compute_capability: 8.9\n"
        )

        self.assertEqual(
            gpu_query_value(output, "name"),
            "NVIDIA GeForce RTX 4070 Ti",
        )
        self.assertEqual(
            gpu_query_value(output, "driver_version"),
            "595.58.03",
        )
        self.assertEqual(
            gpu_query_memory_bytes(output),
            10_150_000_000,
        )

    def test_profile_contract_keeps_timing_fields_separate(self) -> None:
        shape = bench_contract.SpeedTrackShape(
            document_count=8,
            document_vector_count=4,
            query_count=2,
            query_vector_count=3,
            vector_dim=128,
            top_k=5,
            update_document_count=0,
        )

        payload = gpu_profile_contract(shape, candidate_k=5)

        self.assertEqual(payload["timing_fields"], list(GPU_TIMING_FIELDS))
        self.assertIn("copy_cost_bound", {item["id"] for item in payload["hypotheses"]})
        self.assertEqual(payload["shape"]["total_document_vector_count"], 32)

    def test_copy_probe_counts_match_tensor_boundary_counts(self) -> None:
        shape = bench_contract.SpeedTrackShape(
            document_count=8,
            document_vector_count=4,
            query_count=2,
            query_vector_count=3,
            vector_dim=128,
            top_k=5,
            update_document_count=0,
        )

        counts = gpu_copy_probe_counts(shape, candidate_k=5)

        self.assertEqual(counts["int8_count"], 8 * 4 * 128)
        self.assertEqual(counts["int64_count"], 9 + 10)
        self.assertEqual(counts["float32_count"], (2 * 3 * 128) + 32 + 10)

    def test_single_score_probe_counts_keep_candidate_score_explicit(self) -> None:
        shape = bench_contract.SpeedTrackShape(
            document_count=8,
            document_vector_count=4,
            query_count=2,
            query_vector_count=3,
            vector_dim=128,
            top_k=5,
            update_document_count=0,
        )

        counts = gpu_single_score_probe_counts(shape)

        self.assertEqual(counts["query_vector_count"], 3)
        self.assertEqual(counts["document_vector_count"], 4)
        self.assertEqual(counts["query_value_count"], 3 * 128)
        self.assertEqual(counts["token_code_count"], 4 * 128)
        self.assertEqual(counts["candidate_position_count"], 1)
        self.assertEqual(counts["candidate_score_count"], 1)

    def test_candidate_score_probe_counts_match_tensor_boundary_counts(self) -> None:
        shape = bench_contract.SpeedTrackShape(
            document_count=8,
            document_vector_count=4,
            query_count=2,
            query_vector_count=3,
            vector_dim=128,
            top_k=5,
            update_document_count=0,
        )

        counts = gpu_candidate_score_probe_counts(shape, candidate_k=5)

        self.assertEqual(counts["query_value_count"], 2 * 3 * 128)
        self.assertEqual(counts["token_code_count"], 8 * 4 * 128)
        self.assertEqual(counts["token_scale_count"], 8 * 4)
        self.assertEqual(counts["doc_offset_count"], 9)
        self.assertEqual(counts["candidate_position_count"], 10)
        self.assertEqual(counts["candidate_score_count"], 10)

    def test_real_payload_probe_input_manifest_records_binary_counts(self) -> None:
        shape = bench_contract.SpeedTrackShape(
            document_count=2,
            document_vector_count=2,
            query_count=1,
            query_vector_count=2,
            vector_dim=128,
            top_k=1,
            update_document_count=0,
        )
        payload = KayakPlaidI8PayloadSnapshot(
            doc_offsets=np.asarray([0, 2, 4], dtype=np.int64),
            token_codes=np.arange(4 * 128, dtype=np.int16).astype(np.int8),
            token_scales=np.ones(4, dtype=np.float32),
            centroid_token_indices=np.asarray([0, 2], dtype=np.int64),
            centroid_doc_offsets=np.asarray([0, 1, 2], dtype=np.int64),
            centroid_doc_indices=np.asarray([0, 1], dtype=np.int64),
            document_count=2,
            total_vector_count=4,
            vector_dim=128,
        )
        temp_root = REPO_ROOT / ".cache" / "kayak" / "test_gpu_real_payload"

        manifest = write_gpu_i8_real_payload_probe_inputs(
            temp_root,
            shape=shape,
            candidate_k=2,
            queries=np.zeros((1, 2, 128), dtype=np.float32),
            payload=payload,
            candidate_positions_by_query=((0, 1),),
            reference_scores_by_query=((1.0, 2.0),),
        )

        counts = gpu_real_payload_probe_counts(shape, candidate_k=2)
        self.assertEqual(manifest["counts"], counts)
        self.assertEqual(manifest["byte_counts"]["token_codes"], 4 * 128)
        self.assertEqual(manifest["byte_counts"]["candidate_positions"], 2 * 8)
        self.assertTrue((temp_root / "manifest.json").exists())

    def test_real_payload_profile_requires_bridge_success_when_gpu_available(
        self,
    ) -> None:
        capability = MojoGpuCapability(
            status=GPU_STATUS_AVAILABLE,
            probe=CommandResult(("gpu-query",), 0, "ok", ""),
        )

        self.assertEqual(
            real_payload_profile.report_status(
                capability=capability,
                gpu_probe={"status": GPU_REAL_PAYLOAD_STATUS_OK},
                bridge_probe={"status": real_payload_profile.STATUS_OK},
                prepared_bridge_probe={"status": real_payload_profile.STATUS_OK},
                prepared_ndarray_bridge_probe={
                    "status": real_payload_profile.STATUS_OK
                },
                prepared_address_bridge_probe={
                    "status": real_payload_profile.STATUS_OK
                },
                address_serve_probe={"status": real_payload_profile.STATUS_OK},
            ),
            real_payload_profile.STATUS_OK,
        )
        self.assertEqual(
            real_payload_profile.report_status(
                capability=capability,
                gpu_probe={"status": GPU_REAL_PAYLOAD_STATUS_OK},
                bridge_probe={"status": "error"},
                prepared_bridge_probe={"status": real_payload_profile.STATUS_OK},
                prepared_ndarray_bridge_probe={
                    "status": real_payload_profile.STATUS_OK
                },
                prepared_address_bridge_probe={
                    "status": real_payload_profile.STATUS_OK
                },
                address_serve_probe={"status": real_payload_profile.STATUS_OK},
            ),
            real_payload_profile.STATUS_BLOCKED_GPU_BRIDGE_FAILED,
        )
        self.assertEqual(
            real_payload_profile.report_status(
                capability=capability,
                gpu_probe={"status": GPU_REAL_PAYLOAD_STATUS_OK},
                bridge_probe={"status": real_payload_profile.STATUS_OK},
                prepared_bridge_probe={"status": "error"},
                prepared_ndarray_bridge_probe={
                    "status": real_payload_profile.STATUS_OK
                },
                prepared_address_bridge_probe={
                    "status": real_payload_profile.STATUS_OK
                },
                address_serve_probe={"status": real_payload_profile.STATUS_OK},
            ),
            real_payload_profile.STATUS_BLOCKED_GPU_PREPARED_BRIDGE_FAILED,
        )
        self.assertEqual(
            real_payload_profile.report_status(
                capability=capability,
                gpu_probe={"status": GPU_REAL_PAYLOAD_STATUS_OK},
                bridge_probe={"status": real_payload_profile.STATUS_OK},
                prepared_bridge_probe={"status": real_payload_profile.STATUS_OK},
                prepared_ndarray_bridge_probe={"status": "error"},
                prepared_address_bridge_probe={
                    "status": real_payload_profile.STATUS_OK
                },
                address_serve_probe={"status": real_payload_profile.STATUS_OK},
            ),
            real_payload_profile.STATUS_BLOCKED_GPU_PREPARED_NDARRAY_BRIDGE_FAILED,
        )
        self.assertEqual(
            real_payload_profile.report_status(
                capability=capability,
                gpu_probe={"status": GPU_REAL_PAYLOAD_STATUS_OK},
                bridge_probe={"status": real_payload_profile.STATUS_OK},
                prepared_bridge_probe={"status": real_payload_profile.STATUS_OK},
                prepared_ndarray_bridge_probe={
                    "status": real_payload_profile.STATUS_OK
                },
                prepared_address_bridge_probe={"status": "error"},
                address_serve_probe={"status": real_payload_profile.STATUS_OK},
            ),
            real_payload_profile.STATUS_BLOCKED_GPU_PREPARED_ADDRESS_BRIDGE_FAILED,
        )
        self.assertEqual(
            real_payload_profile.report_status(
                capability=capability,
                gpu_probe={"status": GPU_REAL_PAYLOAD_STATUS_OK},
                bridge_probe={"status": real_payload_profile.STATUS_OK},
                prepared_bridge_probe={"status": real_payload_profile.STATUS_OK},
                prepared_ndarray_bridge_probe={
                    "status": real_payload_profile.STATUS_OK
                },
                prepared_address_bridge_probe={
                    "status": real_payload_profile.STATUS_OK
                },
                address_serve_probe={"status": "error"},
            ),
            real_payload_profile.STATUS_BLOCKED_GPU_ADDRESS_SERVE_FAILED,
        )

    def test_gpu_bridge_result_reports_profile_boundary_fields(self) -> None:
        result = MojoGpuI8RerankBridgeResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            host_to_device_mean_seconds=0.003,
            kernel_mean_seconds=0.004,
            device_to_host_mean_seconds=0.005,
            score_delta_max_abs=0.00001,
            candidate_score_count=2,
            scores=(1.0, 2.0),
        )

        payload = result.to_json_ready()

        self.assertEqual(payload["candidate_score_count"], 2)
        self.assertEqual(payload["score_count"], 2)
        self.assertEqual(payload["host_marshalling_seconds"], 0.1)

    def test_prepared_gpu_bridge_result_reports_prepare_and_score_boundaries(
        self,
    ) -> None:
        result = MojoGpuI8PreparedSessionResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            prepare_host_to_device_mean_seconds=0.003,
            score_host_to_device_mean_seconds=0.004,
            kernel_mean_seconds=0.005,
            device_to_host_mean_seconds=0.006,
            score_delta_max_abs=0.00001,
            candidate_score_count=2,
            scores=(1.0, 2.0),
        )

        payload = result.to_json_ready()

        self.assertEqual(payload["candidate_score_count"], 2)
        self.assertEqual(payload["score_count"], 2)
        self.assertEqual(payload["prepare_host_to_device_mean_seconds"], 0.003)
        self.assertEqual(payload["score_host_to_device_mean_seconds"], 0.004)

    def test_address_serve_result_reports_serving_call_boundary(self) -> None:
        result = MojoGpuI8AddressServeResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            score_delta_max_abs=0.00001,
            candidate_score_count=2,
            scores=(1.0, 2.0),
        )

        payload = result.to_json_ready()

        self.assertEqual(payload["candidate_score_count"], 2)
        self.assertEqual(payload["score_count"], 2)
        self.assertEqual(payload["host_marshalling_seconds"], 0.1)
        self.assertEqual(payload["extension_call_seconds"], 0.2)

    def test_address_resident_session_result_reports_iteration_boundary(
        self,
    ) -> None:
        result = MojoGpuI8AddressResidentSessionResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            score_delta_max_abs=0.00001,
            candidate_score_count=2,
            session_iterations=4,
            scores=(1.0, 2.0),
        )

        payload = result.to_json_ready()

        self.assertEqual(payload["candidate_score_count"], 2)
        self.assertEqual(payload["score_count"], 2)
        self.assertEqual(payload["session_iterations"], 4)
        self.assertEqual(payload["extension_call_seconds_per_iteration"], 0.05)

    def test_address_multi_window_result_reports_window_boundary(self) -> None:
        result = MojoGpuI8AddressMultiWindowSessionResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            score_delta_max_abs=0.00001,
            candidate_score_count_per_window=2,
            window_count=4,
            scores=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
        )

        payload = result.to_json_ready()

        self.assertEqual(payload["candidate_score_count_per_window"], 2)
        self.assertEqual(payload["candidate_score_count_total"], 8)
        self.assertEqual(payload["score_count"], 8)
        self.assertEqual(payload["window_count"], 4)
        self.assertEqual(payload["extension_call_seconds_per_window"], 0.05)

    def test_address_session_handle_reports_explicit_prepare_boundary(
        self,
    ) -> None:
        shape = bench_contract.SpeedTrackShape(
            document_count=8,
            document_vector_count=4,
            query_count=2,
            query_vector_count=3,
            vector_dim=128,
            top_k=5,
            update_document_count=0,
        )
        handle = MojoGpuI8AddressSessionHandle(
            target_accelerator="nvidia:sm_89",
            handle=123,
            shape=shape,
            candidate_k=5,
            prepare_host_marshalling_seconds=0.001,
            prepare_extension_call_seconds=0.002,
        )

        payload = handle.to_json_ready()

        self.assertTrue(payload["handle_open"])
        self.assertEqual(payload["document_count"], 8)
        self.assertEqual(payload["document_vector_count"], 4)
        self.assertEqual(payload["query_count"], 2)
        self.assertEqual(payload["query_vector_count"], 3)
        self.assertEqual(payload["candidate_k"], 5)
        self.assertEqual(payload["prepare_extension_call_seconds"], 0.002)

    def test_address_topk_result_reports_return_boundary(self) -> None:
        result = MojoGpuI8AddressTopKResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            score_delta_max_abs=0.00001,
            candidate_score_count=20,
            top_k=5,
            topk_position_match_count=10,
            positions=tuple(range(10)),
            scores=tuple(float(value) for value in range(10)),
        )

        payload = result.to_json_ready()

        self.assertEqual(payload["candidate_score_count"], 20)
        self.assertEqual(payload["top_k"], 5)
        self.assertEqual(payload["topk_position_count"], 10)
        self.assertEqual(payload["topk_position_match_count"], 10)
        self.assertEqual(payload["topk_position_agreement"], 1.0)
        self.assertEqual(payload["score_count"], 10)

    def test_address_topk_no_reference_result_reports_return_boundary(self) -> None:
        result = MojoGpuI8AddressTopKNoReferenceResult(
            host_marshalling_seconds=0.1,
            extension_call_seconds=0.2,
            candidate_score_count=20,
            top_k=5,
            positions=tuple(range(10)),
            scores=tuple(float(value) for value in range(10)),
        )

        payload = result.to_json_ready()

        self.assertEqual(payload["candidate_score_count"], 20)
        self.assertEqual(payload["top_k"], 5)
        self.assertEqual(payload["topk_position_count"], 10)
        self.assertEqual(payload["score_count"], 10)

    def test_address_serve_sweep_case_parser_keeps_vector_counts_explicit(
        self,
    ) -> None:
        case = address_serve_sweep.parse_sweep_case(
            "wide:documents=512,document_vectors=32,"
            "queries=2,query_vectors=16,candidate_k=128"
        )

        self.assertEqual(case.name, "wide")
        self.assertEqual(case.document_count, 512)
        self.assertEqual(case.document_vector_count, 32)
        self.assertEqual(case.query_count, 2)
        self.assertEqual(case.query_vector_count, 16)
        self.assertEqual(case.candidate_k, 128)
        self.assertEqual(
            case.shape(vector_dim=128, top_k=10).total_document_vector_count,
            16_384,
        )

    def test_address_serve_sweep_case_sets_make_wide_topk_explicit(
        self,
    ) -> None:
        case_sets = address_serve_sweep.CASE_SETS
        default_cases = case_sets["default"]
        wide_cases = case_sets["wide_topk"]

        self.assertGreater(
            max(case.candidate_k for case in wide_cases),
            max(case.candidate_k for case in default_cases),
        )
        self.assertTrue(
            any(case.query_vector_count > 16 for case in wide_cases)
        )
        self.assertTrue(
            any(case.document_vector_count > 32 for case in wide_cases)
        )
        for case in wide_cases:
            shape = case.shape(vector_dim=128, top_k=10)
            self.assertEqual(shape.vector_dim, 128)
            self.assertGreaterEqual(case.candidate_k, 10)
            self.assertLessEqual(case.candidate_k, case.document_count)
            self.assertGreater(shape.total_document_vector_count, 0)

    def test_address_serve_sweep_cli_selects_named_case_sets(self) -> None:
        args = address_sweep_script.parse_args(["--case-set", "wide_topk"])

        self.assertEqual(
            address_sweep_script.cases_from_args(args),
            address_serve_sweep.CASE_SETS["wide_topk"],
        )
        self.assertEqual(
            address_sweep_script.case_selection_from_args(args),
            {"source": "wide_topk", "case_count": 5},
        )

    def test_address_serve_sweep_cli_custom_cases_override_case_set(self) -> None:
        args = address_sweep_script.parse_args(
            [
                "--case-set",
                "wide_topk",
                "--case",
                (
                    "smoke:documents=64,document_vectors=8,queries=1,"
                    "query_vectors=4,candidate_k=16"
                ),
            ]
        )

        cases = address_sweep_script.cases_from_args(args)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].name, "smoke")
        self.assertEqual(
            address_sweep_script.case_selection_from_args(args),
            {"source": "custom", "case_count": 1},
        )

    def test_centroid_budget_sweep_parses_sorted_unique_budgets(self) -> None:
        self.assertEqual(
            centroid_budget_sweep.parse_centroid_budgets("32, 8,16,8"),
            (8, 16, 32),
        )

        with self.assertRaisesRegex(ValueError, "positive"):
            centroid_budget_sweep.parse_centroid_budgets("4,0")

    def test_centroid_budget_sweep_filters_non_full_windows(self) -> None:
        cases = centroid_budget_sweep.non_full_candidate_cases(
            address_serve_sweep.CASE_SETS["wide_topk"],
            include_full_window=False,
        )

        self.assertEqual(
            tuple(case.name for case in cases),
            ("query_vectors32", "doc_vectors64", "query_batch4"),
        )

    def test_candidate_window_recall_uses_whole_candidate_window(self) -> None:
        recall = centroid_budget_sweep.candidate_window_recall_at_k(
            candidate_positions_by_query=((4, 1, 2), (9, 7, 0)),
            reference_positions_by_query=((1, 2, 3), (7, 8, 9)),
            k=2,
        )

        self.assertAlmostEqual(recall, 0.75)

    def test_centroid_budget_sweep_adds_baseline_ratios(self) -> None:
        rows = centroid_budget_sweep.add_baseline_ratios(
            [
                {
                    "centroids_per_query_vector": 8,
                    "cpu_i8_candidate_generation": {"mean_seconds": 0.25},
                    "candidate_window_recall_at_k_vs_kayak_exact": 1.0,
                    "recall_at_k_vs_kayak_exact": 0.9,
                    "comparison": {
                        "cpu_candidate_generation_plus_score_seconds": 0.5
                    },
                },
                {
                    "centroids_per_query_vector": 32,
                    "cpu_i8_candidate_generation": {"mean_seconds": 1.0},
                    "candidate_window_recall_at_k_vs_kayak_exact": 1.0,
                    "recall_at_k_vs_kayak_exact": 0.9,
                    "comparison": {
                        "cpu_candidate_generation_plus_score_seconds": 1.0
                    },
                },
            ],
            baseline_budget=32,
        )

        self.assertEqual(
            rows[0]["comparison"][
                "candidate_generation_seconds_vs_baseline_budget"
            ],
            0.25,
        )
        self.assertEqual(
            rows[0]["comparison"][
                "candidate_plus_score_seconds_vs_baseline_budget"
            ],
            0.5,
        )
        self.assertEqual(
            rows[0]["comparison"]["final_recall_delta_vs_baseline_budget"],
            0.0,
        )

    def test_gpu_i8_score_agreement_tolerance_scales_with_query_vectors(
        self,
    ) -> None:
        self.assertEqual(gpu_i8_score_delta_tolerance(8), 1.0e-4)
        self.assertEqual(gpu_i8_score_delta_tolerance(32), 3.2e-4)

        fields = gpu_i8_score_agreement_fields(
            score_delta_max_abs=0.000244140625,
            query_vector_count=32,
        )

        self.assertEqual(fields["score_delta_tolerance"], 3.2e-4)
        self.assertIs(fields["score_agreement_ok"], True)

    def test_address_serve_sweep_comparison_reports_isolated_and_envelope_ratios(
        self,
    ) -> None:
        comparison = address_serve_sweep.comparison_payload(
            cpu_candidate_generation_mean_seconds=0.003,
            cpu_score_mean_seconds=0.002,
            gpu_parsed={"extension_call_seconds": 0.001},
            resident_parsed={"extension_call_seconds_per_iteration": 0.0005},
            cpu_multi_window_candidate_generation_mean_seconds_per_window=0.004,
            cpu_multi_window_score_mean_seconds_per_window=0.002,
            multi_window_parsed={"extension_call_seconds_per_window": 0.001},
            prepared_handle_parsed={
                "score_extension_call_seconds_per_window": 0.00075
            },
            prepared_handle_topk_parsed={
                "score_extension_call_seconds_per_window": 0.0009
            },
            prepared_handle_topk_no_reference_parsed={
                "score_extension_call_seconds_per_window": 0.0004
            },
        )

        self.assertEqual(
            comparison[
                "gpu_address_serve_extension_call_seconds_per_cpu_score_second"
            ],
            0.5,
        )
        self.assertEqual(
            comparison[
                "cpu_candidate_plus_gpu_address_serve_seconds_per_cpu_candidate_plus_score_second"
            ],
            0.8,
        )
        self.assertEqual(
            comparison[
                "gpu_address_resident_iteration_seconds_per_cpu_score_second"
            ],
            0.25,
        )
        self.assertEqual(
            comparison[
                "cpu_candidate_plus_gpu_resident_iteration_seconds_per_cpu_candidate_plus_score_second"
            ],
            0.7,
        )
        self.assertEqual(
            comparison[
                "gpu_address_resident_multi_window_seconds_per_cpu_multi_window_score_second"
            ],
            0.5,
        )
        self.assertEqual(
            comparison[
                "cpu_multi_window_candidate_plus_gpu_resident_multi_window_seconds_per_cpu_multi_window_candidate_plus_score_second"
            ],
            5.0 / 6.0,
        )
        self.assertEqual(
            comparison[
                "gpu_address_prepared_handle_score_seconds_per_cpu_multi_window_score_second"
            ],
            0.375,
        )
        self.assertEqual(
            comparison[
                "cpu_multi_window_candidate_plus_gpu_prepared_handle_score_seconds_per_cpu_multi_window_candidate_plus_score_second"
            ],
            4.75 / 6.0,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_address_prepared_handle_topk_seconds_per_cpu_multi_window_score_second"
            ],
            0.45,
        )
        self.assertEqual(
            comparison[
                "gpu_address_prepared_handle_topk_seconds_per_prepared_handle_score_second"
            ],
            1.2,
        )
        self.assertAlmostEqual(
            comparison[
                "cpu_multi_window_candidate_plus_gpu_prepared_handle_topk_seconds_per_cpu_multi_window_candidate_plus_score_second"
            ],
            4.9 / 6.0,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_address_prepared_handle_topk_no_reference_seconds_per_cpu_multi_window_score_second"
            ],
            0.2,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_address_prepared_handle_topk_no_reference_seconds_per_prepared_handle_score_second"
            ],
            0.0004 / 0.00075,
        )
        self.assertAlmostEqual(
            comparison[
                "gpu_address_prepared_handle_topk_no_reference_seconds_per_validating_topk_second"
            ],
            0.0004 / 0.0009,
        )
        self.assertAlmostEqual(
            comparison[
                "cpu_multi_window_candidate_plus_gpu_prepared_handle_topk_no_reference_seconds_per_cpu_multi_window_candidate_plus_score_second"
            ],
            4.4 / 6.0,
        )

    def test_candidate_score_profile_derives_rates_and_ratios(self) -> None:
        derived = candidate_profile.derive_candidate_score_metrics(
            {
                "candidate_score_count": 64,
                "serial_kernel_mean_seconds": 0.000512,
                "twopass_kernel_mean_seconds": 0.000016,
                "cpu_reference_mean_seconds": 0.002048,
                "host_to_device_mean_seconds": 0.000004,
                "device_to_host_mean_seconds": 0.000002,
            }
        )

        self.assertEqual(
            derived["twopass_kernel_candidate_scores_per_second"],
            4_000_000.0,
        )
        self.assertEqual(
            derived["twopass_kernel_speedup_vs_serial_kernel"],
            32.0,
        )
        self.assertEqual(
            derived["twopass_kernel_speedup_vs_cpu_scalar_reference"],
            128.0,
        )
        self.assertAlmostEqual(
            float(derived["twopass_copy_to_kernel_ratio"]),
            0.375,
        )

    def test_gpu_fastplaid_compare_keeps_scope_ratios_explicit(self) -> None:
        comparison = fastplaid_compare.build_gpu_vs_fastplaid_comparison(
            gpu_probe={
                "status": "ok",
                "parsed": {
                    "candidate_score_count": 256,
                    "host_to_device_mean_seconds": 0.00002,
                    "twopass_kernel_mean_seconds": 0.00003,
                    "device_to_host_mean_seconds": 0.00001,
                },
            },
            fastplaid_row={
                "system_name": "fastplaid",
                "status": "ok",
                "query_batch_mean_seconds": 0.03,
                "query_mean_seconds": 0.015,
            },
        )

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["candidate_score_count_total"], 256)
        self.assertAlmostEqual(
            float(comparison["gpu_kernel_seconds_per_fastplaid_batch_second"]),
            0.001,
        )
        self.assertAlmostEqual(
            float(
                comparison[
                    "gpu_h2d_kernel_d2h_seconds_per_fastplaid_batch_second"
                ]
            ),
            0.002,
        )
        self.assertIn("not an apples-to-apples", comparison["scope_warning"])

    def test_gpu_fastplaid_compare_reports_prepared_topk_boundary(
        self,
    ) -> None:
        comparison = (
            fastplaid_compare.build_gpu_prepared_topk_vs_fastplaid_comparison(
                prepared_handle_topk_row={
                    "status": "ok",
                    "cpu_i8_multi_window_candidate_generation": {
                        "mean_seconds_per_window": 0.001,
                    },
                    "cpu_i8_multi_window_same_candidate_reference": {
                        "mean_seconds_per_window": 0.002,
                    },
                    "parsed": {
                        "candidate_score_count_per_window": 256,
                        "topk_return_count_per_window": 20,
                        "topk_position_agreement": 1.0,
                        "score_delta_max_abs": 0.000061,
                        "score_delta_tolerance": 0.0001,
                        "score_extension_call_seconds_per_window": 0.0006,
                    },
                },
                fastplaid_row={
                    "system_name": "fastplaid",
                    "status": "ok",
                    "query_batch_mean_seconds": 0.03,
                    "query_mean_seconds": 0.015,
                },
            )
        )

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["candidate_score_count_per_window"], 256)
        self.assertEqual(comparison["topk_return_count_per_window"], 20)
        self.assertEqual(comparison["score_delta_tolerance"], 0.0001)
        self.assertAlmostEqual(
            float(
                comparison[
                    "gpu_prepared_handle_topk_seconds_per_fastplaid_batch_second"
                ]
            ),
            0.02,
        )
        self.assertAlmostEqual(
            float(
                comparison[
                    "cpu_candidate_generation_plus_gpu_topk_seconds_per_cpu_candidate_generation_plus_cpu_score_second"
                ]
            ),
            0.0016 / 0.003,
        )
        self.assertIn("not an apples-to-apples", comparison["scope_warning"])

    def test_gpu_fastplaid_compare_reports_no_reference_topk_boundary(
        self,
    ) -> None:
        comparison = (
            fastplaid_compare.build_gpu_prepared_topk_no_reference_vs_fastplaid_comparison(
                prepared_handle_topk_row={
                    "status": "ok",
                    "no_reference_status": "ok",
                    "cpu_i8_multi_window_candidate_generation": {
                        "mean_seconds_per_window": 0.001,
                    },
                    "cpu_i8_multi_window_same_candidate_reference": {
                        "mean_seconds_per_window": 0.002,
                    },
                    "no_reference_parsed": {
                        "candidate_score_count_per_window": 256,
                        "topk_return_count_per_window": 20,
                        "topk_position_agreement": 1.0,
                        "score_extension_call_seconds_per_window": 0.0005,
                        "validation_reference_scores_sent_to_extension": False,
                    },
                },
                fastplaid_row={
                    "system_name": "fastplaid",
                    "status": "ok",
                    "query_batch_mean_seconds": 0.025,
                    "query_mean_seconds": 0.0125,
                },
            )
        )

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["candidate_score_count_per_window"], 256)
        self.assertIs(
            comparison["validation_reference_scores_sent_to_extension"],
            False,
        )
        self.assertAlmostEqual(
            float(
                comparison[
                    "gpu_prepared_handle_topk_seconds_per_fastplaid_batch_second"
                ]
            ),
            0.02,
        )

    def test_gpu_fastplaid_compare_reports_fused_device_topk_boundary(
        self,
    ) -> None:
        comparison = (
            fastplaid_compare.build_gpu_fused_centroid_posting_vs_fastplaid_comparison(
                fused_row={
                    "status": "ok",
                    "recall_at_k_vs_kayak_exact": 0.8,
                    "parsed": {
                        "candidate_score_count_per_window": 1024,
                        "topk_return_count_per_window": 20,
                        "topk_position_agreement": 1.0,
                        "device_topk_position_agreement": 1.0,
                        "topk_score_delta_max_abs": 0.000061,
                        "device_topk_score_delta_max_abs": 0.000061,
                        "score_extension_call_seconds_per_window": 0.00012,
                        "device_topk_score_extension_call_seconds_per_window": 0.00009,
                        "validation_reference_scores_sent_to_extension": False,
                    },
                },
                fastplaid_row={
                    "system_name": "fastplaid",
                    "status": "ok",
                    "query_batch_mean_seconds": 0.003,
                    "query_mean_seconds": 0.0015,
                },
            )
        )

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["candidate_score_count_per_window"], 1024)
        self.assertEqual(comparison["topk_return_count_per_window"], 20)
        self.assertIs(
            comparison["validation_reference_scores_sent_to_extension"],
            False,
        )
        self.assertAlmostEqual(
            float(
                comparison[
                    "gpu_fused_device_topk_seconds_per_fastplaid_batch_second"
                ]
            ),
            0.03,
        )
        self.assertAlmostEqual(
            float(
                comparison[
                    "gpu_fused_device_topk_seconds_per_host_topk_second"
                ]
            ),
            0.75,
        )
        self.assertEqual(comparison["recall_at_k_vs_kayak_exact"], 0.8)

    def test_gpu_fastplaid_compare_reports_hybrid_shortlist_rerank_boundary(
        self,
    ) -> None:
        comparison = (
            fastplaid_compare.build_gpu_hybrid_shortlist_exact_rerank_vs_fastplaid_comparison(
                hybrid_row={
                    "status": "ok",
                    "recall_at_k_vs_kayak_exact": 0.75,
                    "parsed": {
                        "candidate_score_count_per_window": 512,
                        "shortlist_k": 256,
                        "topk_return_count_per_window": 20,
                        "final_topk_position_agreement": 1.0,
                        "exact_score_delta_max_abs": 0.000061,
                        "hybrid_extension_seconds_per_window": 0.0002,
                        "fused_device_topk_seconds_per_window": 0.00008,
                        "exact_rerank_topk_seconds_per_window": 0.00012,
                        "validation_reference_scores_sent_to_extension": False,
                    },
                },
                fastplaid_row={
                    "system_name": "fastplaid",
                    "status": "ok",
                    "query_batch_mean_seconds": 0.004,
                    "query_mean_seconds": 0.002,
                },
            )
        )

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["candidate_score_count_per_window"], 512)
        self.assertEqual(comparison["shortlist_k"], 256)
        self.assertEqual(comparison["topk_return_count_per_window"], 20)
        self.assertIs(
            comparison["validation_reference_scores_sent_to_extension"],
            False,
        )
        self.assertAlmostEqual(
            float(comparison["gpu_hybrid_seconds_per_fastplaid_batch_second"]),
            0.05,
        )
        self.assertAlmostEqual(
            float(comparison["gpu_hybrid_exact_rerank_share"]),
            0.6,
        )
        self.assertEqual(comparison["recall_at_k_vs_kayak_exact"], 0.75)

    def test_gpu_fastplaid_compare_reports_resident_selected_boundary(
        self,
    ) -> None:
        comparison = (
            fastplaid_compare.build_gpu_resident_selected_posting_exact_rerank_vs_fastplaid_comparison(
                resident_row={
                    "status": "ok",
                    "recall_at_k_vs_kayak_exact": 0.76,
                    "parsed": {
                        "candidate_score_count_per_window": 512,
                        "candidate_k": 256,
                        "topk_return_count_per_window": 20,
                        "candidate_position_agreement_min": 1.0,
                        "final_topk_position_agreement": 1.0,
                        "exact_score_delta_max_abs": 0.0,
                        "resident_candidate_seconds_per_window": 0.00016,
                        "resident_candidate_cold_seconds_per_window": 0.00020,
                        "exact_rerank_topk_seconds_per_window": 0.00004,
                        "resident_selected_posting_exact_rerank_seconds_per_window": 0.00020,
                        "resident_selected_posting_exact_rerank_cold_seconds_per_window": 0.00024,
                        "validation_reference_scores_sent_to_extension": False,
                    },
                },
                fastplaid_row={
                    "system_name": "fastplaid",
                    "status": "ok",
                    "query_batch_mean_seconds": 0.004,
                    "query_mean_seconds": 0.002,
                },
            )
        )

        self.assertEqual(comparison["status"], "ok")
        self.assertEqual(comparison["candidate_score_count_per_window"], 512)
        self.assertEqual(comparison["candidate_k"], 256)
        self.assertIs(
            comparison["validation_reference_scores_sent_to_extension"],
            False,
        )
        self.assertAlmostEqual(
            float(
                comparison[
                    "gpu_resident_selected_exact_rerank_seconds_per_fastplaid_batch_second"
                ]
            ),
            0.05,
        )
        self.assertAlmostEqual(
            float(
                comparison[
                    "gpu_resident_selected_exact_rerank_cold_seconds_per_fastplaid_batch_second"
                ]
            ),
            0.06,
        )
        self.assertAlmostEqual(
            float(comparison["gpu_resident_selected_exact_rerank_exact_share"]),
            0.2,
        )
        self.assertEqual(comparison["recall_at_k_vs_kayak_exact"], 0.76)

    def test_copy_probe_parser_keeps_timing_and_roundtrip_fields(self) -> None:
        parsed = parse_gpu_copy_probe_output(
            "status: ok\n"
            "float32_h2d_mean_seconds: 1.25e-06\n"
            "int8_count: 4096\n"
            "int64_roundtrip_ok: True\n"
        )

        self.assertEqual(parsed["status"], "ok")
        self.assertEqual(parsed["int8_count"], 4096)
        self.assertAlmostEqual(
            float(parsed["float32_h2d_mean_seconds"]),
            1.25e-06,
        )
        self.assertIs(parsed["int64_roundtrip_ok"], True)

    def test_profile_decision_status_requires_repeated_complete_evidence(self) -> None:
        shape = bench_contract.SpeedTrackShape(
            document_count=8,
            document_vector_count=4,
            query_count=2,
            query_vector_count=3,
            vector_dim=128,
            top_k=5,
            update_document_count=0,
        )
        row = gpu_profile_row_template(shape, candidate_k=5)

        self.assertEqual(
            gpu_profile_decision_status(row),
            PROFILE_STATUS_EXPLORATORY,
        )

        row.update(
            {
                "build_or_prepare_seconds": 0.1,
                "host_to_device_seconds": 0.01,
                "kernel_seconds": 0.02,
                "device_to_host_seconds": 0.01,
                "end_to_end_seconds": 0.04,
                "score_delta_max_abs": 0.0,
                "candidate_order_agreement": 1.0,
                "recall_at_k_vs_cpu_i8": 1.0,
                "recall_at_k_vs_kayak_exact": 0.9,
                "run_count": 3,
            }
        )

        self.assertEqual(
            gpu_profile_decision_status(row),
            PROFILE_STATUS_DECISION_READY,
        )

        row["score_delta_max_abs"] = 1.0
        self.assertEqual(
            gpu_profile_decision_status(row),
            PROFILE_STATUS_FALSIFIED,
        )


if __name__ == "__main__":
    unittest.main()
