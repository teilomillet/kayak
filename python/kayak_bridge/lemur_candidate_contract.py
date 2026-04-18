"""Owns an auditable artifact contract for reference LEMUR candidate sweeps.

This contract keeps the benchmark output explicit:
- one exact Kayak baseline on the same encoded task
- one reference LEMUR candidate sweep on the same task
- one compact bundle that selects readable winners from that sweep

It does not own task loading or CLI parsing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import kayak

from .kayak_task_benchmark import benchmark_task_with_kayak_exact
from .lemur_candidate_bundle import build_lemur_candidate_bundle
from .lemur_candidate_sweep import benchmark_reference_lemur_candidate_sweep


@dataclass(frozen=True, slots=True)
class LemurCandidateContractBundle:
    contract_name: str
    exact_path: str
    sweep_path: str
    bundle_path: str

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def benchmark_reference_lemur_contract(
    *,
    task: Mapping[str, Any],
    output_root: Path,
    artifact_prefix: str,
    latent_dims: Sequence[int],
    candidate_ks: Sequence[int],
    activation: str = "gelu",
    query_divisor: float = 32.0,
    landmark_count: int | None = None,
    feature_weights: Any | None = None,
    landmark_vectors: Any | None = None,
    apply_layer_norm: bool = True,
    layer_norm_eps: float = 1e-5,
    pinv_rcond: float = 1e-6,
    seed: int = 0,
    exact_backend: str = kayak.NUMPY_REFERENCE_BACKEND,
    rerank_backend: str = kayak.NUMPY_REFERENCE_BACKEND,
    warmup_iterations: int = 2,
    measurement_iterations: int = 25,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)

    exact_path = output_root / f"{artifact_prefix}_kayak_exact_benchmark.json"
    sweep_path = output_root / f"{artifact_prefix}_lemur_candidate_sweep.json"
    bundle_path = output_root / f"{artifact_prefix}_lemur_candidate_bundle.json"
    contract_path = output_root / f"{artifact_prefix}_lemur_contract_bundle.json"

    exact_summary = benchmark_task_with_kayak_exact(
        task,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
        backend=exact_backend,
    ).to_json_ready()
    _write_json(exact_path, exact_summary)

    sweep_summary = benchmark_reference_lemur_candidate_sweep(
        task,
        latent_dims=latent_dims,
        candidate_ks=candidate_ks,
        activation=activation,
        query_divisor=query_divisor,
        landmark_count=landmark_count,
        feature_weights=feature_weights,
        landmark_vectors=landmark_vectors,
        apply_layer_norm=apply_layer_norm,
        layer_norm_eps=layer_norm_eps,
        pinv_rcond=pinv_rcond,
        seed=seed,
        exact_backend=exact_backend,
        rerank_backend=rerank_backend,
        warmup_iterations=warmup_iterations,
        measurement_iterations=measurement_iterations,
    ).to_json_ready()
    _write_json(sweep_path, sweep_summary)

    bundle = build_lemur_candidate_bundle(
        exact_path=str(exact_path),
        sweep_path=str(sweep_path),
        sweep_summary=sweep_summary,
    ).to_json_ready()
    _write_json(bundle_path, bundle)

    contract_bundle = LemurCandidateContractBundle(
        contract_name="reference_lemur_candidate_contract",
        exact_path=str(exact_path),
        sweep_path=str(sweep_path),
        bundle_path=str(bundle_path),
    ).to_json_ready()
    _write_json(contract_path, contract_bundle)

    return {
        "bundle_path": str(contract_path),
        "bundle": contract_bundle,
        "exact": exact_summary,
        "sweep": sweep_summary,
        "lemur_bundle": bundle,
    }
