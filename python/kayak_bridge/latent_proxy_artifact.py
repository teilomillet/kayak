"""Owns export of trained LEMUR checkpoints into Kayak's native latent proxy.

This module owns:
- translating upstream trained `mlp.pt` + `w.pt` checkpoints into a stable
  Kayak-native stage-1 sidecar
- writing the native latent-proxy manifest and binary payload files

This module does not own:
- running the native Mojo search path
- training checkpoints
- exact reranking or task benchmarking
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .trained_reference_lemur import (
    _RandomActivationFeatures,
    _load_upstream_mlp,
    _load_upstream_w,
    _resolve_checkpoint_paths,
)


LATENT_PROXY_ARTIFACT_KIND = "latent_proxy_index"
LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM = "linear_activation_norm"
LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION = "linear_norm_activation"
VECTOR_PAYLOAD_ENCODING_BINARY_LE = "binary_le"


@dataclass(frozen=True, slots=True)
class LatentProxyProjectionBlock:
    order_kind: str
    activation_kind: str
    input_dim: int
    output_dim: int
    linear_rows: np.ndarray
    linear_bias: np.ndarray
    activation_output_scale: float
    layer_norm_affine: bool
    layer_norm_epsilon: float
    layer_norm_weight: np.ndarray
    layer_norm_bias: np.ndarray


@dataclass(frozen=True, slots=True)
class LatentProxyArtifact:
    dataset_id: str
    model_name: str
    vector_scalar_name: str
    input_vector_dim: int
    query_divisor: float
    doc_ids: tuple[str, ...]
    proxy_vectors: np.ndarray
    blocks: tuple[LatentProxyProjectionBlock, ...]


def _require_matrix(values: np.ndarray, *, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be a 2D float32 matrix")
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _require_vector(values: np.ndarray, *, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"{label} must be a 1D float32 vector")
    return np.ascontiguousarray(vector, dtype=np.float32)


def _activation_name_from_module(module: nn.Module) -> str:
    if isinstance(module, nn.ReLU):
        return "relu"
    if isinstance(module, nn.GELU):
        return "gelu"
    if isinstance(module, nn.SiLU):
        return "silu"
    if isinstance(module, nn.Mish):
        return "mish"
    raise ValueError(f"unsupported activation module: {type(module).__name__}")


def _projection_blocks_from_mlp_feature_extractor(
    feature_extractor: nn.Sequential,
) -> tuple[LatentProxyProjectionBlock, ...]:
    modules = list(feature_extractor)
    if len(modules) % 3 != 0:
        raise ValueError("MLP feature extractor must be Linear/LayerNorm/Activation triples")

    blocks: list[LatentProxyProjectionBlock] = []
    for block_index in range(0, len(modules), 3):
        linear = modules[block_index]
        layer_norm = modules[block_index + 1]
        activation = modules[block_index + 2]
        if not isinstance(linear, nn.Linear):
            raise ValueError("MLP feature extractor block must start with nn.Linear")
        if not isinstance(layer_norm, nn.LayerNorm):
            raise ValueError("MLP feature extractor block must include nn.LayerNorm")
        blocks.append(
            LatentProxyProjectionBlock(
                order_kind=LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION,
                activation_kind=_activation_name_from_module(activation),
                input_dim=int(linear.in_features),
                output_dim=int(linear.out_features),
                linear_rows=_require_matrix(
                    linear.weight.detach().cpu().numpy(),
                    label="mlp linear_rows",
                ),
                linear_bias=_require_vector(
                    linear.bias.detach().cpu().numpy(),
                    label="mlp linear_bias",
                ),
                activation_output_scale=1.0,
                layer_norm_affine=True,
                layer_norm_epsilon=float(layer_norm.eps),
                layer_norm_weight=_require_vector(
                    layer_norm.weight.detach().cpu().numpy(),
                    label="mlp layer_norm_weight",
                ),
                layer_norm_bias=_require_vector(
                    layer_norm.bias.detach().cpu().numpy(),
                    label="mlp layer_norm_bias",
                ),
            )
        )
    return tuple(blocks)


def _projection_blocks_from_elm_feature_extractor(
    feature_extractor: nn.Sequential,
) -> tuple[LatentProxyProjectionBlock, ...]:
    modules = list(feature_extractor)
    if len(modules) != 2:
        raise ValueError("ELM feature extractor must be RandomActivationFeatures + LayerNorm")
    random_features = modules[0]
    layer_norm = modules[1]
    if not isinstance(random_features, _RandomActivationFeatures):
        raise ValueError("ELM feature extractor must start with RandomActivationFeatures")
    if not isinstance(layer_norm, nn.LayerNorm):
        raise ValueError("ELM feature extractor must include nn.LayerNorm")
    if layer_norm.elementwise_affine:
        raise ValueError("upstream ELM LayerNorm is expected to be non-affine")

    linear_rows = _require_matrix(
        random_features.weight.detach().cpu().numpy().T,
        label="elm linear_rows",
    )
    linear_bias = np.zeros(linear_rows.shape[0], dtype=np.float32)
    return (
        LatentProxyProjectionBlock(
            order_kind=LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
            activation_kind=str(random_features.activation),
            input_dim=int(linear_rows.shape[1]),
            output_dim=int(linear_rows.shape[0]),
            linear_rows=linear_rows,
            linear_bias=linear_bias,
            activation_output_scale=float(random_features.scale),
            layer_norm_affine=False,
            layer_norm_epsilon=float(layer_norm.eps),
            layer_norm_weight=np.zeros((0,), dtype=np.float32),
            layer_norm_bias=np.zeros((0,), dtype=np.float32),
        ),
    )


def _projection_blocks_from_feature_extractor(
    feature_extractor: nn.Sequential,
    *,
    model_type: str,
) -> tuple[LatentProxyProjectionBlock, ...]:
    if model_type == "mlp":
        return _projection_blocks_from_mlp_feature_extractor(feature_extractor)
    if model_type == "elm":
        return _projection_blocks_from_elm_feature_extractor(feature_extractor)
    raise ValueError(f"unsupported trained LEMUR model_type: {model_type}")


def _write_manifest(entries: list[tuple[str, str]], path: Path) -> None:
    path.write_text(
        "".join(f"{key}\t{value}\n" for key, value in entries),
        encoding="utf-8",
    )


def _write_vector_payload(path: Path, matrix: np.ndarray) -> None:
    _require_matrix(matrix, label=f"{path.name} payload").astype("<f4", copy=False).tofile(
        path
    )


def _artifact_byte_size(root: Path) -> int:
    return sum(
        child.stat().st_size
        for child in root.iterdir()
        if child.is_file()
    )


def build_trained_reference_lemur_artifact(
    index: "LateIndex",
    *,
    index_root: str | Path | None = None,
    mlp_path: str | Path | None = None,
    w_path: str | Path | None = None,
    query_divisor: float = 32.0,
    model_name: str = "",
) -> LatentProxyArtifact:
    if query_divisor <= 0.0:
        raise ValueError("query_divisor must be positive")

    resolved_mlp_path, resolved_w_path = _resolve_checkpoint_paths(
        index_root=index_root,
        mlp_path=mlp_path,
        w_path=w_path,
    )
    model, payload = _load_upstream_mlp(resolved_mlp_path, device=torch.device("cpu"))
    weights = _load_upstream_w(resolved_w_path, device=torch.device("cpu"))
    config = dict(payload["config"])
    input_dim = int(config["input_dim"])
    model_type = str(model.config["model_type"])

    if index.vector_dim != input_dim:
        raise ValueError("trained LEMUR input_dim does not match index vector_dim")
    if weights.shape[0] != index.document_count:
        raise ValueError("trained LEMUR W row count does not match index document_count")

    proxy_vectors = _require_matrix(weights.detach().cpu().numpy(), label="proxy_vectors")
    blocks = _projection_blocks_from_feature_extractor(
        model.feature_extractor,
        model_type=model_type,
    )
    if len(blocks) == 0:
        raise ValueError("trained LEMUR feature extractor produced zero projection blocks")
    if blocks[-1].output_dim != proxy_vectors.shape[1]:
        raise ValueError("trained LEMUR projection output_dim does not match W columns")

    return LatentProxyArtifact(
        dataset_id="",
        model_name=str(model_name),
        vector_scalar_name="Float32",
        input_vector_dim=int(index.vector_dim),
        query_divisor=float(query_divisor),
        doc_ids=tuple(index.doc_ids),
        proxy_vectors=proxy_vectors,
        blocks=blocks,
    )


def export_trained_reference_lemur_artifact(
    index: "LateIndex",
    output_root: str | Path,
    *,
    index_root: str | Path | None = None,
    mlp_path: str | Path | None = None,
    w_path: str | Path | None = None,
    query_divisor: float = 32.0,
    model_name: str = "",
) -> LatentProxyArtifact:
    artifact = build_trained_reference_lemur_artifact(
        index,
        index_root=index_root,
        mlp_path=mlp_path,
        w_path=w_path,
        query_divisor=query_divisor,
        model_name=model_name,
    )
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    (root / "doc_ids.tsv").write_text("".join(f"{doc_id}\n" for doc_id in artifact.doc_ids), encoding="utf-8")
    _write_vector_payload(root / "proxy_vectors.bin", artifact.proxy_vectors)

    for block_index, block in enumerate(artifact.blocks):
        _write_vector_payload(
            root / f"projection_block_{block_index}_linear_rows.bin",
            block.linear_rows,
        )
        _write_vector_payload(
            root / f"projection_block_{block_index}_linear_bias.bin",
            block.linear_bias.reshape(1, -1),
        )
        if block.layer_norm_affine:
            _write_vector_payload(
                root / f"projection_block_{block_index}_layer_norm_weight.bin",
                block.layer_norm_weight.reshape(1, -1),
            )
            _write_vector_payload(
                root / f"projection_block_{block_index}_layer_norm_bias.bin",
                block.layer_norm_bias.reshape(1, -1),
            )

    artifact_byte_size = 0
    while True:
        entries: list[tuple[str, str]] = [
            ("format_version", "2"),
            ("artifact_kind", LATENT_PROXY_ARTIFACT_KIND),
            ("vector_scalar_name", artifact.vector_scalar_name),
            ("dataset_id", artifact.dataset_id),
            ("model_name", artifact.model_name),
            ("vector_payload_encoding", VECTOR_PAYLOAD_ENCODING_BINARY_LE),
            ("input_vector_dim", str(artifact.input_vector_dim)),
            ("latent_dim", str(int(artifact.proxy_vectors.shape[1]))),
            ("document_count", str(len(artifact.doc_ids))),
            ("query_divisor", str(artifact.query_divisor)),
            ("projection_block_count", str(len(artifact.blocks))),
        ]
        for block_index, block in enumerate(artifact.blocks):
            prefix = f"projection_block_{block_index}_"
            entries.extend(
                [
                    (prefix + "order_kind", block.order_kind),
                    (prefix + "activation_kind", block.activation_kind),
                    (prefix + "input_dim", str(block.input_dim)),
                    (prefix + "output_dim", str(block.output_dim)),
                    (
                        prefix + "activation_output_scale",
                        str(block.activation_output_scale),
                    ),
                    (
                        prefix + "layer_norm_affine",
                        "1" if block.layer_norm_affine else "0",
                    ),
                    (
                        prefix + "layer_norm_epsilon",
                        str(block.layer_norm_epsilon),
                    ),
                ]
            )
        entries.append(("artifact_byte_size", str(artifact_byte_size)))
        _write_manifest(entries, root / "manifest.tsv")
        measured = _artifact_byte_size(root)
        if measured == artifact_byte_size:
            break
        artifact_byte_size = measured

    return artifact
