"""Owns internal reference loading and scoring for exported latent-proxy artifacts.

This module owns:
- parsing one on-disk latent-proxy artifact written by the exporter
- reference query projection and proxy scoring over that artifact

This module does not own:
- exporting trained checkpoints
- native Mojo execution
- the public Python SDK candidate-generator surface
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .late_scores import LateScores
from .latent_proxy_artifact import (
    LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM,
    LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION,
    LatentProxyArtifact,
    LatentProxyProjectionBlock,
)


@dataclass(frozen=True, slots=True)
class LoadedLatentProxyReferenceModel:
    """One loaded latent-proxy artifact exposed through the LEMUR-like scorer API."""

    doc_ids: tuple[str, ...]
    vector_dim: int
    latent_dim: int
    activation: str
    query_divisor: float
    apply_layer_norm: bool
    landmark_count: int
    model_type: str
    device: str
    artifact: LatentProxyArtifact

    def __post_init__(self) -> None:
        if self.vector_dim <= 0:
            raise ValueError("vector_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.query_divisor <= 0.0:
            raise ValueError("query_divisor must be positive")
        if len(self.doc_ids) <= 0:
            raise ValueError("doc_ids must not be empty")

    def compute_query_features(self, query: "LateQuery") -> np.ndarray:
        return compute_query_latent_proxy_features(
            query,
            self.artifact,
            device=self.device,
        )

    def similarity_scores(
        self,
        query: "LateQuery",
        *,
        backend: str | None = None,
    ) -> LateScores:
        return latent_proxy_similarity_scores(
            query,
            self.artifact,
            device=self.device,
            backend=backend,
        )


def _read_manifest(root: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (root / "manifest.tsv").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        key, separator, value = line.partition("\t")
        if separator != "\t":
            raise ValueError("latent proxy manifest rows must be tab-separated")
        entries[key] = value
    return entries


def _require_manifest_value(manifest: dict[str, str], key: str) -> str:
    value = manifest.get(key)
    if value is None:
        raise ValueError(f"latent proxy manifest is missing {key}")
    return value


def _read_matrix(path: Path, *, rows: int, cols: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if int(values.size) != rows * cols:
        raise ValueError(f"{path.name} payload shape does not match manifest")
    return np.ascontiguousarray(values.reshape(rows, cols), dtype=np.float32)


def _read_vector(path: Path, *, size: int) -> np.ndarray:
    values = np.fromfile(path, dtype="<f4")
    if int(values.size) != size:
        raise ValueError(f"{path.name} payload shape does not match manifest")
    return np.ascontiguousarray(values.reshape(size), dtype=np.float32)


def _parse_layer_norm_affine(value: str) -> bool:
    if value == "1":
        return True
    if value == "0":
        return False
    raise ValueError("latent proxy layer_norm_affine must be encoded as 0 or 1")


def _activation_label(blocks: tuple[LatentProxyProjectionBlock, ...]) -> str:
    activations = {block.activation_kind for block in blocks}
    if len(activations) == 1:
        return next(iter(activations))
    return "mixed"


def load_latent_proxy_artifact(root: str | Path) -> LatentProxyArtifact:
    artifact_root = Path(root)
    manifest = _read_manifest(artifact_root)
    input_vector_dim = int(_require_manifest_value(manifest, "input_vector_dim"))
    latent_dim = int(_require_manifest_value(manifest, "latent_dim"))
    document_count = int(_require_manifest_value(manifest, "document_count"))
    projection_block_count = int(
        _require_manifest_value(manifest, "projection_block_count")
    )
    doc_ids = tuple(
        line
        for line in (artifact_root / "doc_ids.tsv").read_text(encoding="utf-8").splitlines()
        if line
    )
    if len(doc_ids) != document_count:
        raise ValueError("latent proxy doc_ids count does not match manifest")

    proxy_vectors = _read_matrix(
        artifact_root / "proxy_vectors.bin",
        rows=document_count,
        cols=latent_dim,
    )
    blocks: list[LatentProxyProjectionBlock] = []
    for block_index in range(projection_block_count):
        prefix = f"projection_block_{block_index}_"
        output_dim = int(_require_manifest_value(manifest, prefix + "output_dim"))
        layer_norm_affine = _parse_layer_norm_affine(
            _require_manifest_value(manifest, prefix + "layer_norm_affine")
        )
        if layer_norm_affine:
            layer_norm_weight = _read_vector(
                artifact_root / f"{prefix}layer_norm_weight.bin",
                size=output_dim,
            )
            layer_norm_bias = _read_vector(
                artifact_root / f"{prefix}layer_norm_bias.bin",
                size=output_dim,
            )
        else:
            layer_norm_weight = np.zeros((0,), dtype=np.float32)
            layer_norm_bias = np.zeros((0,), dtype=np.float32)

        blocks.append(
            LatentProxyProjectionBlock(
                order_kind=_require_manifest_value(manifest, prefix + "order_kind"),
                activation_kind=_require_manifest_value(
                    manifest, prefix + "activation_kind"
                ),
                input_dim=int(_require_manifest_value(manifest, prefix + "input_dim")),
                output_dim=output_dim,
                linear_rows=_read_matrix(
                    artifact_root / f"{prefix}linear_rows.bin",
                    rows=output_dim,
                    cols=int(
                        _require_manifest_value(manifest, prefix + "input_dim")
                    ),
                ),
                linear_bias=_read_vector(
                    artifact_root / f"{prefix}linear_bias.bin",
                    size=output_dim,
                ),
                activation_output_scale=float(
                    _require_manifest_value(
                        manifest, prefix + "activation_output_scale"
                    )
                ),
                layer_norm_affine=layer_norm_affine,
                layer_norm_epsilon=float(
                    _require_manifest_value(manifest, prefix + "layer_norm_epsilon")
                ),
                layer_norm_weight=layer_norm_weight,
                layer_norm_bias=layer_norm_bias,
            )
        )

    return LatentProxyArtifact(
        dataset_id=manifest.get("dataset_id", ""),
        model_name=manifest.get("model_name", ""),
        vector_scalar_name=_require_manifest_value(manifest, "vector_scalar_name"),
        input_vector_dim=input_vector_dim,
        query_divisor=float(_require_manifest_value(manifest, "query_divisor")),
        doc_ids=doc_ids,
        proxy_vectors=proxy_vectors,
        blocks=tuple(blocks),
    )


def load_latent_proxy_reference_model(
    root: str | Path,
    *,
    device: str = "cpu",
) -> LoadedLatentProxyReferenceModel:
    artifact = load_latent_proxy_artifact(root)
    return LoadedLatentProxyReferenceModel(
        doc_ids=artifact.doc_ids,
        vector_dim=artifact.input_vector_dim,
        latent_dim=int(artifact.proxy_vectors.shape[1]),
        activation=_activation_label(artifact.blocks),
        query_divisor=artifact.query_divisor,
        apply_layer_norm=True,
        landmark_count=0,
        model_type="latent_proxy_artifact",
        device=device,
        artifact=artifact,
    )


def _activation(values: torch.Tensor, *, activation_kind: str) -> torch.Tensor:
    if activation_kind == "relu":
        return torch.relu(values)
    if activation_kind == "gelu":
        return F.gelu(values, approximate="none")
    if activation_kind == "silu":
        return F.silu(values)
    if activation_kind == "mish":
        return F.mish(values)
    raise ValueError(f"unsupported latent proxy activation: {activation_kind}")


def _layer_norm(
    values: torch.Tensor,
    *,
    block: LatentProxyProjectionBlock,
    device: torch.device,
) -> torch.Tensor:
    weight = None
    bias = None
    if block.layer_norm_affine:
        weight = torch.from_numpy(block.layer_norm_weight).to(
            device=device,
            dtype=torch.float32,
        )
        bias = torch.from_numpy(block.layer_norm_bias).to(
            device=device,
            dtype=torch.float32,
        )
    return F.layer_norm(
        values,
        (block.output_dim,),
        weight=weight,
        bias=bias,
        eps=block.layer_norm_epsilon,
    )


def _project_query_tokens(
    query: "LateQuery",
    artifact: LatentProxyArtifact,
    *,
    device: str,
) -> torch.Tensor:
    if query.vector_dim != artifact.input_vector_dim:
        raise ValueError("query and latent proxy artifact must share vector_dim")

    current = torch.from_numpy(
        np.array(query.as_vector_matrix(), dtype=np.float32, copy=True)
    ).to(device=torch.device(device), dtype=torch.float32)
    torch_device = current.device
    for block in artifact.blocks:
        if current.shape[1] != block.input_dim:
            raise ValueError("latent proxy block input_dim does not match query state")
        linear_rows = torch.from_numpy(block.linear_rows).to(
            device=torch_device,
            dtype=torch.float32,
        )
        linear_bias = torch.from_numpy(block.linear_bias).to(
            device=torch_device,
            dtype=torch.float32,
        )
        current = F.linear(current, linear_rows, linear_bias)
        if block.order_kind == LATENT_PROXY_BLOCK_ORDER_LINEAR_NORM_ACTIVATION:
            current = _layer_norm(current, block=block, device=torch_device)
            current = _activation(current, activation_kind=block.activation_kind)
            current = current * float(block.activation_output_scale)
            continue
        if block.order_kind != LATENT_PROXY_BLOCK_ORDER_LINEAR_ACTIVATION_NORM:
            raise ValueError(f"unsupported latent proxy block order: {block.order_kind}")
        current = _activation(current, activation_kind=block.activation_kind)
        current = current * float(block.activation_output_scale)
        current = _layer_norm(current, block=block, device=torch_device)
    return current


def compute_query_latent_proxy_features(
    query: "LateQuery",
    artifact: LatentProxyArtifact,
    *,
    device: str = "cpu",
) -> np.ndarray:
    token_features = _project_query_tokens(query, artifact, device=device)
    pooled = token_features.sum(dim=0) / float(artifact.query_divisor)
    return pooled.detach().cpu().numpy().astype(np.float32, copy=False)


def latent_proxy_similarity_scores(
    query: "LateQuery",
    artifact: LatentProxyArtifact,
    *,
    device: str = "cpu",
    backend: str | None = None,
) -> LateScores:
    query_features = compute_query_latent_proxy_features(
        query,
        artifact,
        device=device,
    )
    scores = np.matmul(artifact.proxy_vectors, query_features).astype(
        np.float32,
        copy=False,
    )
    effective_backend = (
        "latent_proxy_artifact_reference" if backend is None else backend
    )
    return LateScores.from_values(effective_backend, artifact.doc_ids, scores)
