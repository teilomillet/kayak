"""Owns loading upstream trained LEMUR checkpoints for reference scoring.

This module owns:
- checkpoint-path resolution for upstream `mlp.pt` and `w.pt` artifacts
- reconstruction of the upstream feature-extractor modules from saved config
- latent single-vector scoring over an existing Kayak late index

This module does not own:
- training
- ANN serving
- task-level benchmarking or artifact contracts
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .dtypes import SCORE_DTYPE
from .late_scores import LateScores


def _validate_activation(activation: str) -> str:
    allowed = {"relu", "gelu", "silu", "mish"}
    if activation not in allowed:
        raise ValueError(f"activation must be one of: {', '.join(sorted(allowed))}")
    return activation


def _activation_cls(activation: str) -> type[nn.Module]:
    activation = _validate_activation(activation)
    classes = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "mish": nn.Mish,
    }
    return classes[activation]


class _UpstreamLemurMlp(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        final_hidden_dim: int | None = None,
        num_layers: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if final_hidden_dim is None:
            final_hidden_dim = hidden_dim

        activation_cls = _activation_cls(activation)
        modules: list[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for layer_index, in_dim in enumerate(dims):
            is_last = layer_index == len(dims) - 1
            out_dim = final_hidden_dim if is_last else hidden_dim
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.LayerNorm(out_dim))
            modules.append(activation_cls())

        self.feature_extractor = nn.Sequential(*modules)
        self.output_layer = nn.Linear(final_hidden_dim, output_dim, bias=False)
        self.config = {
            "model_type": "mlp",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "final_hidden_dim": final_hidden_dim,
            "num_layers": num_layers,
            "activation": activation,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return self.output_layer(feats)


class _RandomActivationFeatures(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, activation: str = "gelu") -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if output_dim <= 0:
            raise ValueError("output_dim must be > 0")
        activation = _validate_activation(activation)
        self.register_buffer("weight", torch.randn(input_dim, output_dim))
        self.activation = activation
        self.scale = float((2.0 / float(output_dim)) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = x @ self.weight
        if self.activation == "relu":
            activated = torch.relu(projected)
        elif self.activation == "gelu":
            activated = torch.nn.functional.gelu(projected)
        elif self.activation == "silu":
            activated = torch.nn.functional.silu(projected)
        else:
            activated = torch.nn.functional.mish(projected)
        return self.scale * activated


class _UpstreamLemurElm(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        final_hidden_dim: int,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if final_hidden_dim <= 0:
            raise ValueError("final_hidden_dim must be > 0")
        activation = _validate_activation(activation)
        self.feature_extractor = nn.Sequential(
            _RandomActivationFeatures(
                input_dim=input_dim,
                output_dim=final_hidden_dim,
                activation=activation,
            ),
            nn.LayerNorm(final_hidden_dim, elementwise_affine=False),
        )
        self.output_layer = nn.Linear(final_hidden_dim, output_dim, bias=False)
        self.config = {
            "model_type": "elm",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "final_hidden_dim": final_hidden_dim,
            "activation": activation,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        return self.output_layer(feats)


def _resolve_checkpoint_paths(
    *,
    index_root: str | Path | None,
    mlp_path: str | Path | None,
    w_path: str | Path | None,
) -> tuple[Path, Path]:
    resolved_mlp = None if mlp_path is None else Path(mlp_path)
    resolved_w = None if w_path is None else Path(w_path)
    if index_root is not None:
        root = Path(index_root)
        if resolved_mlp is None:
            resolved_mlp = root / "mlp.pt"
        if resolved_w is None:
            resolved_w = root / "w.pt"
    if resolved_mlp is None or resolved_w is None:
        raise ValueError(
            "provide index_root or both mlp_path and w_path to load a trained LEMUR"
        )
    return resolved_mlp, resolved_w


def _load_upstream_mlp(path: Path, *, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError("mlp checkpoint must contain a dict payload")
    if "config" not in payload or "state_dict" not in payload:
        raise ValueError("mlp checkpoint must contain config and state_dict")

    config = dict(payload["config"])
    model_type = str(config.pop("model_type"))
    if model_type == "mlp":
        model = _UpstreamLemurMlp(**config).to(device)
    elif model_type == "elm":
        model = _UpstreamLemurElm(**config).to(device)
    else:
        raise ValueError(f"unsupported upstream LEMUR model_type: {model_type}")

    model.load_state_dict(payload["state_dict"])
    model.eval()
    model.config["model_type"] = model_type
    return model, payload


def _load_upstream_w(path: Path, *, device: torch.device) -> torch.Tensor:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "W" in payload:
        weights = payload["W"]
    else:
        weights = payload
    if not torch.is_tensor(weights):
        raise ValueError("w checkpoint must contain a tensor or {'W': tensor}")
    if weights.ndim != 2:
        raise ValueError("w checkpoint must contain a 2D tensor")
    return weights.to(device=device, dtype=torch.float32)


@dataclass(frozen=True, slots=True)
class LoadedTrainedLemurReferenceModel:
    """One torch-backed reference model loaded from upstream trained checkpoints."""

    doc_ids: tuple[str, ...]
    vector_dim: int
    latent_dim: int
    activation: str
    query_divisor: float
    apply_layer_norm: bool
    landmark_count: int
    model_type: str
    device: str
    feature_extractor: Any
    document_weights: Any

    def __post_init__(self) -> None:
        if self.vector_dim <= 0:
            raise ValueError("vector_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.query_divisor <= 0.0:
            raise ValueError("query_divisor must be positive")
        if len(self.doc_ids) <= 0:
            raise ValueError("doc_ids must not be empty")
        if self.document_weights.ndim != 2:
            raise ValueError("document_weights must be 2D")
        if self.document_weights.shape[0] != len(self.doc_ids):
            raise ValueError("document_weights rows must match doc_ids")
        if self.document_weights.shape[1] != self.latent_dim:
            raise ValueError("document_weights columns must match latent_dim")

    def compute_query_features(self, query: "LateQuery") -> np.ndarray:
        if query.vector_dim != self.vector_dim:
            raise ValueError("query and model must share the same vector dimension")

        device = torch.device(self.device)
        # Kayak query matrices may expose a read-only NumPy view. Copy once here
        # so the torch bridge stays warning-free and behavior remains explicit.
        token_values = np.array(query.as_vector_matrix(), dtype=np.float32, copy=True)
        token_matrix = torch.from_numpy(token_values).to(
            device=device,
            dtype=torch.float32,
        )
        self.feature_extractor.to(device)
        self.feature_extractor.eval()
        with torch.inference_mode():
            token_features = self.feature_extractor(token_matrix)
            pooled = token_features.sum(dim=0) / float(self.query_divisor)
        return pooled.detach().cpu().numpy().astype(np.float32, copy=False)

    def similarity_scores(
        self,
        query: "LateQuery",
        *,
        backend: str | None = None,
    ) -> LateScores:
        query_features = torch.from_numpy(self.compute_query_features(query)).to(
            device=self.document_weights.device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            scores = query_features @ self.document_weights.T
        effective_backend = (
            "lemur_reference_trained" if backend is None else backend
        )
        return LateScores.from_values(
            effective_backend,
            self.doc_ids,
            scores.detach().cpu().numpy().astype(SCORE_DTYPE, copy=False),
        )


def load_trained_reference_lemur(
    index: "LateIndex",
    *,
    index_root: str | Path | None = None,
    mlp_path: str | Path | None = None,
    w_path: str | Path | None = None,
    query_divisor: float = 32.0,
    device: str = "cpu",
) -> LoadedTrainedLemurReferenceModel:
    """Load an upstream-trained LEMUR checkpoint pair for one Kayak late index."""

    if query_divisor <= 0.0:
        raise ValueError("query_divisor must be positive")

    resolved_mlp_path, resolved_w_path = _resolve_checkpoint_paths(
        index_root=index_root,
        mlp_path=mlp_path,
        w_path=w_path,
    )
    torch_device = torch.device(device)
    model, payload = _load_upstream_mlp(resolved_mlp_path, device=torch_device)
    weights = _load_upstream_w(resolved_w_path, device=torch_device)

    config = dict(payload["config"])
    input_dim = int(config["input_dim"])
    final_hidden_dim = int(config["final_hidden_dim"])
    activation = str(config["activation"])
    model_type = str(model.config["model_type"])

    if index.vector_dim != input_dim:
        raise ValueError(
            "trained LEMUR input_dim does not match index vector_dim"
        )
    if weights.shape[0] != index.document_count:
        raise ValueError(
            "trained LEMUR W row count does not match index document_count"
        )
    if weights.shape[1] != final_hidden_dim:
        raise ValueError(
            "trained LEMUR W column count does not match checkpoint latent_dim"
        )

    return LoadedTrainedLemurReferenceModel(
        doc_ids=index.doc_ids,
        vector_dim=index.vector_dim,
        latent_dim=final_hidden_dim,
        activation=activation,
        query_divisor=float(query_divisor),
        apply_layer_norm=True,
        landmark_count=0,
        model_type=model_type,
        device=str(torch_device),
        feature_extractor=model.feature_extractor,
        document_weights=weights,
    )
