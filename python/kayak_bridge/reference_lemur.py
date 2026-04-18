"""Owns a LEMUR-shaped reference reduction for late interaction.

This module owns:
- an explicit random-feature or caller-supplied feature map over token vectors
- pseudoinverse fitting of one latent document weight vector per document
- latent single-vector scoring for shortlist generation

This module does not own:
- native ANN indexing
- MLP training from the full LEMUR paper
- exact reranking or judged evaluation
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .dtypes import SCORE_DTYPE, VECTOR_DTYPE
from .late_scores import LateScores


def _activation(name: str, values: np.ndarray) -> np.ndarray:
    if name == "relu":
        return np.maximum(values, np.float32(0.0)).astype(np.float32, copy=False)
    if name == "gelu":
        scaled = np.float32(math.sqrt(2.0 / math.pi)) * (
            values + np.float32(0.044715) * values * values * values
        )
        return np.float32(0.5) * values * (
            np.float32(1.0) + np.tanh(scaled).astype(np.float32, copy=False)
        )
    if name == "silu":
        return values / (
            np.float32(1.0) + np.exp(-values).astype(np.float32, copy=False)
        )
    if name == "mish":
        softplus = np.logaddexp(np.float32(0.0), values).astype(
            np.float32, copy=False
        )
        return values * np.tanh(softplus).astype(np.float32, copy=False)
    raise ValueError("activation must be one of: relu, gelu, silu, mish")


def _layer_normalize_rows(values: np.ndarray, eps: float) -> np.ndarray:
    means = values.mean(axis=1, keepdims=True, dtype=np.float32)
    centered = values - means
    variances = np.mean(centered * centered, axis=1, keepdims=True, dtype=np.float32)
    denom = np.sqrt(variances + np.float32(eps)).astype(np.float32, copy=False)
    return centered / denom


def _require_matrix(values: np.ndarray, *, label: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=VECTOR_DTYPE)
    if matrix.ndim != 2:
        raise ValueError(f"{label} must be a 2D float32 matrix")
    if matrix.shape[0] <= 0 or matrix.shape[1] <= 0:
        raise ValueError(f"{label} must be non-empty")
    return np.ascontiguousarray(matrix)


def _sample_landmarks(
    matrix: np.ndarray,
    *,
    landmark_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if landmark_count <= 0:
        raise ValueError("landmark_count must be positive")
    if landmark_count > matrix.shape[0]:
        raise ValueError("landmark_count cannot exceed the corpus token count")
    if landmark_count == matrix.shape[0]:
        return matrix.copy()

    positions = np.sort(
        rng.choice(matrix.shape[0], size=landmark_count, replace=False)
    )
    return np.ascontiguousarray(matrix[positions], dtype=VECTOR_DTYPE)


def _build_feature_weights(
    *,
    vector_dim: int,
    latent_dim: int,
    seed: int,
) -> np.ndarray:
    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive")
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(
        rng.standard_normal((vector_dim, latent_dim), dtype=np.float32),
        dtype=VECTOR_DTYPE,
    )


def _transform_token_matrix(
    token_matrix: np.ndarray,
    *,
    feature_weights: np.ndarray,
    activation: str,
    apply_layer_norm: bool,
    layer_norm_eps: float,
) -> np.ndarray:
    projected = token_matrix @ feature_weights
    features = np.float32(math.sqrt(2.0 / float(feature_weights.shape[1]))) * _activation(
        activation,
        projected.astype(np.float32, copy=False),
    )
    if apply_layer_norm:
        return _layer_normalize_rows(features, layer_norm_eps)
    return np.ascontiguousarray(features, dtype=VECTOR_DTYPE)


def _maxsim_targets(document_matrix: np.ndarray, landmark_matrix: np.ndarray) -> np.ndarray:
    similarities = document_matrix @ landmark_matrix.T
    return similarities.max(axis=0).astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class LemurReferenceModel:
    """One fitted reference model for latent single-vector shortlist scoring."""

    doc_ids: tuple[str, ...]
    vector_dim: int
    latent_dim: int
    activation: str
    query_divisor: float
    apply_layer_norm: bool
    layer_norm_eps: float
    landmark_count: int
    feature_weights: np.ndarray
    document_weights: np.ndarray

    def __post_init__(self) -> None:
        if self.vector_dim <= 0:
            raise ValueError("vector_dim must be positive")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.query_divisor <= 0.0:
            raise ValueError("query_divisor must be positive")
        if self.landmark_count <= 0:
            raise ValueError("landmark_count must be positive")
        if self.feature_weights.shape != (self.vector_dim, self.latent_dim):
            raise ValueError("feature_weights shape must match vector_dim x latent_dim")
        if self.document_weights.shape != (len(self.doc_ids), self.latent_dim):
            raise ValueError(
                "document_weights shape must match document_count x latent_dim"
            )

    def compute_query_features(self, query: "LateQuery") -> np.ndarray:
        if query.vector_dim != self.vector_dim:
            raise ValueError("query and model must share the same vector dimension")

        token_features = _transform_token_matrix(
            query.as_vector_matrix(),
            feature_weights=self.feature_weights,
            activation=self.activation,
            apply_layer_norm=self.apply_layer_norm,
            layer_norm_eps=self.layer_norm_eps,
        )
        return np.sum(token_features, axis=0, dtype=np.float32) / np.float32(
            self.query_divisor
        )

    def similarity_scores(
        self,
        query: "LateQuery",
        *,
        backend: str | None = None,
    ) -> LateScores:
        query_features = self.compute_query_features(query)
        scores = query_features @ self.document_weights.T
        effective_backend = "lemur_reference" if backend is None else backend
        return LateScores.from_values(
            effective_backend,
            self.doc_ids,
            np.asarray(scores, dtype=SCORE_DTYPE),
        )


def fit_reference_lemur(
    index: "LateIndex",
    *,
    latent_dim: int | None = None,
    activation: str = "gelu",
    query_divisor: float = 32.0,
    landmark_count: int | None = None,
    feature_weights: np.ndarray | None = None,
    landmark_vectors: np.ndarray | None = None,
    apply_layer_norm: bool = True,
    layer_norm_eps: float = 1e-5,
    pinv_rcond: float = 1e-6,
    seed: int = 0,
) -> LemurReferenceModel:
    """Fit a reference LEMUR-style reduction on one packed late index.

    This intentionally implements the training-free paper-shaped slice:
    random or caller-supplied token features, then pseudoinverse fitting of the
    document weight matrix against exact MaxSim targets on sampled landmarks.
    """

    if pinv_rcond <= 0.0:
        raise ValueError("pinv_rcond must be positive")

    token_matrix = index.as_packed_token_matrix()
    vector_dim = int(index.vector_dim)

    if feature_weights is None:
        if latent_dim is None:
            raise ValueError("latent_dim is required when feature_weights is omitted")
        normalized_feature_weights = _build_feature_weights(
            vector_dim=vector_dim,
            latent_dim=latent_dim,
            seed=seed,
        )
    else:
        normalized_feature_weights = _require_matrix(
            feature_weights,
            label="feature_weights",
        )
        if normalized_feature_weights.shape[0] != vector_dim:
            raise ValueError(
                "feature_weights must have leading dimension equal to vector_dim"
            )
        if latent_dim is not None and normalized_feature_weights.shape[1] != latent_dim:
            raise ValueError("feature_weights and latent_dim must agree")

    resolved_latent_dim = int(normalized_feature_weights.shape[1])
    if resolved_latent_dim <= 0:
        raise ValueError("latent_dim must be positive")

    rng = np.random.default_rng(seed)
    if landmark_vectors is None:
        resolved_landmark_count = (
            token_matrix.shape[0] if landmark_count is None else landmark_count
        )
        normalized_landmarks = _sample_landmarks(
            token_matrix,
            landmark_count=resolved_landmark_count,
            rng=rng,
        )
    else:
        normalized_landmarks = _require_matrix(
            landmark_vectors,
            label="landmark_vectors",
        )
        if normalized_landmarks.shape[1] != vector_dim:
            raise ValueError(
                "landmark_vectors must have trailing dimension equal to vector_dim"
            )
        resolved_landmark_count = int(normalized_landmarks.shape[0])
        if landmark_count is not None and resolved_landmark_count != landmark_count:
            raise ValueError("landmark_vectors and landmark_count must agree")

    landmark_features = _transform_token_matrix(
        normalized_landmarks,
        feature_weights=normalized_feature_weights,
        activation=activation,
        apply_layer_norm=apply_layer_norm,
        layer_norm_eps=layer_norm_eps,
    )
    projector = np.linalg.pinv(landmark_features, rcond=pinv_rcond).astype(np.float32)

    document_weights = np.empty(
        (index.document_count, resolved_latent_dim),
        dtype=np.float32,
    )
    for document_index in range(index.document_count):
        document_matrix = index.document_token_matrix(document_index)
        targets = _maxsim_targets(document_matrix, normalized_landmarks)
        document_weights[document_index] = projector @ targets

    return LemurReferenceModel(
        doc_ids=index.doc_ids,
        vector_dim=vector_dim,
        latent_dim=resolved_latent_dim,
        activation=activation,
        query_divisor=float(query_divisor),
        apply_layer_norm=apply_layer_norm,
        layer_norm_eps=float(layer_norm_eps),
        landmark_count=resolved_landmark_count,
        feature_weights=np.ascontiguousarray(
            normalized_feature_weights,
            dtype=VECTOR_DTYPE,
        ),
        document_weights=np.ascontiguousarray(document_weights, dtype=VECTOR_DTYPE),
    )
