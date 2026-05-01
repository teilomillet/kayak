"""Universal Scalability Law fitting for measured throughput sweeps.

This module owns the math for USL evidence. It does not run benchmarks or
choose retrieval parameters; callers provide measured concurrency or batch-size
points and decide how to use the fitted curve.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class UslObservation:
    scale: int
    throughput: float

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UslFit:
    base_throughput: float
    alpha: float
    beta: float
    r_squared: float
    observed_peak_scale: int
    observed_peak_throughput: float
    predicted_peak_scale: int
    predicted_peak_throughput: float
    fit_status: str

    def predict(self, scale: int) -> float:
        if scale <= 0:
            raise ValueError("scale must be positive")
        denominator = (
            1.0
            + self.alpha * float(scale - 1)
            + self.beta * float(scale) * float(scale - 1)
        )
        if denominator <= 0.0:
            return math.nan
        return self.base_throughput * float(scale) / denominator

    def to_json_ready(self) -> dict[str, object]:
        return asdict(self)


def fit_usl(
    observations: Sequence[UslObservation],
    *,
    max_predicted_scale: int | None = None,
) -> UslFit:
    """Fit throughput = gamma*N/(1+a(N-1)+bN(N-1)).

    Reason for the linearized fit: it avoids a SciPy dependency and makes the
    model auditable. The transformation N / X(N) = c0 + c1(N-1) +
    c2N(N-1) lets least squares recover gamma, alpha, and beta.
    """

    clean = _clean_observations(observations)
    if len(clean) < 3:
        raise ValueError("USL fit requires at least three positive observations")

    scales = np.asarray([row.scale for row in clean], dtype=np.float64)
    throughputs = np.asarray([row.throughput for row in clean], dtype=np.float64)
    target = scales / throughputs
    design = np.column_stack(
        [
            np.ones_like(scales),
            scales - 1.0,
            scales * (scales - 1.0),
        ]
    )
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    c0, c1, c2 = (float(value) for value in coefficients)
    fit_status = "ok"
    if c0 <= 0.0:
        fit_status = "invalid_nonpositive_base"
        c0 = max(c0, np.finfo(np.float64).eps)
    alpha = c1 / c0
    beta = c2 / c0
    if alpha < 0.0 or beta < 0.0:
        fit_status = "ok_with_negative_contention_terms"

    base = 1.0 / c0
    observed_peak = max(clean, key=lambda row: row.throughput)
    prediction_limit = (
        max(row.scale for row in clean)
        if max_predicted_scale is None
        else max_predicted_scale
    )
    if prediction_limit <= 0:
        raise ValueError("max_predicted_scale must be positive")
    predicted_scale, predicted_throughput = _predicted_peak(
        base_throughput=base,
        alpha=alpha,
        beta=beta,
        max_scale=prediction_limit,
    )
    predictions = np.asarray(
        [
            _predict_throughput(
                base_throughput=base,
                alpha=alpha,
                beta=beta,
                scale=int(scale),
            )
            for scale in scales
        ],
        dtype=np.float64,
    )
    r_squared = _r_squared(observed=throughputs, predicted=predictions)
    if r_squared < 0.0:
        if fit_status == "ok_with_negative_contention_terms":
            fit_status = "poor_explanatory_fit_with_negative_contention_terms"
        else:
            fit_status = "poor_explanatory_fit"
    return UslFit(
        base_throughput=base,
        alpha=alpha,
        beta=beta,
        r_squared=r_squared,
        observed_peak_scale=observed_peak.scale,
        observed_peak_throughput=observed_peak.throughput,
        predicted_peak_scale=predicted_scale,
        predicted_peak_throughput=predicted_throughput,
        fit_status=fit_status,
    )


def _clean_observations(
    observations: Sequence[UslObservation],
) -> tuple[UslObservation, ...]:
    if not observations:
        raise ValueError("USL observations must not be empty")
    ordered = sorted(observations, key=lambda row: row.scale)
    clean: list[UslObservation] = []
    last_scale = 0
    for row in ordered:
        if row.scale <= 0:
            raise ValueError("USL observation scale must be positive")
        if row.throughput <= 0.0 or not math.isfinite(row.throughput):
            raise ValueError("USL observation throughput must be positive and finite")
        if row.scale == last_scale:
            raise ValueError(f"duplicate USL observation scale: {row.scale}")
        clean.append(row)
        last_scale = row.scale
    return tuple(clean)


def _predicted_peak(
    *,
    base_throughput: float,
    alpha: float,
    beta: float,
    max_scale: int,
) -> tuple[int, float]:
    best_scale = 1
    best_throughput = _predict_throughput(
        base_throughput=base_throughput,
        alpha=alpha,
        beta=beta,
        scale=1,
    )
    for scale in range(2, max_scale + 1):
        throughput = _predict_throughput(
            base_throughput=base_throughput,
            alpha=alpha,
            beta=beta,
            scale=scale,
        )
        if math.isfinite(throughput) and throughput > best_throughput:
            best_scale = scale
            best_throughput = throughput
    return best_scale, best_throughput


def _predict_throughput(
    *,
    base_throughput: float,
    alpha: float,
    beta: float,
    scale: int,
) -> float:
    denominator = (
        1.0 + alpha * float(scale - 1) + beta * float(scale) * float(scale - 1)
    )
    if denominator <= 0.0:
        return math.nan
    return base_throughput * float(scale) / denominator


def _r_squared(*, observed: np.ndarray, predicted: np.ndarray) -> float:
    mean_observed = float(np.mean(observed))
    total = float(np.sum(np.square(observed - mean_observed)))
    if total == 0.0:
        return 1.0
    residual = float(np.sum(np.square(observed - predicted)))
    return 1.0 - residual / total
