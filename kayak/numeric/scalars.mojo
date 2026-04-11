comptime STORAGE_FORMAT_VERSION = 2

comptime VECTOR_SCALAR_NAME = "Float32"
comptime SCORE_SCALAR_NAME = "Float32"
comptime METRIC_SCALAR_NAME = "Float64"

comptime VectorScalar = Float32
comptime ScoreScalar = Float32
comptime MetricScalar = Float64


def zero_vector_scalar() -> VectorScalar:
    return VectorScalar(0.0)


def zero_score_scalar() -> ScoreScalar:
    return ScoreScalar(0.0)


def zero_metric_scalar() -> MetricScalar:
    return MetricScalar(0.0)


def min_score_scalar() -> ScoreScalar:
    return ScoreScalar(-3.4028234663852886e38)
