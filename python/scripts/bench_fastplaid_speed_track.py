from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

import kayak
from kayak_bridge.plaid_approx import KayakPlaidApproxConfig, KayakPlaidApproxIndex


FASTPLAID_BLOG_URL = "https://lighton.ai/lighton-blogs/fastplaid"
FASTPLAID_REPO_URL = "https://github.com/lightonai/fast-plaid"


@dataclass(frozen=True, slots=True)
class SpeedTrackShape:
    document_count: int
    document_vector_count: int
    query_count: int
    query_vector_count: int
    vector_dim: int
    top_k: int
    update_document_count: int

    @property
    def total_document_vector_count(self) -> int:
        return self.document_count * self.document_vector_count

    @property
    def total_query_vector_count(self) -> int:
        return self.query_count * self.query_vector_count

    @property
    def initial_document_count(self) -> int:
        return self.document_count - self.update_document_count

    def validate(self) -> None:
        _require_positive("document_count", self.document_count)
        _require_positive("document_vector_count", self.document_vector_count)
        _require_positive("query_count", self.query_count)
        _require_positive("query_vector_count", self.query_vector_count)
        _require_positive("vector_dim", self.vector_dim)
        _require_positive("top_k", self.top_k)
        if self.update_document_count < 0:
            raise ValueError("update_document_count must be non-negative")
        if self.update_document_count >= self.document_count:
            raise ValueError(
                "update_document_count must be smaller than document_count"
            )

    def to_json_ready(self) -> dict[str, int]:
        return {
            "document_count": self.document_count,
            "document_vector_count": self.document_vector_count,
            "document_vector_count_total": self.total_document_vector_count,
            "query_count": self.query_count,
            "query_vector_count": self.query_vector_count,
            "query_vector_count_total": self.total_query_vector_count,
            "vector_dim": self.vector_dim,
            "top_k": self.top_k,
            "initial_document_count": self.initial_document_count,
            "update_document_count": self.update_document_count,
        }


@dataclass(frozen=True, slots=True)
class SyntheticInputs:
    documents: np.ndarray
    queries: np.ndarray
    doc_ids: tuple[str, ...]


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def parse_engine_list(value: str) -> tuple[str, ...]:
    engines = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(engines) == 0:
        raise argparse.ArgumentTypeError("at least one engine is required")

    supported = {"kayak_exact", "kayak_plaid", "fastplaid"}
    unknown = tuple(engine for engine in engines if engine not in supported)
    if unknown:
        supported_text = ", ".join(sorted(supported))
        raise argparse.ArgumentTypeError(
            f"unsupported engine(s): {', '.join(unknown)}; "
            f"supported engines: {supported_text}"
        )
    if (
        ("fastplaid" in engines or "kayak_plaid" in engines)
        and "kayak_exact" not in engines
    ):
        raise argparse.ArgumentTypeError(
            "approximate comparisons require kayak_exact as the exact reference"
        )
    return engines


def _optional_bool(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "expected one of auto, true, false, yes, no, 1, or 0"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Kayak exact Mojo search against optional FastPlaid "
            "on one explicit synthetic multi-vector shape."
        )
    )
    parser.add_argument(
        "--engines",
        type=parse_engine_list,
        default=parse_engine_list("kayak_exact,kayak_plaid,fastplaid"),
        help="Comma-separated engines: kayak_exact,kayak_plaid,fastplaid.",
    )
    parser.add_argument("--document-count", type=int, default=1000)
    parser.add_argument("--document-vector-count", type=int, default=300)
    parser.add_argument("--query-count", type=int, default=8)
    parser.add_argument("--query-vector-count", type=int, default=50)
    parser.add_argument("--vector-dim", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--update-document-count",
        type=int,
        default=0,
        help=(
            "If positive, build FastPlaid on the initial documents and time "
            "one incremental update for the trailing documents."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--normalize-vectors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="L2-normalize synthetic token vectors before indexing.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--measurement-iterations", type=int, default=5)
    parser.add_argument(
        "--kayak-backend",
        choices=(kayak.MOJO_EXACT_CPU_BACKEND, kayak.NUMPY_REFERENCE_BACKEND),
        default=kayak.MOJO_EXACT_CPU_BACKEND,
    )
    parser.add_argument(
        "--fastplaid-device",
        default="cpu",
        help='FastPlaid device string, e.g. "cpu", "cuda", or an empty auto value.',
    )
    parser.add_argument(
        "--fastplaid-low-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to FastPlaid. On GPU this controls index residency.",
    )
    parser.add_argument("--fastplaid-kmeans-niters", type=int, default=4)
    parser.add_argument("--fastplaid-max-points-per-centroid", type=int, default=256)
    parser.add_argument("--fastplaid-nbits", type=int, default=4)
    parser.add_argument("--fastplaid-batch-size", type=int, default=25_000)
    parser.add_argument(
        "--fastplaid-use-triton-kmeans",
        type=_optional_bool,
        default=None,
        help="Forwarded as use_triton_kmeans; use auto to let FastPlaid decide.",
    )
    parser.add_argument("--fastplaid-update-buffer-size", type=int, default=100)
    parser.add_argument("--kayak-plaid-centroid-count", type=int, default=128)
    parser.add_argument("--kayak-plaid-centroids-per-query-vector", type=int, default=32)
    parser.add_argument("--kayak-plaid-candidate-k", type=int, default=160)
    parser.add_argument(
        "--require-fastplaid",
        action="store_true",
        help="Fail instead of writing a skipped FastPlaid row when unavailable.",
    )
    parser.add_argument(
        "--index-root",
        type=Path,
        default=Path(".cache/kayak/fastplaid_speed_track/indexes"),
    )
    parser.add_argument(
        "--overwrite-index-root",
        action="store_true",
        help="Remove existing per-engine index directories before benchmarking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/kayak/fastplaid_speed_track/summary.json"),
    )
    return parser.parse_args()


def build_synthetic_inputs(
    shape: SpeedTrackShape,
    *,
    seed: int,
    normalize_vectors: bool,
) -> SyntheticInputs:
    shape.validate()
    rng = np.random.default_rng(seed)
    documents = rng.standard_normal(
        (
            shape.document_count,
            shape.document_vector_count,
            shape.vector_dim,
        )
    ).astype(np.float32)
    queries = rng.standard_normal(
        (
            shape.query_count,
            shape.query_vector_count,
            shape.vector_dim,
        )
    ).astype(np.float32)
    if normalize_vectors:
        documents = _l2_normalize_last_axis(documents)
        queries = _l2_normalize_last_axis(queries)
    doc_ids = tuple(f"doc-{index:08d}" for index in range(shape.document_count))
    return SyntheticInputs(documents=documents, queries=queries, doc_ids=doc_ids)


def _l2_normalize_last_axis(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, np.float32(1e-12))


def _doc_position(doc_id: str) -> int:
    prefix = "doc-"
    if not doc_id.startswith(prefix):
        raise ValueError(f"unexpected synthetic doc_id: {doc_id}")
    return int(doc_id[len(prefix) :])


def mean_recall_at_k(
    *,
    candidate_positions_by_query: Sequence[Sequence[int]],
    reference_positions_by_query: Sequence[Sequence[int]],
    k: int,
) -> float:
    if len(candidate_positions_by_query) != len(reference_positions_by_query):
        raise ValueError("candidate and reference query counts must match")
    if k <= 0:
        raise ValueError("k must be positive")

    recalls: list[float] = []
    for candidate_positions, reference_positions in zip(
        candidate_positions_by_query,
        reference_positions_by_query,
        strict=True,
    ):
        reference = set(reference_positions[:k])
        denominator = min(k, len(reference))
        if denominator == 0:
            recalls.append(0.0)
            continue
        candidate = set(candidate_positions[:k])
        recalls.append(len(reference & candidate) / float(denominator))
    return float(sum(recalls) / len(recalls))


def build_pairwise_rows(systems: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next(
        (
            system
            for system in systems
            if system["engine"] == "kayak" and system["status"] == "ok"
        ),
        None,
    )
    if baseline is None:
        return []

    rows: list[dict[str, Any]] = []
    baseline_batch_seconds = float(baseline["query_batch_mean_seconds"])
    baseline_qps = float(baseline["query_qps"])
    for system in systems:
        if system is baseline or system["status"] != "ok":
            continue
        candidate_batch_seconds = float(system["query_batch_mean_seconds"])
        candidate_qps = float(system["query_qps"])
        rows.append(
            {
                "baseline": baseline["system_name"],
                "candidate": system["system_name"],
                "query_batch_seconds_ratio_vs_baseline": (
                    candidate_batch_seconds / baseline_batch_seconds
                ),
                "query_qps_ratio_vs_baseline": candidate_qps / baseline_qps,
                "recall_at_k_vs_baseline": system.get(
                    "recall_at_k_vs_kayak_exact"
                ),
                "index_bytes_ratio_vs_baseline": _optional_ratio(
                    system.get("index_bytes"),
                    baseline.get("index_bytes"),
                ),
            }
        )
    return rows


def _optional_ratio(value: object, baseline: object) -> float | None:
    if value is None or baseline is None:
        return None
    denominator = float(baseline)
    if denominator == 0.0:
        return None
    return float(value) / denominator


def _measurement_stats(durations: Sequence[float]) -> dict[str, float]:
    if len(durations) == 0:
        raise ValueError("durations must not be empty")
    ordered = sorted(float(duration) for duration in durations)
    return {
        "min_seconds": ordered[0],
        "median_seconds": ordered[len(ordered) // 2],
        "mean_seconds": float(sum(ordered) / len(ordered)),
        "max_seconds": ordered[-1],
    }


def _path_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


def _fresh_index_root(root: Path, *, overwrite: bool) -> Path:
    if root.exists():
        if not overwrite:
            raise FileExistsError(
                f"index root already exists: {root}; pass --overwrite-index-root"
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def benchmark_kayak_exact(
    *,
    shape: SpeedTrackShape,
    inputs: SyntheticInputs,
    backend: str,
    warmup_iterations: int,
    measurement_iterations: int,
) -> tuple[dict[str, Any], tuple[tuple[int, ...], ...]]:
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    _require_positive("measurement_iterations", measurement_iterations)

    started_at = time.perf_counter()
    index = kayak.documents(inputs.doc_ids, inputs.documents).pack()
    query_batch = kayak.query_batch(inputs.queries)
    if shape.vector_dim == 128:
        index = index.to_layout("hybrid_flat_dim128")
        query_batch = query_batch.to_layout("flat_dim128")
    build_seconds = time.perf_counter() - started_at

    for _ in range(warmup_iterations):
        kayak.search_batch(query_batch, index, k=shape.top_k, backend=backend)

    durations: list[float] = []
    hits_by_query = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        current_hits = kayak.search_batch(
            query_batch,
            index,
            k=shape.top_k,
            backend=backend,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            hits_by_query = current_hits

    if hits_by_query is None:  # pragma: no cover - guarded above.
        raise RuntimeError("Kayak exact benchmark produced no hits")

    reference_positions = tuple(
        tuple(_doc_position(hit.doc_id) for hit in hits)
        for hits in hits_by_query
    )
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    index_bytes = (
        int(index.doc_offsets.nbytes)
        + int(
            index.token_values.nbytes
            if index.token_values is not None
            else index.as_packed_token_matrix().nbytes
        )
    )
    system_name = (
        "kayak_mojo_exact_cpu"
        if backend == kayak.MOJO_EXACT_CPU_BACKEND
        else "kayak_numpy_reference"
    )
    return (
        {
            "system_name": system_name,
            "engine": "kayak",
            "status": "ok",
            "engine_version": "python_sdk",
            "index_kind": f"exact_{index.layout}",
            "backend": backend,
            "vector_metric": "dot_product",
            "build_seconds": build_seconds,
            "index_bytes": index_bytes,
            "query_batch_min_seconds": stats["min_seconds"],
            "query_batch_median_seconds": stats["median_seconds"],
            "query_batch_mean_seconds": batch_mean_seconds,
            "query_batch_max_seconds": stats["max_seconds"],
            "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
            "query_qps": shape.query_count / batch_mean_seconds,
            "recall_at_k_vs_kayak_exact": 1.0,
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
        },
        reference_positions,
    )


def benchmark_kayak_plaid(
    *,
    shape: SpeedTrackShape,
    inputs: SyntheticInputs,
    reference_positions_by_query: Sequence[Sequence[int]],
    config: KayakPlaidApproxConfig,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, Any]:
    config.validate(final_k=shape.top_k)
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    _require_positive("measurement_iterations", measurement_iterations)

    started_at = time.perf_counter()
    index = KayakPlaidApproxIndex.build(
        doc_ids=inputs.doc_ids,
        documents=inputs.documents,
        config=config,
        final_k=shape.top_k,
    )
    build_seconds = time.perf_counter() - started_at

    for _ in range(warmup_iterations):
        index.search_batch_positions(inputs.queries, final_k=shape.top_k)

    durations: list[float] = []
    first_positions = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        current_positions = index.search_batch_positions(
            inputs.queries,
            final_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_positions = current_positions

    if first_positions is None:  # pragma: no cover - guarded above.
        raise RuntimeError("Kayak PLAID approximation produced no result")

    recall = mean_recall_at_k(
        candidate_positions_by_query=first_positions,
        reference_positions_by_query=reference_positions_by_query,
        k=shape.top_k,
    )
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return {
        "system_name": "kayak_plaid_mojo_probe",
        "engine": "kayak",
        "status": "ok",
        "engine_version": "mojo_bridge",
        "index_kind": index.index_kind,
        "backend": "mojo_centroid_postings",
        "vector_metric": "dot_product",
        "build_seconds": build_seconds,
        "index_bytes": index.index_bytes,
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "query_qps": shape.query_count / batch_mean_seconds,
        "recall_at_k_vs_kayak_exact": recall,
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "centroid_count": index.centroid_count,
        "centroids_per_query_vector": config.centroids_per_query_vector,
        "candidate_k": config.candidate_k,
        "payload": config.payload,
        "rerank": index.rerank_kind,
    }


def benchmark_fastplaid(
    *,
    shape: SpeedTrackShape,
    inputs: SyntheticInputs,
    reference_positions_by_query: Sequence[Sequence[int]],
    index_root: Path,
    overwrite_index_root: bool,
    device: str,
    low_memory: bool,
    kmeans_niters: int,
    max_points_per_centroid: int,
    nbits: int,
    batch_size: int,
    use_triton_kmeans: bool | None,
    update_buffer_size: int,
    seed: int,
    warmup_iterations: int,
    measurement_iterations: int,
    require_fastplaid: bool,
) -> dict[str, Any]:
    try:
        search_module = importlib.import_module("fast_plaid.search")
        torch = importlib.import_module("torch")
    except ImportError as exc:
        if require_fastplaid:
            raise
        return _skipped_fastplaid_summary(f"missing dependency: {exc}")

    FastPlaid = getattr(search_module, "FastPlaid", None)
    if FastPlaid is None:
        if require_fastplaid:
            raise RuntimeError("fast_plaid.search.FastPlaid was not found")
        return _skipped_fastplaid_summary(
            "missing API: fast_plaid.search.FastPlaid"
        )

    _require_positive("measurement_iterations", measurement_iterations)
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")

    engine_root = _fresh_index_root(index_root / "fastplaid", overwrite=overwrite_index_root)
    started_at = time.perf_counter()
    fast_plaid = FastPlaid(
        index=str(engine_root),
        device=device,
        low_memory=low_memory,
    )

    initial_count = shape.initial_document_count
    initial_documents = [
        torch.from_numpy(document)
        for document in inputs.documents[:initial_count]
    ]
    fast_plaid.create(
        documents_embeddings=initial_documents,
        kmeans_niters=kmeans_niters,
        max_points_per_centroid=max_points_per_centroid,
        nbits=nbits,
        batch_size=batch_size,
        seed=seed,
        use_triton_kmeans=use_triton_kmeans,
    )
    build_seconds = time.perf_counter() - started_at

    update_seconds = None
    if shape.update_document_count > 0:
        update_documents = [
            torch.from_numpy(document)
            for document in inputs.documents[initial_count:]
        ]
        started_at = time.perf_counter()
        fast_plaid.update(
            documents_embeddings=update_documents,
            batch_size=batch_size,
            kmeans_niters=kmeans_niters,
            max_points_per_centroid=max_points_per_centroid,
            seed=seed,
            buffer_size=update_buffer_size,
            use_triton_kmeans=use_triton_kmeans,
        )
        update_seconds = time.perf_counter() - started_at

    query_tensor = torch.from_numpy(inputs.queries)
    for _ in range(warmup_iterations):
        fast_plaid.search(queries_embeddings=query_tensor, top_k=shape.top_k)

    durations: list[float] = []
    first_result = None
    for measurement_index in range(measurement_iterations):
        started_at = time.perf_counter()
        result = fast_plaid.search(
            queries_embeddings=query_tensor,
            top_k=shape.top_k,
        )
        durations.append(time.perf_counter() - started_at)
        if measurement_index == 0:
            first_result = result

    if first_result is None:  # pragma: no cover - guarded above.
        raise RuntimeError("FastPlaid benchmark produced no result")

    candidate_positions = fastplaid_result_positions(first_result)
    recall = mean_recall_at_k(
        candidate_positions_by_query=candidate_positions,
        reference_positions_by_query=reference_positions_by_query,
        k=shape.top_k,
    )
    stats = _measurement_stats(durations)
    batch_mean_seconds = stats["mean_seconds"]
    return {
        "system_name": "fastplaid",
        "engine": "fastplaid",
        "status": "ok",
        "engine_version": _package_version("fast-plaid"),
        "index_kind": "plaid_kmeans_pq",
        "backend": f"fastplaid:{device or 'auto'}",
        "vector_metric": "dot_product",
        "build_seconds": build_seconds,
        "update_seconds": update_seconds,
        "index_bytes": _path_size_bytes(engine_root),
        "query_batch_min_seconds": stats["min_seconds"],
        "query_batch_median_seconds": stats["median_seconds"],
        "query_batch_mean_seconds": batch_mean_seconds,
        "query_batch_max_seconds": stats["max_seconds"],
        "query_mean_seconds": batch_mean_seconds / float(shape.query_count),
        "query_qps": shape.query_count / batch_mean_seconds,
        "recall_at_k_vs_kayak_exact": recall,
        "warmup_iterations": warmup_iterations,
        "measurement_iterations": measurement_iterations,
        "device": device,
        "low_memory": low_memory,
        "kmeans_niters": kmeans_niters,
        "max_points_per_centroid": max_points_per_centroid,
        "nbits": nbits,
        "batch_size": batch_size,
        "use_triton_kmeans": use_triton_kmeans,
        "update_buffer_size": update_buffer_size,
    }


def _skipped_fastplaid_summary(reason: str) -> dict[str, Any]:
    return {
        "system_name": "fastplaid",
        "engine": "fastplaid",
        "status": "skipped",
        "skip_reason": reason,
        "engine_version": None,
        "index_kind": "plaid_kmeans_pq",
        "backend": None,
    }


def _package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def fastplaid_result_positions(result: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(_fastplaid_hit_position(hit) for hit in query_hits)
        for query_hits in result
    )


def _fastplaid_hit_position(hit: Any) -> int:
    if isinstance(hit, dict):
        for key in ("document_index", "doc_index", "id", "doc_id"):
            if key in hit:
                return int(hit[key])
    if isinstance(hit, (list, tuple)) and len(hit) >= 1:
        return int(hit[0])
    document_index = getattr(hit, "document_index", None)
    if document_index is not None:
        return int(document_index)
    doc_index = getattr(hit, "doc_index", None)
    if doc_index is not None:
        return int(doc_index)
    raise TypeError(f"unsupported FastPlaid hit shape: {hit!r}")


def build_report(
    *,
    shape: SpeedTrackShape,
    inputs: SyntheticInputs,
    systems: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "fastplaid_speed_track",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_urls": {
            "fastplaid_blog": FASTPLAID_BLOG_URL,
            "fastplaid_repo": FASTPLAID_REPO_URL,
        },
        "shape": shape.to_json_ready(),
        "controls": {
            "seed": args.seed,
            "normalize_vectors": args.normalize_vectors,
            "engines": list(args.engines),
            "kayak_backend": args.kayak_backend,
            "fastplaid_device": args.fastplaid_device,
            "fastplaid_low_memory": args.fastplaid_low_memory,
            "fastplaid_kmeans_niters": args.fastplaid_kmeans_niters,
            "fastplaid_max_points_per_centroid": (
                args.fastplaid_max_points_per_centroid
            ),
            "fastplaid_nbits": args.fastplaid_nbits,
            "fastplaid_batch_size": args.fastplaid_batch_size,
            "fastplaid_use_triton_kmeans": args.fastplaid_use_triton_kmeans,
            "fastplaid_update_buffer_size": args.fastplaid_update_buffer_size,
            "kayak_plaid_centroid_count": args.kayak_plaid_centroid_count,
            "kayak_plaid_centroids_per_query_vector": (
                args.kayak_plaid_centroids_per_query_vector
            ),
            "kayak_plaid_candidate_k": args.kayak_plaid_candidate_k,
        },
        "input_bytes": {
            "documents": int(inputs.documents.nbytes),
            "queries": int(inputs.queries.nbytes),
        },
        "systems": list(systems),
        "pairwise_vs_kayak_exact": build_pairwise_rows(systems),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    shape = SpeedTrackShape(
        document_count=args.document_count,
        document_vector_count=args.document_vector_count,
        query_count=args.query_count,
        query_vector_count=args.query_vector_count,
        vector_dim=args.vector_dim,
        top_k=args.top_k,
        update_document_count=args.update_document_count,
    )
    shape.validate()
    if args.warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    _require_positive("measurement_iterations", args.measurement_iterations)

    inputs = build_synthetic_inputs(
        shape,
        seed=args.seed,
        normalize_vectors=args.normalize_vectors,
    )

    systems: list[dict[str, Any]] = []
    kayak_summary, reference_positions = benchmark_kayak_exact(
        shape=shape,
        inputs=inputs,
        backend=args.kayak_backend,
        warmup_iterations=args.warmup_iterations,
        measurement_iterations=args.measurement_iterations,
    )
    systems.append(kayak_summary)

    if "kayak_plaid" in args.engines:
        systems.append(
            benchmark_kayak_plaid(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                config=KayakPlaidApproxConfig(
                    centroid_count=args.kayak_plaid_centroid_count,
                    centroids_per_query_vector=(
                        args.kayak_plaid_centroids_per_query_vector
                    ),
                    candidate_k=args.kayak_plaid_candidate_k,
                ),
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
            )
        )

    if "fastplaid" in args.engines:
        systems.append(
            benchmark_fastplaid(
                shape=shape,
                inputs=inputs,
                reference_positions_by_query=reference_positions,
                index_root=args.index_root,
                overwrite_index_root=args.overwrite_index_root,
                device=args.fastplaid_device,
                low_memory=args.fastplaid_low_memory,
                kmeans_niters=args.fastplaid_kmeans_niters,
                max_points_per_centroid=args.fastplaid_max_points_per_centroid,
                nbits=args.fastplaid_nbits,
                batch_size=args.fastplaid_batch_size,
                use_triton_kmeans=args.fastplaid_use_triton_kmeans,
                update_buffer_size=args.fastplaid_update_buffer_size,
                seed=args.seed,
                warmup_iterations=args.warmup_iterations,
                measurement_iterations=args.measurement_iterations,
                require_fastplaid=args.require_fastplaid,
            )
        )

    report = build_report(shape=shape, inputs=inputs, systems=systems, args=args)
    write_report(args.output, report)

    print(json.dumps(report, indent=2, sort_keys=True))
    for system in systems:
        if system["status"] == "ok":
            print(
                f"{system['system_name']}_query_batch_mean_seconds: "
                f"{system['query_batch_mean_seconds']}"
            )
    print(f"Mean: {systems[0]['query_batch_mean_seconds']}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
