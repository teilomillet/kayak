from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.append(str(PYTHON_ROOT))

from kayak_bridge.json_task_loader import load_task_json
from kayak_bridge.latent_proxy_artifact import (
    export_trained_reference_lemur_artifact,
)
from kayak_bridge.native_latent_proxy_task_benchmark import (
    DEFAULT_COLLECTION_ID,
    DEFAULT_NAMESPACE_ID,
    DEFAULT_SNAPSHOT_ID,
    DEFAULT_TENANT_ID,
    benchmark_materialized_collection_search,
    materialize_native_latent_proxy_collection,
)


def _parse_candidate_ks(value: str | None, *, default_k: int) -> list[int]:
    if value is None:
        return [default_k]
    candidate_ks: list[int] = []
    for part in value.split(","):
        stripped = part.strip()
        if stripped == "":
            continue
        parsed = int(stripped)
        if parsed <= 0:
            raise ValueError("candidate_k values must be positive")
        candidate_ks.append(parsed)
    if not candidate_ks:
        raise ValueError("at least one candidate_k is required")
    return candidate_ks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one exported latent-proxy artifact through Kayak's "
            "native collection runtime."
        )
    )
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path)
    parser.add_argument("--candidate-ks", type=str)
    parser.add_argument(
        "--candidate-generator-kind",
        type=str,
        default="latent_proxy",
        help="Explicit candidate generator, for example latent_proxy or exact_full_scan.",
    )
    parser.add_argument("--collection-id", type=str, default=DEFAULT_COLLECTION_ID)
    parser.add_argument("--tenant-id", type=str, default=DEFAULT_TENANT_ID)
    parser.add_argument("--namespace-id", type=str, default=DEFAULT_NAMESPACE_ID)
    parser.add_argument("--snapshot-id", type=str, default=DEFAULT_SNAPSHOT_ID)
    parser.add_argument(
        "--no-text-corpus",
        action="store_true",
        help="Skip loading document texts into the mirrored collection.",
    )
    parser.add_argument("--index-root", type=Path)
    parser.add_argument("--mlp-path", type=Path)
    parser.add_argument("--w-path", type=Path)
    parser.add_argument("--query-divisor", type=float, default=32.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = load_task_json(str(args.task))
    candidate_ks = _parse_candidate_ks(args.candidate_ks, default_k=int(task["k"]))

    if (
        args.index_root is not None
        or args.mlp_path is not None
        or args.w_path is not None
    ):
        import kayak

        index = kayak.documents(
            [document["doc_id"] for document in task["documents"]],
            [document["vectors"] for document in task["documents"]],
            texts=[document["text"] for document in task["documents"]],
        ).pack()
        export_trained_reference_lemur_artifact(
            index,
            args.artifact_root,
            index_root=args.index_root,
            mlp_path=args.mlp_path,
            w_path=args.w_path,
            query_divisor=args.query_divisor,
            model_name=str(task["model_name"]),
        )

    with tempfile.TemporaryDirectory(
        prefix="kayak-native-latent-proxy-"
    ) as tmp_dir:
        collection_root = args.collection_root
        if collection_root is None:
            collection_root = Path(tmp_dir) / "collection"

        materialization = materialize_native_latent_proxy_collection(
            task_path=args.task,
            artifact_root=args.artifact_root,
            collection_root=collection_root,
            collection_id=args.collection_id,
            tenant_id=args.tenant_id,
            namespace_id=args.namespace_id,
            snapshot_id=args.snapshot_id,
            load_text_corpus=not args.no_text_corpus,
        )
        benchmarks = [
            benchmark_materialized_collection_search(
                task_path=args.task,
                collection_root=collection_root,
                snapshot_id=args.snapshot_id,
                candidate_generator_kind=args.candidate_generator_kind,
                candidate_k=candidate_k,
            )
            for candidate_k in candidate_ks
        ]

        payload = {
            "materialization": materialization,
            "benchmarks": benchmarks,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    print(f"wrote {args.output}")
    print(f"benchmarks: {len(candidate_ks)}")


if __name__ == "__main__":
    main()
