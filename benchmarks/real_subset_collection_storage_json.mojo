from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    RealSliceCollectionStorageSummary,
    build_real_slice_collection_storage_summary,
    ensure_public_benchmark_dataset_collection_mirror,
    load_default_public_benchmark_datasets,
    real_slice_collection_storage_summaries_json,
)
from kayak.collections import (
    SnapshotId,
    load_collection_storage_report,
)


def append_storage_summary(
    mut summaries: List[RealSliceCollectionStorageSummary],
    dataset_id: String,
    collection_name: String,
    collection_root: Path,
    model_name: String,
) raises:
    summaries.append(
        build_real_slice_collection_storage_summary(
            dataset_id,
            collection_name,
            model_name,
            "snapshot-0001",
            load_collection_storage_report(
                collection_root, SnapshotId("snapshot-0001")
            ),
        )
    )


def main() raises:
    var summaries = List[RealSliceCollectionStorageSummary]()

    for dataset in load_default_public_benchmark_datasets():
        append_storage_summary(
            summaries,
            dataset.stored_task.dataset_id,
            dataset.collection_root_stem,
            ensure_public_benchmark_dataset_collection_mirror(
                dataset,
                "collection",
            ),
            dataset.stored_index.model_name,
        )

    var output_root = Path(".cache/kayak")
    makedirs(output_root, exist_ok=True)
    (output_root / "public_real_slice_collection_storage.json").write_text(
        real_slice_collection_storage_summaries_json(summaries)
    )
    print(
        "wrote ",
        String(output_root / "public_real_slice_collection_storage.json"),
    )
