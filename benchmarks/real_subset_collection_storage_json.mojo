from std.collections import List
from std.os import makedirs
from std.pathlib import Path

from kayak.benchmarks import (
    RealSliceCollectionStorageSummary,
    build_real_slice_collection_storage_summary,
    real_slice_collection_storage_summaries_json,
)
from kayak.collections import (
    CollectionId,
    NamespaceId,
    SnapshotId,
    TenantId,
    ensure_one_segment_collection_mirror,
    load_collection_storage_report,
)
from kayak.storage import (
    ensure_browsecomp_plus_gold_real_subset_cache,
    ensure_browsecomp_plus_real_subset_cache,
    ensure_fiqa_real_subset_cache,
    ensure_limit_small_real_subset_cache,
    ensure_scifact_real_subset_cache,
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

    var scifact_cache = ensure_scifact_real_subset_cache()
    append_storage_summary(
        summaries,
        scifact_cache.stored_task.dataset_id,
        "scifact_real_subset",
        ensure_one_segment_collection_mirror(
            Path(".cache/kayak/scifact_real_subset_collection"),
            CollectionId("scifact_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            scifact_cache.stored_index,
        ),
        scifact_cache.stored_index.model_name,
    )

    var fiqa_cache = ensure_fiqa_real_subset_cache()
    append_storage_summary(
        summaries,
        fiqa_cache.stored_task.dataset_id,
        "fiqa_real_subset",
        ensure_one_segment_collection_mirror(
            Path(".cache/kayak/fiqa_real_subset_collection"),
            CollectionId("fiqa_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            fiqa_cache.stored_index,
        ),
        fiqa_cache.stored_index.model_name,
    )

    var limit_small_cache = ensure_limit_small_real_subset_cache()
    append_storage_summary(
        summaries,
        limit_small_cache.stored_task.dataset_id,
        "limit_small_real_subset",
        ensure_one_segment_collection_mirror(
            Path(".cache/kayak/limit_small_real_subset_collection"),
            CollectionId("limit_small_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            limit_small_cache.stored_index,
        ),
        limit_small_cache.stored_index.model_name,
    )

    var browsecomp_cache = ensure_browsecomp_plus_real_subset_cache()
    append_storage_summary(
        summaries,
        browsecomp_cache.stored_task.dataset_id,
        "browsecomp_plus_real_subset",
        ensure_one_segment_collection_mirror(
            Path(".cache/kayak/browsecomp_plus_real_subset_collection"),
            CollectionId("browsecomp_plus_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            browsecomp_cache.stored_index,
        ),
        browsecomp_cache.stored_index.model_name,
    )

    var browsecomp_gold_cache = ensure_browsecomp_plus_gold_real_subset_cache()
    append_storage_summary(
        summaries,
        browsecomp_gold_cache.stored_task.dataset_id,
        "browsecomp_plus_gold_real_subset",
        ensure_one_segment_collection_mirror(
            Path(".cache/kayak/browsecomp_plus_gold_real_subset_collection"),
            CollectionId("browsecomp_plus_gold_real_subset"),
            TenantId("public"),
            NamespaceId("benchmark"),
            SnapshotId("snapshot-0001"),
            1,
            browsecomp_gold_cache.stored_index,
        ),
        browsecomp_gold_cache.stored_index.model_name,
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
