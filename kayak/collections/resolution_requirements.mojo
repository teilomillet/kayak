from std.collections import List

from .validation import require_non_empty_string


struct SnapshotLoadRequirements(Copyable):
    var load_all_search_artifacts: Bool
    var search_artifact_families: List[String]
    var load_text_corpus: Bool

    def __init__(
        out self,
        load_all_search_artifacts: Bool,
        read search_artifact_families: List[String],
        load_text_corpus: Bool,
    ) raises:
        var validated_families = List[String]()
        for family in search_artifact_families:
            var normalized = require_non_empty_string(
                family, "search_artifact family"
            )
            var already_present = False
            for existing in validated_families:
                if existing == normalized:
                    already_present = True
                    break

            if not already_present:
                validated_families.append(normalized)

        self.load_all_search_artifacts = load_all_search_artifacts
        self.search_artifact_families = validated_families^
        self.load_text_corpus = load_text_corpus

    def should_load_search_artifact_family(self, family: String) -> Bool:
        if self.load_all_search_artifacts:
            return True

        for required_family in self.search_artifact_families:
            if required_family == family:
                return True

        return False


def load_all_snapshot_requirements(
    load_text_corpus: Bool = True
) raises -> SnapshotLoadRequirements:
    return SnapshotLoadRequirements(True, [], load_text_corpus)


def exact_only_snapshot_requirements(
    load_text_corpus: Bool = False
) raises -> SnapshotLoadRequirements:
    return SnapshotLoadRequirements(False, [], load_text_corpus)


def search_artifact_snapshot_requirements(
    artifact_family: String, load_text_corpus: Bool = False
) raises -> SnapshotLoadRequirements:
    return SnapshotLoadRequirements(
        False,
        [require_non_empty_string(artifact_family, "artifact_family")],
        load_text_corpus,
    )
