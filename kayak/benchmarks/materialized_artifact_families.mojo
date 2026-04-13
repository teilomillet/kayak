from std.collections import List

from kayak.planning import StageArtifactMaterialization


def materialized_artifact_families(
    read materializations: List[StageArtifactMaterialization]
) -> List[String]:
    var families = List[String]()

    for materialization in materializations:
        var already_seen = False
        for family in families:
            if family == materialization.family:
                already_seen = True
                break

        if not already_seen:
            families.append(materialization.family.copy())

    return families^
