from std.collections import List

from kayak.numeric import ScoreScalar


def centroid_shortlist_entry_is_worse(
    score: ScoreScalar,
    centroid_index: Int,
    other_score: ScoreScalar,
    other_centroid_index: Int,
) -> Bool:
    if score < other_score:
        return True

    if score > other_score:
        return False

    # Equal scores keep earlier centroids, so the larger centroid index is the
    # eviction candidate when the shortlist overflows.
    return centroid_index > other_centroid_index


def centroid_shortlist_entry_ranks_before(
    score: ScoreScalar,
    centroid_index: Int,
    other_score: ScoreScalar,
    other_centroid_index: Int,
) -> Bool:
    if score > other_score:
        return True

    if score < other_score:
        return False

    return centroid_index < other_centroid_index


def swap_centroid_shortlist_entries(
    mut centroid_indices: List[Int],
    mut centroid_scores: List[ScoreScalar],
    lhs: Int,
    rhs: Int,
):
    var index_value = centroid_indices[lhs]
    centroid_indices[lhs] = centroid_indices[rhs]
    centroid_indices[rhs] = index_value

    var score_value = centroid_scores[lhs]
    centroid_scores[lhs] = centroid_scores[rhs]
    centroid_scores[rhs] = score_value


def sift_up_worst_first_centroid_shortlist(
    mut centroid_indices: List[Int],
    mut centroid_scores: List[ScoreScalar],
    start_position: Int,
):
    var position = start_position

    while position > 0:
        var parent = (position - 1) // 2
        if not centroid_shortlist_entry_is_worse(
            centroid_scores[position],
            centroid_indices[position],
            centroid_scores[parent],
            centroid_indices[parent],
        ):
            break

        swap_centroid_shortlist_entries(
            centroid_indices,
            centroid_scores,
            position,
            parent,
        )
        position = parent


def sift_down_worst_first_centroid_shortlist(
    mut centroid_indices: List[Int],
    mut centroid_scores: List[ScoreScalar],
    shortlist_count: Int,
    start_position: Int,
):
    var position = start_position

    while True:
        var left = position * 2 + 1
        if left >= shortlist_count:
            return

        var next = left
        var right = left + 1
        if (
            right < shortlist_count
            and centroid_shortlist_entry_is_worse(
                centroid_scores[right],
                centroid_indices[right],
                centroid_scores[left],
                centroid_indices[left],
            )
        ):
            next = right

        if not centroid_shortlist_entry_is_worse(
            centroid_scores[next],
            centroid_indices[next],
            centroid_scores[position],
            centroid_indices[position],
        ):
            return

        swap_centroid_shortlist_entries(
            centroid_indices,
            centroid_scores,
            position,
            next,
        )
        position = next


def insert_top_bound_centroid_match(
    mut centroid_indices: List[Int],
    mut centroid_scores: List[ScoreScalar],
    centroid_index: Int,
    centroid_score: ScoreScalar,
    bound: Int,
):
    if bound <= 0:
        return

    var shortlist_count = len(centroid_scores)
    if shortlist_count < bound:
        centroid_indices.append(centroid_index)
        centroid_scores.append(centroid_score)
        sift_up_worst_first_centroid_shortlist(
            centroid_indices,
            centroid_scores,
            shortlist_count,
        )
        return

    # The root is the current eviction candidate:
    # lowest score, and on equal scores the latest centroid index.
    if centroid_score <= centroid_scores[0]:
        return

    centroid_indices[0] = centroid_index
    centroid_scores[0] = centroid_score
    sift_down_worst_first_centroid_shortlist(
        centroid_indices,
        centroid_scores,
        shortlist_count,
        0,
    )


def sort_top_bound_centroid_matches_descending(
    mut centroid_indices: List[Int],
    mut centroid_scores: List[ScoreScalar],
):
    for sorted_end in range(1, len(centroid_scores)):
        var centroid_index = centroid_indices[sorted_end]
        var centroid_score = centroid_scores[sorted_end]
        var insert_at = sorted_end

        while (
            insert_at > 0
            and centroid_shortlist_entry_ranks_before(
                centroid_score,
                centroid_index,
                centroid_scores[insert_at - 1],
                centroid_indices[insert_at - 1],
            )
        ):
            centroid_indices[insert_at] = centroid_indices[insert_at - 1]
            centroid_scores[insert_at] = centroid_scores[insert_at - 1]
            insert_at -= 1

        centroid_indices[insert_at] = centroid_index
        centroid_scores[insert_at] = centroid_score
