from std.collections import List

from .collection_hit import CollectionHit


struct ScoreHistogram(Copyable):
    var bin_count: Int
    var min_score: Float64
    var max_score: Float64
    var counts: List[Int]

    def __init__(
        out self,
        bin_count: Int,
        min_score: Float64,
        max_score: Float64,
        read counts: List[Int],
    ) raises:
        if bin_count < 0:
            raise Error("score histogram bin_count must be non-negative")

        if len(counts) != bin_count:
            raise Error("score histogram counts length must match bin_count")

        self.bin_count = bin_count
        self.min_score = min_score
        self.max_score = max_score
        self.counts = counts.copy()


def empty_score_histogram() raises -> ScoreHistogram:
    return ScoreHistogram(0, 0.0, 0.0, [])


def build_score_histogram(
    read hits: List[CollectionHit], bin_count: Int
) raises -> ScoreHistogram:
    if bin_count <= 0:
        raise Error("score histogram bin_count must be positive")

    if len(hits) == 0:
        return empty_score_histogram()

    var min_score = Float64(hits[0].score)
    var max_score = Float64(hits[0].score)

    for hit in hits:
        var score = Float64(hit.score)
        if score < min_score:
            min_score = score
        if score > max_score:
            max_score = score

    var counts = List[Int]()
    for _ in range(bin_count):
        counts.append(0)

    if min_score == max_score:
        counts[0] = len(hits)
        return ScoreHistogram(bin_count, min_score, max_score, counts)

    for hit in hits:
        var normalized = (Float64(hit.score) - min_score) / (max_score - min_score)
        var index = Int(normalized * Float64(bin_count))
        if index >= bin_count:
            index = bin_count - 1
        counts[index] += 1

    return ScoreHistogram(bin_count, min_score, max_score, counts)
