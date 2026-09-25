"""Score ordered IDs without prescribing a model, embedding layout, or score scale."""

import math
from collections.abc import Mapping, Sequence
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _check_ids(identifiers: Sequence[str]) -> None:
    if isinstance(identifiers, str) or any(
        not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers
    ):
        raise ValueError("document IDs must be nonblank strings in a sequence")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("document IDs must be unique")


class RankedOutput(BaseModel):
    """An adapter's best-first IDs and optional native scores; empty output is valid.

    Order is authoritative. Scores need not be probabilities or share a scale
    with another backend. Provenance is caller-supplied model/index/recipe
    identity, not independently verified execution evidence.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)
    ids: list[str]
    scores: dict[str, float] | None = None
    provenance: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_ids_and_scores(self) -> Self:
        _check_ids(self.ids)
        if self.scores is not None and self.scores.keys() != set(self.ids):
            raise ValueError("scores must contain exactly the returned document IDs")
        return self


class RankingAssessment(BaseModel):
    """Metrics over supplied judgments; missing labels and unseen ranks stay unknown."""

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)
    k: int
    ranked_ids: list[str]
    relevance: dict[str, int]
    unjudged_ids: list[str]
    metrics: dict[str, float | None]
    unavailable_reasons: dict[str, str]
    complete: bool = True


def assess_ranking(
    ranked_ids: Sequence[str], relevance: Mapping[str, int], *, k: int, complete: bool = True
) -> RankingAssessment:
    """Pure ranking evaluation with explicit unknowns and no model calls.

    Positive integer grades mean relevant; zero means judged irrelevant; absent
    IDs are unjudged. Recall uses only the known positive IDs in `relevance`,
    including positives missing from the ranked list. It is not total-corpus
    recall unless the judgments cover that corpus. nDCG uses linear grade gain.
    Set complete=False for a recorded prefix with an unobserved tail. A cutoff
    within that prefix remains assessable; an unseen rank is not an irrelevant
    document. Known hits and a recovered set of all known positives stay known.
    """
    _check_ids(ranked_ids)
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if type(complete) is not bool:
        raise ValueError("complete must be a boolean")
    _check_ids(list(relevance))
    if any(type(grade) is not int or grade < 0 for grade in relevance.values()):
        raise ValueError("relevance grades must be nonnegative integers")
    ordered = list(ranked_ids)
    judgments = dict(relevance)
    positives = {identifier for identifier, grade in judgments.items() if grade > 0}
    unjudged = [identifier for identifier in ordered if identifier not in judgments]
    prefix = ordered[:k]
    prefix_complete = complete or k <= len(ordered)
    prefix_unknown = any(identifier not in judgments for identifier in prefix)
    hits = sum(identifier in positives for identifier in prefix)
    first_positive = next(
        (position for position, identifier in enumerate(ordered, 1) if identifier in positives),
        None,
    )

    top1 = None
    if ordered and ordered[0] in judgments:
        top1 = float(ordered[0] in positives)
    elif not ordered and positives and complete:
        top1 = 0.0
    hit_at_k = None
    if hits:
        hit_at_k = 1.0
    elif prefix_complete and not prefix_unknown and judgments:
        hit_at_k = 0.0
    recall = None
    if positives and (prefix_complete or hits == len(positives)):
        recall = hits / len(positives)
    reciprocal_rank = None
    if first_positive is not None:
        if all(identifier in judgments for identifier in ordered[:first_positive]):
            reciprocal_rank = 1 / first_positive
    elif positives and not unjudged and complete:
        reciprocal_rank = 0.0

    ndcg = None
    if positives and prefix_complete and not prefix_unknown:
        ideal = sorted((judgments[identifier] for identifier in positives), reverse=True)[:k]
        # Scaling gains by the maximum cancels in the ratio and avoids overflow
        # when a caller uses large integer grades.
        maximum = ideal[0]
        ideal_dcg = math.fsum(
            (grade / maximum) / math.log2(position + 1) for position, grade in enumerate(ideal, 1)
        )
        actual_dcg = math.fsum(
            (judgments[identifier] / maximum) / math.log2(position + 1)
            for position, identifier in enumerate(prefix, 1)
        )
        ndcg = actual_dcg / ideal_dcg
    unavailable: dict[str, str] = {}
    if top1 is None:
        unavailable["top1"] = "first result is unjudged or no judged first-result outcome exists"
    if hit_at_k is None:
        unavailable["hit_at_k"] = (
            "no known hit and missing observations or judgments for the prefix"
        )
    if recall is None:
        unavailable["known_recall_at_k"] = (
            "no known positive judgments; denominator is zero"
            if not positives
            else "unobserved prefix positions may contain additional known positives"
        )
    if reciprocal_rank is None:
        unavailable["reciprocal_rank"] = "first relevant rank is unknown under these judgments"
    if ndcg is None:
        unavailable["ndcg_at_k"] = (
            "requires positive judgments and a fully observed, fully judged prefix"
        )
    if not prefix or not prefix_complete:
        unavailable["judged_fraction_at_k"] = (
            "no returned documents in the prefix" if not prefix else "prefix is not fully observed"
        )
    return RankingAssessment(
        k=k,
        ranked_ids=ordered,
        relevance=judgments,
        unjudged_ids=unjudged,
        metrics={
            "top1": top1,
            "hit_at_k": hit_at_k,
            "known_recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank,
            "ndcg_at_k": ndcg,
            "judged_fraction_at_k": (
                sum(identifier in judgments for identifier in prefix) / len(prefix)
                if prefix and prefix_complete
                else None
            ),
        },
        unavailable_reasons=unavailable,
        complete=complete,
    )


def ranking_metrics(relevant: Sequence[str], ordered: Sequence[str], k: int) -> dict[str, float]:
    """Binary metrics for fully judged cases; every other returned ID is irrelevant.

    This is the starter-case convention. Use `assess_ranking` when some returned
    documents are unjudged. Empty output contributes zero; empty positives have
    no recall denominator and are rejected instead of reporting a false score.
    """
    _check_ids(relevant)
    if not relevant:
        raise ValueError("binary ranking cases need at least one relevant document")
    judgments = dict.fromkeys(ordered, 0)
    judgments.update(dict.fromkeys(relevant, 1))
    assessed = assess_ranking(ordered, judgments, k=k)
    metrics: dict[str, float] = {}
    for name, source in (
        ("top1", "top1"),
        ("hit_at_k", "hit_at_k"),
        ("recall_at_k", "known_recall_at_k"),
        ("mrr", "reciprocal_rank"),
    ):
        value = assessed.metrics[source]
        if value is None:
            raise ValueError("binary ranking metrics require complete judgments")
        metrics[name] = value
    return metrics
