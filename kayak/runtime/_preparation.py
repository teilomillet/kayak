"""Pure CLM input preparation and token-budget policy, without model dependencies."""

from collections.abc import Sequence

from ..decisions import MAX_TOTAL_TOKENS, DecisionRequest
from ..errors import InputError


def prepare_texts(request: DecisionRequest) -> list[str]:
    """Return all question states, then candidates, in their original order.

    The clm-choice-v1 recipe strips state/instructions and preserves candidate
    descriptions verbatim. IDs never enter the encoder input. The returned list
    belongs to the caller; preparing it does not mutate the validated request.
    """
    state = request.state.strip()
    texts = [
        f"{state}\n\n{question.instructions.strip()}" for question in request.questions.values()
    ]
    for question in request.questions.values():
        texts.extend(question.criteria.values())
    return texts


def validate_token_lengths(lengths: Sequence[int], *, max_length: int) -> int:
    """Check actual tokenizer row lengths before execution and return their total.

    Raises InputError for an empty/oversized sequence or an excessive request.
    No truncation or padding is performed here.
    """
    if any(length == 0 or length > max_length for length in lengths):
        raise InputError(
            f"each prepared state or candidate must contain 1..{max_length} "
            "tokens; input is never silently truncated"
        )
    total = sum(lengths)
    if total > MAX_TOTAL_TOKENS:
        raise InputError(f"request exceeds {MAX_TOTAL_TOKENS} prepared input tokens")
    return total
