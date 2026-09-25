"""The documented input recipe and limits, exercised without inference libraries."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from kayak import Choice, InputError
from kayak.decisions import request_from
from kayak.runtime._preparation import prepare_texts, validate_token_lengths


def test_prepared_order_matches_the_documented_choice_recipe() -> None:
    request = request_from(
        "\t café 🚣 \n",
        {
            "question-id-not-text": Choice(
                instructions=" Pick a route. \n",
                criteria={"candidate-id-not-text": "  billing\t", "second": "support"},
            ),
            "another-question": Choice(
                instructions="\t Pick urgency. ", criteria={"third": "\nurgent  "}
            ),
        },
    )
    assert prepare_texts(request) == [
        "café 🚣\n\nPick a route.",
        "café 🚣\n\nPick urgency.",
        "  billing\t",
        "support",
        "\nurgent  ",
    ]


@given(text=st.text(max_size=64), candidate=st.text(max_size=64))
def test_preparation_preserves_candidate_text_and_owns_its_output(
    text: str, candidate: str
) -> None:
    # The prefix guarantees nonblank text without excluding unusual Unicode.
    question = Choice(instructions="Choose.", criteria={"option": f"x{candidate}"})
    request = request_from(f"x{text}", {"question": question})
    padded = request_from(
        f" \t{request.state}\n",
        {"question": Choice(instructions=" \tChoose.\n", criteria=question.criteria)},
    )
    original = request.model_dump_json()
    texts = prepare_texts(request)
    assert texts == prepare_texts(padded)
    assert texts[1:] == [f"x{candidate}"]
    assert request.model_dump_json() == original
    texts.clear()
    assert prepare_texts(request) == prepare_texts(padded)
    assert request.model_dump_json() == original


@pytest.mark.parametrize("length", [1, 2048])
def test_sequence_token_limits_are_inclusive(length: int) -> None:
    assert validate_token_lengths([length], max_length=2048) == length


@pytest.mark.parametrize("length", [0, 2049])
def test_empty_and_oversized_sequences_are_rejected(length: int) -> None:
    with pytest.raises(InputError, match="never silently truncated"):
        validate_token_lengths([1, length, 1], max_length=2048)


@pytest.mark.parametrize("last_length", [2047, 2048])
def test_total_token_limit_accepts_values_up_to_the_boundary(last_length: int) -> None:
    lengths = [2048] * 7 + [last_length]
    assert validate_token_lengths(lengths, max_length=2048) == 14336 + last_length


def test_total_token_limit_rejects_one_token_over_budget() -> None:
    with pytest.raises(InputError, match="16384"):
        validate_token_lengths([2048] * 8 + [1], max_length=2048)


@given(lengths=st.lists(st.integers(min_value=1, max_value=2048), min_size=1, max_size=8))
def test_token_accounting_is_additive_and_does_not_mutate_lengths(lengths: list[int]) -> None:
    before = lengths.copy()
    split = len(lengths) // 2
    left = validate_token_lengths(lengths[:split], max_length=2048)
    right = validate_token_lengths(lengths[split:], max_length=2048)
    assert validate_token_lengths(lengths, max_length=2048) == left + right
    assert lengths == before
