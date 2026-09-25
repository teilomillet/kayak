"""Resident resources, serialized execution, and explicit model lifetime."""

from __future__ import annotations

from collections.abc import KeysView, Mapping
from dataclasses import dataclass
from threading import Lock
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self

from .. import decisions, judgments, ranking
from ..bundle import ModelSpec
from ..decisions import Choice, ChoiceAnswer, DecisionResult, ModelInfo
from ..errors import InferenceError, InputError, ModelClosedError
from ..judgments import JudgmentQuestion, JudgmentResult
from ..ranking import RankingResult
from ._preparation import prepare_texts, validate_token_lengths

if TYPE_CHECKING:
    from torch import Tensor
    from torch import device as TorchDevice
    from torch import dtype as TorchDtype

    from .._heads import ProjectionHead


@dataclass(frozen=True)
class _CandidateCache:
    token_ids: tuple[tuple[int, ...], ...]
    embeddings: Tensor


class Model:
    """A resident CLM. Calls are serialized; close waits for active inference."""

    def __init__(
        self,
        *,
        spec: ModelSpec,
        tokenizer: Tokenizer,
        encoder: Encoder,
        state_head: ProjectionHead,
        action_head: ProjectionHead,
        scale: float,
        batch_size: int,
    ) -> None:
        self.spec: ModelSpec = spec
        self._tokenizer: Tokenizer | None = tokenizer
        self._encoder: Encoder | None = encoder
        self._state_head: ProjectionHead | None = state_head
        self._action_head: ProjectionHead | None = action_head
        self._scale = scale
        self._batch_size = batch_size
        self._candidate_cache: _CandidateCache | None = None
        self._lock = Lock()
        self.info = ModelInfo(
            id=spec.model_id,
            revision=spec.revision,
            fingerprint=spec.fingerprint,
            encoder=spec.encoder_id,
            encoder_revision=spec.encoder_revision,
            device=str(encoder.device),
            dtype=str(encoder.dtype).removeprefix("torch."),
        )

    def decide(
        self, *, state: str, questions: Mapping[str, Choice | Mapping[str, object]]
    ) -> DecisionResult:
        request = decisions.request_from(state, questions)
        texts = prepare_texts(request)
        with self._lock:
            if self._encoder is None:
                raise ModelClosedError("model has been closed")
            assert self._tokenizer is not None
            assert self._state_head is not None and self._action_head is not None
            try:
                import torch
                from torch.nn import functional as F

                input_ids = self._tokenizer(texts, add_special_tokens=True, truncation=False)[
                    "input_ids"
                ]
                input_tokens = validate_token_lengths(
                    [len(row) for row in input_ids], max_length=self.spec.max_length
                )
                question_count = len(request.questions)
                # Singleton rows have no padding or batch neighbors to change
                # their encoder result. Keep only the last ordered candidate block.
                candidate_token_ids = (
                    tuple(tuple(row) for row in input_ids[question_count:])
                    if self._batch_size == 1
                    else None
                )
                cached = self._candidate_cache
                if cached is not None and cached.token_ids == candidate_token_ids:
                    encoder_input_ids = input_ids[:question_count]
                else:
                    self._candidate_cache = cached = None
                    encoder_input_ids = input_ids
                with torch.inference_mode():
                    normalized_batches: list[Tensor] = []
                    for start in range(0, len(encoder_input_ids), self._batch_size):
                        batch = self._tokenizer.pad(
                            {"input_ids": encoder_input_ids[start : start + self._batch_size]},
                            padding=True,
                            return_tensors="pt",
                        ).to(self._encoder.device)
                        hidden_states = self._encoder(**batch, use_cache=False).last_hidden_state
                        # Loading fixes right padding; each row uses its own final real token.
                        last_token_positions = batch["attention_mask"].sum(dim=1) - 1
                        last_token_embeddings = hidden_states[
                            torch.arange(len(last_token_positions), device=hidden_states.device),
                            last_token_positions,
                        ]
                        last_token_embeddings = last_token_embeddings.float()
                        normalized_batches.append(
                            last_token_embeddings
                            / (last_token_embeddings.norm(dim=-1, keepdim=True) + 1e-12)
                        )
                    if cached is not None:
                        normalized_batches.append(cached.embeddings)
                    embeddings = torch.cat(normalized_batches)
                    # Preparation places question states before all candidate rows.
                    state_embeddings = F.normalize(
                        self._state_head(embeddings[:question_count]), dim=-1
                    )
                    candidate_embeddings = F.normalize(
                        self._action_head(embeddings[question_count:]), dim=-1
                    )
                    answers: dict[str, ChoiceAnswer] = {}
                    candidate_offset = 0
                    for question_index, (question_id, question) in enumerate(
                        request.questions.items()
                    ):
                        candidate_count = len(question.criteria)
                        question_candidates = candidate_embeddings[
                            candidate_offset : candidate_offset + candidate_count
                        ]
                        scores: list[float] = (
                            self._scale * (question_candidates @ state_embeddings[question_index])
                        ).tolist()
                        answers[question_id] = decisions.answer_from_scores(
                            list(question.criteria), scores
                        )
                        candidate_offset += candidate_count
                    result = DecisionResult(
                        model=self.info, answers=answers, input_tokens=input_tokens
                    )
                    if candidate_token_ids is not None and cached is None:
                        # Commit only after a valid result. A failed call must not
                        # retain invalid vectors; the next call can recompute them.
                        # Clone the candidate rows without retaining the state rows.
                        self._candidate_cache = _CandidateCache(
                            candidate_token_ids, embeddings[question_count:].clone()
                        )
                return result
            except (InputError, InferenceError):
                raise
            except Exception as exc:
                raise InferenceError(f"CLM inference failed ({type(exc).__name__})") from exc

    def rank(
        self, *, state: str, instructions: str, candidates: Mapping[str, str]
    ) -> RankingResult:
        """Rank supplied descriptions; the caller owns selection policy and execution."""
        request = ranking.request_from(state, instructions, candidates)
        result = self.decide(state=request.state, questions=request.questions)
        return ranking.result_from(result)

    def judge(
        self, *, state: str, questions: Mapping[str, JudgmentQuestion | Mapping[str, object]]
    ) -> JudgmentResult:
        """Evaluate named typed judgments through the same serialized Choice computation."""
        request = judgments.request_from(state, questions)
        decision = request.as_decision()
        result = self.decide(state=decision.state, questions=decision.questions)
        return request.decode(result)

    def close(self) -> None:
        with self._lock:
            self._encoder = self._state_head = self._action_head = self._tokenizer = None
            self._candidate_cache = None

    def __enter__(self) -> Self:
        if self._encoder is None:
            raise ModelClosedError("model has been closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class TensorBatch(Protocol):
    """The tensor-only output of tokenizer.pad(return_tensors='pt')."""

    def to(self, device: TorchDevice) -> Self: ...
    def keys(self) -> KeysView[str]: ...
    def __getitem__(self, key: str) -> Tensor: ...


class Tokenizer(Protocol):
    """Text preparation and padding needed by the resident model."""

    padding_side: str
    pad_token_id: int | None

    def __call__(
        self, texts: list[str], *, add_special_tokens: bool, truncation: bool
    ) -> Mapping[str, list[list[int]]]: ...

    def pad(
        self, inputs: dict[str, list[list[int]]], *, padding: bool, return_tensors: str
    ) -> TensorBatch: ...


class EncoderOutput(Protocol):
    @property
    def last_hidden_state(self) -> Tensor: ...


class Encoder(Protocol):
    """The encoder operations consumed by Model; loading returns the concrete model."""

    @property
    def device(self) -> TorchDevice: ...

    @property
    def dtype(self) -> TorchDtype: ...

    def __call__(self, *, use_cache: bool, **inputs: Tensor) -> EncoderOutput: ...
