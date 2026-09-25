"""Read a user-owned classification suite without downloading data or loading models."""

import json
from pathlib import Path

from pydantic import ValidationError

from ._schema import Suite


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key; keys must be unique")
        result[key] = value
    return result


def load_suite(path: str | Path) -> Suite:
    """Validate the complete file; preserve examples, descriptions, order, and provenance.

    Duplicate JSON keys are rejected instead of silently replacing a candidate.
    Validation errors identify fields without printing evaluation input values.
    Token limits still require the model's tokenizer when a run executes.
    """
    raw = Path(path).read_bytes()
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_keys)
        return Suite.model_validate(payload)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        details = [
            f"  {'.'.join(map(str, error['loc'])) or 'suite'}: {error['msg']}"
            for error in errors[:10]
        ]
        if len(errors) > 10:
            details.append(f"  ... and {len(errors) - 10} more validation errors")
        # An uncaught chained ValidationError would print the original input values.
        raise ValueError(f"invalid suite {path}:\n" + "\n".join(details)) from None
    except ValueError as exc:
        raise ValueError(f"invalid suite {path}: {exc}") from exc
