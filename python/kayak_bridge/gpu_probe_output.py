from __future__ import annotations


def parse_key_value_probe_output(stdout: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in stdout.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip()
        normalized_key = key.strip()
        if value in {"True", "False"}:
            parsed[normalized_key] = value == "True"
            continue
        parsed_number = _parse_number(value)
        parsed[normalized_key] = parsed_number if parsed_number is not None else value
    return parsed


def _parse_number(value: str) -> int | float | None:
    try:
        if any(char in value for char in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return None
