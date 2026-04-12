# Validation helpers for hosted collection contracts.


def require_non_empty_string(value: String, field_name: String) raises -> String:
    if value.byte_length() == 0:
        raise Error(field_name + " must not be empty")

    return value.copy()


def require_positive_int(value: Int, field_name: String) raises -> Int:
    if value <= 0:
        raise Error(field_name + " must be positive")

    return value


def require_non_negative_int(value: Int, field_name: String) raises -> Int:
    if value < 0:
        raise Error(field_name + " must be non-negative")

    return value
