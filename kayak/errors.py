"""Failures shared by direct inference and the HTTP client."""


class KayakError(Exception):
    """Base class for Kayak failures."""

    def __init__(self, *args: object, request_id: str | None = None) -> None:
        super().__init__(*args)
        self.request_id = request_id


class InputError(KayakError, ValueError):
    """A request cannot be evaluated without changing its meaning."""


class ModelLoadError(KayakError):
    """The model artifacts or execution environment are incompatible."""


class InferenceError(KayakError):
    """Inference failed or returned invalid scores."""


class ModelClosedError(KayakError):
    """The loaded model has been closed."""


class TransportError(KayakError):
    """A server could not be reached or returned an invalid response."""


class RemoteError(KayakError):
    """A server rejected a request or could not complete it."""

    def __init__(
        self, message: str, *, status_code: int, code: str, request_id: str | None = None
    ) -> None:
        super().__init__(message, request_id=request_id)
        self.status_code = status_code
        self.code = code
