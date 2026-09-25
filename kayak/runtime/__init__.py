"""Public local runtime API; importing it does not load inference libraries."""

from ._loading import load
from ._model import Model

__all__ = ["Model", "load"]
