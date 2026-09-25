"""The Rust tokenizers surface used to build the offline test tokenizer."""

from .models import WordLevel
from .pre_tokenizers import Whitespace

class Tokenizer:
    pre_tokenizer: Whitespace
    def __init__(self, model: WordLevel) -> None: ...
