"""The pyperf 2.x surface used by our benchmark harness (upstream is untyped)."""

from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Mapping

class Runner:
    argparser: ArgumentParser
    metadata: dict[str, str | int | float]
    def __init__(
        self,
        *,
        program_args: tuple[str, ...],
        add_cmdline_args: Callable[[list[str], Namespace], None],
    ) -> None: ...
    def parse_args(self) -> Namespace: ...
    def bench_func(
        self, name: str, func: Callable[[], object], *, metadata: Mapping[str, object]
    ) -> object: ...
