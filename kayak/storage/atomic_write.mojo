from std.ffi import CStringSlice, c_int, external_call
from std.os import remove
from std.pathlib import Path


def atomic_write_temp_path(path: Path) -> Path:
    return Path(path.__fspath__() + ".atomic-write.tmp")


def rename_path_atomic(source: Path, target: Path) raises:
    var source_text = String(source.__fspath__() + "\0")
    var target_text = String(target.__fspath__() + "\0")
    var result = external_call["rename", c_int](
        CStringSlice(source_text),
        CStringSlice(target_text),
    )
    if result != 0:
        raise Error(
            "atomic rename failed: "
            + source.__fspath__()
            + " -> "
            + target.__fspath__()
        )


def write_text_atomic(path: Path, text: String) raises:
    var temp_path = atomic_write_temp_path(path)
    if temp_path.exists():
        remove(temp_path)

    temp_path.write_text(text)
    rename_path_atomic(temp_path, path)
