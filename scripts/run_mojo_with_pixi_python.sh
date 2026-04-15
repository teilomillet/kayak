#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

python_executable="$(
  python - <<'PY'
import sys
print(sys.executable)
PY
)"

python_library="$(
  python - <<'PY'
from pathlib import Path
import sys
import sysconfig

libdir = Path(sysconfig.get_config_var("LIBDIR") or "")
version_prefix = f"libpython{sys.version_info.major}.{sys.version_info.minor}"

preferred_candidates = []
for pattern in (
    version_prefix + ".dylib",
    version_prefix + ".so",
    version_prefix + ".so.*",
    "libpython*.dylib",
    "libpython*.so",
    "libpython*.so.*",
):
    preferred_candidates.extend(sorted(libdir.glob(pattern)))

for candidate in preferred_candidates:
    if candidate.suffix == ".a":
        continue
    print(candidate)
    raise SystemExit(0)

ldlibrary = sysconfig.get_config_var("LDLIBRARY")
if ldlibrary:
    library_path = libdir / ldlibrary
    if library_path.exists() and library_path.suffix != ".a":
        print(library_path)
        raise SystemExit(0)

raise SystemExit(
    "Could not locate a shared libpython for Mojo under " + str(libdir)
)
PY
)"

export MOJO_PYTHON="${python_executable}"
export MOJO_PYTHON_LIBRARY="${python_library}"

if command -v mojo >/dev/null 2>&1; then
  exec mojo "$@"
fi

if command -v pixi >/dev/null 2>&1; then
  exec pixi run --manifest-path "${repo_root}/pyproject.toml" mojo "$@"
fi

echo "Could not find \`mojo\` or \`pixi\` on PATH" >&2
exit 1
