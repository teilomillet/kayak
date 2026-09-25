"""Check four client/server pairs from two non-editable installations; no inference.

Requires --baseline-python, --current-python, and a fresh --output directory.
See docs/compatibility.md for the pinned baseline and environment setup.
"""

import argparse
import hashlib
import json
import signal
import subprocess
from pathlib import Path
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/api_v1"


class Arguments(argparse.Namespace):
    baseline_python: Path
    current_python: Path
    output: Path


def package_digest(directory: Path) -> str:
    """Identify installed Python, model manifests, and the typing marker, without importing them."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if path.is_file() and (path.suffix in {".py", ".json"} or path.name == "py.typed"):
            digest.update(path.relative_to(directory).as_posix().encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_bytes())
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"expected a JSON object: {path}")
    return dict(value)


def check_installation(receipt: dict[str, object], expected_digest: str) -> None:
    package = receipt.get("package")
    if not isinstance(package, str):
        raise ValueError("process did not identify its installed package")
    actual_digest = package_digest(Path(package))
    receipt["package_sha256"] = actual_digest
    if actual_digest != expected_digest:
        raise ValueError(f"wrong or stale installed package: {package}")


def run_pair(
    server_python: Path,
    client_python: Path,
    work: Path,
    server_digest: str,
    client_digest: str,
    *,
    extensions: bool,
) -> dict[str, object]:
    work.mkdir()
    result: dict[str, object] = {"status": "failed"}
    with (work / "server.log").open("w") as server_log:
        process = subprocess.Popen(
            [
                str(server_python),
                "-I",
                str(ROOT / "scripts/compatibility/server.py"),
                "--fixtures",
                str(FIXTURES),
                "--work",
                str(work),
            ],
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=work,
        )
        try:
            deadline = monotonic() + 10
            address = work / "address.json"
            while not address.exists():
                if process.poll() is not None or monotonic() > deadline:
                    raise RuntimeError("server did not announce an address; inspect server.log")
                sleep(0.02)
            server = read_object(address)
            result["server"] = server
            check_installation(server, server_digest)
            url = server.get("url")
            if not isinstance(url, str) or not url.startswith("http://127.0.0.1:"):
                raise ValueError("server must bind loopback")
            with (work / "client.json").open("w") as output, (work / "client.log").open("w") as log:
                completed = subprocess.run(
                    [
                        str(client_python),
                        "-I",
                        str(ROOT / "scripts/compatibility/client.py"),
                        "--url",
                        url,
                        "--fixtures",
                        str(FIXTURES),
                        "--work",
                        str(work),
                        *(["--extensions"] if extensions else []),
                    ],
                    stdout=output,
                    stderr=log,
                    text=True,
                    timeout=30,
                    cwd=work,
                )
            if completed.returncode != 0:
                raise RuntimeError(f"client exited {completed.returncode}; inspect client.log")
            client = read_object(work / "client.json")
            result["client"] = client
            check_installation(client, client_digest)
            result["status"] = "passed"
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            result["error"] = str(exc)
        finally:
            # Release controlled work even when the probe fails, then require cleanup.
            (work / "release").touch()
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
                result.update(status="failed", shutdown_error="server required a kill")
            result["server_returncode"] = process.returncode
            if process.returncode not in (0, -signal.SIGTERM) or not (work / "closed").exists():
                result.update(status="failed", shutdown_error="model did not close cleanly")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--current-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(namespace=Arguments())
    baseline = read_object(FIXTURES / "baseline.json")
    baseline_digest = baseline.get("package_sha256")
    if not isinstance(baseline_digest, str) or len(baseline_digest) != 64:
        parser.error("baseline must record its package SHA-256")
    # Keep venv executable paths: resolving their symlinks would discard the environment.
    interpreters = {
        "baseline": args.baseline_python.absolute(),
        "current": args.current_python.absolute(),
    }
    if len(set(interpreters.values())) != 2 or not all(p.is_file() for p in interpreters.values()):
        parser.error("provide two distinct installed Python environments")
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    digests = {"baseline": baseline_digest, "current": package_digest(ROOT / "kayak")}
    pairs: dict[str, object] = {}
    report: dict[str, object] = {
        "baseline": baseline,
        "fixtures_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (FIXTURES / "request.json", FIXTURES / "response.json")
        },
        "pairs": pairs,
    }
    failed = False
    for server_name, server_python in interpreters.items():
        for client_name, client_python in interpreters.items():
            name = f"{client_name}-client_{server_name}-server"
            try:
                result = run_pair(
                    server_python,
                    client_python,
                    output / name,
                    digests[server_name],
                    digests[client_name],
                    extensions=client_name == "current",
                )
            except OSError as exc:
                result = {"status": "failed", "error": str(exc)}
            pairs[name] = result
            failed |= result["status"] != "passed"
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name}: {result['status']}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
