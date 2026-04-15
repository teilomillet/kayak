from __future__ import annotations

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import urllib.error
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_MODULE = "kayak_engine.server"


def http_json(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


class HostedEngineServer:
    def __init__(self, root: Path):
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        python_root = str(REPO_ROOT / "python")
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = (
            python_root
            if existing_pythonpath in (None, "")
            else f"{python_root}{os.pathsep}{existing_pythonpath}"
        )
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                SERVER_MODULE,
                "--root",
                str(root),
                "--port",
                "0",
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.base_url = self._wait_until_ready()

    def _wait_until_ready(self) -> str:
        assert self.process.stdout is not None
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                stdout = self.process.stdout.read() if self.process.stdout else ""
                stderr = self.process.stderr.read() if self.process.stderr else ""
                raise RuntimeError(
                    "hosted engine server exited before becoming ready\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                )
            ready, _, _ = select.select([self.process.stdout], [], [], 0.5)
            if not ready:
                continue
            line = self.process.stdout.readline().strip()
            if "listening on http://" not in line:
                continue
            return line.rsplit(" ", 1)[-1]
        self.close()
        raise RuntimeError("timed out waiting for hosted engine server startup")

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.stderr is not None:
            self.process.stderr.close()

    def __enter__(self) -> "HostedEngineServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = exc_type
        _ = exc
        _ = tb
        self.close()
