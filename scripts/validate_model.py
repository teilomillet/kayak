"""Run explicit full-model checks on a CUDA or macOS/MPS machine.

Requires `uv sync --extra serve`. Never substitutes a smaller model. The
optional reference endpoint is an independently running, pinned vLLM encoder.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import socket
import statistics
import threading
import time
from importlib.metadata import version
from pathlib import Path
from typing import TypedDict, cast

import httpx
import numpy as np
import torch
import uvicorn

import kayak
from kayak._heads import load_heads
from kayak.bundle import DEFAULT_MODEL, ModelSpec, resolve, sha256
from kayak.server import DecisionModel, create_app

Case = tuple[str, str, dict[str, str]]


class HeadFixture(TypedDict):
    checkpoint_sha256: str
    seed: int
    scores: list[list[float]]
    source_commit: str
    absolute_tolerance: float


class EmbeddingEntry(TypedDict):
    index: int
    embedding: list[float]


class EmbeddingResponse(TypedDict):
    data: list[EmbeddingEntry]


class Arguments(argparse.Namespace):
    device: str
    dtype: str
    cache_dir: str | None
    local_files_only: bool
    heads_only: bool
    repeats: int
    output: Path
    reference_emb_url: str | None
    reference_emb_model: str
    reference_atol: float | None


CASES: list[Case] = [
    (
        "What causes tides on Earth?",
        "Choose the best answer.",
        {
            "moon": "The Moon's gravitational pull.",
            "plants": "Photosynthesis in plants.",
            "shape": "Because the Earth is round.",
        },
    ),
    (
        "I was charged twice for my subscription.",
        "Which team should handle this request?",
        {
            "billing": "Charges, invoices, and refunds",
            "technical": "Bugs and service outages",
        },
    ),
]


def timing_summary(seconds: list[float]) -> dict[str, object]:
    """Keep the first call separate and retain the spread of subsequent calls."""
    if len(seconds) < 2:
        raise ValueError("at least two runs are required")
    warm = seconds[1:]
    return {
        "first_seconds": seconds[0],
        "warm_runs": len(warm),
        "warm_median_seconds": statistics.median(warm),
        "warm_min_seconds": min(warm),
        "warm_max_seconds": max(warm),
    }


def memory_snapshot(device: str) -> dict[str, int | None]:
    """Report process peak RSS and allocator observations, in bytes.

    MPS values are current snapshots, not peaks. CUDA's peak covers allocations
    since process start. These counters overlap and must not be added together.
    """
    result: dict[str, int | None] = {"process_peak_rss_bytes": None}
    if platform.system() in {"Linux", "Darwin"}:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result["process_peak_rss_bytes"] = rss if platform.system() == "Darwin" else rss * 1024
    if device.startswith("cuda"):
        result["cuda_peak_tensor_bytes"] = torch.cuda.max_memory_allocated()
    elif device.split(":", 1)[0] == "mps":
        result["mps_current_tensor_bytes"] = torch.mps.current_allocated_memory()
        result["mps_current_driver_bytes"] = torch.mps.driver_allocated_memory()
    return result


def head_reference(checkpoint: Path, spec: ModelSpec) -> dict[str, object]:
    """Check a portable recorded reference and same-platform FP32 runtime parity."""
    fixture = cast(
        HeadFixture,
        json.loads(
            (Path(__file__).resolve().parents[1] / "tests/fixtures/released_heads.json").read_text()
        ),
    )
    if (
        spec.checkpoint_sha256 != fixture["checkpoint_sha256"]
        or sha256(checkpoint) != fixture["checkpoint_sha256"]
    ):
        raise ValueError("the recorded head reference applies only to the selected release")
    rng = np.random.default_rng(fixture["seed"])
    states = rng.standard_normal((2, 4096)).astype(np.float32)
    actions = rng.standard_normal((5, 4096)).astype(np.float32)
    states /= np.linalg.norm(states, axis=-1, keepdims=True) + 1e-12
    actions /= np.linalg.norm(actions, axis=-1, keepdims=True) + 1e-12
    states_tensor, actions_tensor = torch.from_numpy(states), torch.from_numpy(actions)
    recorded = torch.tensor(fixture["scores"], dtype=torch.float64)
    atol = fixture["absolute_tolerance"]
    # The Linux FP32 snapshot is not portable at 1e-5: upstream itself differs
    # by 2.48e-5 on macOS. FP64 checks that snapshot without widening its tolerance.
    stable = released_scores(checkpoint, states_tensor, actions_tensor, dtype=torch.float64)
    recorded_error = score_error(stable, recorded, atol, "recorded head reference")
    reference = released_scores(checkpoint, states_tensor, actions_tensor, dtype=torch.float32)
    sh, ah, scale = load_heads(checkpoint, spec.hidden_size, "cpu", spec.encoder_id)
    with torch.inference_mode():
        zs = torch.nn.functional.normalize(sh(states_tensor), dim=-1)
        za = torch.nn.functional.normalize(ah(actions_tensor), dim=-1)
        actual = torch.stack([scale * (za @ state) for state in zs])
    runtime_error = score_error(actual, reference, atol, "FP32 runtime head reference")
    return {
        "status": "passed",
        "max_abs_error": runtime_error,
        "runtime_reference_dtype": "float32",
        "recorded_reference_dtype": "float64",
        "recorded_reference_max_abs_error": recorded_error,
        "historical_fp32_max_abs_error": float((actual.double() - recorded).abs().max().item()),
        "source_commit": fixture["source_commit"],
        "atol": atol,
    }


def score_error(actual: torch.Tensor, expected: torch.Tensor, atol: float, label: str) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(f"{label}: score shapes differ")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{label}: non-finite scores")
    difference = float((actual.double() - expected.double()).abs().max().item())
    if difference > atol:
        raise AssertionError(f"{label}: scores differ by {difference} (atol={atol})")
    return difference


def released_scores(
    checkpoint: Path, states: torch.Tensor, actions: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    """Functional reference from pinned upstream heads.py; never calls Kayak heads.

    Apache-2.0, revision 7956937c58ed5839c06ddc4dc6b6b61c3a3e4094.
    The caller checks the release hash; this implements its fixed architecture.
    """
    contents: dict[str, object] = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_config = {
        "model": "Qwen/Qwen3-8B",
        "hidden_size": 4096,
        "projection_dim": 512,
        "width": 1536,
        "depth": 3,
        "activation": "gelu",
        "layernorm": True,
        "residual": False,
    }
    if contents["cfg"] != expected_config:
        raise ValueError("reference requires the pinned released head architecture")
    with torch.inference_mode():
        state_weights = cast(dict[str, torch.Tensor], contents["state_head"])
        action_weights = cast(dict[str, torch.Tensor], contents["action_head"])
        zs = project_reference(states.to(dtype=dtype), state_weights)
        za = project_reference(actions.to(dtype=dtype), action_weights)
        # Keep the upstream FP32 temperature operation in both comparisons.
        scale = torch.as_tensor(contents["logit_scale"], dtype=torch.float32)
        scale_value = float(scale.exp().clamp(max=100.0).item())
        return torch.stack([scale_value * (za @ state) for state in zs])


def project_reference(inputs: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = {name: value.to(dtype=inputs.dtype) for name, value in weights.items()}
    hidden = torch.nn.functional.gelu(
        torch.nn.functional.linear(inputs, weights["inp.weight"], weights["inp.bias"])
    )
    hidden = torch.nn.functional.linear(
        hidden, weights["hidden.0.weight"], weights["hidden.0.bias"]
    )
    hidden = torch.nn.functional.layer_norm(
        hidden,
        (hidden.shape[-1],),
        weights["norms.0.weight"],
        weights["norms.0.bias"],
        eps=1e-5,
    )
    projected = torch.nn.functional.linear(
        torch.nn.functional.gelu(hidden), weights["out.weight"], weights["out.bias"]
    )
    return torch.nn.functional.normalize(projected, dim=-1)


def reference_scores(
    url: str,
    served_name: str,
    state: str,
    question: kayak.Choice,
    checkpoint: Path,
    spec: ModelSpec,
) -> list[float]:
    """Use independent encoder outputs with the already checked projection heads."""
    texts = [f"{state.strip()}\n\n{question.instructions.strip()}", *question.criteria.values()]
    response = httpx.post(
        url,
        timeout=120,
        json={
            "model": served_name,
            "input": texts,
            "encoding_format": "float",
        },
    )
    response.raise_for_status()
    payload = cast(EmbeddingResponse, response.json())
    entries = sorted(payload["data"], key=lambda row: row["index"])
    if [row["index"] for row in entries] != list(range(len(texts))):
        raise ValueError("reference embedding indices do not match the texts")
    vectors = np.array([row["embedding"] for row in entries], dtype=np.float32)
    if vectors.shape != (len(texts), spec.hidden_size) or not np.isfinite(vectors).all():
        raise ValueError("reference embeddings have an invalid shape or value")
    vectors /= np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-12
    sh, ah, scale = load_heads(checkpoint, spec.hidden_size, "cpu", spec.encoder_id)
    with torch.inference_mode():
        zs = torch.nn.functional.normalize(sh(torch.from_numpy(vectors[:1])), dim=-1)
        za = torch.nn.functional.normalize(ah(torch.from_numpy(vectors[1:])), dim=-1)
        return cast(list[float], (scale * (za @ zs[0])).tolist())


def validate_http(model: DecisionModel, case: Case, direct: kayak.DecisionResult) -> bool:
    state, instructions, criteria = case
    app = create_app(lambda: model, api_key="local-validation")
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", limit_concurrency=16))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 180
            while not server.started:
                if not thread.is_alive() or time.monotonic() > deadline:
                    raise RuntimeError("validation service failed to become ready")
                time.sleep(0.05)
            with kayak.Client(
                base_url=f"http://127.0.0.1:{port}", api_key="local-validation"
            ) as client:
                remote = client.decide(
                    state=state,
                    questions={
                        "decision": kayak.Choice(instructions=instructions, criteria=criteria)
                    },
                )
            if remote.model != direct.model or remote.answers != direct.answers:
                raise AssertionError("local and HTTP decisions differ")
            return True
        finally:
            server.should_exit = True
            thread.join(timeout=180)
            if thread.is_alive():
                raise RuntimeError("validation service did not finish draining inference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument(
        "--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto"
    )
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--heads-only", action="store_true")
    parser.add_argument(
        "--repeats", type=int, default=3, help="runs per case, including first call"
    )
    parser.add_argument("--output", type=Path, default=Path("validation/clm.json"))
    parser.add_argument("--reference-emb-url")
    parser.add_argument("--reference-emb-model", default="qwen3-8b")
    parser.add_argument(
        "--reference-atol",
        type=float,
        help="explicit acceptance tolerance for an independent encoder comparison",
    )
    args = parser.parse_args(namespace=Arguments())
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    if args.reference_emb_url and (
        args.reference_atol is None
        or not math.isfinite(args.reference_atol)
        or args.reference_atol < 0
    ):
        parser.error("--reference-emb-url requires an explicit nonnegative --reference-atol")
    report: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "requested_device": args.device,
        "requested_dtype": args.dtype,
        "repeats": args.repeats,
        "versions": {k: version(k) for k in ("kayak", "torch", "transformers")},
        "torch_num_threads": torch.get_num_threads(),
        "full_model": "not_run",
        "encoder_reference": "not_run",
        "http": "not_run",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        spec, checkpoint = resolve(
            DEFAULT_MODEL, cache_dir=args.cache_dir, local_files_only=args.local_files_only
        )
        report["model_spec"] = spec.model_dump()
        report["heads_reference"] = head_reference(checkpoint, spec)
        if args.heads_only:
            return
        started = time.monotonic()
        with kayak.load(
            device=args.device,
            dtype=args.dtype,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        ) as model:
            report["load_seconds"] = time.monotonic() - started
            report["model"] = model.info.model_dump()
            report["memory_after_load"] = memory_snapshot(model.info.device)
            cases: list[dict[str, object]] = []
            report["cases"] = cases
            for state, instructions, criteria in CASES:
                question = kayak.Choice(instructions=instructions, criteria=criteria)
                runs: list[kayak.DecisionResult] = []
                elapsed: list[float] = []
                for _ in range(args.repeats):
                    started = time.monotonic()
                    runs.append(model.decide(state=state, questions={"decision": question}))
                    elapsed.append(time.monotonic() - started)
                scores = [list(r.answers["decision"].scores.values()) for r in runs]
                entry: dict[str, object] = {
                    "state": state,
                    "seconds": elapsed,
                    "timing": timing_summary(elapsed),
                    "memory": memory_snapshot(model.info.device),
                    "scores": scores,
                    "choice": runs[0].answers["decision"].choice,
                    "choices": [r.answers["decision"].choice for r in runs],
                }
                cases.append(entry)
                if any(r.answers != runs[0].answers for r in runs[1:]):
                    raise AssertionError("repeated inference differs; inspect the hardware/runtime")
                if args.reference_emb_url:
                    expected = reference_scores(
                        args.reference_emb_url,
                        args.reference_emb_model,
                        state,
                        question,
                        checkpoint,
                        spec,
                    )
                    error = float(np.max(np.abs(np.array(scores[0]) - expected)))
                    entry["reference_scores"] = expected
                    entry["reference_max_abs_error"] = error
                    entry["reference_choice_agrees"] = int(np.argmax(scores[0])) == int(
                        np.argmax(expected)
                    )
                    assert args.reference_atol is not None  # Validated by the CLI above.
                    if error > args.reference_atol or not entry["reference_choice_agrees"]:
                        raise AssertionError(f"encoder reference disagreement: {entry}")
            try:
                model.decide(state="word " * 3000, questions={"decision": question})
            except kayak.InputError:
                report["overflow_rejected"] = True
            else:
                raise AssertionError("overlong input was accepted")
            report["full_model"] = "passed"
            if args.reference_emb_url:
                report["encoder_reference"] = {
                    "status": "passed",
                    "atol": args.reference_atol,
                    "assumption": "operator pins the reference encoder to the manifest revision",
                }
            report["http"] = "passed" if validate_http(model, CASES[-1], runs[0]) else "failed"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Validation report: {args.output}")


if __name__ == "__main__":
    main()
