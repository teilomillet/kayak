from __future__ import annotations

import csv
import math
import pathlib
from collections import defaultdict


DEFAULT_PATH = pathlib.Path(".cache/kayak/profile_exact_cpu_usl.tsv")


def usl_capacity(n: float, alpha: float, beta: float) -> float:
    return n / (1.0 + alpha * (n - 1.0) + beta * n * (n - 1.0))


def fit_alpha_beta(rows: list[dict[str, float]]) -> tuple[float, float]:
    s_aa = 0.0
    s_ab = 0.0
    s_bb = 0.0
    s_ay = 0.0
    s_by = 0.0

    for row in rows:
        n = row["work_items"]
        if n <= 1.0:
            continue

        capacity = row["relative_capacity"]
        a = n - 1.0
        b = n * (n - 1.0)
        y = (n / capacity) - 1.0

        s_aa += a * a
        s_ab += a * b
        s_bb += b * b
        s_ay += a * y
        s_by += b * y

    det = (s_aa * s_bb) - (s_ab * s_ab)
    if det == 0.0:
        raise ValueError("USL fit is singular; need at least two nontrivial load points")

    alpha = ((s_ay * s_bb) - (s_by * s_ab)) / det
    beta = ((s_by * s_aa) - (s_ay * s_ab)) / det
    return alpha, beta


def r_squared(rows: list[dict[str, float]], alpha: float, beta: float) -> float:
    ys = [row["relative_capacity"] for row in rows]
    mean_y = sum(ys) / len(ys)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum(
        (row["relative_capacity"] - usl_capacity(row["work_items"], alpha, beta)) ** 2
        for row in rows
    )
    if ss_tot == 0.0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def load_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    path = DEFAULT_PATH
    if not path.exists():
        raise SystemExit(
            f"missing USL dataset: {path}. Run `pixi run bench_profile_cpu_usl` first."
        )
    rows = load_rows(path)

    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    x1_by_group: dict[tuple[str, str], float] = {}

    for raw in rows:
        key = (raw["benchmark_name"], raw["workload_name"])
        row = {
            "work_items": float(raw["work_items"]),
            "mean_seconds": float(raw["mean_seconds"]),
            "throughput_per_second": float(raw["throughput_per_second"]),
            "relative_capacity": float(raw["relative_capacity"]),
        }
        grouped[key].append(row)
        if raw["work_items"] == "1":
            x1_by_group[key] = row["throughput_per_second"]

    print(f"USL fit source: {path}")
    print("")

    for key in sorted(grouped):
        benchmark_name, workload_name = key
        series = sorted(grouped[key], key=lambda row: row["work_items"])
        alpha, beta = fit_alpha_beta(series)
        fit_r2 = r_squared(series, alpha, beta)
        x1 = x1_by_group[key]

        nmax: str
        if beta > 0.0 and (1.0 - alpha) > 0.0:
            nmax = f"{math.sqrt((1.0 - alpha) / beta):.3f}"
        else:
            nmax = "none"

        print(f"{benchmark_name} / {workload_name}")
        print(f"  X(1)={x1:.3f} ops/s")
        print(f"  alpha={alpha:.6f}")
        print(f"  beta={beta:.6f}")
        print(f"  R^2={fit_r2:.6f}")
        print(f"  Nmax={nmax}")
        print("")


if __name__ == "__main__":
    main()
