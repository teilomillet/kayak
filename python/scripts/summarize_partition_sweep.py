from __future__ import annotations

import csv
import pathlib
from collections import defaultdict


DEFAULT_PATH = pathlib.Path(".cache/kayak/profile_exact_cpu_partition_sweep.tsv")


def load_rows(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main() -> None:
    path = DEFAULT_PATH
    if not path.exists():
        raise SystemExit(
            f"missing partition sweep dataset: {path}. Run `pixi run bench_profile_cpu_partitions` first."
        )

    rows = load_rows(path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["workload_name"]].append(row)

    print(f"Partition sweep source: {path}")
    print("")

    for workload_name in sorted(grouped):
        series = sorted(grouped[workload_name], key=lambda row: int(row["work_items"]))
        best = min(series, key=lambda row: float(row["mean_seconds"]))
        print(workload_name)
        print(
            f"  best_work_items={best['work_items']} "
            f"mean_seconds={float(best['mean_seconds']):.9f} "
            f"throughput={float(best['throughput_per_second']):.3f} ops/s"
        )
        print("  candidates:")
        for row in series:
            print(
                f"    N={int(row['work_items']):>2} "
                f"mean={float(row['mean_seconds']):.9f} "
                f"throughput={float(row['throughput_per_second']):.3f}"
            )
        print("")


if __name__ == "__main__":
    main()
