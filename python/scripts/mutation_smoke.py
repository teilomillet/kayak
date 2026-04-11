from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


MUTANTS = [
    {
        "name": "maxsim_subtracts_best_similarity",
        "path": REPO_ROOT / "kayak/scoring/maxsim.mojo",
        "before": "        total += best_similarity\n",
        "after": "        total -= best_similarity\n",
        "command": ["pixi", "run", "test_battle"],
    },
    {
        "name": "maxsim_prefers_lower_similarity",
        "path": REPO_ROOT / "kayak/scoring/maxsim.mojo",
        "before": "            if similarity > best_similarity:\n",
        "after": "            if similarity < best_similarity:\n",
        "command": ["pixi", "run", "test_battle"],
    },
    {
        "name": "topk_reverses_equal_score_stability",
        "path": REPO_ROOT / "kayak/search/topk.mojo",
        "before": "    while insert_at < len(hits) and hits[insert_at].score >= hit.score:\n",
        "after": "    while insert_at < len(hits) and hits[insert_at].score > hit.score:\n",
        "command": ["pixi", "run", "test_battle"],
    },
    {
        "name": "topk_rejects_zero_k",
        "path": REPO_ROOT / "kayak/search/topk.mojo",
        "before": "    if k < 0:\n",
        "after": "    if k <= 0:\n",
        "command": ["pixi", "run", "test_battle"],
    },
    {
        "name": "pack_documents_accepts_mismatched_dimensions",
        "path": REPO_ROOT / "kayak/index/builder.mojo",
        "before": "        if document.vector_dim != vector_dim:\n",
        "after": "        if document.vector_dim == vector_dim:\n",
        "command": ["pixi", "run", "test_battle"],
    },
    {
        "name": "storage_accepts_scalar_mismatch",
        "path": REPO_ROOT / "kayak/storage/manifest.mojo",
        "before": "    if stored_vector_scalar_name != VECTOR_SCALAR_NAME:\n",
        "after": "    if stored_vector_scalar_name == VECTOR_SCALAR_NAME:\n",
        "command": ["pixi", "run", "test_storage_invariants"],
    },
    {
        "name": "recall_uses_hits_count_denominator",
        "path": REPO_ROOT / "kayak/eval/metrics.mojo",
        "before": "    return MetricScalar(found_count) / MetricScalar(len(relevant_doc_ids))\n",
        "after": "    return MetricScalar(found_count) / MetricScalar(len(hits))\n",
        "command": ["pixi", "run", "test_eval_battle"],
    },
    {
        "name": "reciprocal_rank_shifts_rank_by_one",
        "path": REPO_ROOT / "kayak/eval/metrics.mojo",
        "before": "            return MetricScalar(1.0) / MetricScalar(index + 1)\n",
        "after": "            return MetricScalar(1.0) / MetricScalar(index + 2)\n",
        "command": ["pixi", "run", "test_eval_battle"],
    },
    {
        "name": "evaluate_task_accepts_zero_queries",
        "path": REPO_ROOT / "kayak/eval/evaluate.mojo",
        "before": "    if len(task.queries) == 0:\n",
        "after": "    if len(task.queries) < 0:\n",
        "command": ["pixi", "run", "test_eval_battle"],
    },
    {
        "name": "choose_primary_value_maps_recall_to_success",
        "path": REPO_ROOT / "kayak/eval/evaluate.mojo",
        "before": "    if primary_metric == \"recall\":\n        return mean_recall_at_k\n",
        "after": "    if primary_metric == \"recall\":\n        return success_rate_at_k\n",
        "command": ["pixi", "run", "test_eval_battle"],
    },
]


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def tail(text: str, line_count: int = 12) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-line_count:])


def apply_mutant(mutant: dict[str, object]) -> tuple[bool, str]:
    path = Path(mutant["path"])
    before = str(mutant["before"])
    after = str(mutant["after"])
    command = [str(part) for part in mutant["command"]]

    original = path.read_text()
    occurrences = original.count(before)
    if occurrences != 1:
        return False, f"expected exactly one replacement target, found {occurrences}"

    path.write_text(original.replace(before, after, 1))

    try:
        result = run_command(command)
    finally:
        path.write_text(original)

    output = "\n".join(
        chunk for chunk in [result.stdout.strip(), result.stderr.strip()] if chunk
    )
    if result.returncode == 0:
        return False, tail(output)

    return True, tail(output)


def main() -> int:
    print("curated mutation smoke for kayak")
    print(f"repo: {REPO_ROOT}")

    killed = 0
    survived = 0

    for mutant in MUTANTS:
        name = str(mutant["name"])
        print(f"\n==> {name}")
        mutant_killed, detail = apply_mutant(mutant)

        if mutant_killed:
            killed += 1
            print("status: KILLED")
        else:
            survived += 1
            print("status: SURVIVED")

        if detail:
            print(detail)

    print("\nsummary")
    print(f"killed: {killed}")
    print(f"survived: {survived}")

    return 0 if survived == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
