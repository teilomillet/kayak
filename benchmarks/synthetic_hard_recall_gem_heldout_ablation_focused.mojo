from kayak.benchmarks import (
    focused_gem_heldout_ablation_run_options,
    write_synthetic_hard_recall_gem_heldout_ablation,
)


def main() raises:
    write_synthetic_hard_recall_gem_heldout_ablation(
        focused_gem_heldout_ablation_run_options()
    )
