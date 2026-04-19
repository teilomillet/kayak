from kayak.benchmarks import (
    adaptive_probe_gem_heldout_ablation_run_options,
    write_synthetic_hard_recall_gem_heldout_ablation,
)


def main() raises:
    write_synthetic_hard_recall_gem_heldout_ablation(
        adaptive_probe_gem_heldout_ablation_run_options()
    )
