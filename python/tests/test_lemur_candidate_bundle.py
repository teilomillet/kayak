from __future__ import annotations

from kayak_bridge.lemur_candidate_bundle import build_lemur_candidate_bundle


def test_build_lemur_candidate_bundle_picks_quality_and_speed_winners() -> None:
    bundle = build_lemur_candidate_bundle(
        exact_path="/tmp/kayak_exact.json",
        sweep_path="/tmp/lemur_sweep.json",
        sweep_summary={
            "dataset_id": "dataset://a",
            "family": "mock",
            "slice_name": "slice",
            "primary_metric": "ndcg",
            "exact_primary_value": 0.80,
            "exact_mean_search_seconds": 0.01,
            "rows": [
                {
                    "latent_dim": 16,
                    "candidate_k": 10,
                    "landmark_count": 32,
                    "fit_seconds": 0.2,
                    "primary_value": 0.78,
                    "mean_search_seconds": 0.004,
                    "mean_exact_topk_shortlist_recall": 0.8,
                    "primary_ratio_vs_exact": 0.975,
                    "mean_search_seconds_ratio_vs_exact": 0.4,
                },
                {
                    "latent_dim": 32,
                    "candidate_k": 20,
                    "landmark_count": 32,
                    "fit_seconds": 0.3,
                    "primary_value": 0.80,
                    "mean_search_seconds": 0.006,
                    "mean_exact_topk_shortlist_recall": 1.0,
                    "primary_ratio_vs_exact": 1.0,
                    "mean_search_seconds_ratio_vs_exact": 0.6,
                },
                {
                    "latent_dim": 64,
                    "candidate_k": 30,
                    "landmark_count": 64,
                    "fit_seconds": 0.5,
                    "primary_value": 0.82,
                    "mean_search_seconds": 0.009,
                    "mean_exact_topk_shortlist_recall": 1.0,
                    "primary_ratio_vs_exact": 1.025,
                    "mean_search_seconds_ratio_vs_exact": 0.9,
                },
            ],
        },
    )

    assert bundle.candidate_count == 3
    assert bundle.best_quality_candidate_name == "lemur_latent64_k30_landmarks64"
    assert (
        bundle.fastest_full_shortlist_recall_candidate_name
        == "lemur_latent32_k20_landmarks32"
    )
    assert (
        bundle.fastest_no_primary_regression_candidate_name
        == "lemur_latent32_k20_landmarks32"
    )
    assert bundle.rows[1].primary_ratio_vs_exact == 1.0
