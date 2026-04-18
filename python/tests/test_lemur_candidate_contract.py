from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import kayak

from kayak_bridge.lemur_candidate_contract import (
    LemurCandidateContractBundle,
    benchmark_reference_lemur_contract,
)


def test_contract_bundle_serializes_paths() -> None:
    bundle = LemurCandidateContractBundle(
        contract_name="reference_lemur_candidate_contract",
        exact_path="/tmp/exact.json",
        sweep_path="/tmp/sweep.json",
        bundle_path="/tmp/bundle.json",
    )

    payload = bundle.to_json_ready()

    assert payload["contract_name"] == "reference_lemur_candidate_contract"
    assert payload["exact_path"] == "/tmp/exact.json"


def test_reference_lemur_contract_writes_expected_artifacts() -> None:
    task = {
        "dataset_id": "dataset://tiny",
        "model_name": "unit-test-model",
        "family": "tiny_family",
        "slice_name": "tiny_slice",
        "primary_metric": "ndcg",
        "k": 1,
        "nominal_query_vector_count": 2,
        "nominal_document_vector_count": 2,
        "vector_dim": 2,
        "documents": [
            {
                "doc_id": "doc-a",
                "text": "alpha beta",
                "vector_count": 2,
                "vectors": [[1.0, 0.0], [0.0, 1.0]],
            },
            {
                "doc_id": "doc-b",
                "text": "alpha alpha",
                "vector_count": 2,
                "vectors": [[1.0, 0.0], [1.0, 0.0]],
            },
        ],
        "queries": [
            {
                "query_id": "q-1",
                "text": "alpha beta",
                "relevant_doc_ids": ["doc-a"],
                "vector_count": 2,
                "vectors": [[1.0, 0.0], [0.0, 1.0]],
            }
        ],
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_root = Path(tmp_dir)
        result = benchmark_reference_lemur_contract(
            task=task,
            output_root=output_root,
            artifact_prefix="tiny",
            latent_dims=[2],
            candidate_ks=[1, 2],
            feature_weights=np.eye(2, dtype=np.float32),
            landmark_vectors=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            activation="relu",
            query_divisor=1.0,
            apply_layer_norm=False,
            exact_backend=kayak.NUMPY_REFERENCE_BACKEND,
            rerank_backend=kayak.NUMPY_REFERENCE_BACKEND,
            warmup_iterations=0,
            measurement_iterations=1,
        )

        contract_path = Path(result["bundle_path"])
        exact_path = output_root / "tiny_kayak_exact_benchmark.json"
        sweep_path = output_root / "tiny_lemur_candidate_sweep.json"
        bundle_path = output_root / "tiny_lemur_candidate_bundle.json"

        assert contract_path.exists()
        assert exact_path.exists()
        assert sweep_path.exists()
        assert bundle_path.exists()

        with contract_path.open("r", encoding="utf-8") as handle:
            contract_payload = json.load(handle)
        with bundle_path.open("r", encoding="utf-8") as handle:
            bundle_payload = json.load(handle)

        assert contract_payload["contract_name"] == "reference_lemur_candidate_contract"
        assert bundle_payload["candidate_count"] == 2
        assert bundle_payload["best_quality_candidate_name"] != ""
