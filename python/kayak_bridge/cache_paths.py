from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = REPO_ROOT / ".cache"
HF_HOME = CACHE_ROOT / "huggingface"
TORCH_EXTENSIONS_DIR = CACHE_ROOT / "torch_extensions"
IR_DATASETS_HOME = CACHE_ROOT / "ir_datasets"


def configure_local_caches() -> None:
    HF_HOME.mkdir(parents=True, exist_ok=True)
    (HF_HOME / "hub").mkdir(parents=True, exist_ok=True)
    TORCH_EXTENSIONS_DIR.mkdir(parents=True, exist_ok=True)
    IR_DATASETS_HOME.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(HF_HOME / "hub"))
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(TORCH_EXTENSIONS_DIR))
    os.environ.setdefault("IR_DATASETS_HOME", str(IR_DATASETS_HOME))
