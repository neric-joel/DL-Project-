"""Filesystem layout. Everything large lives under data/ and is ignored by git."""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("ASRSHIFT_DATA", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
AUDIO_CACHE_DIR = DATA_DIR / "audio"
MODEL_CACHE_DIR = DATA_DIR / "models"
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
