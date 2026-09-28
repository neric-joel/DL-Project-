"""Load the study configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from asrshift.paths import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)
