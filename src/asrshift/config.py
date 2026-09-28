"""Load the study configuration.

A config may start with ``extends: <other.yaml>`` (relative to its own folder); it is then merged
over that file, key by key, so a variant only lists what it changes.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from asrshift.paths import REPO_ROOT

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    parent = cfg.pop("extends", None)
    return _merge(load_config(path.parent / parent), cfg) if parent else cfg
