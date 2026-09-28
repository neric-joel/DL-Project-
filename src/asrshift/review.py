"""Apply the fitted review policies to a new transcript.

``asrshift evaluate`` exports the fitted policies to ``results/models/policies_<model>.json``:
the standardised logistic-regression coefficients of P3, the Platt parameters of P4 for each PriMock
unit type, and the thresholds of each operating point. Plain JSON, so no pickles are loaded.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from asrshift.features import unit_features
from asrshift.paths import RESULTS_DIR


def policy_path(model: str = "small") -> Path:
    return RESULTS_DIR / "models" / f"policies_{model.replace('/', '_')}.json"


def export(path: Path, features: list[str], p3_pipeline, platt_by_type: dict, thresholds: dict, label: str) -> None:
    scaler, lr = p3_pipeline[0], p3_pipeline[-1]
    doc = {
        "label": label,
        "features": features,
        "P3": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
               "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])},
        "P4": {k: {"a": v.a, "b": v.b} for k, v in platt_by_type.items()},
        "thresholds": thresholds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def probabilities(feats: dict, domain: str = "window", model: str = "small") -> dict:
    doc = _load(str(policy_path(model)))
    x = np.array([feats[f] for f in doc["features"]], float)
    p3 = doc["P3"]
    z = float(((x - np.array(p3["mean"])) / np.array(p3["scale"])) @ np.array(p3["coef"]) + p3["intercept"])
    out = {"P1": 1.0 - feats["conf"], "P3": float(1 / (1 + np.exp(-z)))}
    key = "turn" if domain.startswith("turn") else "window"
    if key in doc["P4"] and domain != "eka":
        a, b = doc["P4"][key]["a"], doc["P4"][key]["b"]
        pz = np.log(np.clip(out["P3"], 1e-6, 1 - 1e-6) / np.clip(1 - out["P3"], 1e-6, 1))
        out["P4"] = float(1 / (1 + np.exp(-(a * pz + b))))
    return out


def assess(rec: dict, domain: str = "window", model: str = "small") -> dict:
    """Risk under each policy and the decision at each exported operating point."""
    doc = _load(str(policy_path(model)))
    feats = unit_features(rec)
    probs = probabilities(feats, domain, model)
    key = "turn" if domain.startswith("turn") else "window"
    decisions = {}
    for pol, p in probs.items():
        th = doc["thresholds"].get(pol, {})
        th = th.get(key, th) if pol == "P4" else th
        decisions[pol] = {point: ("review" if p >= t else "accept") for point, t in th.items() if isinstance(t, (int, float))}
    return {"confidence": feats["conf"], "risk": probs, "decisions": decisions}
