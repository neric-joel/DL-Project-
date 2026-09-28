"""Build the unit manifest for both datasets.

Writes ``data/interim/manifest.parquet`` (everything needed to load audio and score a unit) and
``results/splits/*.csv`` (which unit is in which split; small, committed for transparency).
"""

from __future__ import annotations

import pandas as pd

from asrshift import data_eka, data_primock
from asrshift.paths import INTERIM_DIR, RESULTS_DIR

MANIFEST = INTERIM_DIR / "manifest.parquet"


def build(cfg: dict) -> pd.DataFrame:
    eka = data_eka.build_manifest(cfg["eka"]["fractions"], cfg["seed"])
    pm_cfg = cfg["primock57"]
    pm = data_primock.build_manifest(
        max_window_s=pm_cfg["max_window_s"], turn_pad_s=pm_cfg["turn_pad_s"],
        window_pad_s=pm_cfg["window_pad_s"], min_turn_s=pm_cfg["min_turn_s"],
        recal_consultations=pm_cfg["recalibration_consultations"],
    )
    df = pd.concat([eka, pm], ignore_index=True, sort=False)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MANIFEST, index=False)

    out = RESULTS_DIR / "splits"
    out.mkdir(parents=True, exist_ok=True)
    eka[["unit_id", "split", "dropped_reason", "group_id", "recording_context", "duration"]].to_csv(
        out / "eka_units.csv", index=False)
    pm[["unit_id", "split", "subset", "consultation", "start", "end", "n_speaker_turns"]].round(3).to_csv(
        out / "primock57_units.csv", index=False)
    return df


def load() -> pd.DataFrame:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} missing; run `asrshift prepare`")
    return pd.read_parquet(MANIFEST)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    kept = df[df["dropped_reason"].fillna("") == ""]
    return (kept.groupby(["dataset", "subset", "split"])
            .agg(units=("unit_id", "size"), groups=("group_id", "nunique"), hours=("duration", lambda s: s.sum() / 3600))
            .round(3))
