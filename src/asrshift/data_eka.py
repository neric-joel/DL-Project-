"""Eka Medical ASR Evaluation Dataset (English subset): manifest and leakage-safe split.

Eka ``session_id`` is almost one per clip (3,524 sessions for 3,619 clips) while 56 speakers each
record dozens of sessions, and 228 prompt texts are read by more than one speaker. A split by
session alone would put the same voices and sentences on both sides, so we split by *speaker*
(clips without a speaker, the conversation recordings, are grouped by session), stratified by
recording context, and then drop the few clips whose session or prompt text would cross splits.
The result is disjoint in speakers, sessions and texts.
"""

from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from asrshift.paths import RAW_DIR

META_COLUMNS = [
    "file_name", "duration", "text", "session_id", "speaker",
    "type_concept", "recording_context", "medical_entities", "md5_text",
]
SPLITS = ("calibration", "validation", "test")


def shard_paths(root: Path = RAW_DIR) -> list[str]:
    paths = sorted(glob.glob(str(root / "eka" / "en" / "test-*.parquet")))
    if not paths:
        raise FileNotFoundError(f"no Eka shards under {root / 'eka' / 'en'}; run `asrshift download`")
    return paths


def load_metadata(root: Path = RAW_DIR) -> pd.DataFrame:
    frames = []
    for shard in shard_paths(root):
        df = pd.read_parquet(shard, columns=META_COLUMNS)
        df["shard"] = Path(shard).name
        df["row"] = np.arange(len(df))
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["unit_id"] = "eka_" + df["file_name"].map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:12])
    if df["unit_id"].duplicated().any():
        raise ValueError("unit_id collision in Eka metadata")
    # The split is computed on the raw key; some speaker IDs are e-mail addresses, so only a hash of
    # them (group_id) is written to any output.
    df["group_key"] = np.where(df["speaker"].notna(), "spk:" + df["speaker"].astype(str), "ses:" + df["session_id"])
    spk_hash = df["speaker"].map(lambda s: hashlib.sha1(s.encode()).hexdigest()[:10] if isinstance(s, str) else None)
    df["group_id"] = np.where(df["speaker"].notna(), "spk:" + spk_hash.astype(str), "ses:" + df["session_id"])
    return df


def _assign_groups(sizes: pd.Series, targets: dict[str, float], seed: int) -> dict[str, str]:
    """Greedy balanced assignment: biggest groups first, each to the split furthest below target."""
    rng = np.random.default_rng(seed)
    order = sizes.sample(frac=1.0, random_state=int(rng.integers(1 << 31))).sort_values(ascending=False, kind="stable")
    total = float(sizes.sum())
    filled = {s: 0.0 for s in targets}
    out = {}
    for gid, n in order.items():
        deficit = {s: targets[s] * total - filled[s] for s in targets}
        best = max(deficit, key=lambda s: (deficit[s] / max(targets[s], 1e-9), -list(targets).index(s)))
        out[gid] = best
        filled[best] += n
    return out


def split(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    """Speaker-disjoint split stratified by recording context, then session/text de-duplication."""
    df = df.copy()
    key = "group_key" if "group_key" in df else "group_id"
    assignment: dict[str, str] = {}
    for ctx, part in df.groupby("recording_context", sort=True):
        sizes = part.groupby(key).size()
        assignment.update(_assign_groups(sizes, fractions, seed + sum(map(ord, ctx))))
    df["split"] = df[key].map(assignment)

    # A session or prompt text seen in more than one split is kept only in the first split of
    # this priority order; the test split is kept whole so the in-domain evaluation is not thinned.
    priority = {"test": 0, "calibration": 1, "validation": 2}
    df["dropped_reason"] = ""
    for key in ("session_id", "md5_text"):
        best = df[df["dropped_reason"] == ""].groupby(key)["split"].agg(lambda s: min(s, key=priority.get))
        clash = (df["dropped_reason"] == "") & (df["split"] != df[key].map(best))
        df.loc[clash, "dropped_reason"] = f"{key}_crosses_split"
    return df


def parse_entities(raw: str) -> list[dict]:
    """``[[text, type, [[start, end], ...]], ...]`` -> list of {text, type, start, end}."""
    out = []
    for text, etype, spans in json.loads(raw or "[]"):
        for start, end in spans or [[None, None]]:
            out.append({"text": text, "type": etype, "start": start, "end": end})
    out.sort(key=lambda e: (e["start"] is None, e["start"] or 0))
    return out


def build_manifest(fractions: dict[str, float], seed: int, root: Path = RAW_DIR) -> pd.DataFrame:
    df = split(load_metadata(root), fractions, seed)
    df["dataset"] = "eka"
    df["subset"] = df["recording_context"]
    df["reference"] = df["text"]
    df["start"] = 0.0
    df["end"] = df["duration"]
    df["audio_source"] = df["shard"] + "#" + df["row"].astype(str)
    df["n_entities"] = df["medical_entities"].map(lambda r: len(parse_entities(r)))
    cols = [
        "unit_id", "dataset", "subset", "split", "group_id", "group_key", "session_id", "speaker", "duration",
        "start", "end", "audio_source", "reference", "type_concept", "recording_context",
        "medical_entities", "n_entities", "md5_text", "dropped_reason",
    ]
    return df[cols].reset_index(drop=True)
