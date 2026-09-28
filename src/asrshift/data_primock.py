"""PriMock57: utterance parsing, the two target unit types, and the consultation split.

Two unit types are built from the utterance-level TextGrids:

* **turns**: one utterance, cut from that speaker's own channel. Conversational speech, but one
  speaker and usually a few seconds long, so close to Eka in shape.
* **windows**: the doctor and patient channels mixed into one track, cut into stretches of at most
  ``max_window_s`` seconds. Cuts are only placed in gaps where nobody is speaking, so a window holds
  several speaker turns, the pauses between them, and any overlapping speech. This is the unit a
  chunked clinical scribe would transcribe, and the main target domain of the study.

All units from one consultation stay in the same split.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from asrshift.paths import RAW_DIR
from asrshift.textnorm import has_unintelligible, reference_tokens

_INTERVAL = re.compile(
    r"intervals\s*\[\d+\]:\s*xmin\s*=\s*([\d.eE+-]+)\s*xmax\s*=\s*([\d.eE+-]+)\s*text\s*=\s*\"((?:[^\"]|\"\")*)\"",
    re.S,
)
_NAME = re.compile(r"(day\d+_consultation\d+)_(doctor|patient)\.TextGrid$")


def parse_textgrid(path: Path) -> list[tuple[float, float, str]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    return [(float(a), float(b), t.replace('""', '"').strip()) for a, b, t in _INTERVAL.findall(raw)]


def load_utterances(root: Path = RAW_DIR) -> pd.DataFrame:
    rows = []
    tdir = root / "primock57" / "transcripts"
    paths = sorted(tdir.glob("*.TextGrid"))
    if not paths:
        raise FileNotFoundError(f"no TextGrids under {tdir}; run `asrshift download`")
    for path in paths:
        m = _NAME.search(path.name)
        if not m:
            continue
        consultation, role = m.groups()
        for k, (start, end, text) in enumerate(parse_textgrid(path)):
            if text:
                rows.append({"consultation": consultation, "role": role, "idx": k,
                             "start": start, "end": end, "text": text})
    df = pd.DataFrame(rows).sort_values(["consultation", "start", "role"]).reset_index(drop=True)
    df["n_ref_words"] = df["text"].map(lambda t: sum(1 for w in reference_tokens(t) if w != "<*>"))
    df["has_unin"] = df["text"].map(has_unintelligible)
    return df


def channel_duration(consultation: str, role: str, root: Path = RAW_DIR) -> float:
    import soundfile as sf

    return sf.info(str(root / "primock57" / "audio" / f"{consultation}_{role}.wav")).duration


def build_turns(utts: pd.DataFrame, pad_s: float, min_dur_s: float, durations: dict) -> pd.DataFrame:
    rows = []
    for u in utts.itertuples(index=False):
        if u.n_ref_words == 0 or (u.end - u.start) < min_dur_s:
            continue
        dur_total = durations[(u.consultation, u.role)]
        start, end = max(0.0, u.start - pad_s), min(dur_total, u.end + pad_s)
        rows.append({
            "unit_id": f"pm_turn_{u.consultation}_{u.role}_{u.idx:03d}",
            "subset": f"turn_{u.role}",
            "group_id": u.consultation,
            "consultation": u.consultation,
            "start": start, "end": end, "duration": end - start,
            "audio_source": f"{u.consultation}:{u.role}",
            "reference": u.text,
            "n_utterances": 1, "n_speaker_turns": 1, "roles": u.role,
            "speech_s": u.end - u.start, "overlap_s": 0.0, "has_unin": u.has_unin,
        })
    return pd.DataFrame(rows)


def _blocks(utts: pd.DataFrame) -> list[dict]:
    """Merge utterances (both speakers) into blocks that overlap in time; cuts go between blocks."""
    blocks: list[dict] = []
    for u in utts.sort_values("start").itertuples(index=False):
        if blocks and u.start < blocks[-1]["end"]:
            b = blocks[-1]
            b["end"] = max(b["end"], u.end)
            b["utts"].append(u)
        else:
            blocks.append({"start": u.start, "end": u.end, "utts": [u]})
    return blocks


def build_windows(utts: pd.DataFrame, max_window_s: float, pad_s: float, durations: dict) -> pd.DataFrame:
    rows = []
    for consultation, part in utts.groupby("consultation", sort=True):
        total = min(durations[(consultation, "doctor")], durations[(consultation, "patient")])
        blocks = _blocks(part)
        windows: list[list[dict]] = []
        for b in blocks:
            if windows and b["end"] - windows[-1][0]["start"] <= max_window_s:
                windows[-1].append(b)
            else:
                windows.append([b])
        bounds = [(w[0]["start"], w[-1]["end"]) for w in windows]
        for k, w in enumerate(windows):
            s0, e0 = bounds[k]
            prev_end = bounds[k - 1][1] if k else 0.0
            next_start = bounds[k + 1][0] if k + 1 < len(bounds) else total
            # Padding never reaches into a neighbouring window and never pushes a window past
            # max_window_s, so Whisper decodes it in one pass instead of a sub-second leftover.
            room = max(0.0, (max_window_s - (e0 - s0)) / 2)
            start = max(0.0, s0 - min(pad_s, (s0 - prev_end) / 2, room))
            end = min(total, e0 + min(pad_s, (next_start - e0) / 2, room))
            us = sorted((u for b in w for u in b["utts"]), key=lambda u: (u.start, u.role))
            # one utterance per line: the scorer normalises each line on its own, so numbers from
            # different utterances are never merged
            reference = "\n".join(u.text for u in us)
            if sum(u.n_ref_words for u in us) == 0:
                continue
            roles = [u.role for u in us]
            turns = 1 + sum(1 for a, b in zip(roles, roles[1:]) if a != b)
            speech = sum(u.end - u.start for u in us)
            overlap = max(0.0, speech - sum(b["end"] - b["start"] for b in w))
            rows.append({
                "unit_id": f"pm_win_{consultation}_{k:03d}",
                "subset": "window",
                "group_id": consultation,
                "consultation": consultation,
                "start": start, "end": end, "duration": end - start,
                "audio_source": f"{consultation}:mix",
                "reference": reference,
                "n_utterances": len(us), "n_speaker_turns": turns,
                "roles": "+".join(sorted(set(roles))),
                "speech_s": speech, "overlap_s": overlap,
                "has_unin": any(u.has_unin for u in us),
            })
    return pd.DataFrame(rows)


def split_consultations(consultations: list[str], n_recal_per_day: int, seed: int) -> dict[str, str]:
    """Recalibration set: ``n_recal_per_day`` consultations drawn from each recording day."""
    rng = np.random.default_rng(seed)
    out = {c: "test" for c in consultations}
    by_day: dict[str, list[str]] = {}
    for c in sorted(consultations):
        by_day.setdefault(c.split("_")[0], []).append(c)
    for day in sorted(by_day):
        for c in rng.choice(by_day[day], size=min(n_recal_per_day, len(by_day[day])), replace=False):
            out[str(c)] = "recalibration"
    return out


def build_manifest(max_window_s: float, turn_pad_s: float, window_pad_s: float, min_turn_s: float,
                   recal_consultations: list[str], root: Path = RAW_DIR) -> pd.DataFrame:
    utts = load_utterances(root)
    durations = {(c, r): channel_duration(c, r, root) for c in utts["consultation"].unique() for r in ("doctor", "patient")}
    turns = build_turns(utts, turn_pad_s, min_turn_s, durations)
    windows = build_windows(utts, max_window_s, window_pad_s, durations)
    df = pd.concat([windows, turns], ignore_index=True)
    df["dataset"] = "primock57"
    recal = set(recal_consultations)
    df["split"] = np.where(df["consultation"].isin(recal), "recalibration", "test")
    df["dropped_reason"] = ""
    return df
