"""Speech-energy features from PriMock57's two channels, aimed at words Whisper drops in overlapping speech.

Each consultation has one recording per speaker. Per 10 ms hop we keep the RMS energy (dB, 25 ms frames)
of each channel and whether that channel carries speech: above its own noise floor (the 20th percentile of
its frame energies) by ``MARGIN_DB``. A hop is *overlap* when both channels carry speech.

For a turn, "own" is the speaker's channel and "other" the other one; for a window (both channels mixed)
"own" is either channel and "other" is overlap. Word and gap times in the committed results are spread
evenly over each unit (there are no word timestamps), so every feature is a share over a tolerant
window around the item. Eka clips have one channel: every feature is NaN there and ``two_ch`` is 0.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from asrshift.paths import INTERIM_DIR, RAW_DIR

SR = 16000
FRAME_S, HOP_S = 0.025, 0.010
FLOOR_PCT, MARGIN_DB = 20, 12.0
WORD_PAD_S, GAP_PAD_S = 0.3, 0.6
CACHE_DIR = INTERIM_DIR / "audio_features"

WORD_FEATURES = ["a_ov", "a_other", "a_own", "u_ov", "u_own_s", "u_wps", "two_ch"]
GAP_FEATURES = ["g_ov", "g_other", "g_own", "u_ov", "u_own_s", "u_wps", "two_ch"]
ROLES = ("doctor", "patient")


def frame_db(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """RMS energy in dB of 25 ms frames every 10 ms."""
    frame, hop = int(FRAME_S * sr), int(HOP_S * sr)
    if len(x) < frame:
        x = np.pad(x, (0, frame - len(x)))
    n = 1 + (len(x) - frame) // hop
    sq = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(n) * hop
    rms = np.sqrt(np.maximum(sq[starts + frame] - sq[starts], 0.0) / frame)
    return 20 * np.log10(np.maximum(rms, 1e-6))


def speech_mask(db: np.ndarray) -> np.ndarray:
    """Speech = more than ``MARGIN_DB`` above the channel's own noise floor."""
    return db > np.percentile(db, FLOOR_PCT) + MARGIN_DB


class Track:
    """Per-hop speech masks of a two-channel recording, with O(1) shares over any time span."""

    def __init__(self, doctor: np.ndarray, patient: np.ndarray):
        n = min(len(doctor), len(patient))
        self.sp = {"doctor": doctor[:n].astype(bool), "patient": patient[:n].astype(bool)}
        self.sp["any"] = self.sp["doctor"] | self.sp["patient"]
        self.sp["both"] = self.sp["doctor"] & self.sp["patient"]
        self.n = n
        self._cum = {k: np.concatenate([[0], np.cumsum(v)]) for k, v in self.sp.items()}

    @classmethod
    def from_audio(cls, doctor: np.ndarray, patient: np.ndarray, sr: int = SR) -> "Track":
        return cls(speech_mask(frame_db(doctor, sr)), speech_mask(frame_db(patient, sr)))

    def share(self, key: str, a, b) -> np.ndarray:
        """Share of hops in [a, b) seconds where ``key`` holds (vectorised over a and b)."""
        ia = np.clip(np.floor(np.asarray(a, float) / HOP_S).astype(int), 0, self.n)
        ib = np.clip(np.ceil(np.asarray(b, float) / HOP_S).astype(int), 0, self.n)
        ib = np.maximum(ib, ia + 1)
        c = self._cum[key]
        return (c[np.minimum(ib, self.n)] - c[ia]) / (ib - ia)

    def seconds(self, key: str, a: float, b: float) -> float:
        return float(self.share(key, a, b)) * max(b - a, 0.0)


def _cache_path(consultation: str) -> Path:
    return CACHE_DIR / f"{consultation}.parquet"


def has_channels(consultation: str, root: Path = RAW_DIR) -> bool:
    a = root / "primock57" / "audio"
    return all((a / f"{consultation}_{r}.wav").exists() for r in ROLES)


@lru_cache(maxsize=8)
def track(consultation: str, root: Path = RAW_DIR) -> Track | None:
    """The consultation's speech masks, computed once and cached under data/interim/audio_features/."""
    p = _cache_path(consultation)
    if p.exists():
        df = pd.read_parquet(p)
        return Track(df["doctor"].to_numpy(), df["patient"].to_numpy())
    if not has_channels(consultation, root):
        return None
    masks = {}
    for r in ROLES:
        x, sr = sf.read(str(root / "primock57" / "audio" / f"{consultation}_{r}.wav"), dtype="float32")
        masks[r] = speech_mask(frame_db(x if x.ndim == 1 else x.mean(1), sr))
    n = min(len(masks["doctor"]), len(masks["patient"]))
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({r: masks[r][:n] for r in ROLES}).to_parquet(p)
    return Track(masks["doctor"][:n], masks["patient"][:n])


def unit_features(tr: Track | None, role: str | None, start: float, end: float, word_t0, word_t1, gap_t,
                  n_words: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audio features for one unit's words and gaps. Times are absolute (seconds in the consultation).

    ``role`` is the turn's speaker, or None for a window (both channels)."""
    word_t0, word_t1, gap_t = (np.asarray(x, float) for x in (word_t0, word_t1, gap_t))
    if tr is None:
        nan_w = {k: np.full(len(word_t0), np.nan) for k in WORD_FEATURES}
        nan_g = {k: np.full(len(gap_t), np.nan) for k in GAP_FEATURES}
        nan_w["two_ch"], nan_g["two_ch"] = np.zeros(len(word_t0)), np.zeros(len(gap_t))
        return pd.DataFrame(nan_w), pd.DataFrame(nan_g)
    if role in ROLES:
        own, other = role, ROLES[1 - ROLES.index(role)]
    else:
        own, other = "any", "both"
    own_s = tr.seconds(own, start, end)
    unit = {"u_ov": float(tr.share("both", start, end)), "u_own_s": own_s,
            "u_wps": n_words / own_s if own_s > 0.05 else np.nan, "two_ch": 1.0}
    a, b = word_t0 - WORD_PAD_S, word_t1 + WORD_PAD_S
    words = pd.DataFrame({"a_ov": tr.share("both", a, b), "a_other": tr.share(other, a, b),
                          "a_own": tr.share(own, a, b), **{k: np.full(len(word_t0), v) for k, v in unit.items()}})
    a, b = gap_t - GAP_PAD_S, gap_t + GAP_PAD_S
    gaps = pd.DataFrame({"g_ov": tr.share("both", a, b), "g_other": tr.share(other, a, b),
                         "g_own": tr.share(own, a, b), **{k: np.full(len(gap_t), v) for k, v in unit.items()}})
    return words, gaps
