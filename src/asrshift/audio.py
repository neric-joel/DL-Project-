"""Audio for a manifest row, as 16 kHz mono float32.

* Eka: ``audio_source = "<shard>#<row>"``; the parquet cell holds MP3 bytes, decoded with PyAV.
* PriMock57: ``audio_source = "<consultation>:<doctor|patient|mix>"`` plus ``start``/``end``.
  ``mix`` is the sum of the two channels (peak-limited to avoid clipping).
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

from asrshift.paths import RAW_DIR

SR = 16_000


@lru_cache(maxsize=2)
def _eka_shard(path: str):
    return pq.read_table(path, columns=["audio"]).column("audio")


def eka_audio(audio_source: str, root: Path = RAW_DIR) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    shard, row = audio_source.split("#")
    blob = _eka_shard(str(root / "eka" / "en" / shard))[int(row)].as_py()
    if isinstance(blob, dict):
        blob = blob["bytes"]
    return decode_audio(io.BytesIO(blob), sampling_rate=SR)


@lru_cache(maxsize=4)
def _channel(consultation: str, role: str, root: str) -> np.ndarray:
    audio, sr = sf.read(str(Path(root) / "primock57" / "audio" / f"{consultation}_{role}.wav"), dtype="float32")
    if sr != SR:
        raise ValueError(f"expected {SR} Hz, got {sr}")
    return audio


@lru_cache(maxsize=2)
def _mix(consultation: str, root: str) -> np.ndarray:
    d, p = _channel(consultation, "doctor", root), _channel(consultation, "patient", root)
    n = min(len(d), len(p))
    mix = d[:n] + p[:n]
    peak = float(np.max(np.abs(mix))) if n else 0.0
    return mix / peak * 0.99 if peak > 0.99 else mix


def primock_audio(audio_source: str, start: float, end: float, root: Path = RAW_DIR) -> np.ndarray:
    consultation, role = audio_source.split(":")
    full = _mix(consultation, str(root)) if role == "mix" else _channel(consultation, role, str(root))
    return np.ascontiguousarray(full[int(round(start * SR)): int(round(end * SR))])


def load_unit_audio(row, root: Path = RAW_DIR) -> np.ndarray:
    if row["dataset"] == "eka":
        return eka_audio(row["audio_source"], root)
    return primock_audio(row["audio_source"], float(row["start"]), float(row["end"]), root)
