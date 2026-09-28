from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asrshift import data_eka, data_primock
from asrshift.features import FEATURES, unit_features
from asrshift.paths import INTERIM_DIR, RAW_DIR

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 10
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "Doctor"
        xmin = 0
        xmax = 10
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 1.5
            text = ""
        intervals [2]:
            xmin = 1.5
            xmax = 4.25
            text = "Take it ""twice"" a day. <UNIN/>"
        intervals [3]:
            xmin = 4.25
            xmax = 10
            text = ""
'''


def test_parse_textgrid(tmp_path: Path):
    p = tmp_path / "day1_consultation01_doctor.TextGrid"
    p.write_text(TEXTGRID)
    iv = data_primock.parse_textgrid(p)
    assert len(iv) == 3 and iv[1] == (1.5, 4.25, 'Take it "twice" a day. <UNIN/>')


U = namedtuple("U", "consultation role idx start end text n_ref_words has_unin")


def _utts(spans):
    rows = [U("c1", role, i, s, e, "word " * 3, 3, False)._asdict() for i, (role, s, e) in enumerate(spans)]
    return pd.DataFrame(rows)


def test_windows_never_cut_inside_overlapping_speech():
    utts = _utts([("doctor", 0, 10), ("patient", 9, 20), ("doctor", 21, 28), ("patient", 29, 40), ("doctor", 41, 45)])
    w = data_primock.build_windows(utts, max_window_s=30, pad_s=0.3, durations={("c1", "doctor"): 50, ("c1", "patient"): 50})
    # blocks: [0,20] [21,28] [29,40] [41,45] -> windows [0..28], [29..45]
    assert len(w) == 2
    assert w.iloc[0]["n_utterances"] == 3 and w.iloc[0]["overlap_s"] == pytest.approx(1.0)
    assert w.iloc[0]["n_speaker_turns"] == 3
    assert w.iloc[0]["end"] <= 28.5 and w.iloc[1]["start"] >= 28.5
    assert (w["duration"] <= 30.6).all()


def test_long_single_block_becomes_its_own_window():
    utts = _utts([("doctor", 0, 35), ("patient", 36, 38)])
    w = data_primock.build_windows(utts, 30, 0.3, {("c1", "doctor"): 40, ("c1", "patient"): 40})
    assert len(w) == 2 and w.iloc[0]["duration"] > 30


def test_recalibration_split_is_stratified_by_day():
    cons = [f"day{d}_consultation{c:02d}" for d in range(1, 6) for c in range(1, 12)]
    s = data_primock.split_consultations(cons, 2, seed=598)
    recal = [c for c, v in s.items() if v == "recalibration"]
    assert len(recal) == 10
    assert {c.split("_")[0] for c in recal} == {f"day{d}" for d in range(1, 6)}


def test_eka_split_is_disjoint_in_speakers_sessions_and_texts():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(400):
        spk = f"s{rng.integers(0, 20)}" if i < 350 else None
        rows.append({"file_name": f"f{i}", "session_id": f"ses{i // 2}", "speaker": spk,
                     "recording_context": "narration_entity" if i % 3 else "narration_sentence",
                     "md5_text": f"t{rng.integers(0, 300)}"})
    df = pd.DataFrame(rows)
    df["group_id"] = np.where(df["speaker"].notna(), "spk:" + df["speaker"].astype(str), "ses:" + df["session_id"])
    out = data_eka.split(df, {"calibration": 0.5, "validation": 0.2, "test": 0.3}, seed=1)
    kept = out[out["dropped_reason"] == ""]
    for key in ("speaker", "session_id", "md5_text"):
        assert (kept.dropna(subset=[key]).groupby(key)["split"].nunique() <= 1).all()
    assert set(kept["split"]) == {"calibration", "validation", "test"}


def test_empty_hypothesis_gets_least_confident_features():
    f = unit_features({"text": "", "segments": [], "words": [], "audio_s": 3.0})
    assert f["empty_hyp"] == 1 and f["conf"] == 0 and set(FEATURES) <= set(f)


def test_features_from_segments():
    rec = {"text": " take it twice", "audio_s": 2.0,
           "segments": [{"avg_logprob": -0.2, "n_tokens": 4, "no_speech_prob": 0.01, "compression_ratio": 1.1,
                         "temperature": 0.0}],
           "words": [{"word": " take", "probability": 0.9}, {"word": " it", "probability": 0.4},
                     {"word": " twice", "probability": 0.8}]}
    f = unit_features(rec)
    assert f["conf"] == pytest.approx(np.exp(-0.2)) and f["word_frac_low"] == pytest.approx(1 / 3)
    assert f["words_per_s"] == pytest.approx(1.5) and f["fallback"] == 0


@pytest.mark.data
@pytest.mark.skipif(not (INTERIM_DIR / "manifest.parquet").exists(), reason="manifest not built")
def test_real_manifest_has_no_leakage():
    m = pd.read_parquet(INTERIM_DIR / "manifest.parquet")
    kept = m[m["dropped_reason"].fillna("") == ""]
    eka = kept[kept["dataset"] == "eka"]
    for key in ("speaker", "session_id", "md5_text"):
        assert (eka.dropna(subset=[key]).groupby(key)["split"].nunique() <= 1).all()
    pm = kept[kept["dataset"] == "primock57"]
    assert (pm.groupby("consultation")["split"].nunique() == 1).all()
    assert pm["consultation"].nunique() == 57 and (RAW_DIR / "primock57").exists()
