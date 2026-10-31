import numpy as np
import pandas as pd
import pytest

from asrshift import audio_features as af

SR = af.SR


def _tone(seconds, on, amp=0.3, sr=SR):
    """A 200 Hz tone during the ``on`` spans (seconds), faint noise elsewhere."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1e-4, int(seconds * sr)).astype(np.float32)
    t = np.arange(len(x)) / sr
    for a, b in on:
        m = (t >= a) & (t < b)
        x[m] += amp * np.sin(2 * np.pi * 200 * t[m]).astype(np.float32)
    return x


@pytest.fixture(scope="module")
def tr():
    # doctor speaks 0-4 s, patient 3-6 s: overlap 3-4 s, silence 6-8 s
    return af.Track.from_audio(_tone(8, [(0, 4)]), _tone(8, [(3, 6)]))


def test_frame_db_hops():
    db = af.frame_db(np.zeros(SR, np.float32))
    assert len(db) == 1 + (SR - int(af.FRAME_S * SR)) // int(af.HOP_S * SR)
    assert np.all(db <= -100)


def test_speech_and_overlap_detected(tr):
    assert tr.share("doctor", 0.5, 2.5) == pytest.approx(1.0, abs=0.02)
    assert tr.share("patient", 0.5, 2.5) == pytest.approx(0.0, abs=0.02)
    assert tr.share("both", 3.1, 3.9) == pytest.approx(1.0, abs=0.03)
    assert tr.share("both", 0.5, 2.5) == pytest.approx(0.0, abs=0.02)
    assert tr.share("any", 6.2, 7.8) == pytest.approx(0.0, abs=0.02)
    assert tr.seconds("both", 0, 8) == pytest.approx(1.0, abs=0.1)


def test_turn_features_use_own_and_other_channel(tr):
    # a doctor turn 0-4 s with two words; the second sits in the overlap
    w, g = af.unit_features(tr, "doctor", 0.0, 4.0, [0.5, 3.3], [1.0, 3.7], [0.0, 1.0, 3.7], n_words=2)
    assert list(w.columns) == af.WORD_FEATURES and list(g.columns) == af.GAP_FEATURES
    assert w["a_ov"].iloc[0] < 0.05 and w["a_ov"].iloc[1] > 0.9
    assert w["a_other"].iloc[1] > 0.9 and w["a_own"].iloc[0] > 0.9
    assert g["g_ov"].iloc[2] > 0.5 > g["g_ov"].iloc[1]
    assert w["u_ov"].iloc[0] == pytest.approx(0.25, abs=0.03)
    assert w["u_own_s"].iloc[0] == pytest.approx(4.0, abs=0.1)
    assert w["u_wps"].iloc[0] == pytest.approx(0.5, abs=0.05)
    assert (w["two_ch"] == 1).all()


def test_window_features_treat_overlap_as_other(tr):
    w, _ = af.unit_features(tr, None, 0.0, 8.0, [3.3], [3.7], [8.0], n_words=1)
    assert w["a_other"].iloc[0] == pytest.approx(w["a_ov"].iloc[0])
    assert w["u_own_s"].iloc[0] == pytest.approx(6.0, abs=0.15)


def test_single_channel_units_get_nan(tr):
    w, g = af.unit_features(None, None, 0.0, 3.0, [0.1, 0.5], [0.4, 0.9], [0.0, 0.4, 0.9], n_words=2)
    assert len(w) == 2 and len(g) == 3
    assert w.drop(columns="two_ch").isna().all().all() and g.drop(columns="two_ch").isna().all().all()
    assert (w["two_ch"] == 0).all() and (g["two_ch"] == 0).all()


def test_empty_unit_still_has_a_gap(tr):
    w, g = af.unit_features(tr, "patient", 3.0, 6.0, [], [], [3.0], n_words=0)
    assert len(w) == 0 and len(g) == 1
    assert np.isnan(g["u_wps"].iloc[0]) or g["u_wps"].iloc[0] == 0


def test_track_cache_roundtrip(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    audio = root / "primock57" / "audio"
    audio.mkdir(parents=True)
    import soundfile as sf
    sf.write(str(audio / "day9_consultation99_doctor.wav"), _tone(3, [(0, 2)]), SR)
    sf.write(str(audio / "day9_consultation99_patient.wav"), _tone(3, [(1, 3)]), SR)
    monkeypatch.setattr(af, "CACHE_DIR", tmp_path / "cache")
    af.track.cache_clear()
    t1 = af.track("day9_consultation99", root)
    assert (tmp_path / "cache" / "day9_consultation99.parquet").exists()
    af.track.cache_clear()
    t2 = af.track("day9_consultation99", root)        # from the cache this time
    assert t1.share("both", 0, 3) == pytest.approx(t2.share("both", 0, 3))
    assert af.track("day9_consultation98", root) is None
    af.track.cache_clear()
