import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from asrshift import verifier as v
from asrshift.config import load_config
from asrshift.textnorm import WILDCARD, scoring_tokens


@pytest.fixture(scope="module")
def w():
    return v.Weights.from_config(load_config())


def _labels(ref_text, hyp_text, w):
    hyp, _ = v.hyp_tokens(hyp_text.split())
    return hyp, v.labels(scoring_tokens(ref_text, reference=True), hyp, w)


def test_weights_follow_config(w):
    cfg = load_config()["verifier"]["weights"]
    assert (w.crit, w.med, w.cont, w.func) == (cfg["critical"], cfg["medical"], cfg["content"], cfg["function"])
    assert "not" not in w.function_words and "the" in w.function_words and "yeah" in w.function_words


def test_categories(w):
    toks = scoring_tokens("take 500 mg of paracetamol not the blue one")
    cats = dict(zip(toks, v.categories(toks, w)))
    assert cats["500"] == "crit" and cats["not"] == "crit"
    assert cats["the"] == "func" and cats["blue"] == "cont"


def test_deleted_negation_is_a_heavy_gap(w):
    hyp, lab = _labels("I have no chest pain", "I have chest pain", w)
    assert hyp == ["i", "have", "chest", "pain"]
    assert lab["gap_b"].tolist() == [0, 0, w.crit, 0, 0]       # before "chest"
    assert lab["gap_cr"].sum() == 1 and lab["word_b"].sum() == 0


def test_substitution_costs_the_heavier_word(w):
    _, lab = _labels("take 500 mg", "take 50 mg", w)
    assert lab["word_b"].tolist() == [0, w.crit, 0] and lab["word_cr"].tolist() == [0, 1, 0]


def test_insertion_and_filler(w):
    _, lab = _labels("chest pain", "the chest pain", w)
    assert lab["word_b"].tolist() == [w.func, 0, 0]


def test_wildcard_absorbs_without_cost(w):
    ref = scoring_tokens("I was <UNIN/> at home", reference=True)
    assert WILDCARD in ref
    hyp, _ = v.hyp_tokens("I was mumbling something at home".split())
    lab = v.labels(ref, hyp, w)
    assert lab["word_b"].sum() == 0 and lab["gap_b"].sum() == 0
    assert lab["status"].count("W") == 2


def test_hyp_tokens_keep_display_owner():
    toks, owner = v.hyp_tokens(["I've", "got", "pain"])
    assert toks == ["i", "have", "got", "pain"] and owner == [0, 0, 1, 2]


def test_gap_after_indices():
    owner = [0, 0, 1, 2]                      # "I've got pain" -> i have got pain
    scores = np.array([0.1, 0.9, 0.2, 0.3, 0.4])
    gaps = v._gaps(owner, scores, [[] for _ in scores], [0.5, 0.8, 1.2], start=0.0)
    after = {g["after"]: g["score"] for g in gaps}
    assert set(after) == {-1, 0, 1, 2}
    assert after[0] == 0.9                    # the gap inside "I've" and the one after it merge


def _toy_verifier(w):
    rng = np.random.default_rng(0)
    variants = {}
    for name in ("second", "solo"):
        wf, gf = v.variant_features(name)
        Xw, Xg = pd.DataFrame(rng.random((50, len(wf))), columns=wf), pd.DataFrame(rng.random((50, len(gf))), columns=gf)
        variants[name] = {"word_features": wf, "gap_features": gf,
                          "words": HistGradientBoostingRegressor(loss="poisson", max_iter=5).fit(Xw, rng.random(50)),
                          "gaps": HistGradientBoostingRegressor(loss="poisson", max_iter=5).fit(Xg, rng.random(50))}
    return v.Verifier(weights=w, variants=variants, meta={})


@pytest.mark.parametrize("second", [None, "I've got a pain in my chest"])
def test_score_shape(w, second):
    words = [{"word": " I've", "start": 0.0, "end": 0.3, "probability": 0.9},
             {"word": " got", "start": 0.3, "end": 0.5, "probability": 0.4},
             {"word": " pain", "start": 0.5, "end": 0.9, "probability": 0.8}]
    unit = {"mean_logprob": -0.3, "word_p_mean": 0.7, "nsp_max": 0.1, "cr_max": 1.2, "n_segments": 1,
            "words_per_s": 3.3, "log_duration": 0.0, "kind": 1}
    out = _toy_verifier(w).score(words, unit, second)
    assert [x["text"] for x in out["words"]] == ["I've", "got", "pain"]
    assert all(set(x) == {"text", "t0", "t1", "p", "score", "critical"} for x in out["words"])
    assert {g["after"] for g in out["gaps"]} <= {-1, 0, 1, 2} and -1 in {g["after"] for g in out["gaps"]}
    assert all(g["score"] >= 0 for g in out["gaps"])
    if second:
        assert any("chest" in g["suggestion"] for g in out["gaps"])


def test_empty_transcript(w):
    out = _toy_verifier(w).score([], {"kind": 1, "start": 2.0})
    assert out["words"] == [] and len(out["gaps"]) == 1 and out["gaps"][0]["after"] == -1


def test_training_uses_no_test_units():
    sets = {v._set({"split": s, "dataset": d, "subset": sub})
            for s in ("calibration", "validation", "recalibration", "test")
            for d, sub in (("eka", "narration_sentence"), ("primock57", "window"), ("primock57", "turn_doctor"))}
    assert sets == {"train", "eka_test", "pm_window_test", "pm_turn_test"}
    assert v._set({"split": "test", "dataset": "primock57", "subset": "window"}) != "train"
    assert "test" not in v.TRAIN_SPLITS


def test_hybrid_reviews_whole_unit_when_cheaper():
    # one 2 s unit with many flagged words: listening to all of it beats clip after clip
    n = 8
    it = {"unit": np.array(["u"] * n), "a": np.linspace(0, 1.6, n), "b": np.linspace(0.4, 2.0, n),
          "pred": np.ones(n), "true": np.ones(n), "crit": np.zeros(n)}
    caught = v._hybrid(it, {"u": 2.0}, [2.0])
    assert caught[0, 0] == n


def test_variant_features():
    full_w, full_g = v.variant_features("full")
    second_w, second_g = v.variant_features("second")
    solo_w, solo_g = v.variant_features("solo")
    assert set(second_w) < set(full_w) and "other_p" in full_w and "other_p" not in second_w
    assert not set(solo_w) & set(v.SECOND_FEATURES_W + v.SECOND_P_FEATURES_W)
    assert not set(solo_g) & set(v.SECOND_FEATURES_G + v.SECOND_P_FEATURES_G)
    v3_w, _ = v.variant_features("v3")
    assert "dis_run" not in v3_w and "agree" in v3_w


def test_second_opinion_probabilities_and_extras(w):
    hyp = ["i", "have", "chest", "pain"]
    other = ["i", "have", "no", "chest", "pain"]
    so = v._second_opinion(hyp, other, w, other_p=np.array([0.9, 0.8, 0.6, 0.7, 0.95]))
    assert so["agree"].tolist() == [1, 1, 1, 1]
    assert so["other_p"].tolist() == [0.9, 0.8, 0.7, 0.95]
    assert so["extra"][2] == 1 and so["extra_crit"][2] == 1 and so["extra_p"][2] == 0.6
    assert so["suggestion"][2] == ["no"]
    none = v._second_opinion(hyp, None, w)
    assert np.isnan(none["agree"]).all() and np.isnan(none["other_p"]).all()


def test_new_word_features(w):
    hyp = ["take", "the", "tablets", "daily"]
    wf, gf, _ = v.features(hyp, np.array([0.9, 0.4, 0.3, 0.9]), {"kind": 2},
                           ["take", "two", "tablets", "daily"], w)
    assert wf["dis_run"].tolist() == [0, 1, 0, 0] and len(gf) == len(hyp) + 1
    assert gf["dis_near"].tolist() == [0, 1, 1, 0, 0]
    assert np.isnan(wf["other_p"]).all()                  # no probabilities passed for the second model


def test_hybrid_scale_zero_never_buys_whole_units():
    n = 8
    it = {"unit": np.array(["u"] * n), "a": np.linspace(0, 1.6, n), "b": np.linspace(0.4, 2.0, n),
          "pred": np.ones(n), "true": np.ones(n), "crit": np.zeros(n)}
    # with the whole-unit option switched off, 2 s only buys the clips that fit
    assert v._hybrid(it, {"u": 2.0}, [2.0], scale=0.0)[0, 0] == v._queue(it, [2.0])[0, 0]


def test_crossfit_never_scores_a_consultation_with_its_own_data(monkeypatch):
    cfg = load_config()
    wf, gf = v.variant_features("full")
    rng = np.random.default_rng(1)

    def frame(n, cols, cons):
        df = pd.DataFrame(rng.random((n, len(cols))), columns=cols)
        return df.assign(b=rng.random(n), consultation=rng.choice(cons, n))

    cons = [f"c{i}" for i in range(10)]
    tests = {"window": (frame(200, wf, cons), frame(220, gf, cons), None),
             "turn": (frame(300, wf, cons), frame(320, gf, cons), None)}
    train_w, train_g = frame(100, wf, ["eka"]), frame(110, gf, ["eka"])
    tests["window"][0]["probe"] = np.arange(200)
    fits = []

    class Recorder:
        """Remembers which consultations it trained on and predicts 1 for any it has seen."""

        def fit(self, X, y):
            self.cons = set(X.attrs.get("cons", []))
            fits.append(self)
            return self

        def predict(self, X):
            return np.zeros(len(X))

    monkeypatch.setattr(v, "_regressor", lambda cfg_, params=None, head=None: Recorder())
    real_concat = pd.concat

    def tagging_concat(frames, *a, **k):
        out = real_concat(frames, *a, **k)
        if "consultation" in out:
            out.attrs["cons"] = set(out["consultation"])
        return out

    monkeypatch.setattr(v.pd, "concat", tagging_concat)
    toy = v.Verifier(weights=v.Weights(), variants={}, meta={}, committed_variant="full")
    preds = v._crossfit_preds(toy, train_w, train_g, tests, cfg, n_folds=5)
    for key, (W, G, _) in tests.items():
        assert len(preds[key][0]) == len(W) and len(preds[key][1]) == len(G)
    # 5 folds x (words, gaps); each model saw the training data and exactly 4/5 of the consultations
    assert len(fits) == 10
    assert all("eka" in f.cons and len(f.cons - {"eka"}) == 8 for f in fits)


def test_audio_variant_adds_audio_features():
    from asrshift import audio_features as af

    wf, gf = v.variant_features("full")
    awf, agf = v.variant_features("full+audio")
    assert awf[:len(wf)] == wf and agf[:len(gf)] == gf
    assert set(awf) - set(wf) == set(af.WORD_FEATURES) and set(agf) - set(gf) == set(af.GAP_FEATURES)
    assert v.variant_features("solo+audio")[0][-len(af.WORD_FEATURES):] == af.WORD_FEATURES


def test_audio_columns_nan_for_eka():
    row = {"dataset": "eka", "consultation": None, "start": 0.0, "end": 3.0, "duration": 3.0, "subset": "x", "roles": None}
    aw, ag = v.audio_columns(row, 4)
    assert len(aw) == 4 and len(ag) == 5
    assert aw["a_ov"].isna().all() and (ag["two_ch"] == 0).all()
