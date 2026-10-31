"""Word-and-gap verifier: which words a human should check, and where words were probably dropped.

Exploratory (Amendment 7, not part of the confirmatory study). The review policies P1-P4 decide per
transcript; the verifier ranks *items* inside a transcript so a reviewer listens only to short clips:

* a **word** item: one hypothesis word that may be wrong (substituted or inserted);
* a **gap** item: the boundary before each hypothesis token (and after the last one), where reference
  words may have been deleted. On PriMock57 windows more than half of all errors are deletions, which
  no word flag can point at.

Errors are weighted by clinical harm (``verifier.weights`` in the config): numbers, doses and
negations (``score.is_critical``) 10, lexicon medical terms 5, other content words 1, function words
and fillers 0.2. A substitution costs the heavier of its two words, a deletion the reference word's
weight, an insertion the hypothesis word's. Both models predict the *expected weighted error* an item
would fix (Poisson gradient boosting), so the queue puts a possibly missed "not" ahead of five unsure
"and"s. Labels come from the study's own wildcard alignment (``asrshift.align``): hypothesis words
absorbed by an ``<UNIN/>`` wildcard are neither errors nor reviewable gains.

Two variants are fitted. ``second`` adds features from a second model's transcript of the same audio
(whisper-large-v3-turbo): where the two disagree, and which words the second model wrote in a gap.
``solo`` uses whisper-small's own decoder statistics only, for live decoding without a second model.

With ``verifier.audio_features`` on, ``full+audio`` also uses speech energy from PriMock57's two
channels (``asrshift.audio_features``): overlapping speech around a word or gap, and how much speech
the unit holds per hypothesis word. It scores the committed transcripts of consultations whose audio
is downloaded; everything else falls back to the variants without audio.

Training uses only units that were never test data: Eka calibration + validation and the PriMock57
recalibration consultations. The committed per-word table has no timestamps, so for training and for
the review curves word times are spread evenly over each unit; live decoding uses Whisper's own.

    asrshift verifier        # fit, then write results/models/verifier_small.joblib and the curves
"""

from __future__ import annotations

import difflib
import heapq
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from asrshift import audio_features as af
from asrshift import score, terms
from asrshift.align import align, ops
from asrshift.config import load_config
from asrshift.paths import RAW_DIR, RESULTS_DIR
from asrshift.textnorm import collapse_repeats, scoring_tokens, strip_tags, tokens

CATS = ("crit", "med", "cont", "func")          # heaviest first under every weight setting used
TRAIN_SPLITS = ("calibration", "validation", "recalibration")
SECOND_FEATURES_W = ["agree", "agree_win", "other_w", "dis_run", "extra_near"]
SECOND_FEATURES_G = ["extra", "extra_w", "extra_crit", "dis_near"]
# the second model's own word probabilities exist for the committed results only, not in live decoding
SECOND_P_FEATURES_W = ["other_p"]
SECOND_P_FEATURES_G = ["extra_p"]
UNIT_FEATURES = ["n_words", "mean_logprob", "word_p_mean", "nsp_max", "cr_max", "n_segments", "words_per_s",
                 "log_duration", "kind"]
WORD_FEATURES = ["p", "p_prev", "p_next", "own_w", "pos", *SECOND_FEATURES_W, *UNIT_FEATURES]
GAP_FEATURES = ["left_p", "right_p", "pos", "edge", *SECOND_FEATURES_G, *UNIT_FEATURES]
# v3's feature sets, kept so the comparison with the previous verifier can be reproduced
V3_WORD_FEATURES = ["p", "p_prev", "p_next", "own_w", "pos", "agree", "agree_win", "other_w", *UNIT_FEATURES]
V3_GAP_FEATURES = ["left_p", "right_p", "pos", "edge", "extra", "extra_w", "extra_crit", *UNIT_FEATURES]
KIND = {"eka": 0, "turn": 1, "window": 2}
_UNIN = re.compile(r"<\s*(?:UNIN|INAUDIBLE_SPEECH)\s*/?\s*>", re.IGNORECASE)


def model_path(model: str) -> Path:
    return RESULTS_DIR / "models" / f"verifier_{model.replace('/', '_')}.joblib"


def curves_path(model: str) -> Path:
    return RESULTS_DIR / "tables" / model.replace("/", "_") / "verifier" / "curves.json"


# ---- token categories and weights ---------------------------------------------------------------

@dataclass(frozen=True)
class Weights:
    crit: float = 10.0
    med: float = 5.0
    cont: float = 1.0
    func: float = 0.2
    function_words: frozenset = frozenset()

    @classmethod
    def from_config(cls, cfg: dict) -> "Weights":
        v = cfg.get("verifier", {})
        w = v.get("weights", {})
        fillers = set(v.get("fillers", []))
        function = (set(ENGLISH_STOP_WORDS) | fillers) - score.NEGATION
        return cls(crit=w.get("critical", 10.0), med=w.get("medical", 5.0), cont=w.get("content", 1.0),
                   func=w.get("function", 0.2), function_words=frozenset(function))

    def of(self, cat: str) -> float:
        return {"crit": self.crit, "med": self.med, "cont": self.cont, "func": self.func}.get(cat, 0.0)


@lru_cache(maxsize=1)
def _matcher() -> "terms.Matcher | None":
    lex = terms.load()
    return terms.Matcher(lex) if lex is not None else None


def categories(toks: list[str], w: Weights) -> list[str]:
    """Category of each normalised token: crit (number/dose/negation), med (lexicon term), func or cont."""
    med = np.zeros(len(toks), bool)
    m = _matcher()
    if m is not None:
        for s, e, _term, _type in m.find(toks):
            med[s:e] = True
    out = []
    for i, t in enumerate(toks):
        if score.is_critical(toks, i):
            out.append("crit")
        elif med[i]:
            out.append("med")
        elif t in w.function_words:
            out.append("func")
        else:
            out.append("cont")
    return out


def hyp_tokens(words: list[str]) -> tuple[list[str], list[int]]:
    """Normalised hypothesis tokens and, for each, the index of the display word it came from.

    Each display word is normalised on its own and repeats are collapsed, as in the study's
    word-level table (``score._word_level``)."""
    toks, owner = [], []
    for i, w in enumerate(words):
        for t in tokens(w):
            toks.append(t)
            owner.append(i)
    kept, idx = collapse_repeats(toks)
    return kept, [owner[k] for k in idx]


# ---- labels from the wildcard alignment ----------------------------------------------------------

def labels(ref: list[str], hyp: list[str], w: Weights) -> dict:
    """Per hypothesis token: status, weighted error fixed by checking it, critical flag.
    Per gap (len(hyp) + 1): weighted and critical deletions there."""
    a = align(ref, hyp)
    rc, hc = categories(ref, w), categories(hyp, w)
    m = len(hyp)
    word_b, word_cr = np.zeros(m), np.zeros(m, int)
    gap_b, gap_cr, gap_n = np.zeros(m + 1), np.zeros(m + 1, int), np.zeros(m + 1, int)
    i = j = 0                                       # reference / hypothesis position, in alignment order
    for op, _r, h in ops(a):
        if op == "C":
            i, j = i + 1, j + 1
        elif op == "S":
            word_b[j] = max(w.of(rc[i]), w.of(hc[j]))
            word_cr[j] = rc[i] == "crit" or hc[j] == "crit"
            i, j = i + 1, j + 1
        elif op == "I":
            word_b[j], word_cr[j] = w.of(hc[j]), hc[j] == "crit"
            j += 1
        elif op == "D":                             # the gap before hypothesis token j
            gap_b[j] += w.of(rc[i])
            gap_cr[j] += rc[i] == "crit"
            gap_n[j] += 1
            i += 1
        else:                                       # wildcard: absorbed tokens are neither errors nor gains
            j += len(h.split(" ")) if h else 0
            i += 1
    return {"status": list(a.hyp_status), "word_b": word_b, "word_cr": word_cr, "gap_b": gap_b, "gap_cr": gap_cr, "gap_n": gap_n,
            "ref_weight": float(sum(w.of(c) for c, s in zip(rc, a.ref_status) if s != "W"))}


# ---- features -------------------------------------------------------------------------------------

def _second_opinion(hyp: list[str], other: list[str] | None, w: Weights, other_p: np.ndarray | None = None) -> dict:
    """Where a second model's transcript agrees with ``hyp``, and what it wrote in each gap.

    ``other_p`` (the second model's probability for each of its tokens) is optional: it exists for the
    committed results, not for a transcript passed in as text."""
    m = len(hyp)
    agree, other_w, op = np.zeros(m), np.zeros(m), np.full(m, np.nan)
    extra, extra_w, extra_crit = np.zeros(m + 1), np.zeros(m + 1), np.zeros(m + 1)
    extra_p = np.full(m + 1, np.nan)
    suggestion = [[] for _ in range(m + 1)]
    out = {"agree": agree, "other_w": other_w, "other_p": op, "extra": extra, "extra_w": extra_w,
           "extra_crit": extra_crit, "extra_p": extra_p, "suggestion": suggestion}
    if other is None:
        for k in ("agree", "other_w", "extra", "extra_w", "extra_crit"):
            out[k] = np.full_like(out[k], np.nan)
        return out
    oc = categories(other, w)
    has_p = other_p is not None and len(other_p) == len(other)
    for tag, a0, a1, b0, b1 in difflib.SequenceMatcher(a=hyp, b=other, autojunk=False).get_opcodes():
        if tag == "equal":
            agree[a0:a1] = 1
            if has_p:
                op[a0:a1] = other_p[b0:b1]
            continue
        if tag == "replace":
            other_w[a0:a1] = max(w.of(c) for c in oc[b0:b1])
            if has_p:
                op[a0:a1] = float(np.mean(other_p[b0:b1]))
            if b1 - b0 <= a1 - a0:
                continue
            at, b0 = a1, b0 + (a1 - a0)
        elif tag == "insert":
            at = a0
        else:
            continue
        extra[at] += b1 - b0
        extra_w[at] += sum(w.of(c) for c in oc[b0:b1])
        extra_crit[at] = float(any(c == "crit" for c in oc[b0:b1]))
        if has_p:
            extra_p[at] = float(np.mean(other_p[b0:b1]))
        suggestion[at] += other[b0:b1]
    return out


def _run_length(flag: np.ndarray) -> np.ndarray:
    """For each position, the length of the run of True values it belongs to (0 where False)."""
    out = np.zeros(len(flag))
    i = 0
    while i < len(flag):
        if not flag[i]:
            i += 1
            continue
        j = i
        while j < len(flag) and flag[j]:
            j += 1
        out[i:j] = j - i
        i = j
    return out


def features(hyp: list[str], p: np.ndarray, unit: dict, other: list[str] | None, w: Weights,
             other_p: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Word and gap feature tables for one unit (``unit`` holds the decoder statistics)."""
    m = len(hyp)
    so = _second_opinion(hyp, other, w, other_p)
    pad = np.concatenate([[1.0], p, [1.0]])
    u = {k: float(unit[k]) if unit.get(k) is not None and pd.notna(unit.get(k)) else np.nan
         for k in UNIT_FEATURES if k != "n_words"}
    u["n_words"] = float(m)
    hc = categories(hyp, w)
    idx = np.arange(m)
    agree = so["agree"]
    dis = (agree == 0) if other is not None else np.zeros(m, bool)
    extra = so["extra"]
    words = pd.DataFrame({
        "p": p, "p_prev": pad[:m], "p_next": pad[2:], "own_w": [w.of(c) for c in hc],
        "pos": idx / max(m - 1, 1), "agree": agree,
        "agree_win": pd.Series(agree).rolling(5, center=True, min_periods=1).mean().to_numpy() if m else agree,
        "other_w": so["other_w"], "other_p": so["other_p"],
        "dis_run": _run_length(dis) if other is not None else np.full(m, np.nan),
        "extra_near": (extra[:m] + extra[1:]) if m else np.zeros(0), **u})
    left = np.concatenate([[0.0], dis.astype(float)])
    right = np.concatenate([dis.astype(float), [0.0]])
    gaps = pd.DataFrame({
        "left_p": pad[:m + 1], "right_p": pad[1:m + 2], "pos": np.arange(m + 1) / max(m, 1),
        "edge": [float(i in (0, m)) for i in range(m + 1)],
        "extra": extra, "extra_w": so["extra_w"], "extra_crit": so["extra_crit"], "extra_p": so["extra_p"],
        "dis_near": (left + right) if other is not None else np.full(m + 1, np.nan), **u})
    return words, gaps, so["suggestion"]


# ---- committed results -> training and evaluation tables ------------------------------------------

def _kind(row) -> int:
    return KIND["eka"] if row["dataset"] == "eka" else KIND["window" if row["subset"] == "window" else "turn"]


def _set(row) -> str:
    if row["split"] in TRAIN_SPLITS:
        return "train"
    if row["dataset"] == "eka":
        return "eka_test"
    return "pm_window_test" if row["subset"] == "window" else "pm_turn_test"


def _probs(model: str) -> dict:
    words = pd.read_parquet(score.words_path(model))
    return {k: g["probability"].to_numpy(float) for k, g in words.groupby("unit_id", sort=False)}


@lru_cache(maxsize=2)
def _committed(model: str, second: str) -> tuple[pd.DataFrame, dict, dict, dict]:
    u = score.load(model)
    other = score.load(second)[["unit_id", "hypothesis"]]
    try:
        other_probs = _probs(second)
    except FileNotFoundError:
        other_probs = {}
    return u, _probs(model), dict(zip(other["unit_id"], other["hypothesis"].fillna(""))), other_probs


def _other_tokens(text: str, probs: np.ndarray | None) -> tuple[list[str], np.ndarray | None]:
    """The second model's tokens, with its word probabilities when they line up with them."""
    toks, _ = hyp_tokens(text.split())
    if probs is not None and len(probs) == len(toks):
        return toks, probs
    return scoring_tokens(text), None


def _p_full(probs: np.ndarray | None, status: list[str], conf) -> np.ndarray:
    """Committed per-token probabilities skip wildcard-absorbed tokens; fill those (or everything, when the
    token sets differ) with the unit's confidence."""
    c = float(conf) if pd.notna(conf) else 0.5
    if probs is None or len(probs) != sum(st != "W" for st in status):
        return np.full(len(status), c)
    it = iter(probs)
    return np.array([next(it) if st != "W" else c for st in status])


def _role(row) -> str | None:
    """The speaker whose channel a turn came from; None for windows (both channels mixed) and Eka."""
    return row["roles"] if row.get("subset") != "window" and row.get("roles") in af.ROLES else None


def audio_columns(row: dict, n_words: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Audio features of a committed unit's tokens and gaps, at the same evenly spread times the
    training tables use. NaN (and ``two_ch`` 0) for Eka and for consultations without audio."""
    start = float(row["start"]) if pd.notna(row.get("start")) else 0.0
    dur = float(row["duration"]) if pd.notna(row.get("duration")) else 1.0
    end = float(row["end"]) if pd.notna(row.get("end")) else start + dur
    tr = af.track(row["consultation"]) if row.get("dataset") == "primock57" else None
    tpw = dur / max(n_words, 1)
    k = np.arange(n_words)
    return af.unit_features(tr, _role(row), start, end, start + k * tpw, start + (k + 1) * tpw,
                            start + np.arange(n_words + 1) * tpw, n_words)


def use_audio(cfg: dict) -> bool:
    return bool(cfg["verifier"].get("audio_features", False))


def unit_tables(model: str, second: str, w: Weights, sets: tuple[str, ...],
                audio: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Word, gap and unit tables (features, labels, evenly spread times) for the chosen sets; with
    ``audio`` also the two-channel speech-energy features."""
    u, probs, other_hyp, other_probs = _committed(model, second)
    W, G, U = [], [], []
    for row in u.to_dict("records"):
        s = _set(row)
        if s not in sets:
            continue
        disp = str(row.get("hypothesis") or "").split()
        hyp, _ = hyp_tokens(disp)
        lab = labels(scoring_tokens(row["reference"], reference=True), hyp, w)
        p_full = _p_full(probs.get(row["unit_id"]), lab["status"], row["conf"])
        unit = {**row, "kind": _kind(row)}
        other, op = _other_tokens(other_hyp.get(row["unit_id"], ""), other_probs.get(row["unit_id"]))
        wf, gf, _ = features(hyp, p_full, unit, other, w, op)
        if audio:
            aw, ag = audio_columns(row, len(hyp))
            wf, gf = pd.concat([wf, aw.drop(columns=[c for c in aw if c in wf])], axis=1), \
                pd.concat([gf, ag.drop(columns=[c for c in ag if c in gf])], axis=1)
        dur = float(row["duration"]) if pd.notna(row["duration"]) else 1.0
        tpw = dur / max(len(hyp), 1)
        group = row.get("consultation") if pd.notna(row.get("consultation")) else row.get("group_id")
        meta = {"unit_id": row["unit_id"], "set": s, "consultation": row.get("consultation"), "group": group}
        W.append(wf.assign(b=lab["word_b"], cr=lab["word_cr"], t0=np.arange(len(hyp)) * tpw,
                           t1=(np.arange(len(hyp)) + 1) * tpw, **meta))
        G.append(gf.assign(b=lab["gap_b"], cr=lab["gap_cr"], t0=np.arange(len(hyp) + 1) * tpw,
                           t1=np.arange(len(hyp) + 1) * tpw, **meta))
        U.append({**meta, "duration": dur, "conf": row["conf"], "ref_weight": lab["ref_weight"]})
    return pd.concat(W, ignore_index=True), pd.concat(G, ignore_index=True), pd.DataFrame(U)


# ---- fitting --------------------------------------------------------------------------------------

def _regressor(cfg: dict, params: dict | None = None, head: str | None = None) -> HistGradientBoostingRegressor:
    """Poisson gradient boosting with the config's shared settings, then the word or gap head's own
    (``verifier.model_words`` / ``verifier.model_gaps``, tuned by grouped CV on the training data)."""
    vc = cfg["verifier"]
    own = vc.get(f"model_{head}", {}) if head else {}
    return HistGradientBoostingRegressor(loss="poisson", random_state=cfg["seed"], **{**vc["model"], **own, **(params or {})})


def variant_features(variant: str) -> tuple[list[str], list[str]]:
    """Feature lists of each fitted variant.

    ``full`` (committed results: second transcript plus its word probabilities), ``second`` (a second
    transcript as text, as in live decoding), ``solo`` (whisper-small only) and ``v3`` (the previous
    feature set, kept for the comparison). A ``+audio`` suffix adds the two-channel audio features."""
    if variant.endswith("+audio"):
        wf, gf = variant_features(variant[:-len("+audio")])
        return wf + [f for f in af.WORD_FEATURES if f not in wf], gf + [f for f in af.GAP_FEATURES if f not in gf]
    second_w, second_g = SECOND_FEATURES_W + SECOND_P_FEATURES_W, SECOND_FEATURES_G + SECOND_P_FEATURES_G
    if variant == "full":
        return WORD_FEATURES + SECOND_P_FEATURES_W, GAP_FEATURES + SECOND_P_FEATURES_G
    if variant == "second":
        return list(WORD_FEATURES), list(GAP_FEATURES)
    if variant == "solo":
        return ([f for f in WORD_FEATURES if f not in second_w], [f for f in GAP_FEATURES if f not in second_g])
    if variant == "v3":
        return list(V3_WORD_FEATURES), list(V3_GAP_FEATURES)
    raise ValueError(variant)


def fit_variants(W: pd.DataFrame, G: pd.DataFrame, cfg: dict, variants=("full", "second", "solo")) -> dict:
    out = {}
    for name in variants:
        wf, gf = variant_features(name)
        out[name] = {"word_features": wf, "gap_features": gf,
                     "words": _regressor(cfg, head="words").fit(W[wf], W["b"]),
                     "gaps": _regressor(cfg, head="gaps").fit(G[gf], G["b"])}
    return out


def train(model: str = "small", second: str = "large-v3-turbo", cfg: dict | None = None,
          crossfit: bool = False) -> dict:
    """Fit the variants on the training splits, save them and the review curves, return metadata.

    ``crossfit`` also writes curves in which each PriMock57 test consultation is scored by a model that
    additionally trained on the other test consultations (5 folds by consultation): what a clinic gains by
    labelling some of its own consultations. The saved model never sees a test consultation."""
    cfg = cfg or load_config()
    w = Weights.from_config(cfg)
    audio = use_audio(cfg)
    W, G, _ = unit_tables(model, second, w, ("train",), audio=audio)
    committed = "full+audio" if audio else "full"
    fitted = fit_variants(W, G, cfg, ("full+audio", "full", "second", "solo") if audio else ("full", "second", "solo"))
    vc = cfg["verifier"]
    meta = {"model": model, "second": second, "weights": {c: w.of(c) for c in CATS},
            "train_units": int(W["unit_id"].nunique()), "train_words": int(len(W)), "train_gaps": int(len(G)),
            "hyperparameters": {"shared": vc["model"], "words": vc.get("model_words", {}), "gaps": vc.get("model_gaps", {})},
            "committed_variant": committed,
            "features": {"words": fitted[committed]["word_features"], "gaps": fitted[committed]["gap_features"]}}
    path = model_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"meta": meta, "variants": fitted, "weights": w, "committed_variant": committed}, path, compress=3)
    load.cache_clear()
    curves = write_curves(model, second, cfg, crossfit=crossfit)
    i60 = curves["x_seconds"].index(60) if 60 in curves["x_seconds"] else 12
    meta["curves"] = {k: {m: curves[k]["weighted"][m][i60] for m in curves[k]["weighted"]} for k in ("window", "turn")}
    if crossfit:
        meta["curves_crossfit"] = {k: {m: curves["crossfit"][k]["weighted"][m][i60] for m in curves["crossfit"][k]["weighted"]}
                                   for k in ("window", "turn")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


@dataclass
class Verifier:
    weights: Weights
    variants: dict
    meta: dict
    committed_variant: str = "second"

    def predict(self, words: pd.DataFrame, gaps: pd.DataFrame, variant: str) -> tuple[np.ndarray, np.ndarray]:
        v = self.variants[variant]
        sw = v["words"].predict(words[v["word_features"]]) if len(words) else np.zeros(0)
        sg = v["gaps"].predict(gaps[v["gap_features"]]) if len(gaps) else np.zeros(0)
        return sw, sg

    def score(self, words: list[dict], unit: dict, second_text: str | None = None) -> dict:
        """Score one decoded unit.

        ``words`` are faster-whisper words (``word``, ``start``, ``end``, ``probability``), ``unit`` the
        decoder statistics (``features.unit_features`` plus ``kind``), ``second_text`` another model's
        transcript of the same audio or None. Scores are expected weighted errors an item would fix."""
        disp = [str(x["word"]).strip() for x in words]
        hyp, owner = hyp_tokens(disp)
        wp = np.array([float(x.get("probability", 1.0)) for x in words])
        p = wp[owner] if owner else np.zeros(0)
        other = scoring_tokens(second_text) if second_text is not None else None
        wf, gf, sugg = features(hyp, p, unit, other, self.weights)
        sw, sg = self.predict(wf, gf, "second" if other is not None else "solo")
        crit = {owner[k] for k, c in enumerate(categories(hyp, self.weights)) if c == "crit"}
        best = np.zeros(len(disp))
        for k, s in enumerate(sw):
            best[owner[k]] = max(best[owner[k]], s)
        out_w = [{"text": d, "t0": float(x.get("start", 0.0)), "t1": float(x.get("end", 0.0)),
                  "p": float(x.get("probability", 1.0)), "score": round(float(best[i]), 4), "critical": i in crit}
                 for i, (d, x) in enumerate(zip(disp, words))]
        start = float(words[0].get("start", 0.0)) if words else float(unit.get("start", 0.0))
        gaps = _gaps(owner, sg, sugg, [x["t1"] for x in out_w], start)
        return {"words": out_w, "gaps": gaps}


def _gaps(owner: list[int], scores: np.ndarray, sugg: list[list[str]], word_ends: list[float], start: float,
          truth: np.ndarray | None = None) -> list[dict]:
    """Merge token gaps onto display positions: ``after`` = display word the gap follows (-1 = before all)."""
    merged: dict[int, dict] = {}
    for k in range(len(scores)):
        after = -1 if k == 0 else owner[k - 1]
        if k and k < len(owner) and owner[k] == owner[k - 1]:
            after = owner[k]                    # inside one display word ("I've" -> "i have"): attach to it
        g = merged.setdefault(after, {"after": after, "t": word_ends[after] if after >= 0 else start,
                                      "score": 0.0, "suggestion": []})
        g["score"] = max(g["score"], float(scores[k]))
        g["suggestion"] += sugg[k]
        if truth is not None:
            g["truth"] = round(g.get("truth", 0.0) + float(truth[k]), 2)
    return [{**g, "t": round(g["t"], 3), "score": round(g["score"], 4), "suggestion": " ".join(g["suggestion"])}
            for g in merged.values()]


@lru_cache(maxsize=2)
def load(model: str = "small") -> Verifier:
    path = model_path(model)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `asrshift verifier`")
    d = joblib.load(path)
    return Verifier(weights=d["weights"], variants=d["variants"], meta=d["meta"],
                    committed_variant=d.get("committed_variant", "second"))


# ---- consultations from the committed results -----------------------------------------------------

def _has_audio(consultation: str) -> bool:
    a = RAW_DIR / "primock57" / "audio"
    return (a / f"{consultation}_doctor.wav").exists() and (a / f"{consultation}_patient.wav").exists()


def list_consultations(model: str = "small") -> list[dict]:
    u = score.load(model)
    pm = u[u["dataset"] == "primock57"]
    out = []
    for c, g in pm.groupby("consultation"):
        turns, wins = g[g["subset"].str.startswith("turn")], g[g["subset"] == "window"]
        out.append({"id": c, "split": str(g["split"].iloc[0]), "duration": round(float(g["end"].max()), 2),
                    "n_turns": int(len(turns)), "n_windows": int(len(wins)), "has_audio": _has_audio(c)})
    return out


def _spread(words: list[str], start: float, end: float) -> list[tuple[float, float]]:
    """Word times spread by character length over the unit (the committed table has no timestamps)."""
    lens = [len(x) + 1 for x in words]
    tot, s0 = sum(lens) or 1, start + 0.1
    span, acc, out = max(0.2, end - 0.1 - s0), 0, []
    for n in lens:
        t0 = s0 + span * acc / tot
        acc += n
        out.append((round(t0, 3), round(s0 + span * acc / tot, 3)))
    return out


def consultation_payload(consultation: str, unit: str = "turn", model: str = "small",
                         second: str = "large-v3-turbo") -> dict:
    """Every turn (or window) of one consultation with words, verifier scores, gaps and the truth."""
    v = load(model)
    u, probs, other_hyp, other_probs = _committed(model, second)
    g = u[(u["consultation"] == consultation)
          & (u["subset"].str.startswith("turn") if unit == "turn" else u["subset"] == "window")]
    if g.empty:
        raise KeyError(consultation)
    w = v.weights
    out_units = []
    for row in g.sort_values("start").to_dict("records"):
        disp = str(row.get("hypothesis") or "").split()
        hyp, owner = hyp_tokens(disp)
        lab = labels(scoring_tokens(row["reference"], reference=True), hyp, w)
        p = _p_full(probs.get(row["unit_id"]), lab["status"], row["conf"])
        other, op = _other_tokens(other_hyp.get(row["unit_id"], ""), other_probs.get(row["unit_id"]))
        wf, gf, sugg = features(hyp, p, {**row, "kind": _kind(row)}, other, w, op)
        variant = v.committed_variant
        if variant.endswith("+audio"):
            if af.has_channels(consultation):
                aw, ag = audio_columns(row, len(hyp))
                wf, gf = pd.concat([wf, aw.drop(columns=[c for c in aw if c in wf])], axis=1), \
                    pd.concat([gf, ag.drop(columns=[c for c in ag if c in gf])], axis=1)
            else:
                variant = variant[:-len("+audio")]
        sw, sg = v.predict(wf, gf, variant)
        times = _spread(disp, float(row["start"]), float(row["end"]))
        cats = categories(hyp, w)
        dw = [{"text": d, "t0": t0, "t1": t1, "p": None, "score": 0.0, "critical": False, "wrong": None}
              for d, (t0, t1) in zip(disp, times)]
        for k, i in enumerate(owner):
            x = dw[i]
            x["p"] = round(float(p[k]) if x["p"] is None else min(x["p"], float(p[k])), 3)
            x["score"] = round(max(x["score"], float(sw[k])), 4)
            x["critical"] = x["critical"] or cats[k] == "crit"
            if lab["status"][k] != "W":
                x["wrong"] = max(x["wrong"] or 0, int(lab["status"][k] in ("S", "I")))
        ref_text = " ".join(strip_tags(_UNIN.sub(" [inaudible] ", str(row["reference"]))).split())
        out_units.append({"id": row["unit_id"], "role": row["roles"], "start": round(float(row["start"]), 3),
                          "end": round(float(row["end"]), 3), "reference": ref_text, "words": dw,
                          "gaps": _gaps(owner, sg, sugg, [t1 for _, t1 in times], float(row["start"]) + 0.1,
                                        truth=lab["gap_b"])})
    return {"id": consultation, "split": str(g["split"].iloc[0]), "unit": unit,
            "duration": round(float(g["end"].max()), 3), "units": out_units}


# ---- review curves: errors left against seconds of listening per consultation ----------------------

class _Cover:
    """Union of listened intervals inside one unit."""

    def __init__(self):
        self.iv: list[tuple[float, float]] = []
        self.length = 0.0

    def marginal(self, a: float, b: float) -> float:
        new = b - a
        for x, y in self.iv:
            if y > a and x < b:
                new -= min(b, y) - max(a, x)
        return max(new, 0.0)

    def add(self, a: float, b: float) -> None:
        keep = []
        for x, y in self.iv:
            if y < a or x > b:
                keep.append((x, y))
            else:
                a, b = min(a, x), max(b, y)
        self.iv = sorted(keep + [(a, b)])
        self.length = sum(y - x for x, y in self.iv)


def _caught(steps: list[tuple[float, float, float]], budgets: list[float]) -> np.ndarray:
    """(cost, weighted, critical) steps in review order -> what is caught within each budget."""
    if not steps:
        return np.zeros((len(budgets), 2))
    s = np.array(steps)
    c, bw, bc = np.cumsum(s[:, 0]), np.cumsum(s[:, 1]), np.cumsum(s[:, 2])
    k = np.searchsorted(c, np.asarray(budgets) + 1e-9, side="right")
    return np.column_stack([np.where(k > 0, bw[k - 1], 0.0), np.where(k > 0, bc[k - 1], 0.0)])


def _queue(it: dict, budgets: list[float]) -> np.ndarray:
    """Items in a fixed order (predicted benefit per standalone second), overlapping clips charged once."""
    order = np.argsort(-(it["pred"] / (it["b"] - it["a"])), kind="stable")
    cov, steps = {}, []
    for i in order:
        c = cov.setdefault(it["unit"][i], _Cover())
        steps.append((c.marginal(it["a"][i], it["b"][i]), it["true"][i], it["crit"][i]))
        c.add(it["a"][i], it["b"][i])
    return _caught(steps, budgets)


def _hybrid(it: dict, durs: dict, budgets: list[float], scale: float = 1.0) -> np.ndarray:
    """Items and whole units compete on predicted benefit per marginal second (greedy, lazy updates).

    A whole unit's predicted benefit (its remaining items' predictions) is multiplied by ``scale``.
    Calibrating it (observed / predicted on out-of-fold training data, about 1.1-1.2) did not help, so
    the curves use 1."""
    n = len(it["unit"])
    done = np.zeros(n, bool)
    by_unit: dict[str, list[int]] = {}
    for i, x in enumerate(it["unit"]):
        by_unit.setdefault(x, []).append(i)
    rem = {x: float(it["pred"][ix].sum()) for x, ix in by_unit.items()}
    cov, closed = {x: _Cover() for x in by_unit}, set()
    heap = [(-(it["pred"][i] / (it["b"][i] - it["a"][i])), i) for i in range(n)]
    heapq.heapify(heap)
    steps, spent = [], 0.0
    while spent < max(budgets) + 1e-9:
        item = None
        while heap:
            neg, i = heap[0]
            if done[i] or it["unit"][i] in closed:
                heapq.heappop(heap)
                continue
            mc = cov[it["unit"][i]].marginal(it["a"][i], it["b"][i])
            r = it["pred"][i] / max(mc, 1e-3)
            if abs(-neg - r) > 1e-9:
                heapq.heapreplace(heap, (-r, i))
                continue
            item = (r, i, mc)
            break
        whole = None
        for x in by_unit:
            rest = durs[x] - cov[x].length
            if x in closed or rest <= 1e-6:
                continue
            if whole is None or scale * rem[x] / rest > whole[0]:
                whole = (scale * rem[x] / rest, x, rest)
        if item is None and whole is None:
            break
        if whole is not None and (item is None or whole[0] > item[0]):
            _, x, rest = whole
            ix = [i for i in by_unit[x] if not done[i]]
            steps.append((rest, float(it["true"][ix].sum()), float(it["crit"][ix].sum())))
            done[ix] = True
            closed.add(x)
            spent += rest
        else:
            _, i, mc = item
            heapq.heappop(heap)
            x = it["unit"][i]
            cov[x].add(it["a"][i], it["b"][i])
            done[i] = True
            rem[x] -= it["pred"][i]
            steps.append((mc, float(it["true"][i]), float(it["crit"][i])))
            spent += mc
    return _caught(steps, budgets)


METHODS = ("window_review", "whisper_conf", "verifier", "hybrid", "perfect")


def curve_parts(W: pd.DataFrame, G: pd.DataFrame, U: pd.DataFrame, pad: float, budgets: list[float],
                scale: float = 1.0) -> dict:
    """Per consultation: weighted and critical errors caught within each budget, by method.

    ``W`` and ``G`` carry a ``pred`` column (the verifier's expected weighted error). Returns
    ``{"consultations": [...], "total": (n, 2) array, method: (n, len(budgets), 2) array}``."""
    durs = dict(zip(U["unit_id"], U["duration"]))
    uw = W.groupby("unit_id")["b"].sum().add(G.groupby("unit_id")["b"].sum(), fill_value=0)
    uc = W.groupby("unit_id")["cr"].sum().add(G.groupby("unit_id")["cr"].sum(), fill_value=0)
    Wg, Gg = dict(tuple(W.groupby("consultation"))), dict(tuple(G.groupby("consultation")))
    cons = sorted(U["consultation"].unique())
    out = {"consultations": cons, "total": np.zeros((len(cons), 2)),
           **{m: np.zeros((len(cons), len(budgets), 2)) for m in METHODS}}
    for n, (c, uc_) in enumerate(sorted(U.groupby("consultation"), key=lambda x: x[0])):
        uc_ = uc_.sort_values("conf")
        out["window_review"][n] = _caught([(r.duration, uw.get(r.unit_id, 0.0), uc.get(r.unit_id, 0.0))
                                           for r in uc_.itertuples()], budgets)
        Wc, Gc = Wg.get(c, W.iloc[:0]), Gg.get(c, G.iloc[:0])
        out["total"][n] = [Wc["b"].sum() + Gc["b"].sum(), Wc["cr"].sum() + Gc["cr"].sum()]
        unit_ids = np.concatenate([Wc["unit_id"].to_numpy(), Gc["unit_id"].to_numpy()])
        dur = np.array([durs[x] for x in unit_ids])
        a = np.maximum(np.concatenate([Wc["t0"], Gc["t0"]]) - pad, 0)
        b = np.maximum(np.minimum(np.concatenate([Wc["t1"], Gc["t1"]]) + pad, dur), a + 1e-3)
        it = {"unit": unit_ids, "a": a, "b": b, "pred": np.concatenate([Wc["pred"], Gc["pred"]]),
              "true": np.concatenate([Wc["b"], Gc["b"]]), "crit": np.concatenate([Wc["cr"], Gc["cr"]])}
        out["verifier"][n] = _queue(it, budgets)
        out["hybrid"][n] = _hybrid(it, durs, budgets, scale)
        out["perfect"][n] = _queue({**it, "pred": it["true"] + 1e-6}, budgets)
        nw = len(Wc)
        out["whisper_conf"][n] = _queue({k: x[:nw] for k, x in {**it, "pred": np.concatenate(
            [1 - Wc["p"].to_numpy(), np.zeros(len(Gc))])}.items()}, budgets)
    return out


def _summarise(parts: dict) -> dict:
    tot = parts["total"].sum(0)
    return {"weighted": {m: [round(float(1 - x / tot[0]), 4) for x in parts[m].sum(0)[:, 0]] for m in METHODS},
            "critical": {m: [round(float(1 - x / max(tot[1], 1)), 4) for x in parts[m].sum(0)[:, 1]] for m in METHODS}}


def _crossfit_preds(v: "Verifier", train_w: pd.DataFrame, train_g: pd.DataFrame, tests: dict, cfg: dict,
                    n_folds: int = 5) -> dict:
    """Out-of-fold predictions for the PriMock57 test consultations: fold k is scored by a model trained
    on the training splits plus every test consultation outside fold k (windows and turns alike)."""
    cons = sorted(set().union(*(set(W["consultation"]) for W, _, _ in tests.values())))
    rng = np.random.default_rng(cfg["seed"])
    fold = dict(zip(rng.permutation(cons), np.arange(len(cons)) % n_folds))
    wf, gf = variant_features(v.committed_variant)
    preds = {k: (np.zeros(len(W)), np.zeros(len(G))) for k, (W, G, _) in tests.items()}
    for f in range(n_folds):
        extra_w = pd.concat([W[W["consultation"].map(fold) != f] for W, _, _ in tests.values()])
        extra_g = pd.concat([G[G["consultation"].map(fold) != f] for _, G, _ in tests.values()])
        Wtr, Gtr = pd.concat([train_w, extra_w]), pd.concat([train_g, extra_g])
        mw, mg = _regressor(cfg, head="words").fit(Wtr[wf], Wtr["b"]), _regressor(cfg, head="gaps").fit(Gtr[gf], Gtr["b"])
        for k, (W, G, _) in tests.items():
            iw, ig = (W["consultation"].map(fold) == f).to_numpy(), (G["consultation"].map(fold) == f).to_numpy()
            if iw.any():
                preds[k][0][iw] = mw.predict(W.loc[iw, wf])
            if ig.any():
                preds[k][1][ig] = mg.predict(G.loc[ig, gf])
    return preds


def write_curves(model: str = "small", second: str = "large-v3-turbo", cfg: dict | None = None,
                 crossfit: bool = False) -> dict:
    """Share of weighted / critical errors left after B seconds of listening per test consultation."""
    cfg = cfg or load_config()
    vc = cfg["verifier"]
    pad, budgets = float(vc["clip_pad_s"]), list(range(0, int(vc["curve_max_s"]) + 1, int(vc["curve_step_s"])))
    v = load(model)
    out: dict = {"x_seconds": budgets, "meta": {"model": model, "second": second, "clip_pad_s": pad,
                                                "variant": v.committed_variant}}
    audio = v.committed_variant.endswith("+audio")
    tests = {key: unit_tables(model, second, v.weights, (s,), audio=audio)
             for key, s in (("window", "pm_window_test"), ("turn", "pm_turn_test"))}
    for key, (W, G, U) in tests.items():
        sw, sg = v.predict(W, G, v.committed_variant)
        parts = curve_parts(W.assign(pred=sw), G.assign(pred=sg), U, pad, budgets)
        out[key] = _summarise(parts)
        out["meta"][key] = {"consultations": int(U["consultation"].nunique()), "units": int(len(U)),
                            "mean_audio_s": round(float(U.groupby("consultation")["duration"].sum().mean()), 1),
                            "weighted_errors": round(float(parts["total"][:, 0].sum()), 1),
                            "critical_errors": int(parts["total"][:, 1].sum())}
    if crossfit:
        Wtr, Gtr, _ = unit_tables(model, second, v.weights, ("train",), audio=audio)
        preds = _crossfit_preds(v, Wtr, Gtr, tests, cfg)
        out["crossfit"] = {"note": "each test consultation scored by a model that also trained on the other "
                                   "test consultations (5 folds by consultation); not the deployed model"}
        for key, (W, G, U) in tests.items():
            pw, pg = preds[key]
            out["crossfit"][key] = _summarise(curve_parts(W.assign(pred=pw), G.assign(pred=pg), U, pad, budgets))
    path = curves_path(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def review_curves(model: str = "small") -> dict:
    path = curves_path(model)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `asrshift verifier`")
    return json.loads(path.read_text())


def unit_stats(rec: dict, kind: str = "turn") -> dict:
    """Decoder statistics for ``Verifier.score`` from a decoder record (segments, words, text, audio_s)."""
    from asrshift.features import unit_features

    f = unit_features(rec)
    f["kind"] = KIND[kind]
    return f
