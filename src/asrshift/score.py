"""Score every transcript against its reference and attach labels and features.

Produces one row per unit (``data/interim/units_<model>.parquet``) with WER components, clinically
critical token counts (numbers, negations), Eka entity accuracy, the binary labels used for
calibration, and the confidence features. A word-level table (``words_<model>.parquet``) pairs each
hypothesis word's probability with whether it was aligned as correct.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from asrshift import data_eka, infer, prepare, terms
from asrshift.align import align
from asrshift.features import unit_features
from asrshift.paths import INTERIM_DIR, RESULTS_DIR
from asrshift.textnorm import WILDCARD, collapse_repeats, reference_tokens, scoring_tokens, strip_tags, tokens

NEGATION = frozenset({
    "no", "not", "never", "nothing", "none", "nobody", "neither", "nor", "without", "nil",
    "deny", "denies", "denied",
})
_DIGIT = re.compile(r"\d")
_ORDINAL = re.compile(r"^\d+(st|nd|rd|th)$")  # the normaliser turns "first"/"second" into "1st"/"2nd"
NUMBER_WORDS = frozenset({"once", "twice", "thrice", "half", "od", "bd", "tds", "qds", "prn"})
# The normaliser writes the digit 1 as "one"; it counts as a number only before one of these.
ONE_CONTEXT = frozenset({
    "mg", "mcg", "ml", "kg", "g", "gram", "grams", "tablet", "tablets", "capsule", "capsules", "puff",
    "puffs", "drop", "drops", "dose", "doses", "day", "days", "week", "weeks", "month", "months",
    "year", "years", "hour", "hours", "minute", "minutes", "time", "times", "daily",
})


def is_number(toks: list[str], i: int) -> bool:
    t = toks[i]
    if t == WILDCARD or _ORDINAL.match(t):
        return False
    if _DIGIT.search(t) or t in NUMBER_WORDS:
        return True
    return t == "one" and i + 1 < len(toks) and toks[i + 1] in ONE_CONTEXT


def is_critical(toks: list[str], i: int) -> bool:
    return toks[i] in NEGATION or is_number(toks, i)

LABELS = {"err": 0.10, "any_err": 0.0, "severe_err": 0.30}
PRIMARY_LABEL = "err"


def units_path(model: str):
    return INTERIM_DIR / f"units_{model.replace('/', '_')}.parquet"


def words_path(model: str):
    """Word-level table used by evaluation: the committed copy, so every route (local, Docker, Colab)
    evaluates exactly the same numbers. Falls back to the local intermediate file."""
    committed = RESULTS_DIR / "units" / f"words_{model.replace('/', '_')}.parquet"
    return committed if committed.exists() else INTERIM_DIR / f"words_{model.replace('/', '_')}.parquet"


def _find(seq: list[str], sub: list[str], occurrence: int) -> int:
    """Start index of the ``occurrence``-th (0-based) contiguous match of ``sub`` in ``seq``, or -1."""
    if not sub:
        return -1
    seen = 0
    for i in range(len(seq) - len(sub) + 1):
        if seq[i:i + len(sub)] == sub:
            if seen == occurrence:
                return i
            seen += 1
    return -1


def _contains(seq: list[str], sub: list[str]) -> bool:
    return _find(seq, sub, 0) >= 0


def _occurrences(seq: list[str], sub: list[str]) -> list[int]:
    return [i for i in range(len(seq) - len(sub) + 1) if seq[i:i + len(sub)] == sub] if sub else []


def entity_scores(raw_entities: str, ref: list[str], ref_status: list[str], hyp: list[str],
                  ref_text: str = "") -> dict:
    """Eka entities: correct if all of the entity's reference tokens are aligned as correct.

    Duplicate annotations (same text and span) are counted once. Each entity is located in the
    normalised reference at the occurrence nearest to its annotated character offset (the number of
    normalised tokens before that offset), and no occurrence is used twice. An entity that cannot be
    located in the normalised reference (normalisation can merge it with neighbouring words) is not
    scored; it is counted in ``n_ent_unmapped`` instead.
    """
    out: dict = {"n_ent": 0, "n_ent_correct": 0, "n_ent_unmapped": 0}
    used: set[int] = set()
    for ent in data_eka.parse_entities(raw_entities):
        etoks = collapse_repeats(tokens(ent["text"]))[0]
        if not etoks:
            continue
        cands = [p for p in _occurrences(ref, etoks) if p not in used]
        if not cands:
            out["n_ent_unmapped"] += 1
            continue
        if ent["start"] is not None and ref_text:
            expected = len(tokens(ref_text[: ent["start"]]))
            pos = min(cands, key=lambda p: (abs(p - expected), p))
        else:
            pos = cands[0]
        used.add(pos)
        ok = all(s == "C" for s in ref_status[pos:pos + len(etoks)])
        t = ent["type"]
        out["n_ent"] += 1
        out["n_ent_correct"] += int(ok)
        out[f"n_ent_{t}"] = out.get(f"n_ent_{t}", 0) + 1
        out[f"n_ent_{t}_correct"] = out.get(f"n_ent_{t}_correct", 0) + int(ok)
    return out


def _critical(ref: list[str], ref_status: list[str], hyp: list[str], hyp_status: list[str]) -> dict:
    """Numbers and negations. Reference side: recognised or missed. Hypothesis side: false alarms
    (substituted or inserted critical tokens, e.g. a wrong dose or a hallucinated "not")."""
    out = {}
    for kind, pred in (("num", is_number), ("neg", lambda toks, i: toks[i] in NEGATION)):
        ref_st = [s for i, s in enumerate(ref_status) if pred(ref, i)]
        hyp_st = [s for i, s in enumerate(hyp_status) if s != "W" and pred(hyp, i)]
        out[f"n_{kind}"] = len(ref_st)
        out[f"n_{kind}_correct"] = ref_st.count("C")
        out[f"n_{kind}_hyp"] = len(hyp_st)
        out[f"n_{kind}_false"] = sum(s in ("S", "I") for s in hyp_st)
    out["n_crit_missed"] = (out["n_num"] - out["n_num_correct"]) + (out["n_neg"] - out["n_neg_correct"])
    out["n_crit_false"] = out["n_num_false"] + out["n_neg_false"]
    # the digit-only definition of the first protocol draft, kept for transparency
    digit = [s for t, s in zip(ref, ref_status) if t != WILDCARD and _DIGIT.search(t)]
    out["n_digit"], out["n_digit_correct"] = len(digit), digit.count("C")
    return out


def _word_level(rec: dict, ref: list[str]) -> list[tuple[str, float, str, int]]:
    """(token, probability, status, is_critical) for each normalised hypothesis token.

    Each Whisper word is normalised on its own and its probability copied to every token it yields.
    This can differ from normalising the whole text (spoken "twenty five" -> "20 5" vs "25"), so the
    unit-level scores always come from the whole-text alignment.
    """
    toks, probs = [], []
    for w in rec.get("words") or []:
        for t in tokens(w["word"]):
            toks.append(t)
            probs.append(w["probability"])
    kept, idx = collapse_repeats(toks)
    a = align(ref, kept)
    return [(kept[j], probs[i], s, int(is_critical(kept, j)))
            for j, (i, s) in enumerate(zip(idx, a.hyp_status)) if s != "W"]


def score_unit(row: dict, rec: dict, lexicon: pd.DataFrame | None = None) -> tuple[dict, list[tuple[float, int]]]:
    ref = scoring_tokens(row["reference"], reference=True)
    hyp = scoring_tokens(rec.get("text", ""))
    a = align(ref, hyp)
    # The same score without collapsing repetitions, reported for transparency only.
    raw = align(reference_tokens(row["reference"]), tokens(strip_tags(rec.get("text", ""))))
    out = {
        "n_ref": a.n_ref, "n_hyp": len(hyp), "hits": a.hits, "sub": a.substitutions, "del": a.deletions,
        "ins": a.insertions, "absorbed": a.absorbed, "errors": a.errors, "wer": a.wer,
        "n_ref_nocollapse": raw.n_ref, "errors_nocollapse": raw.errors,
        "n_wildcards": len(a.wild_runs), "max_wild_run": max(a.wild_runs, default=0), "wild_excess": a.wild_excess(3),
        "hypothesis": rec.get("text", ""),
    }
    out.update(_critical(ref, a.ref_status, hyp, a.hyp_status))
    if row["dataset"] == "eka":
        out.update(entity_scores(row.get("medical_entities") or "[]", ref, a.ref_status, hyp, row["reference"]))
    if lexicon is not None:  # the same lexicon and matcher on both datasets (shared vocabulary)
        out.update(terms.term_scores(ref, a.ref_status, lexicon))
    out.update(unit_features(rec))
    words = _word_level(rec, ref)
    # agreement check between the per-word (word-level analysis) and whole-text (WER) alignments
    out["wl_tokens"] = len(words)
    out["wl_wrong"] = sum(s in ("S", "I") for _, _, s, _ in words)
    return out, words


def run(cfg: dict, model: str | None = None) -> pd.DataFrame:
    model = model or cfg["asr"]["model"]
    manifest = prepare.load()
    manifest = manifest[manifest["dropped_reason"].fillna("") == ""]
    outputs = infer.load_outputs(model).set_index("unit_id")
    missing = set(manifest["unit_id"]) - set(outputs.index)
    if missing:
        print(f"warning: {len(missing)} units have no ASR output yet and are skipped")
    rows, word_rows = [], []
    lexicon = terms.load()
    for row in tqdm(manifest.to_dict("records"), desc="score", mininterval=10):
        if row["unit_id"] not in outputs.index:
            continue
        rec = outputs.loc[row["unit_id"]].to_dict()
        scored, words = score_unit(row, rec, lexicon)
        if scored["n_ref"] == 0:
            continue
        rows.append({**row, **scored, "decode_s": rec.get("decode_s")})
        word_rows.extend((row["unit_id"], t, p, s, int(s == "C"), c) for t, p, s, c in words)
    df = pd.DataFrame(rows)
    for name, thr in LABELS.items():
        df[name] = (df["wer"] > thr).astype(int)
    df["crit_err"] = ((df["n_crit_missed"] + df["n_crit_false"]) > 0).astype(int)
    # upper bound with each wildcard capped at 3 absorbed words (Amendment 2)
    df["err_c3"] = ((df["errors"] + df["wild_excess"]) / df["n_ref"] > LABELS["err"]).astype(int)
    if "n_ent" in df:
        df["ent_err"] = np.where(df["n_ent"].fillna(0) > 0, (df["n_ent_correct"] < df["n_ent"]).astype(float), np.nan)
    df["domain"] = np.where(df["dataset"] == "eka", "eka", "primock57_" + df["subset"].str.replace("turn_.*", "turn", regex=True))
    df.to_parquet(units_path(model), index=False)
    words = pd.DataFrame(word_rows, columns=["unit_id", "token", "probability", "status", "correct", "critical"])
    words.to_parquet(INTERIM_DIR / f"words_{model.replace('/', '_')}.parquet", index=False)
    _export(df, words, model)
    return df


def _export(df: pd.DataFrame, words: pd.DataFrame, model: str) -> None:
    """Compact per-unit and per-word tables committed with the repo, so evaluation, figures and the
    report can be rerun without the audio or a GPU."""
    private = {"speaker", "group_key"}  # raw speaker IDs (some are e-mail addresses) stay local
    keep = [c for c in df.columns if c not in {"medical_entities", "md5_text", "dropped_reason", "audio_source"} | private]
    out = RESULTS_DIR / "units"
    out.mkdir(parents=True, exist_ok=True)
    tag = model.replace("/", "_")
    df[keep].to_csv(out / f"units_{tag}.csv.gz", index=False, float_format="%.10g")
    compact = words[["unit_id", "probability", "correct", "critical"]].astype(
        {"probability": "float32", "correct": "int8", "critical": "int8"})
    compact.to_parquet(out / f"words_{tag}.parquet", index=False, compression="zstd")


def load(model: str) -> pd.DataFrame:
    """Per-unit table used by evaluation: the committed CSV (identical inputs on every route), else the
    local parquet intermediate."""
    committed = RESULTS_DIR / "units" / f"units_{model.replace('/', '_')}.csv.gz"
    if committed.exists():
        return pd.read_csv(committed, low_memory=False)
    return pd.read_parquet(units_path(model))


def overview(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["domain", "split"])
    return pd.DataFrame({
        "units": g.size(),
        "wer": g.apply(lambda x: x["errors"].sum() / x["n_ref"].sum()),
        "err_rate": g["err"].mean(),
        "any_err_rate": g["any_err"].mean(),
        "mean_conf": g["conf"].mean(),
    }).round(3)
