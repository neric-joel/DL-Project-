"""Transcript Review Console: a local web app over the study's results.

    uvicorn app.server:app --port 8765          (from the repository root)

Everything shown comes from the pipeline's own outputs: per-unit scores (results/units), policy
scores and operating points (results/tables), fitted policies (results/models). Audio playback and
"Run Whisper live" also need the downloaded datasets (data/) and, for live decoding, the model.
"""

from __future__ import annotations

import io
import json
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from asrshift import review, score
from asrshift.align import align, ops
from asrshift.paths import FIGURES_DIR, INTERIM_DIR, RESULTS_DIR
from asrshift.textnorm import scoring_tokens

MODEL = "small"
TABLES = RESULTS_DIR / "tables" / MODEL / "err"
STATIC = Path(__file__).parent / "static"

EVAL_SETS = {
    "eka_test": "Eka test · short medical clips",
    "pm_turn_test": "PriMock57 turns · one speaker",
    "pm_window_test": "PriMock57 windows · conversation",
}
POLICIES = {
    "P1": "P1 · Raw confidence",
    "P3": "P3 · Calibrated on Eka (frozen)",
    "P4": "P4 · Recalibrated on PriMock57",
}
POINTS = {
    "budget_0.10": "Review budget 10%",
    "risk_0.20": "Accepted error ≤ 20%",
    "plugin_0.20": "Plug-in rule, 20%",
}

app = FastAPI(title="Transcript Review Console")


@lru_cache(maxsize=1)
def units() -> pd.DataFrame:
    return score.load(MODEL).set_index("unit_id")


@lru_cache(maxsize=1)
def unit_scores() -> pd.DataFrame:
    return pd.read_csv(TABLES / "unit_scores.csv.gz")


@lru_cache(maxsize=1)
def operating_points() -> pd.DataFrame:
    return pd.read_csv(TABLES / "operating_points.csv")


@lru_cache(maxsize=1)
def manifest() -> pd.DataFrame | None:
    p = INTERIM_DIR / "manifest.parquet"
    return pd.read_parquet(p).set_index("unit_id") if p.exists() else None


@lru_cache(maxsize=1)
def decoder_outputs() -> dict:
    """unit_id -> Whisper words with probabilities (local inference output, if present)."""
    from asrshift import infer

    out = {}
    for path in infer.all_output_paths(MODEL):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[rec["unit_id"]] = rec.get("words") or []
    return out


def _scores(eval_set: str) -> pd.DataFrame:
    s = unit_scores()
    return s[s["eval"] == eval_set]


def _op(eval_set: str, policy: str, point: str) -> pd.Series:
    o = operating_points()
    row = o[(o["eval"] == eval_set) & (o["policy"] == policy) & (o["point"] == point)]
    if row.empty:
        raise HTTPException(404, f"{policy} has no operating point {point} on {eval_set}")
    return row.iloc[0]


def _decide(risk: np.ndarray, point: str, threshold: float) -> np.ndarray:
    """Same rule as asrshift.evaluate: budgets review when risk >= t, risk targets and plug-in when risk > t."""
    return risk >= threshold if point.startswith("budget") else risk > threshold


def _num(x):
    """JSON-safe float: NaN and +-inf (an infeasible threshold) become None."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _check(eval_set: str, policy: str, point: str) -> None:
    if eval_set not in EVAL_SETS or policy not in POLICIES or point not in POINTS:
        raise HTTPException(422, "unknown evaluation set, policy or operating point")


def _note(policy: str, point: str) -> str:
    if policy == "P4" and not point.startswith("plugin"):
        return ("Platt recalibration keeps P3's ranking, so at this operating point P4 is P3's score with the "
                "threshold re-tuned on 10 PriMock57 consultations.")
    if point.startswith("plugin"):
        return "Plug-in rule: accepts transcripts while the model's own predicted error rate stays at 20%. Needs no labels."
    return ""


@app.get("/api/meta")
def meta():
    o = operating_points()
    available = {f"{r.eval}|{r.policy}|{r.point}": bool(r.feasible) for r in o.itertuples()
                 if r.policy in POLICIES and r.point in POINTS}
    return {"eval_sets": EVAL_SETS, "policies": POLICIES, "points": POINTS, "available": available,
            "live": manifest() is not None}


@app.get("/api/summary")
def summary(eval_set: str = Query(...), policy: str = Query(...), point: str = Query(...)):
    _check(eval_set, policy, point)
    r = _op(eval_set, policy, point)
    d = units()
    sc = _scores(eval_set)
    sub = d.loc[sc["unit_id"]]
    review_mask = _decide(sc[policy].to_numpy(float), point, r["threshold"])
    acc = ~review_mask
    crit = sub["crit_err"].to_numpy(int)
    return {
        "units": int(len(sub)),
        "review_rate": _num(r["review_rate"]),
        "review_rate_ci": [_num(r["review_rate_lo"]), _num(r["review_rate_hi"])],
        "err_caught": _num(r["err_caught"]),
        "accepted_err_rate": _num(r["accepted_err_rate"]),
        "accepted_err_ci": [_num(r["accepted_err_rate_lo"]), _num(r["accepted_err_rate_hi"])],
        "accepted_crit_rate": _num(crit[acc].mean()) if acc.any() else None,
        "promised_review": _num(r["target"]) if point.startswith("budget") else None,
        "promised_accepted_err": _num(r["target"]) if not point.startswith("budget") else None,
        "base_rate": _num(sub["err"].mean()),
        "feasible": bool(r["feasible"]),
        "note": _note(policy, point),
    }


@app.get("/api/queue")
def queue(eval_set: str, policy: str, point: str, show: str = "all", limit: int = Query(200, ge=1, le=1000)):
    _check(eval_set, policy, point)
    if show not in {"all", "review", "accepted_errors", "accepted_critical"}:
        raise HTTPException(422, "unknown filter")
    r = _op(eval_set, policy, point)
    sc = _scores(eval_set)
    d = units().loc[sc["unit_id"]]
    risk = sc[policy].to_numpy(float)
    rev = _decide(risk, point, r["threshold"])
    df = pd.DataFrame({
        "unit_id": sc["unit_id"].to_numpy(), "risk": risk, "decision": np.where(rev, "review", "accept"),
        "wer": d["wer"].to_numpy(float), "err": d["err"].to_numpy(int), "crit_err": d["crit_err"].to_numpy(int),
        "conf": d["conf"].to_numpy(float), "duration": d["duration"].to_numpy(float),
        "n_ref": d["n_ref"].to_numpy(int), "hypothesis": d["hypothesis"].fillna("").str.slice(0, 140).to_numpy(),
    })
    if show == "accepted_errors":
        df = df[(df["decision"] == "accept") & (df["err"] == 1)]
    elif show == "review":
        df = df[df["decision"] == "review"]
    elif show == "accepted_critical":
        df = df[(df["decision"] == "accept") & (df["crit_err"] == 1)]
    df = df.sort_values("risk", ascending=False) if show != "accepted_errors" else df.sort_values("risk")
    return {"total": int(len(df)), "rows": df.head(limit).round(4).to_dict("records")}


@app.get("/api/unit/{unit_id}")
def unit(unit_id: str, eval_set: str, point: str = "budget_0.10"):
    _check(eval_set, "P3", point)
    d = units()
    if unit_id not in d.index:
        raise HTTPException(404, "unknown unit")
    u = d.loc[unit_id]
    ref = scoring_tokens(u["reference"], reference=True)
    hyp = scoring_tokens(u["hypothesis"] if isinstance(u["hypothesis"], str) else "")
    a = align(ref, hyp)
    sc = _scores(eval_set)[lambda x: x["unit_id"] == unit_id]
    risks = {}
    for pol in POLICIES:
        if pol in sc and sc[pol].notna().any():
            try:
                r = _op(eval_set, pol, point)
            except HTTPException:
                continue
            v = float(sc[pol].iloc[0])
            risks[pol] = {"risk": _num(v), "threshold": _num(r["threshold"]), "feasible": bool(r["feasible"]),
                          "label": ("P4 · P3 score, threshold re-tuned on PriMock57"
                                    if pol == "P4" and not point.startswith("plugin") else POLICIES[pol]),
                          "decision": "review" if _decide(np.array([v]), point, r["threshold"])[0] else "accept"}
    return {
        "in_set": not sc.empty,
        "unit_id": unit_id, "dataset": u["dataset"], "subset": u["subset"], "duration": float(u["duration"]),
        "reference": u["reference"], "hypothesis": u["hypothesis"] if isinstance(u["hypothesis"], str) else "",
        "wer": float(u["wer"]), "err": int(u["err"]), "conf": float(u["conf"]),
        "errors": {"sub": int(u["sub"]), "del": int(u["del"]), "ins": int(u["ins"]), "n_ref": int(u["n_ref"])},
        "critical": {"missed": int(u["n_crit_missed"]), "false": int(u["n_crit_false"])},
        "ops": [{"op": o, "ref": r_, "hyp": h} for o, r_, h in ops(a)],
        "words": [{"w": w["word"], "p": round(w["probability"], 3)} for w in decoder_outputs().get(unit_id, [])],
        "risks": risks, "audio": manifest() is not None,
    }


@app.get("/api/audio/{unit_id}")
def audio(unit_id: str):
    m = manifest()
    if m is None or unit_id not in m.index:
        raise HTTPException(404, "audio not available (run `asrshift download` and `asrshift prepare`)")
    from asrshift.audio import load_unit_audio

    row = m.loc[unit_id].to_dict() | {"unit_id": unit_id}
    buf = io.BytesIO()
    sf.write(buf, load_unit_audio(row), 16_000, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


_model_lock = threading.Lock()
_decode_lock = threading.Lock()  # one GPU decode at a time
_model = {}


def _whisper():
    with _model_lock:
        if "m" not in _model:
            from asrshift import infer
            from asrshift.config import load_config

            cfg = load_config()
            _model["cfg"] = cfg
            _model["m"] = infer.load_model(cfg["asr"])[0]
        return _model["m"], _model["cfg"]


@app.post("/api/transcribe/{unit_id}")
def transcribe(unit_id: str):
    """Decode the unit again, right now, and score it with the exported policies."""
    import time

    from asrshift import infer
    from asrshift.audio import load_unit_audio

    m = manifest()
    if m is None or unit_id not in m.index:
        raise HTTPException(404, "audio not available")
    model, cfg = _whisper()
    row = m.loc[unit_id].to_dict() | {"unit_id": unit_id}
    audio_arr = load_unit_audio(row)
    with _decode_lock:
        t0 = time.perf_counter()
        rec = infer.transcribe(model, audio_arr, cfg["asr"]["decode"])
        elapsed = time.perf_counter() - t0
    domain = "eka" if row["dataset"] == "eka" else ("turn" if str(row["subset"]).startswith("turn") else "window")
    verdict = review.assess(rec, domain=domain, model=MODEL)
    risk = {k: _num(v) for k, v in verdict["risk"].items()}
    return {"text": rec["text"], "seconds": round(elapsed, 2), "audio_s": rec["audio_s"],
            "words": [{"w": w["word"], "p": round(w["probability"], 3)} for w in rec["words"]],
            "confidence": _num(verdict["confidence"]), "risk": risk, "decisions": verdict["decisions"]}


@app.post("/api/warmup")
def warmup():
    _whisper()
    return {"ok": True}


@app.get("/figures/{name}")
def figure(name: str):
    p = (FIGURES_DIR / name).resolve()
    if p.parent != FIGURES_DIR.resolve() or not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
