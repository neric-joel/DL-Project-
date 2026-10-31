"""Transcript Review Console: a local web app over the study's results.

    uvicorn app.server:app --port 8765          (from the repository root)

The Second Listen UI is served at "/" and uses the /api/v2 endpoints (trained verifier, whole
consultations, live decoding as server-sent events); the original console lives at "/classic/".

Everything shown comes from the pipeline's own outputs: per-unit scores (results/units), policy
scores and operating points (results/tables), fitted policies (results/models). Audio playback and
"Run Whisper live" also need the downloaded datasets (data/) and, for live decoding, the model.
"""

from __future__ import annotations

import io
import json
import re
import threading
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from asrshift import review, score
from asrshift.align import align, ops
from asrshift.paths import DATA_DIR, FIGURES_DIR, INTERIM_DIR, REPO_ROOT, RESULTS_DIR
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
    # the review queue is riskiest-first; mistakes that slipped through are most-confident-first
    df = df.sort_values("risk", ascending=show.startswith("accepted"))
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


# ---------------------------------------------------------------------------------------------
# v2: Second Listen. Word and gap flags from the trained verifier (asrshift.verifier), whole
# consultations instead of single units, live decoding streamed as server-sent events.
# ---------------------------------------------------------------------------------------------

CONSULTATION_ID = re.compile(r"^day\d+_consultation\d+$")
AUDIO_DIR = DATA_DIR / "audio"
UPLOAD_DIR = DATA_DIR / "uploads"
SECOND_MODEL_CONFIG = REPO_ROOT / "configs" / "large-v3-turbo.yaml"


def _verifier():
    """Imported on first use, so the v1 endpoints work without a trained verifier."""
    from asrshift import verifier

    return verifier


# saved transcripts the Review and Compare views can show; live decoding always runs whisper-small
TRANSCRIPT_MODELS = {
    "turbo-lora-mixed": "large-v3-turbo fine-tuned (LoRA, PriMock57 + Eka)",
    "small": "whisper-small",
}


def _available(name: str) -> bool:
    return (RESULTS_DIR / "units" / f"units_{name}.csv.gz").exists() and         (RESULTS_DIR / "models" / f"verifier_{name}.joblib").exists()


def _transcript_model(model: str | None) -> str:
    """The requested saved-transcript model, or the best one with results on disk."""
    if model is None:
        return next((m for m in TRANSCRIPT_MODELS if _available(m)), "small")
    if model not in TRANSCRIPT_MODELS:
        raise HTTPException(422, f"model must be one of {sorted(TRANSCRIPT_MODELS)}")
    if not _available(model):
        raise HTTPException(503, f"no results for {model}; see experiments/lora_pilot/README.md")
    return model


def _consultation(consultation_id: str) -> str:
    if not CONSULTATION_ID.match(consultation_id):
        raise HTTPException(404, "unknown consultation")
    return consultation_id


@lru_cache(maxsize=1)
def committed_turns() -> pd.DataFrame:
    """PriMock57 turn boundaries from the committed results, so live decoding needs no `asrshift prepare`."""
    cols = ["unit_id", "dataset", "subset", "consultation", "roles", "start", "end", "split"]
    d = pd.read_csv(RESULTS_DIR / "units" / f"units_{MODEL}.csv.gz", usecols=cols, low_memory=False)
    return d[(d["dataset"] == "primock57") & d["subset"].str.startswith("turn")].sort_values(["consultation", "start"])


@app.get("/api/v2/models")
def v2_models():
    default = _transcript_model(None)
    return {"default": default, "live": MODEL,
            "models": [{"id": m, "label": label, "available": _available(m)} for m, label in TRANSCRIPT_MODELS.items()]}


@app.get("/api/v2/consultations")
def v2_consultations(model: str | None = None):
    return _verifier().list_consultations(model=_transcript_model(model))


@app.get("/api/v2/consultation/{consultation_id}")
def v2_consultation(consultation_id: str, unit: str = "turn", model: str | None = None):
    _consultation(consultation_id)
    if unit not in {"turn", "window"}:
        raise HTTPException(422, "unit must be turn or window")
    model = _transcript_model(model)
    try:
        return {**_verifier().consultation_payload(consultation_id, unit=unit, model=model), "model": model}
    except (KeyError, ValueError):
        raise HTTPException(404, "unknown consultation")
    except FileNotFoundError as e:
        raise HTTPException(503, f"verifier not trained: {e}")


def _mix_path(consultation_id: str) -> Path:
    """The two channels summed, written once to data/audio/ and served from there."""
    path = AUDIO_DIR / f"{consultation_id}_mix.wav"
    if not path.exists():
        from asrshift.audio import primock_audio

        try:
            mix = primock_audio(f"{consultation_id}:mix", 0.0, 1e9)
        except (FileNotFoundError, OSError, RuntimeError):
            raise HTTPException(404, "audio not available (run `asrshift download`)")
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.wav")
        sf.write(tmp, mix, 16_000, format="WAV", subtype="PCM_16")
        tmp.replace(path)
    return path


@app.get("/api/v2/consultation/{consultation_id}/audio")
def v2_audio(consultation_id: str, request: Request):
    return _ranged_file(_mix_path(_consultation(consultation_id)), request, "audio/wav")


def _ranged_file(path: Path, request: Request, media_type: str) -> Response:
    """A file with HTTP Range support, so the browser can seek in long audio without loading all of it."""
    size = path.stat().st_size
    rng = request.headers.get("range")
    m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
    if not m or (not m.group(1) and not m.group(2)):
        return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})
    if m.group(1):
        start = int(m.group(1))
        end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
    else:  # "bytes=-N": the last N bytes
        start, end = max(0, size - int(m.group(2))), size - 1
    if start >= size or start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    with open(path, "rb") as f:
        f.seek(start)
        body = f.read(end - start + 1)
    return Response(body, status_code=206, media_type=media_type,
                    headers={"Content-Range": f"bytes {start}-{end}/{size}", "Accept-Ranges": "bytes"})


@app.get("/api/v2/curves")
def v2_curves(model: str | None = None):
    try:
        return _verifier().review_curves(model=_transcript_model(model))
    except FileNotFoundError as e:
        raise HTTPException(503, f"verifier not trained: {e}")


OVERVIEW = RESULTS_DIR / "tables" / "turbo-lora-mixed" / "overview.json"


@app.get("/api/v2/overview")
def v2_overview():
    """Overall metrics of the final system and the starting point, for the results page
    (written by experiments/lora_pilot/analysis/overview.py)."""
    if not OVERVIEW.exists():
        raise HTTPException(503, "results/tables/turbo-lora-mixed/overview.json missing; run "
                                 "experiments/lora_pilot/analysis/overview.py")
    return json.loads(OVERVIEW.read_text())


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_second = {}


def _second_whisper():
    """large-v3-turbo for the second opinion, loaded only when a client asks for it."""
    with _model_lock:
        if "m" not in _second:
            from asrshift import infer
            from asrshift.config import load_config

            cfg = load_config(SECOND_MODEL_CONFIG)
            _second["cfg"] = cfg
            _second["m"] = infer.load_model(cfg["asr"])[0]
        return _second["m"], _second["cfg"]


def _decode_and_score(audio_arr: np.ndarray, offset: float, kind: int, decode_cfg: dict, second: bool,
                      vad: bool = False) -> tuple[list[dict], dict, float]:
    """Decode one stretch of audio, score it with the verifier, shift times to the consultation clock.

    Returns the decoder segments (relative times), the verifier output and the decode time in seconds.
    """
    import time

    from asrshift import infer

    model, cfg = _whisper()
    dec = dict(decode_cfg or cfg["asr"]["decode"])
    if vad:
        dec["vad_filter"] = True
    with _decode_lock:
        t0 = time.perf_counter()
        rec = infer.transcribe(model, audio_arr, dec)
        second_text = None
        if second:
            m2, cfg2 = _second_whisper()
            second_text = infer.transcribe(m2, audio_arr, dict(cfg2["asr"]["decode"]) | ({"vad_filter": True} if vad else {}))["text"]
        elapsed = time.perf_counter() - t0
    out = _scored(rec, offset, kind, second_text)
    return rec["segments"], out, elapsed


def _scored(rec: dict, offset: float, kind: int, second_text: str | None) -> dict:
    from asrshift.features import unit_features

    feats = unit_features(rec)
    unit = {k: feats.get(k) for k in ("mean_logprob", "word_p_mean", "nsp_max", "cr_max", "n_segments",
                                      "words_per_s", "log_duration")} | {"kind": kind}
    words = [w | {"start": w["start"] + offset, "end": w["end"] + offset} for w in rec["words"]]
    return _verifier().load(MODEL).score(words, unit, second_text)


class _Job:
    """Events of one decoding run. Kept after the run ends, so a late or reconnecting client gets them all."""

    def __init__(self):
        self.events: list[str] = []
        self.done = False
        self.cancelled = threading.Event()
        self.cond = threading.Condition()

    def put(self, event: str, data: dict) -> None:
        with self.cond:
            self.events.append(_sse(event, data))
            self.done = self.done or event in ("done", "error")
            self.cond.notify_all()

    def stream(self, cancel_on_disconnect: bool):
        i = 0
        try:
            while True:
                with self.cond:
                    while i >= len(self.events) and not self.done:
                        if not self.cond.wait(timeout=15):
                            break
                    batch, i = self.events[i:], len(self.events)
                    finished = self.done and not batch
                if batch:
                    yield "".join(batch)
                elif finished:
                    return
                else:
                    yield ": keep-alive\n\n"
        finally:
            if cancel_on_disconnect and not self.done:
                self.cancelled.set()


def _event_stream(job: _Job, cancel_on_disconnect: bool) -> StreamingResponse:
    return StreamingResponse(job.stream(cancel_on_disconnect), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _run_live(job: _Job, consultation_id: str, second: bool) -> None:
    import time

    from asrshift.audio import primock_audio

    try:
        turns = committed_turns()
        turns = turns[turns["consultation"] == consultation_id]
        if turns.empty:
            raise KeyError(consultation_id)
        _, cfg = _whisper()
        t_start, decoded_audio = time.perf_counter(), 0.0
        for r in turns.itertuples():
            if job.cancelled.is_set():
                return
            role = str(r.roles)
            arr = primock_audio(f"{consultation_id}:{role}", float(r.start), float(r.end))
            _, out, elapsed = _decode_and_score(arr, float(r.start), 1, cfg["asr"]["decode"], second)
            decoded_audio += float(r.end) - float(r.start)
            job.put("turn", {"id": r.unit_id, "role": role, "start": float(r.start), "end": float(r.end),
                             "words": out["words"], "gaps": out["gaps"], "decode_s": round(elapsed, 3)})
            wall = time.perf_counter() - t_start
            job.put("progress", {"decoded_until": float(r.end),
                                 "realtime_factor": round(decoded_audio / wall, 3) if wall > 0 else None})
        job.put("done", {})
    except KeyError:
        job.put("error", {"message": f"no turns for {consultation_id}"})
    except (FileNotFoundError, OSError):
        job.put("error", {"message": "audio not available (run `asrshift download`)"})
    except Exception as e:  # noqa: BLE001 - the client must hear about any failure, not a silent stream
        job.put("error", {"message": f"{type(e).__name__}: {e}"})


@app.get("/api/v2/live/{consultation_id}")
def v2_live(consultation_id: str, second: int = Query(0, ge=0, le=1)):
    """Decode the consultation turn by turn (each speaker's own channel) and stream the scored turns."""
    _consultation(consultation_id)
    job = _Job()
    # A closed or reloaded page does not always cancel its run (the stream's cleanup only runs once the
    # server next writes to it), and every run queues on the one Whisper model. One viewer, one live run:
    # a new run cancels the older ones, so it never waits behind decodes nobody is watching.
    with _live_lock:
        for old in _live_jobs:
            old.cancelled.set()
        _live_jobs[:] = [job]
    threading.Thread(target=_run_live, args=(job, consultation_id, bool(second)), daemon=True).start()
    return _event_stream(job, cancel_on_disconnect=True)


_live_lock = threading.Lock()
_live_jobs: list[_Job] = []


_jobs: dict[str, _Job] = {}


def _run_upload(job: _Job, path: Path, second: bool) -> None:
    import time

    from faster_whisper.audio import decode_audio

    from asrshift import infer

    try:
        arr = decode_audio(str(path), sampling_rate=16_000)
        model, cfg = _whisper()
        t0 = time.perf_counter()
        dec = dict(cfg["asr"]["decode"]) | {"vad_filter": True}
        with _decode_lock:
            rec = infer.transcribe(model, arr, dec)
        segs = rec["segments"]
        for i, s in enumerate(segs):
            if job.cancelled.is_set():
                return
            a, b = int(s["start"] * 16_000), int(s["end"] * 16_000)
            words = [w for w in rec["words"] if s["start"] <= w["start"] < s["end"] + 1e-3]
            seg_rec = {"text": s["text"], "segments": [s], "audio_s": s["end"] - s["start"],
                       "words": [w | {"start": w["start"] - s["start"], "end": w["end"] - s["start"]} for w in words]}
            second_text = None
            if second:
                m2, cfg2 = _second_whisper()
                with _decode_lock:
                    second_text = infer.transcribe(m2, arr[a:b], dict(cfg2["asr"]["decode"]))["text"]
            out = _scored(seg_rec, s["start"], 1, second_text)
            job.put("turn", {"id": f"seg_{i:04d}", "role": "speaker", "start": s["start"], "end": s["end"],
                             "words": out["words"], "gaps": out["gaps"], "decode_s": None})
            wall = time.perf_counter() - t0
            job.put("progress", {"decoded_until": s["end"],
                                 "realtime_factor": round(s["end"] / wall, 3) if wall > 0 else None})
        job.put("done", {"audio_s": rec["audio_s"]})
    except Exception as e:  # noqa: BLE001
        job.put("error", {"message": f"{type(e).__name__}: {e}"})


@app.post("/api/v2/upload")
async def v2_upload(file: UploadFile = File(...), second: int = Query(0, ge=0, le=1)):
    """Store an audio file and start decoding it; events arrive on /api/v2/jobs/{job}/events."""
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}:
        raise HTTPException(415, "upload a wav, mp3, m4a, ogg, flac or webm file")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{job_id}{suffix}"
    path.write_bytes(await file.read())
    job = _jobs[job_id] = _Job()
    threading.Thread(target=_run_upload, args=(job, path, bool(second)), daemon=True).start()
    return {"job": job_id}


@app.get("/api/v2/jobs/{job_id}/events")
def v2_job_events(job_id: str):
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return _event_stream(job, cancel_on_disconnect=False)


app.mount("/classic", StaticFiles(directory=STATIC / "classic", html=True, check_dir=False), name="classic")
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
