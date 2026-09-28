"""Run faster-whisper over the manifest and store the raw decoder outputs.

One JSON line per unit with the text, every segment's decoder statistics (average log-probability,
no-speech probability, compression ratio, temperature used, token count) and every word with its
probability and timestamps. Confidence features are derived from these later, so nothing here
depends on labels. The run is resumable: units already in the output file are skipped.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from asrshift.audio import SR, load_unit_audio
from asrshift.paths import INTERIM_DIR, MODEL_CACHE_DIR
from asrshift.runtime import setup_cuda_dlls

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def output_path(model_name: str, shard: str | None = None) -> Path:
    """Main output file, or a per-process shard (two processes never append to one file)."""
    stem = f"asr_{model_name.replace('/', '_')}"
    return INTERIM_DIR / (f"{stem}.shard-{shard}.jsonl" if shard else f"{stem}.jsonl")


def all_output_paths(model_name: str) -> list[Path]:
    """This model's files only (a model called e.g. "small.en" has its own, non-matching stem)."""
    stem = f"asr_{model_name.replace('/', '_')}"
    return sorted(INTERIM_DIR.glob(f"{stem}.jsonl")) + sorted(INTERIM_DIR.glob(f"{stem}.shard-*.jsonl"))


def _done_ids(paths: list[Path]) -> set[str]:
    done = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["unit_id"])
                except (json.JSONDecodeError, KeyError):
                    continue  # a line cut short by an interrupted run; the unit is simply redone
    return done


def load_model(asr_cfg: dict, device: str | None = None, compute_type: str | None = None):
    setup_cuda_dlls()
    import ctranslate2
    from faster_whisper import WhisperModel

    device = device or asr_cfg["device"]
    if device == "cuda" and ctranslate2.get_cuda_device_count() == 0:
        print("no CUDA device found; falling back to CPU int8")
        device, compute_type = "cpu", "int8"
    compute_type = compute_type or (asr_cfg["compute_type"] if device == "cuda" else "int8")
    model = WhisperModel(
        asr_cfg.get("model_repo", asr_cfg["model"]), device=device, compute_type=compute_type,
        download_root=str(MODEL_CACHE_DIR), revision=asr_cfg.get("model_revision"),
    )
    return model, device, compute_type


def transcribe(model, audio, decode_cfg: dict) -> dict:
    kw = dict(decode_cfg)
    kw["temperature"] = tuple(kw["temperature"])
    segments, info = model.transcribe(audio, **kw)
    segs, words = [], []
    for s in segments:
        segs.append({
            "start": round(s.start, 3), "end": round(s.end, 3), "text": s.text,
            "avg_logprob": s.avg_logprob, "no_speech_prob": s.no_speech_prob,
            "compression_ratio": s.compression_ratio, "temperature": s.temperature,
            "n_tokens": len(s.tokens),
        })
        for w in s.words or []:
            words.append({"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word,
                          "probability": w.probability})
    return {"text": "".join(s["text"] for s in segs).strip(), "segments": segs, "words": words,
            "audio_s": round(len(audio) / SR, 3)}


def run(manifest: pd.DataFrame, cfg: dict, out: Path | None = None, limit: int | None = None,
        device: str | None = None, compute_type: str | None = None, shard: str | None = None) -> Path:
    import ctranslate2
    import faster_whisper

    asr_cfg = cfg["asr"]
    out = out or output_path(asr_cfg["model"], shard)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _done_ids([out] + all_output_paths(asr_cfg["model"]))
    todo = manifest[~manifest["unit_id"].isin(done)]
    # Group by audio file so the per-file caches in asrshift.audio are hit in order.
    todo = todo.assign(_k=todo["audio_source"].str.split(r"[#:]").str[0]).sort_values(["dataset", "_k", "start"])
    if limit:
        todo = todo.head(limit)
    if todo.empty:
        print(f"{out.name}: nothing to do ({len(done)} units present)")
        return out

    ctranslate2.set_random_seed(cfg["seed"])
    model, device, compute_type = load_model(asr_cfg, device, compute_type)
    meta = {
        "model": asr_cfg["model"], "model_repo": asr_cfg.get("model_repo"), "revision": asr_cfg.get("model_revision"),
        "device": device, "compute_type": compute_type, "faster_whisper": faster_whisper.__version__,
        "ctranslate2": ctranslate2.__version__, "python": platform.python_version(), "decode": asr_cfg["decode"],
    }
    meta_path = out.with_suffix(".meta.json")
    runs = json.loads(meta_path.read_text()) if meta_path.exists() else []
    runs.append({**meta, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "units": int(len(todo))})
    meta_path.write_text(json.dumps(runs, indent=2))

    # A run killed mid-write leaves a truncated last line; start the next record on a fresh line so
    # the truncated one is the only casualty (its unit is redone because it never parsed).
    if out.exists() and out.stat().st_size:
        with open(out, "rb") as fb:
            fb.seek(-1, os.SEEK_END)
            needs_newline = fb.read(1) != b"\n"
        if needs_newline:
            with open(out, "a", encoding="utf-8") as f:
                f.write("\n")
    with open(out, "a", encoding="utf-8") as f:
        for row in tqdm(todo.to_dict("records"), desc=f"asr {asr_cfg['model']}", mininterval=10):
            audio = load_unit_audio(row)
            t0 = time.perf_counter()
            rec = transcribe(model, audio, asr_cfg["decode"])
            rec = {"unit_id": row["unit_id"], "decode_s": round(time.perf_counter() - t0, 3), **rec}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    return out


def load_outputs(model_name: str) -> pd.DataFrame:
    """All decoder outputs for a model, across the main file and any shards."""
    rows = []
    for path in all_output_paths(model_name):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows)
    return df.drop_duplicates("unit_id", keep="last").reset_index(drop=True)
