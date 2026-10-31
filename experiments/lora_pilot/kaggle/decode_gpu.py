"""Decode PriMock57 windows with a CTranslate2 model through asrshift.infer.transcribe (word probabilities
kept), on a GPU when there is one. Resumable.

usage (repo root): python decode_gpu.py MODEL_DIR OUT_JSONL CONSULTATION [CONSULTATION ...]
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
from faster_whisper import WhisperModel

from asrshift.audio import primock_audio
from asrshift.config import load_config
from asrshift.infer import transcribe

model_dir, out, cons = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
u = pd.read_csv("results/units/units_small.csv.gz", low_memory=False)
w = u[(u.subset == "window") & u.consultation.isin(cons)].sort_values(["consultation", "start"])
done = set()
if out.exists():
    done = {json.loads(l)["unit_id"] for l in out.open(encoding="utf-8") if l.strip()}
dec = load_config()["asr"]["decode"]
try:
    model = WhisperModel(model_dir, device="cuda", compute_type="float16")
    print("decoding on cuda float16", flush=True)
except Exception as e:  # missing cuDNN/cuBLAS for CTranslate2: fall back to CPU rather than fail the run
    print(f"cuda unavailable for CTranslate2 ({e}); decoding on cpu int8", flush=True)
    model = WhisperModel(model_dir, device="cpu", compute_type="int8")
todo = w[~w.unit_id.isin(done)]
print(f"{len(w)} windows, {len(done)} already done, {len(todo)} to decode", flush=True)
t0 = time.time()
with out.open("a", encoding="utf-8") as f:
    for k, r in enumerate(todo.itertuples()):
        t = time.time()
        rec = transcribe(model, primock_audio(f"{r.consultation}:mix", r.start, r.end), dec)
        rec.update(unit_id=r.unit_id, decode_s=round(time.time() - t, 2))
        f.write(json.dumps(rec) + "\n"); f.flush()
        if k % 100 == 0:
            print(f"{k}/{len(todo)} windows, {time.time() - t0:.0f} s", flush=True)
print(f"done: {len(todo)} windows in {time.time() - t0:.0f} s", flush=True)
