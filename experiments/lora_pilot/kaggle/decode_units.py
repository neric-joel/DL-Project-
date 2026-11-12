"""Decode manifest units with a local CTranslate2 model through asrshift.infer.run, exactly as the
committed transcripts were made (same audio loading per unit, same decoding settings). Resumable.

usage (repo root): python decode_units.py MODEL_DIR NAME OUT_JSONL SET [SET ...]
SET: eka_calibration, eka_validation, eka_test, pm_turn_recalibration, pm_turn_test,
     pm_window_recalibration, pm_window_test
CONS (environment, optional): comma-separated PriMock57 consultations to keep (Eka units are unaffected)
"""
import os
import sys
from pathlib import Path

from asrshift import infer, prepare
from asrshift.config import load_config

model_dir, name, out, sets = sys.argv[1], sys.argv[2], Path(sys.argv[3]), set(sys.argv[4:])
cfg = load_config()
cfg["asr"].update(model=name, model_repo=model_dir, model_revision=None, device="cuda", compute_type="float16")
m = prepare.load()
m = m[m["dropped_reason"].fillna("") == ""]
kind = m["subset"].where(m["dataset"] == "primock57", "").str.replace(r"turn_.*", "turn", regex=True)
key = (m["dataset"].map({"eka": "eka", "primock57": "pm"}) + kind.map(lambda k: f"_{k}" if k else "")
       + "_" + m["split"])
m = m[key.isin(sets)]
if os.environ.get("CONS"):
    m = m[(m["dataset"] != "primock57") | m["consultation"].isin(os.environ["CONS"].split(","))]
print(f"{len(m)} units: {key[m.index].value_counts().to_dict()}", flush=True)
infer.run(m, cfg, out=out)
print("done", flush=True)
