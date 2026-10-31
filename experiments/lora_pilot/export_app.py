"""Write the fine-tuned turbo's results in the committed-results format, so the verifier code and the
Second Listen app read them like whisper-small's:

    results/units/units_turbo-lora-mixed.csv.gz      per-unit scores (same columns as units_small)
    results/units/words_turbo-lora-mixed.parquet     per-token probability / correct / critical
    results/models/verifier_turbo-lora-mixed.joblib  verifier trained on pooled errors (+ .json metadata)
    results/tables/turbo-lora-mixed/verifier/curves.json

The model is whisper-large-v3-turbo with a LoRA adapter trained on the 8 PriMock57 training
consultations and Eka calibration (train.py --eka). DECODES is its decoder records (jsonl), in which the
8 PriMock57 training consultations come from the 4 out-of-fold models (train.py --holdout), so no unit is
scored by a model that trained on it, except Eka calibration, which is not decoded.

The verifier trains on whisper-small's training transcripts (the saved small verifier's data) pooled with
this model's: Eka validation and all 10 PriMock57 recalibration consultations. Pooling beat the small
verifier and a verifier trained on this model's transcripts alone (experiments/lora_pilot/README.md).

    python experiments/lora_pilot/export_app.py DECODES
"""

from __future__ import annotations

import json
import sys

import joblib
import numpy as np
import pandas as pd

from asrshift import score, terms, verifier
from asrshift.config import load_config
from asrshift.paths import RESULTS_DIR

NAME = "turbo-lora-mixed"
SECOND = "large-v3-turbo"


def main():
    cfg = load_config()
    recs = {}
    for line in open(sys.argv[1], encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[r["unit_id"]] = r
    small, lex = score.load("small"), terms.load()
    rows, words = [], []
    for row in small[small.unit_id.isin(recs)].to_dict("records"):
        out, ws = score.score_unit(row, recs[row["unit_id"]], lex)
        rows.append({**row, **out})
        words += [(row["unit_id"], p, int(s == "C"), c) for _, p, s, c in ws]
    u = pd.DataFrame(rows)[small.columns]
    w = pd.DataFrame(words, columns=["unit_id", "probability", "correct", "critical"]).astype(
        {"probability": "float32", "correct": "int8", "critical": "int8"})
    u.to_csv(RESULTS_DIR / "units" / f"units_{NAME}.csv.gz", index=False)
    w.to_parquet(RESULTS_DIR / "units" / f"words_{NAME}.parquet", index=False, compression="zstd")
    print(f"{len(u)} units, {len(w)} tokens; {u.consultation.nunique()} PriMock57 consultations", flush=True)

    wt = verifier.Weights.from_config(cfg)
    Ws, Gs, _ = verifier.unit_tables("small", SECOND, wt, ("train",))
    Wm, Gm, _ = verifier.unit_tables(NAME, SECOND, wt, ("train",))
    W, G = pd.concat([Ws, Wm], ignore_index=True), pd.concat([Gs, Gm], ignore_index=True)
    variants = ("full", "second", "solo")
    fitted = verifier.fit_variants(W, G, cfg, variants)
    vc = cfg["verifier"]
    meta = {"model": NAME, "second": SECOND, "weights": {c: wt.of(c) for c in verifier.CATS},
            "train_units": int(Ws.unit_id.nunique() + Wm.unit_id.nunique()),   # same unit ids, two transcripts each "train_units_small": int(Ws.unit_id.nunique()),
            "train_units_own": int(Wm.unit_id.nunique()), "train_words": int(len(W)), "train_gaps": int(len(G)),
            "hyperparameters": {"shared": vc["model"], "words": vc.get("model_words", {}), "gaps": vc.get("model_gaps", {})},
            "committed_variant": "full",
            "features": {"words": fitted["full"]["word_features"], "gaps": fitted["full"]["gap_features"]}}
    path = verifier.model_path(NAME)
    joblib.dump({"meta": meta, "variants": fitted, "weights": wt, "committed_variant": "full"}, path, compress=3)
    verifier.load.cache_clear()
    curves = verifier.write_curves(NAME, SECOND, cfg)
    i60 = curves["x_seconds"].index(60)
    meta["curves"] = {k: {m: curves[k]["weighted"][m][i60] for m in curves[k]["weighted"]} for k in ("window", "turn")}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"verifier: {meta['train_units']} training units ({meta['train_units_small']} small + "
          f"{meta['train_units_own']} own); weighted errors left at 60 s: "
          + json.dumps({k: round(v["verifier"], 3) for k, v in meta["curves"].items()}), flush=True)


if __name__ == "__main__":
    main()
