"""Step 2: the verifier on LoRA transcripts, end to end, against the original small + verifier.

usage (repo root, repo venv): python verify_lora.py LORA_JSONL OUT_DIR

Same 12 test consultations, windows only (the LoRA pilot decoded windows). Reported in ABSOLUTE weighted
and critical errors left per consultation after B seconds of listening, because the two transcripts
start from different error counts (a share-of-errors-left curve would hide the LoRA gain).
  (a) the deployed verifier (trained on whisper-small's errors, Eka cal/val + recalibration) applied as is
  (b) cross-fit: 5 folds by consultation over the 12, training on the protocol (a) data plus the other folds
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from asrshift import verifier
from asrshift.config import load_config
from lora_tables import install, load_lora

jsonl, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config(); vc = cfg["verifier"]
pad = float(vc["clip_pad_s"])
BUD = [0, 15, 30, 60, 120, 240, 300]
u_lora, probs, _ = load_lora(jsonl)
install(u_lora, probs)
cons = sorted(u_lora[u_lora.split == "test"].consultation.unique())
v = verifier.load("small")
var = v.committed_variant


def tables(model):
    W, G, U = verifier.unit_tables(model, "large-v3-turbo", v.weights, ("pm_window_test",))
    keep = lambda d: d[d.consultation.isin(cons)].reset_index(drop=True)
    return keep(W), keep(G), keep(U)


T = {m: tables(m) for m in ("small", "lora")}
assert T["small"][2].unit_id.nunique() == T["lora"][2].unit_id.nunique(), [len(t[2]) for t in T.values()]
P = {m: v.predict(W, G, var) for m, (W, G, U) in T.items()}

Wtr, Gtr, _ = verifier.unit_tables("small", "large-v3-turbo", v.weights, ("train",))
cf = {m: verifier._crossfit_preds(v, Wtr, Gtr, {m: T[m]}, cfg)[m] for m in T}   # separately, so folds never mix transcripts


def parts(m, preds):
    W, G, U = T[m]
    return verifier.curve_parts(W.assign(pred=preds[0]), G.assign(pred=preds[1]), U, pad, BUD)


R = {(m, how): parts(m, P[m] if how == "a" else cf[m]) for m in T for how in ("a", "b")}
rng = np.random.default_rng(598)
draws = [rng.integers(0, len(cons), len(cons)) for _ in range(1000)]
res = {"budgets": BUD, "consultations": cons}
for k, name in ((0, "weighted"), (1, "critical")):
    print(f"\n== {name} errors left per consultation (mean over {len(cons)}), windows")
    print(f"{'':<34}" + "".join(f"{b:>8}s" for b in BUD))
    rows = {}
    for (m, how), p in R.items():
        tot = p["total"][:, k]
        for meth in ("window_review", "whisper_conf", "verifier", "hybrid", "perfect"):
            if how == "b" and meth in ("window_review", "whisper_conf", "perfect"):
                continue
            left = tot[:, None] - p[meth][:, :, k]
            rows[f"{m} {meth} ({how})"] = left
            print(f"{m + ' ' + meth + ' (' + how + ')':<34}" + "".join(f"{x:>9.1f}" for x in left.mean(0)))
    print("  difference lora - small (verifier), 95% CI:")
    for how in ("a", "b"):
        for meth in ("verifier", "hybrid"):
            d = rows[f"lora {meth} ({how})"] - rows[f"small {meth} ({how})"]
            lo, hi = np.quantile([d[s].mean(0) for s in draws], [0.025, 0.975], axis=0)
            print(f"    {meth} ({how})".ljust(34) + "".join(f"{e:>+6.1f}[{a:+.0f},{b:+.0f}]" for e, a, b in zip(d.mean(0), lo, hi)))
            res[f"{name}_{meth}_{how}_diff"] = dict(mean=d.mean(0).round(2).tolist(), lo=lo.round(2).tolist(), hi=hi.round(2).tolist())
    res[name] = {r: x.mean(0).round(2).tolist() for r, x in rows.items()}
for m, (W, G, U) in T.items():
    for how, pr in (("a", P[m]), ("b", cf[m])):
        aw = roc_auc_score(W.b > 0, pr[0]); ag = roc_auc_score(G.b > 0, pr[1])
        res[f"auroc_{m}_{how}"] = [round(aw, 4), round(ag, 4)]
        print(f"AUROC words/gaps {m} ({how}): {aw:.3f} / {ag:.3f}")
json.dump(res, open(OUT / "summary.json", "w"), indent=1)
print("wrote", OUT / "summary.json")
