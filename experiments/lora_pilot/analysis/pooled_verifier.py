"""Verifier trained on pooled errors: whisper-small's training transcripts (the old verifier's data) plus
the mixed turbo LoRA's held-out training transcripts (Eka validation, the 2 dev consultations, the 8
out-of-fold consultations). Scored on the mixed model's test transcripts against the old verifier.

usage (repo root, repo venv): python pooled_verifier.py MIXED_OOF_JSONL OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from asrshift import verifier
from asrshift.config import load_config
from lora_tables import load_lora

jsonl, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
w8 = verifier.Weights.from_config(cfg)
u_mx, p_mx, _ = load_lora(jsonl)
real = verifier._committed
_, _, other_hyp, other_probs = real("small", "large-v3-turbo")
verifier._committed = lambda m, sec: (u_mx, p_mx, other_hyp, other_probs) if m == "mixed" else real(m, sec)
pad = float(cfg["verifier"]["clip_pad_s"])
BUD = [0, 15, 30, 60, 120, 240]
rng = np.random.default_rng(598)

Ws, Gs, _ = verifier.unit_tables("small", "large-v3-turbo", w8, ("train",))
Wm, Gm, _ = verifier.unit_tables("mixed", "large-v3-turbo", w8, ("train",))
print(f"training units: small {Ws.unit_id.nunique()}, mixed {Wm.unit_id.nunique()}", flush=True)


def fit(W, G):
    return verifier.Verifier(weights=w8, variants=verifier.fit_variants(W, G, cfg, ("full",)), meta={},
                             committed_variant="full")


V = {"old (saved)": verifier.load("small"),
     "refit small only": fit(Ws, Gs),
     "pooled small + mixed": fit(pd.concat([Ws, Wm], ignore_index=True), pd.concat([Gs, Gm], ignore_index=True))}
res = {}
for key, s in (("window", "pm_window_test"), ("turn", "pm_turn_test"), ("eka", "eka_test")):
    W, G, U = verifier.unit_tables("mixed", "large-v3-turbo", w8, (s,))
    for d in (W, G, U):   # Eka: one group = one "consultation" for the budget
        d["consultation"] = d["consultation"].where(d["consultation"].notna(), d["group"])
    n = U.consultation.nunique()
    draws = [rng.integers(0, n, n) for _ in range(1000)]
    print(f"\n== {s}: {n} consultations/groups")
    left, det = {}, {}
    for name, v in V.items():
        sw, sg = v.predict(W, G, v.committed_variant)
        p = verifier.curve_parts(W.assign(pred=sw), G.assign(pred=sg), U, pad, BUD)
        for meth in ("verifier", "hybrid"):
            left[(name, meth)] = np.stack([p["total"][:, k][:, None] - p[meth][:, :, k] for k in (0, 1)])
        yw, yg = W.b.to_numpy() > 0, G.b.to_numpy() > 0
        det[name] = dict(words_AUROC=roc_auc_score(yw, sw), words_AUPRC=average_precision_score(yw, sw),
                         gaps_AUROC=roc_auc_score(yg, sg), gaps_AUPRC=average_precision_score(yg, sg))
        print(f"  {name:<22}" + "  ".join(f"{k} {x:.3f}" for k, x in det[name].items()), flush=True)
    res[f"{key}_detection"] = det
    for meth in ("verifier", "hybrid"):
        for k, kn in ((0, "weighted"), (1, "critical")):
            print(f"  [{meth}] {kn} errors left per consultation" + "".join(f"{b:>9}s" for b in BUD))
            for name in V:
                print(f"    {name:<40}" + "".join(f"{y:>10.2f}" for y in left[(name, meth)][k].mean(0)))
            d = left[("pooled small + mixed", meth)][k] - left[("old (saved)", meth)][k]
            lo, hi = np.quantile([d[i].mean(0) for i in draws], [0.025, 0.975], axis=0)
            print(f"    {'pooled - old (95% CI)':<40}" + "".join(f"{e:+5.1f}[{l:+.1f},{h:+.1f}]" for e, l, h in zip(d.mean(0), lo, hi)))
            res[f"{key}_{meth}_{kn}"] = {**{n_: left[(n_, meth)][k].mean(0).round(3).tolist() for n_ in V},
                                         "pooled_minus_old": [d.mean(0).round(3).tolist(), lo.round(3).tolist(), hi.round(3).tolist()]}
json.dump(res, open(OUT / "summary.json", "w"), indent=1)
print("\nwrote", OUT / "summary.json")
