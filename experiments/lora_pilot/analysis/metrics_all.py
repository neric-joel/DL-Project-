"""Classification-style metrics for the final systems.

usage (repo root, repo venv): python metrics_all.py PM_ONLY_JSONL MIXED_OOF_JSONL OUT_DIR

A. Transcripts (small, turbo, turbo+LoRA PriMock-only, turbo+LoRA mixed) per test set: word accuracy
   (hits / reference words), substitution / deletion / insertion rates, and precision / recall / F1 of
   numbers and negations (precision = share of the numbers the model wrote that are in the reference).
   95% CIs by consultation (Eka: group) bootstrap.
B. Error detectors on test words (label: the word is wrong) and gaps (label: words were dropped there):
   the verifier on small and on the mixed model, and Whisper's own word probability. AUROC, AUPRC, and at
   two operating points - the F1-best threshold chosen on training data, and flagging the top 10% -
   accuracy, precision, recall, specificity, F1 and recall of critical errors.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from asrshift import score, verifier
from asrshift.config import load_config
from lora_tables import load_lora

pm_jsonl, mix_jsonl, OUT = sys.argv[1], sys.argv[2], Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
w8 = verifier.Weights.from_config(cfg)
rng = np.random.default_rng(598)
u_pm, p_pm, _ = load_lora(pm_jsonl)
u_mx, p_mx, _ = load_lora(mix_jsonl)
models = {"small": score.load("small"), "turbo": score.load("large-v3-turbo"), "tl_primock": u_pm, "tl_mixed": u_mx}
test_ids = set(u_pm[u_pm.split == "test"].unit_id) & set(u_mx[u_mx.split == "test"].unit_id)
SETS = ("pm_window_test", "pm_turn_test", "eka_test")
res = {}

# ---- A. transcripts ---------------------------------------------------------------------------------
COLS = ["n_ref", "hits", "sub", "del", "ins", "n_num", "n_num_correct", "n_num_hyp", "n_num_false",
        "n_neg", "n_neg_correct", "n_neg_hyp", "n_neg_false"]


def prf(ok_ref, n_ref, n_hyp, false):
    p = (n_hyp - false) / n_hyp if n_hyp else np.nan
    r = ok_ref / n_ref if n_ref else np.nan
    return p, r, 2 * p * r / (p + r) if p + r else np.nan


MET = {
    "word accuracy": lambda c: c.hits / c.n_ref,
    "substitution rate": lambda c: c["sub"] / c.n_ref,
    "deletion rate": lambda c: c["del"] / c.n_ref,
    "insertion rate": lambda c: c["ins"] / c.n_ref,
    "number precision": lambda c: prf(c.n_num_correct, c.n_num, c.n_num_hyp, c.n_num_false)[0],
    "number recall": lambda c: prf(c.n_num_correct, c.n_num, c.n_num_hyp, c.n_num_false)[1],
    "number F1": lambda c: prf(c.n_num_correct, c.n_num, c.n_num_hyp, c.n_num_false)[2],
    "negation precision": lambda c: prf(c.n_neg_correct, c.n_neg, c.n_neg_hyp, c.n_neg_false)[0],
    "negation recall": lambda c: prf(c.n_neg_correct, c.n_neg, c.n_neg_hyp, c.n_neg_false)[1],
    "negation F1": lambda c: prf(c.n_neg_correct, c.n_neg, c.n_neg_hyp, c.n_neg_false)[2],
}
rows = []
for m, u in models.items():
    u = u[u.unit_id.isin(test_ids)].copy()
    u["set"] = u.apply(verifier._set, axis=1)
    u["group"] = u.consultation.where(u.dataset == "primock57", u.group_id)
    rows.append(u[["set", "group", *COLS]].assign(model=m))
D = pd.concat(rows, ignore_index=True)
for s in SETS:
    G = {m: D[(D.model == m) & (D.set == s)].groupby("group")[COLS].sum() for m in models}
    idx = G["turbo"].index
    draws = [rng.choice(idx, len(idx)) for _ in range(1000)]
    print(f"\n== A. transcripts, {s} ({len(idx)} groups)")
    print(f"  {'':<20}" + "".join(f"{m:>12}" for m in models) + "   mixed 95% CI")
    res[s] = {}
    for name, f in MET.items():
        v = {m: float(f(G[m].loc[idx].sum())) for m in models}
        lo, hi = np.nanquantile([f(G["tl_mixed"].loc[x].sum()) for x in draws], [0.025, 0.975])
        res[s][name] = {**v, "mixed_ci": [float(lo), float(hi)]}
        print(f"  {name:<20}" + "".join(f"{v[m]:>12.4f}" for m in models) + f"   [{lo:.4f}, {hi:.4f}]")

# ---- B. error detectors -----------------------------------------------------------------------------
real = verifier._committed
_, _, other_hyp, other_probs = real("small", "large-v3-turbo")
verifier._committed = lambda m, sec: (u_mx, p_mx, other_hyp, other_probs) if m == "mixed" else real(m, sec)
v = verifier.load("small")


def scores(model, sets):
    W, G, _ = verifier.unit_tables(model, "large-v3-turbo", w8, sets)
    sw, sg = v.predict(W, G, v.committed_variant)
    return W, G, np.asarray(sw), np.asarray(sg)


def at(y, crit, s, thr):
    f = s >= thr
    tp, fp, fn, tn = (f & y).sum(), (f & ~y).sum(), (~f & y).sum(), (~f & ~y).sum()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(flagged=float(f.mean()), accuracy=float((tp + tn) / len(y)), precision=float(p), recall=float(r),
                specificity=float(tn / max(tn + fp, 1)), F1=float(2 * p * r / max(p + r, 1e-12)),
                critical_recall=float((f & crit).sum() / max(crit.sum(), 1)))


def best_thr(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[np.argmax(f[:-1])])


# thresholds chosen on training data: the mixed model's out-of-fold / held-out training transcripts (the
# verifier never saw them); for small the training transcripts are the verifier's own training data
train = {m: scores(m, ("train",)) for m in ("small", "mixed")}
thr = {}
for m, (W, G, sw, sg) in train.items():
    thr[(m, "verifier", "words")] = best_thr(W.b.to_numpy() > 0, sw)
    thr[(m, "verifier", "gaps")] = best_thr(G.b.to_numpy() > 0, sg)
    thr[(m, "word prob", "words")] = best_thr(W.b.to_numpy() > 0, 1 - W.p.to_numpy())
res["thresholds"] = {" / ".join(k): v_ for k, v_ in thr.items()}
KEYS = ["flagged", "accuracy", "precision", "recall", "specificity", "F1", "critical_recall"]
for s in SETS:
    print(f"\n== B. error detection, {s}")
    for m in ("small", "mixed"):
        W, G, sw, sg = scores(m, (s,))
        items = {("verifier", "words"): (W, sw), ("verifier", "gaps"): (G, sg), ("word prob", "words"): (W, 1 - W.p.to_numpy())}
        for (det, kind), (T, sc) in items.items():
            y, crit = T.b.to_numpy() > 0, T.cr.to_numpy() > 0
            out = {"n": int(len(y)), "error_rate": float(y.mean()), "AUROC": float(roc_auc_score(y, sc)),
                   "AUPRC": float(average_precision_score(y, sc)),
                   "at_train_F1_threshold": at(y, crit, sc, thr[(m, det, kind)]),
                   "top_10pct": at(y, crit, sc, np.quantile(sc, 0.9))}
            res.setdefault(f"detect_{s}", {})[f"{m} | {det} | {kind}"] = out
            print(f"  {m:<6} {det:<9} {kind:<5} n={out['n']:>6}  errors {out['error_rate']:.1%}  "
                  f"AUROC {out['AUROC']:.3f}  AUPRC {out['AUPRC']:.3f}")
            for op in ("at_train_F1_threshold", "top_10pct"):
                print(f"      {op:<22}" + "  ".join(f"{k} {out[op][k]:.3f}" for k in KEYS))
json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
print("\nwrote", OUT / "metrics.json")
