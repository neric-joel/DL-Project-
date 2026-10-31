"""Critical-word second check on top of the verifier, for the mixed turbo LoRA.

A word is flagged when the model wrote a critical word (number, negation, drug: weight 10) that turbo
does not have at that place (agree == 0), and a gap is flagged when turbo has a critical word there that
the model dropped (extra_crit). Flagged items have their predicted benefit multiplied by k before the
review queue is built (k = inf: all flags first). k is chosen on training data (Eka validation, the 2 dev
consultations and the 8 out-of-fold consultations), then the test sets are scored once.

usage (repo root, repo venv): python crit_check.py MIXED_OOF_JSONL OUT_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
v = verifier.load("small")
pad = float(cfg["verifier"]["clip_pad_s"])
BUD = [0, 15, 30, 60, 120, 240]
KS = [1, 2, 5, 20, np.inf]
rng = np.random.default_rng(598)


def tables(sets):
    W, G, U = verifier.unit_tables("mixed", "large-v3-turbo", w8, sets)
    sw, sg = v.predict(W, G, v.committed_variant)
    fw = ((W.own_w >= w8.crit) & (W.agree == 0)).to_numpy()
    fg = (G.extra_crit > 0).to_numpy()
    # Eka has no consultation: one Eka group = one "consultation" for the review budget
    for d in (W, G, U):
        d["consultation"] = d["consultation"].where(d["consultation"].notna(), d["group"])
    return W, G, U, sw, sg, fw, fg


def left(T, k, meth):
    W, G, U, sw, sg, fw, fg = T
    boost = (lambda s, f: s + 1e6 * f) if np.isinf(k) else (lambda s, f: s * np.where(f, k, 1.0))
    p = verifier.curve_parts(W.assign(pred=boost(sw, fw)), G.assign(pred=boost(sg, fg)), U, pad, BUD)
    return np.stack([p["total"][:, j][:, None] - p[meth][:, :, j] for j in (0, 1)])  # (2, cons, budgets)


def flag_stats(T, name):
    W, G, _, _, _, fw, fg = T
    n_cons = W.consultation.nunique()
    crit_tot = W.cr.sum() + G.cr.sum()
    caught = W.cr[fw].sum() + G.cr[fg].sum()
    prec = ((W.b[fw] > 0).sum() + (G.b[fg] > 0).sum()) / max(fw.sum() + fg.sum(), 1)
    print(f"  {name}: {(fw.sum() + fg.sum()) / n_cons:.1f} flags per consultation; flags hold "
          f"{caught / crit_tot:.1%} of critical errors; {prec:.1%} of flags are real errors", flush=True)
    return dict(flags_per_cons=float((fw.sum() + fg.sum()) / n_cons), crit_share=float(caught / crit_tot),
                precision=float(prec))


summary = {}
# ---- choose k on training data ------------------------------------------------------------------------
TR = tables(("train",))
print(f"training data: {TR[2].consultation.nunique()} consultations/groups, {len(TR[2])} units", flush=True)
summary["train_flags"] = flag_stats(TR, "train")
best = {}
for meth in ("verifier", "hybrid"):
    print(f"\n[train, {meth}] mean errors left per consultation/group (weighted | critical) at {BUD[1:]} s")
    score = {}
    for k in KS:
        x = left(TR, k, meth).mean(1)
        # choose on the summed weighted errors left over the 15-120 s budgets (critical errors count 10 there)
        score[k] = float(x[0, 1:5].sum())
        print(f"  k={k:<5} " + " ".join(f"{a:6.1f}" for a in x[0, 1:]) + "  | " + " ".join(f"{a:5.2f}" for a in x[1, 1:]))
    best[meth] = min(score, key=score.get)
    print(f"  chosen k = {best[meth]}")
summary["chosen_k"] = {m: str(k) for m, k in best.items()}

# ---- test, once ------------------------------------------------------------------------------------------
for key, s in (("window", "pm_window_test"), ("turn", "pm_turn_test"), ("eka", "eka_test")):
    T = tables((s,))
    print(f"\n== {s}: {T[2].consultation.nunique()} consultations/groups")
    summary[f"{key}_flags"] = flag_stats(T, s)
    n = T[2].consultation.nunique()
    draws = [rng.integers(0, n, n) for _ in range(1000)]
    for meth in ("verifier", "hybrid"):
        base, new, allk = left(T, 1, meth), left(T, best[meth], meth), left(T, np.inf, meth)
        print(f"  [{meth}]{'':<24}" + "".join(f"{b:>8}s" for b in BUD))
        for j, jn in ((0, "weighted"), (1, "critical")):
            for nm, x in (("no boost", base), (f"k={best[meth]} (chosen)", new), ("k=inf", allk)):
                print(f"   {jn:<9}{nm:<22}" + "".join(f"{y:>9.2f}" for y in x[j].mean(0)))
            d = new[j] - base[j]
            lo, hi = np.quantile([d[i].mean(0) for i in draws], [0.025, 0.975], axis=0)
            print(f"   {jn:<9}{'chosen - no boost':<22}" + "".join(f"{e:+5.2f}[{l:+.1f},{h:+.1f}]" for e, l, h in zip(d.mean(0), lo, hi)))
            summary[f"{key}_{meth}_{jn}"] = {"no_boost": base[j].mean(0).round(3).tolist(),
                                             "chosen": new[j].mean(0).round(3).tolist(),
                                             "k_inf": allk[j].mean(0).round(3).tolist(),
                                             "diff_ci": [d.mean(0).round(3).tolist(), lo.round(3).tolist(), hi.round(3).tolist()]}
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
print("\nwrote", OUT / "summary.json")
