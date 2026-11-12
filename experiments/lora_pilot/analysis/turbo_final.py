"""Fine-tuned turbo (seed 598): transcription quality and confidence on every test set, then the verifier
retrained on its own errors.

usage (repo root, repo venv): python turbo_final.py ALL_JSONL OUT_DIR

A. Per test set (Eka test, PriMock57 turns, PriMock57 windows): WER, weighted WER, critical error rate,
   number / negation / medical-term recall for small, turbo and turbo+LoRA, with bootstrap CIs by
   consultation (Eka: by group). Then whether the model's confidence still ranks errors (AUROC of unit
   confidence for a unit with WER above the study's error threshold, and of word probability for a
   wrong word).
B. Verifier for turbo+LoRA, protocol (a) training data minus the 8 consultations the LoRA trained on
   (their errors are in-sample): Eka calibration/validation + the 2 dev recalibration consultations.
   Second opinion: original turbo. Reported in absolute errors left per test consultation.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from asrshift import score, terms, verifier
from asrshift.config import load_config
from asrshift.textnorm import scoring_tokens
from lora_tables import install, load_lora

jsonl, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
w8 = verifier.Weights.from_config(cfg)
recal = cfg["primock57"]["recalibration_consultations"]
LORA_TRAIN = set(recal[:-2])
ERR = score.LABELS["err"]
rng = np.random.default_rng(598)

u_tl, probs, recs = load_lora(jsonl)
lex = terms.load()
models = {"small": score.load("small"), "turbo": score.load("large-v3-turbo"), "turbo_lora": u_tl}
test_ids = set(u_tl[u_tl.split == "test"].unit_id)
print(f"turbo_lora units: {len(u_tl)}, test units {len(test_ids)}", flush=True)


def setname(r):
    return verifier._set(r)


# ---- A. transcription quality per set ---------------------------------------------------------------
rows = []
for m, u in models.items():
    u = u[u.unit_id.isin(test_ids)]
    for r in u.to_dict("records"):
        ref = scoring_tokens(r["reference"], reference=True)
        hyp = scoring_tokens(str(r.get("hypothesis") or ""))
        lab = verifier.labels(ref, hyp, w8)
        rows.append(dict(model=m, set=setname(r), group=r["consultation"] if r["dataset"] == "primock57" else r["group_id"],
                         unit_id=r["unit_id"], n_ref=r["n_ref"], errors=r["errors"],
                         w_err=float(lab["word_b"].sum() + lab["gap_b"].sum()), w_ref=lab["ref_weight"],
                         crit_err=int(lab["word_cr"].sum() + lab["gap_cr"].sum()),
                         crit_ref=sum(c == "crit" for c in verifier.categories(ref, w8)),
                         n_num=r["n_num"], n_num_ok=r["n_num_correct"], n_neg=r["n_neg"], n_neg_ok=r["n_neg_correct"],
                         n_term=r.get("n_term", 0) or 0, n_term_ok=r.get("n_term_correct", 0) or 0,
                         conf=r["conf"], err=int(r["wer"] > ERR)))
D = pd.DataFrame(rows)
D.to_csv(OUT / "per_unit.csv", index=False)
MET = {"WER": lambda c: c.errors / c.n_ref, "weighted WER": lambda c: c.w_err / c.w_ref,
       "critical error rate": lambda c: c.crit_err / c.crit_ref, "number recall": lambda c: c.n_num_ok / c.n_num,
       "negation recall": lambda c: c.n_neg_ok / c.n_neg, "medical-term recall": lambda c: c.n_term_ok / c.n_term}
summary = {}
for s in ("eka_test", "pm_turn_test", "pm_window_test"):
    G = {m: D[(D.model == m) & (D.set == s)].groupby("group")[["n_ref", "errors", "w_err", "w_ref", "crit_err", "crit_ref",
                                                              "n_num", "n_num_ok", "n_neg", "n_neg_ok", "n_term", "n_term_ok"]].sum()
         for m in models}
    idx = G["turbo"].index.intersection(G["turbo_lora"].index)
    draws = [rng.choice(idx, len(idx)) for _ in range(1000)]
    print(f"\n== {s}: {len(idx)} groups, {int(G['turbo'].loc[idx].n_ref.sum())} reference words")
    print(f"{'':<22}{'small':>9}{'turbo':>9}{'tl':>9}   tl - turbo (95% CI)")
    summary[s] = {}
    for name, f in MET.items():
        v = {m: float(f(G[m].loc[idx].sum())) for m in models}
        d = [f(G["turbo_lora"].loc[x].sum()) - f(G["turbo"].loc[x].sum()) for x in draws]
        lo, hi = np.nanquantile(d, [0.025, 0.975])
        summary[s][name] = dict(**v, diff=v["turbo_lora"] - v["turbo"], lo=float(lo), hi=float(hi))
        print(f"  {name:<20}{v['small']:>9.4f}{v['turbo']:>9.4f}{v['turbo_lora']:>9.4f}   {v['turbo_lora'] - v['turbo']:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    for m in models:
        x = D[(D.model == m) & (D.set == s)]
        summary[s][f"auroc_unit_conf_{m}"] = float(roc_auc_score(x.err, -x.conf)) if x.err.nunique() > 1 else None
    print("  unit confidence AUROC (WER > %.2f):" % ERR, {m: round(summary[s][f'auroc_unit_conf_{m}'], 3) for m in models})

# word-level: does the word probability still separate right from wrong words?
for s in ("eka_test", "pm_turn_test", "pm_window_test"):
    for m in models:
        if m == "turbo_lora":
            u = u_tl[u_tl.unit_id.isin(test_ids)]
            ws = []
            for r in u.to_dict("records"):
                if setname(r) == s:
                    _, words = score.score_unit(r, recs[r["unit_id"]], None)
                    ws += [(p, int(st != "C")) for _, p, st, _ in words]
            ws = pd.DataFrame(ws, columns=["p", "wrong"])
        else:
            wdf = pd.read_parquet(score.words_path(m if m == "small" else "large-v3-turbo"))
            u = models[m]
            keep = set(u[u.unit_id.isin(test_ids)].unit_id[u[u.unit_id.isin(test_ids)].apply(setname, axis=1) == s])
            wdf = wdf[wdf.unit_id.isin(keep)]
            ws = pd.DataFrame({"p": wdf.probability, "wrong": 1 - wdf.correct})
        summary[s][f"auroc_word_p_{m}"] = float(roc_auc_score(ws.wrong, -ws.p))
    print(f"  {s} word-probability AUROC:", {m: round(summary[s][f'auroc_word_p_{m}'], 3) for m in models})

# ---- B. verifier retrained on turbo+LoRA errors -----------------------------------------------------
install(u_tl, probs)
v_old = verifier.load("small")
W, G, U = verifier.unit_tables("lora", "large-v3-turbo", w8, ("train",))
keep = ~W.consultation.isin(LORA_TRAIN)
Wtr, Gtr = W[keep], G[~G.consultation.isin(LORA_TRAIN)]
print(f"\nverifier training: {Wtr.unit_id.nunique()} units ({W.unit_id.nunique() - Wtr.unit_id.nunique()} dropped as LoRA training data)")
fitted = verifier.fit_variants(Wtr, Gtr, cfg, ("full",))
v_new = verifier.Verifier(weights=w8, variants=fitted, meta={}, committed_variant="full")
pad = float(cfg["verifier"]["clip_pad_s"])
BUD = [0, 15, 30, 60, 120, 240]
for key, s in (("window", "pm_window_test"), ("turn", "pm_turn_test")):
    T = {"small": verifier.unit_tables("small", "large-v3-turbo", w8, (s,)), "tl": verifier.unit_tables("lora", "large-v3-turbo", w8, (s,))}
    cons = sorted(set(T["small"][2].consultation) & set(T["tl"][2].consultation))
    T = {k: tuple(d[d.consultation.isin(cons)].reset_index(drop=True) for d in t) for k, t in T.items()}
    runs = {"small + verifier": ("small", v_old), "tl + old verifier": ("tl", v_old), "tl + new verifier": ("tl", v_new)}
    left, auc = {}, {}
    for name, (m, v) in runs.items():
        Wt, Gt, Ut = T[m]
        sw, sg = v.predict(Wt, Gt, v.committed_variant)
        p = verifier.curve_parts(Wt.assign(pred=sw), Gt.assign(pred=sg), Ut, pad, BUD)
        for meth in ("verifier", "hybrid"):
            left[f"{name} [{meth}]"] = np.stack([p["total"][:, k][:, None] - p[meth][:, :, k] for k in (0, 1)])
        auc[name] = (round(roc_auc_score(Wt.b > 0, sw), 3), round(roc_auc_score(Gt.b > 0, sg), 3))
    draws = [rng.integers(0, len(cons), len(cons)) for _ in range(1000)]
    print(f"\n== {key}s: {len(cons)} consultations; errors left per consultation after B seconds")
    for k, kn in ((0, "weighted"), (1, "critical")):
        print(f"  {kn:<32}" + "".join(f"{b:>8}s" for b in BUD))
        for name, x in left.items():
            print(f"  {name:<32}" + "".join(f"{y:>9.1f}" for y in x[k].mean(0)))
        for a, b in (("small + verifier [hybrid]", "tl + new verifier [hybrid]"), ("tl + old verifier [hybrid]", "tl + new verifier [hybrid]")):
            d = left[b][k] - left[a][k]
            lo, hi = np.quantile([d[s_].mean(0) for s_ in draws], [0.025, 0.975], axis=0)
            print(f"    {b.split(' [')[0]} - {a.split(' [')[0]}".ljust(34) + "".join(f"{e:>+6.1f}[{l:+.0f},{h:+.0f}]" for e, l, h in zip(d.mean(0), lo, hi)))
        summary[f"{key}_{kn}_left"] = {n: x[k].mean(0).round(2).tolist() for n, x in left.items()}
    print("  AUROC words/gaps:", auc)
    summary[f"{key}_auroc"] = auc
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
print("\nwrote", OUT / "summary.json")
