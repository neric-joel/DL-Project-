"""Mixed-data turbo LoRA (PriMock57 training consultations + Eka calibration) against the PriMock-only one,
then a verifier trained on out-of-fold transcripts.

usage (repo root, repo venv): python turbo_final2.py PM_ONLY_JSONL MIXED_JSONL FOLDS_JSONL OUT_DIR

A. Per test set: WER, weighted WER, critical error rate, number / negation / medical-term recall for small,
   turbo, turbo+LoRA (PriMock only) and turbo+LoRA (mixed), with consultation (Eka: group) bootstrap CIs
   of mixed minus PriMock-only and mixed minus turbo; AUROC of unit confidence and word probability.
B. Verifier for the mixed model trained on its own realistic errors: the 8 PriMock57 training
   consultations from the 4 out-of-fold models, the 2 dev consultations and Eka validation from the main
   model (Eka calibration is its training data and is left out). Second opinion: original turbo.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from asrshift import score, verifier
from asrshift.config import load_config
from asrshift.textnorm import scoring_tokens
from lora_tables import load_lora

pm_jsonl, mix_jsonl, fold_jsonl, OUT = sys.argv[1], sys.argv[2], sys.argv[3], Path(sys.argv[4])
OUT.mkdir(parents=True, exist_ok=True)
cfg = load_config()
w8 = verifier.Weights.from_config(cfg)
recal = cfg["primock57"]["recalibration_consultations"]
LORA_TRAIN = set(recal[:-2])
ERR = score.LABELS["err"]
rng = np.random.default_rng(598)

# mixed records: out-of-fold transcripts replace the main model's for its 8 PriMock57 training consultations
folds = [json.loads(l) for l in open(fold_jsonl, encoding="utf-8") if l.strip()]
fold_ids = {r["unit_id"] for r in folds}
comb = OUT / "mixed_oof.jsonl"
with open(comb, "w", encoding="utf-8") as f:
    for l in open(mix_jsonl, encoding="utf-8"):
        if l.strip() and json.loads(l)["unit_id"] not in fold_ids:
            f.write(l.rstrip() + "\n")
    for r in folds:
        f.write(json.dumps(r) + "\n")
u_pm, p_pm, recs_pm = load_lora(pm_jsonl)
u_mx, p_mx, recs_mx = load_lora(str(comb))
oof_cons = set(u_mx[u_mx.unit_id.isin(fold_ids)].consultation)
assert oof_cons == LORA_TRAIN, oof_cons
models = {"small": score.load("small"), "turbo": score.load("large-v3-turbo"), "turbo_lora": u_pm, "mixed": u_mx}
LORA = {"turbo_lora": (u_pm, p_pm, recs_pm), "mixed": (u_mx, p_mx, recs_mx)}
test_ids = set(u_pm[u_pm.split == "test"].unit_id) & set(u_mx[u_mx.split == "test"].unit_id)
print(f"PriMock-only units {len(u_pm)}, mixed units {len(u_mx)} ({len(fold_ids)} out-of-fold), "
      f"common test units {len(test_ids)}", flush=True)
SETS = ("eka_test", "pm_turn_test", "pm_window_test")

# ---- A. transcription quality per set ---------------------------------------------------------------
rows = []
for m, u in models.items():
    u = u[u.unit_id.isin(test_ids)]
    for r in u.to_dict("records"):
        ref = scoring_tokens(r["reference"], reference=True)
        hyp = scoring_tokens(str(r.get("hypothesis") or ""))
        lab = verifier.labels(ref, hyp, w8)
        rows.append(dict(model=m, set=verifier._set(r),
                         group=r["consultation"] if r["dataset"] == "primock57" else r["group_id"],
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
COLS = ["n_ref", "errors", "w_err", "w_ref", "crit_err", "crit_ref", "n_num", "n_num_ok", "n_neg", "n_neg_ok",
        "n_term", "n_term_ok"]
summary = {}
for s in SETS:
    G = {m: D[(D.model == m) & (D.set == s)].groupby("group")[COLS].sum() for m in models}
    idx = G["turbo"].index.intersection(G["turbo_lora"].index).intersection(G["mixed"].index)
    draws = [rng.choice(idx, len(idx)) for _ in range(1000)]
    print(f"\n== {s}: {len(idx)} groups, {int(G['turbo'].loc[idx].n_ref.sum())} reference words")
    print(f"{'':<22}{'small':>9}{'turbo':>9}{'tl':>9}{'mixed':>9}   mixed - tl (95% CI)       mixed - turbo (95% CI)")
    summary[s] = {}
    for name, f in MET.items():
        v = {m: float(f(G[m].loc[idx].sum())) for m in models}
        summary[s][name] = dict(**v)
        txt = ""
        for b in ("turbo_lora", "turbo"):
            d = [f(G["mixed"].loc[x].sum()) - f(G[b].loc[x].sum()) for x in draws]
            lo, hi = np.nanquantile(d, [0.025, 0.975])
            summary[s][name][f"mixed_minus_{b}"] = [v["mixed"] - v[b], float(lo), float(hi)]
            txt += f"   {v['mixed'] - v[b]:+.4f} [{lo:+.4f}, {hi:+.4f}]"
        print(f"  {name:<20}" + "".join(f"{v[m]:>9.4f}" for m in models) + txt)
    for m in models:
        x = D[(D.model == m) & (D.set == s)]
        summary[s][f"auroc_unit_conf_{m}"] = float(roc_auc_score(x.err, -x.conf)) if x.err.nunique() > 1 else None
    print("  unit confidence AUROC (WER > %.2f):" % ERR, {m: round(summary[s][f'auroc_unit_conf_{m}'], 3) for m in models})

# word-level: does the word probability still separate right from wrong words?
for s in SETS:
    for m in models:
        if m in LORA:
            u_l, _, recs = LORA[m]
            ws = []
            for r in u_l[u_l.unit_id.isin(test_ids)].to_dict("records"):
                if verifier._set(r) == s:
                    _, words = score.score_unit(r, recs[r["unit_id"]], None)
                    ws += [(p, int(st != "C")) for _, p, st, _ in words]
            ws = pd.DataFrame(ws, columns=["p", "wrong"])
        else:
            wdf = pd.read_parquet(score.words_path(m if m == "small" else "large-v3-turbo"))
            u = models[m][models[m].unit_id.isin(test_ids)]
            keep = set(u.unit_id[u.apply(verifier._set, axis=1) == s])
            wdf = wdf[wdf.unit_id.isin(keep)]
            ws = pd.DataFrame({"p": wdf.probability, "wrong": 1 - wdf.correct})
        summary[s][f"auroc_word_p_{m}"] = float(roc_auc_score(ws.wrong, -ws.p))
    print(f"  {s} word-probability AUROC:", {m: round(summary[s][f'auroc_word_p_{m}'], 3) for m in models})

# ---- B. verifier trained on the mixed model's out-of-fold errors ---------------------------------------
real = verifier._committed
_, _, other_hyp, other_probs = real("small", "large-v3-turbo")
TABLES = {"lora": (u_pm, p_pm), "mixed": (u_mx, p_mx)}
verifier._committed = lambda m, sec: (*TABLES[m], other_hyp, other_probs) if m in TABLES else real(m, sec)
v_old = verifier.load("small")
W, G, _ = verifier.unit_tables("mixed", "large-v3-turbo", w8, ("train",))
print(f"\nverifier training: {W.unit_id.nunique()} units "
      f"({W[W.dataset == 'eka'].unit_id.nunique() if 'dataset' in W else '?'} Eka)", flush=True)
v_new = verifier.Verifier(weights=w8, variants=verifier.fit_variants(W, G, cfg, ("full",)), meta={},
                          committed_variant="full")
pad = float(cfg["verifier"]["clip_pad_s"])
BUD = [0, 15, 30, 60, 120, 240]
for key, s in (("window", "pm_window_test"), ("turn", "pm_turn_test")):
    T = {m: verifier.unit_tables(m, "large-v3-turbo", w8, (s,)) for m in ("small", "lora", "mixed")}
    cons = sorted(set.intersection(*(set(t[2].consultation) for t in T.values())))
    T = {k: tuple(d[d.consultation.isin(cons)].reset_index(drop=True) for d in t) for k, t in T.items()}
    runs = {"small + verifier": ("small", v_old), "tl + old verifier": ("lora", v_old),
            "mixed + old verifier": ("mixed", v_old), "mixed + oof verifier": ("mixed", v_new)}
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
        for a, b in (("tl + old verifier [verifier]", "mixed + old verifier [verifier]"),
                     ("mixed + old verifier [verifier]", "mixed + oof verifier [verifier]"),
                     ("mixed + old verifier [hybrid]", "mixed + oof verifier [hybrid]"),
                     ("tl + old verifier [hybrid]", "mixed + oof verifier [hybrid]")):
            d = left[b][k] - left[a][k]
            lo, hi = np.quantile([d[s_].mean(0) for s_ in draws], [0.025, 0.975], axis=0)
            print(f"    {b}  minus  {a}")
            print(" " * 34 + "".join(f"{e:>+6.1f}[{l:+.0f},{h:+.0f}]" for e, l, h in zip(d.mean(0), lo, hi)))
        summary[f"{key}_{kn}_left"] = {n: x[k].mean(0).round(2).tolist() for n, x in left.items()}
    print("  AUROC words/gaps:", auc)
    summary[f"{key}_auroc"] = auc
json.dump(summary, open(OUT / "summary.json", "w"), indent=1, default=float)
print("\nwrote", OUT / "summary.json")
