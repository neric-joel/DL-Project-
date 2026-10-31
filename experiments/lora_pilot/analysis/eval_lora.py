"""Full evaluation of the LoRA pilot against base whisper-small and the committed transcripts.

usage (from the repo root, repo venv): python eval_lora.py OUT_DIR

Systems, all on the same 277 test windows of 12 PriMock57 consultations:
  small_gpu  committed whisper-small transcripts (fp16, GPU)
  base       whisper-small, int8 CPU (the pilot's baseline)
  lora       whisper-small + LoRA, int8 CPU
  turbo      committed large-v3-turbo transcripts
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from asrshift import terms
from asrshift.align import align, ops
from asrshift.config import load_config
from asrshift.score import score_unit
from asrshift.textnorm import scoring_tokens
from asrshift.verifier import Weights, categories, labels

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PILOT = Path("experiments/lora_pilot/out")
ALL = os.environ.get("EVAL_ALL") == "1"          # all 47 test consultations: committed models + extras only
TEST = sorted(pd.read_csv(PILOT / "hyp_base.csv").consultation.unique())

small = pd.read_csv("results/units/units_small.csv.gz", low_memory=False)
win = small[(small.subset == "window") & (small.split == "test") & (ALL | small.consultation.isin(TEST))].copy()
win = win.sort_values(["consultation", "start"]).reset_index(drop=True)
turbo = pd.read_csv("results/units/units_large-v3-turbo.csv.gz", low_memory=False).set_index("unit_id").hypothesis
hyps = {
    "small_gpu": win.set_index("unit_id").hypothesis,
    "base": pd.read_csv(PILOT / "hyp_base.csv").set_index("unit_id").hypothesis,
    "lora": pd.read_csv(PILOT / "hyp_lora.csv").set_index("unit_id").hypothesis,
    "turbo": turbo,
}
if ALL:
    del hyps["base"], hyps["lora"]
for extra in sys.argv[2:]:                      # name=path to another hypothesis csv (unit_id, hypothesis)
    k, path = extra.split("=", 1)
    hyps[k] = pd.read_csv(path).set_index("unit_id").hypothesis
w8 = Weights.from_config(load_config())
lex = terms.load()


def runs(seq, sym, k=4):
    """Number of runs of at least k consecutive `sym` ops, and the words in them."""
    n = words = cur = 0
    for s in seq + ["#"]:
        if s == sym:
            cur += 1
        else:
            if cur >= k:
                n, words = n + 1, words + cur
            cur = 0
    return n, words


rows, status = [], {}
for name, H in hyps.items():
    for r in win.to_dict("records"):
        hyp = str(H.get(r["unit_id"], "") if pd.notna(H.get(r["unit_id"], "")) else "")
        sc, _ = score_unit(r, {"text": hyp}, lex)
        rt, ht = scoring_tokens(r["reference"], reference=True), scoring_tokens(hyp)
        a = align(rt, ht)
        lab = labels(rt, ht, w8)
        rc = categories(rt, w8)
        seq = [op for op, _, _ in ops(a)]
        dr, drw = runs(seq, "D")
        ir, irw = runs(seq, "I")
        status[(name, r["unit_id"])] = a.ref_status
        rows.append(dict(system=name, unit_id=r["unit_id"], consultation=r["consultation"],
                         n_ref=sc["n_ref"], errors=sc["errors"], sub=sc["sub"], dele=sc["del"], ins=sc["ins"],
                         w_err=float(lab["word_b"].sum() + lab["gap_b"].sum()), w_ref=lab["ref_weight"],
                         crit_err=int(lab["word_cr"].sum() + lab["gap_cr"].sum()),
                         crit_ref=sum(c == "crit" and s != "W" for c, s in zip(rc, a.ref_status)),
                         med_ref=sum(c == "med" and s != "W" for c, s in zip(rc, a.ref_status)),
                         med_ok=sum(c == "med" and s == "C" for c, s in zip(rc, a.ref_status)),
                         n_num=sc["n_num"], n_num_ok=sc["n_num_correct"], n_neg=sc["n_neg"], n_neg_ok=sc["n_neg_correct"],
                         crit_false=sc["n_crit_false"], n_term=sc.get("n_term", 0), n_term_ok=sc.get("n_term_correct", 0),
                         drop_runs=dr, drop_words=drw, ins_runs=ir, ins_words=irw,
                         unit_wer=sc["wer"]))
D = pd.DataFrame(rows)
D.to_csv(OUT / "per_window.csv", index=False)

SUMS = ["n_ref", "errors", "sub", "dele", "ins", "w_err", "w_ref", "crit_err", "crit_ref", "med_ref", "med_ok",
        "n_num", "n_num_ok", "n_neg", "n_neg_ok", "crit_false", "n_term", "n_term_ok", "drop_runs", "drop_words",
        "ins_runs", "ins_words"]
METRICS = {
    "WER": lambda c: c.errors / c.n_ref,
    "weighted WER": lambda c: c.w_err / c.w_ref,
    "substitution rate": lambda c: c["sub"] / c.n_ref,
    "deletion rate": lambda c: c.dele / c.n_ref,
    "insertion rate": lambda c: c.ins / c.n_ref,
    "critical error rate": lambda c: c.crit_err / c.crit_ref,
    "medical word error rate": lambda c: 1 - c.med_ok / c.med_ref,
    "medical term recall (study)": lambda c: c.n_term_ok / c.n_term,
    "number recall (study)": lambda c: c.n_num_ok / c.n_num,
    "negation recall (study)": lambda c: c.n_neg_ok / c.n_neg,
    "false numbers/negations per 1k words": lambda c: 1000 * c.crit_false / c.n_ref,
    "dropped chunks (>=4 words) per 1k words": lambda c: 1000 * c.drop_runs / c.n_ref,
    "share of deletions in dropped chunks": lambda c: c.drop_words / c.dele,
    "hallucinated chunks (>=4 words) per 1k words": lambda c: 1000 * c.ins_runs / c.n_ref,
}
C = {s: g.groupby("consultation")[SUMS].sum() for s, g in D.groupby("system")}
pooled = {s: {m: float(f(c.sum())) for m, f in METRICS.items()} for s, c in C.items()}
for s, g in D.groupby("system"):
    pooled[s]["windows with WER > 20%"] = float((g.unit_wer > 0.2).mean())

rng = np.random.default_rng(598)
idx = np.array(C["small_gpu"].index)
draws = [rng.choice(idx, size=len(idx), replace=True) for _ in range(1000)]


def ci(a, b, f):
    v = [f(C[b].loc[s].sum()) - f(C[a].loc[s].sum()) for s in draws]
    return float(f(C[b].sum()) - f(C[a].sum())), float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


EXTRA = [k for k in hyps if k not in ("small_gpu", "base", "lora", "turbo")]
PAIRS = ([] if ALL else [("base", "lora"), ("turbo", "lora")]) + [(b, k) for k in EXTRA for b in (("small_gpu", "turbo") if ALL else ("lora", "turbo"))]
diffs = {pair: {m: ci(*pair, f) for m, f in METRICS.items()} for pair in PAIRS}

# churn: reference words base got wrong that lora gets right, and the reverse
fixed = broken = both_wrong = total = 0
for uid in ([] if ALL else win.unit_id):
    sb, sl = status[("base", uid)], status[("lora", uid)]
    for x, y in zip(sb, sl):
        if x == "W" or y == "W":
            continue
        total += 1
        fixed += x != "C" and y == "C"
        broken += x == "C" and y != "C"
        both_wrong += x != "C" and y != "C"
churn = dict(ref_words=total, fixed=fixed, broken=broken, wrong_in_both=both_wrong)

# per consultation wins on each headline metric
wins = {} if ALL else {m: int(sum(METRICS[m](C["lora"].loc[k]) < METRICS[m](C["base"].loc[k]) for k in idx))
        for m in ["WER", "weighted WER", "critical error rate", "deletion rate", "substitution rate"]}

summary = dict(pooled=pooled, diffs={f"{b}-{a}": d for (a, b), d in diffs.items()}, churn=churn, wins_of_12=wins)
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
T = pd.DataFrame(pooled)[list(hyps)]
print(T.to_string(float_format=lambda x: f"{x:.4f}"))
for k, d in summary["diffs"].items():
    print(f"\n{k} (95% CI, consultation bootstrap)")
    for m, (e, lo, hi) in d.items():
        print(f"  {m:<46}{e:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
print("\nchurn base -> lora:", churn)
print("lora better than base, consultations of 12:", wins)
pd.DataFrame([dict(unit_id=u, reference=r, **{s: str(hyps[s].get(u, "")) for s in hyps})
              for u, r in zip(win.unit_id, win.reference)]).to_csv(OUT / "texts.csv", index=False)
