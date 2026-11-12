"""Step 5: word-level combination of LoRA-small and large-v3-turbo transcripts (two-system ROVER with
word confidences). The rule is chosen on the two dev consultations (held out from LoRA training, never
test) and then applied once to the 12 test consultations.

usage (repo root, repo venv): python combine.py LORA_JSONL OUT_DIR

Each window: align the two token sequences (edit distance). Per slot
  agree                 keep
  disagree (a vs b)     turbo's word if p_turbo - p_lora > delta
  LoRA-only word        keep if p_lora >= tau_a   (tau_c instead for numbers / negations)
  turbo-only word       add  if p_turbo >= tau_b
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from asrshift import score, verifier
from asrshift.align import align
from asrshift.config import load_config
from asrshift.textnorm import scoring_tokens
from lora_tables import install, load_lora

jsonl, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
w8 = verifier.Weights.from_config(load_config())
u_lora, _, recs = load_lora(jsonl)
other_hyp, other_probs = install(u_lora, {})
turbo_units = score.load("large-v3-turbo").set_index("unit_id")
DEV = ["day5_consultation04", "day5_consultation07"]


def lora_tokens(uid, text):
    disp = str(text or "").split()
    toks, owner = verifier.hyp_tokens(disp)
    wp = [w["probability"] for w in recs[uid]["words"]]
    if len(wp) == len(disp):
        return toks, np.array([wp[i] for i in owner])
    return toks, np.full(len(toks), np.exp(np.mean([s["avg_logprob"] for s in recs[uid]["segments"]] or [-1])))


def turbo_tokens(uid):
    toks, p = verifier._other_tokens(str(other_hyp.get(uid, "") or ""), other_probs.get(uid))
    if p is None:
        c = turbo_units.loc[uid, "conf"] if uid in turbo_units.index else 0.5
        p = np.full(len(toks), float(c) if pd.notna(c) else 0.5)
    return toks, p


def pair_align(a, b):
    n, m = len(a), len(b)
    D = np.zeros((n + 1, m + 1), int); D[:, 0] = np.arange(n + 1); D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j - 1] + (a[i - 1] != b[j - 1]), D[i - 1, j] + 1, D[i, j - 1] + 1)
    i, j, out = n, m, []
    while i or j:
        if i and j and D[i, j] == D[i - 1, j - 1] + (a[i - 1] != b[j - 1]):
            out.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif i and D[i, j] == D[i - 1, j] + 1:
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    return out[::-1]


def slots(row):
    a, pa = lora_tokens(row.unit_id, row.hypothesis)
    b, pb = turbo_tokens(row.unit_id)
    out = []
    for i, j in pair_align(a, b):
        ta, tb = (a[i] if i is not None else None), (b[j] if j is not None else None)
        crit = ta is not None and score.is_critical(a, i)
        out.append((ta, pa[i] if i is not None else 0.0, tb, pb[j] if j is not None else 0.0, crit))
    return out


def combine(sl, delta, tau_a, tau_b, tau_c):
    out = []
    for ta, pa, tb, pb, crit in sl:
        if ta is not None and tb is not None:
            out.append(ta if ta == tb or pb - pa <= delta else tb)
        elif ta is not None:
            if pa >= (tau_c if crit else tau_a):
                out.append(ta)
        elif pb >= tau_b:
            out.append(tb)
    return out


def wwer(rows, toks):
    e = r = n_err = n_ref = 0.0
    for row, t in zip(rows, toks):
        ref = scoring_tokens(row.reference, reference=True)
        lab = verifier.labels(ref, t, w8)
        e += lab["word_b"].sum() + lab["gap_b"].sum(); r += lab["ref_weight"]
        a = align(ref, t); n_err += a.errors; n_ref += a.n_ref
    return e / r, n_err / n_ref


win = u_lora[u_lora.subset == "window"]
dev = list(win[win.consultation.isin(DEV)].itertuples())
test = list(win[win.split == "test"].itertuples())
S_dev, S_test = [slots(r) for r in dev], [slots(r) for r in test]
print(f"{len(dev)} dev windows, {len(test)} test windows", flush=True)

grid = list(itertools.product([-1.0, -0.2, 0.0, 0.2, 0.5, 1.01], [0.0, 0.3, 0.5, 0.7], [0.5, 0.7, 0.9, 1.01],
                              [0.0, 0.5, 0.8, 1.01]))
rows = []
for g in grid:
    ww, wr = wwer(dev, [combine(s, *g) for s in S_dev])
    rows.append(dict(delta=g[0], tau_a=g[1], tau_b=g[2], tau_c=g[3], wwer=ww, wer=wr))
R = pd.DataFrame(rows).sort_values(["wwer", "wer"])
R.to_csv(OUT / "dev_grid.csv", index=False)
lora_only = R[(R.delta == 1.01) & (R.tau_a == 0) & (R.tau_b == 1.01) & (R.tau_c == 0)].iloc[0]
best = R.iloc[0]
print("dev: LoRA alone wwer %.4f wer %.4f" % (lora_only.wwer, lora_only.wer))
print("dev: best rule", best.to_dict())
print("dev top 5:\n", R.head(5).to_string(index=False))
g = (best.delta, best.tau_a, best.tau_b, best.tau_c)
hyp = [" ".join(combine(s, *g)) for s in S_test]
pd.DataFrame(dict(unit_id=[r.unit_id for r in test], consultation=[r.consultation for r in test],
                  hypothesis=hyp)).to_csv(OUT / "hyp_combo.csv", index=False)
json.dump(dict(rule=dict(zip(["delta", "tau_a", "tau_b", "tau_c"], map(float, g))),
               dev_lora=[float(lora_only.wwer), float(lora_only.wer)], dev_best=[float(best.wwer), float(best.wer)]),
          open(OUT / "rule.json", "w"), indent=1)
print("wrote", OUT / "hyp_combo.csv")
