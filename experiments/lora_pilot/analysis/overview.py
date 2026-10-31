"""Overall metrics of the final system (large-v3-turbo + mixed LoRA, pooled verifier) for the results page,
with the starting point (whisper-small + its verifier) beside it.

usage (repo root, repo venv): python experiments/lora_pilot/analysis/overview.py
writes results/tables/turbo-lora-mixed/overview.json

Transcripts, per test set: word accuracy, WER, weighted WER, critical error rate, deletions, and
precision / recall / F1 of numbers and negations. Mistake spotting, per test set: AUROC and AUPRC of the
verifier for wrong words and for gaps with dropped words, and what flagging the riskiest 10% of words
catches (precision, recall, F1, accuracy, share of critical errors), against Whisper's own word
probability. Listening: errors left per consultation against seconds of listening, from the saved curves.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from asrshift import score, verifier
from asrshift.config import load_config
from asrshift.paths import RESULTS_DIR
from asrshift.textnorm import scoring_tokens

FINAL, START, SECOND = "turbo-lora-mixed", "small", "large-v3-turbo"
SETS = ("pm_window_test", "pm_turn_test", "eka_test")
cfg = load_config()
w8 = verifier.Weights.from_config(cfg)


def transcripts(model: str) -> dict:
    u = score.load(model)
    u = u[u.split == "test"]
    out = {}
    for s in SETS:
        g = u[u.apply(verifier._set, axis=1) == s]
        t = g[["n_ref", "hits", "errors", "sub", "del", "ins", "n_num", "n_num_correct", "n_num_hyp", "n_num_false",
               "n_neg", "n_neg_correct", "n_neg_hyp", "n_neg_false"]].sum()
        w_err = w_ref = crit_err = crit_ref = 0.0
        for r in g.to_dict("records"):
            ref = scoring_tokens(r["reference"], reference=True)
            lab = verifier.labels(ref, scoring_tokens(str(r.get("hypothesis") or "")), w8)
            w_err += lab["word_b"].sum() + lab["gap_b"].sum(); w_ref += lab["ref_weight"]
            crit_err += lab["word_cr"].sum() + lab["gap_cr"].sum()
            crit_ref += sum(c == "crit" for c in verifier.categories(ref, w8))

        def prf(ok, n, hyp, false):
            p, r = (hyp - false) / hyp, ok / n
            return round(100 * p, 1), round(100 * r, 1), round(100 * 2 * p * r / (p + r), 1)

        out[s] = {"words": int(t.n_ref), "consultations": int(g.consultation.nunique() if s != "eka_test" else g.group_id.nunique()),
                  "word_accuracy": round(100 * t.hits / t.n_ref, 1), "wer": round(100 * t.errors / t.n_ref, 2),
                  "weighted_wer": round(100 * w_err / w_ref, 2), "critical_error_rate": round(100 * crit_err / crit_ref, 1),
                  "deletions": round(100 * t["del"] / t.n_ref, 2), "substitutions": round(100 * t["sub"] / t.n_ref, 2),
                  "insertions": round(100 * t.ins / t.n_ref, 2),
                  "numbers": dict(zip(("precision", "recall", "f1"), prf(t.n_num_correct, t.n_num, t.n_num_hyp, t.n_num_false))),
                  "negations": dict(zip(("precision", "recall", "f1"), prf(t.n_neg_correct, t.n_neg, t.n_neg_hyp, t.n_neg_false)))}
    return out


def flagged(y, crit, s, share=0.10) -> dict:
    f = s >= np.quantile(s, 1 - share)
    tp, fp, fn, tn = (f & y).sum(), (f & ~y).sum(), (~f & y).sum(), (~f & ~y).sum()
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": round(100 * p, 1), "recall": round(100 * r, 1), "f1": round(100 * 2 * p * r / max(p + r, 1e-12), 1),
            "accuracy": round(100 * (tp + tn) / len(y), 1), "critical_recall": round(100 * (f & crit).sum() / max(crit.sum(), 1), 1)}


def detection(model: str) -> dict:
    v = verifier.load(model)
    out = {}
    for s in SETS:
        W, G, _ = verifier.unit_tables(model, SECOND, w8, (s,))
        sw, sg = v.predict(W, G, v.committed_variant)
        yw, cw, yg, cg = W.b.to_numpy() > 0, W.cr.to_numpy() > 0, G.b.to_numpy() > 0, G.cr.to_numpy() > 0
        conf = 1 - W.p.to_numpy()
        out[s] = {"wrong_word_rate": round(100 * yw.mean(), 1),
                  "words": {"auroc": round(roc_auc_score(yw, sw), 3), "auprc": round(average_precision_score(yw, sw), 3),
                            "top10": flagged(yw, cw, sw)},
                  "gaps": {"auroc": round(roc_auc_score(yg, sg), 3), "auprc": round(average_precision_score(yg, sg), 3)},
                  "whisper_confidence": {"auroc": round(roc_auc_score(yw, conf), 3), "auprc": round(average_precision_score(yw, conf), 3),
                                         "top10": flagged(yw, cw, conf)}}
        print(model, s, out[s]["words"]["auroc"], flush=True)
    return out


def listening() -> dict:
    out = {}
    for key in ("window", "turn"):
        meth = "verifier" if key == "window" else "hybrid"
        d = {}
        for name, m in (("final", FINAL), ("start", START)):
            c = verifier.review_curves(m)
            meta, n = c["meta"][key], c["meta"][key]["consultations"]
            d["x_seconds"] = c["x_seconds"]
            for e in ("weighted", "critical"):
                tot = meta[f"{e}_errors"] / n
                d.setdefault(e, {})[f"{name}_flags"] = [round(v * tot, 2) for v in c[key][e][meth]]
                d[e][f"{name}_whole"] = [round(v * tot, 2) for v in c[key][e]["window_review"]]
        out[key] = d
    return out


def main():
    res = {"final": {"transcripts": transcripts(FINAL), "detection": detection(FINAL)},
           "start": {"transcripts": transcripts(START), "detection": detection(START)},
           "listening": listening(),
           "meta": {"final": "whisper-large-v3-turbo + LoRA (PriMock57 training consultations + Eka calibration), pooled verifier",
                    "start": "whisper-small + its verifier", "flag_share": 0.10}}
    path = RESULTS_DIR / "tables" / FINAL / "overview.json"
    path.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", path)


if __name__ == "__main__":
    main()
