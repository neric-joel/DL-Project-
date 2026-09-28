"""Run every policy on every evaluation set and write the result tables.

Evaluation sets: Eka test (in-domain reference), PriMock57 test windows (main target) and PriMock57
test turns (control). Thresholds always come from a tuning set disjoint from the evaluation set:
Eka validation for the frozen policies, the PriMock recalibration consultations for the
recalibrated/re-tuned ones. The plug-in rule is the exception by design: it reads only the policy's
own probabilities on the (unlabelled) evaluation stream. Intervals are 95% cluster-bootstrap
intervals over speakers/sessions (Eka) or consultations (PriMock), holding the fitted policies and
thresholds fixed; variability from the recalibration sample is measured by ``learning_curve``.

Outputs: ``results/tables/<model>/`` (CSV) and ``results/metrics/<model>.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from asrshift import metrics as M
from asrshift import score
from asrshift.features import FEATURES
from asrshift.paths import RESULTS_DIR
from asrshift.policies import Platt, build_policies, crossfit_ceiling, fit_logreg

BUDGETS = (0.01, 0.05, 0.10, 0.20, 0.30)
RISK_TARGETS = (0.10, 0.20, 0.30)
MAIN_BUDGET = 0.10
EVAL_SETS = ("eka_test", "pm_window_test", "pm_turn_test")
LENGTH_BINS = ((1, 9), (10, 30), (31, 50), (51, 10_000))
MATCHED_BIN = (25, 41)
LONG_WINDOW_S = 30.6
RISK_N_MIN = 30  # a risk-target threshold must accept at least this many tuning units
RESIDUAL_KEYS = ("sub", "del", "ins", "n_num", "n_num_correct", "n_neg", "n_neg_correct", "crit_err")
P4_NOTE = "P3 score with thresholds re-tuned on PriMock57 recalibration"


def _subsets(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    eka = df[df["dataset"] == "eka"]
    pm = df[df["dataset"] == "primock57"]
    win, turn = pm[pm["subset"] == "window"], pm[pm["subset"].str.startswith("turn")]
    return {
        "eka_calibration": eka[eka["split"] == "calibration"],
        "eka_validation": eka[eka["split"] == "validation"],
        "eka_test": eka[eka["split"] == "test"],
        "pm_window_recal": win[win["split"] == "recalibration"],
        "pm_window_test": win[win["split"] == "test"],
        "pm_turn_recal": turn[turn["split"] == "recalibration"],
        "pm_turn_test": turn[turn["split"] == "test"],
    }


def _extra(d: pd.DataFrame) -> dict:
    out = {k: d[k].fillna(0).to_numpy(float) for k in RESIDUAL_KEYS if k in d}
    if d["dataset"].iloc[0] == "eka" and "n_ent" in d:
        out["n_ent"] = d["n_ent"].fillna(0).to_numpy(float)
        out["n_ent_correct"] = d["n_ent_correct"].fillna(0).to_numpy(float)
    if "n_term" in d and d["n_term"].fillna(0).sum() > 0:
        out["n_term"] = d["n_term"].fillna(0).to_numpy(float)
        out["n_term_correct"] = d["n_term_correct"].fillna(0).to_numpy(float)
    return out


def _take(extra: dict, ix: np.ndarray) -> dict:
    return {k: v[ix] for k, v in extra.items()}


def _rate(num: np.ndarray, den: np.ndarray):
    return lambda ix: float(num[ix].sum() / den[ix].sum()) if den[ix].sum() else float("nan")


# --------------------------------------------------------------------------------------------- ASR

def asr_summary(sets: dict, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for name in ("eka_calibration", "eka_validation", "eka_test", "pm_window_recal", "pm_window_test",
                 "pm_turn_recal", "pm_turn_test"):
        d = sets[name]
        bs = M.ClusterBootstrap(d["group_id"].to_numpy(), n_boot, seed)
        n_ref = d["n_ref"].to_numpy(float)
        col = lambda c: d[c].fillna(0).to_numpy(float)  # noqa: E731
        stats = {
            "wer": _rate(col("errors"), n_ref),
            "sub_rate": _rate(col("sub"), n_ref),
            "del_rate": _rate(col("del"), n_ref),
            "ins_rate": _rate(col("ins"), n_ref),
            "err_rate": lambda ix, y=col("err"): float(y[ix].mean()),
            "crit_err_rate": lambda ix, y=col("crit_err"): float(y[ix].mean()),
            "number_acc": _rate(col("n_num_correct"), col("n_num")),
            "negation_acc": _rate(col("n_neg_correct"), col("n_neg")),
            "number_false_rate": _rate(col("n_num_false"), col("n_num_hyp")),
            "negation_false_rate": _rate(col("n_neg_false"), col("n_neg_hyp")),
        }
        if d["dataset"].iloc[0] == "eka":
            stats["entity_acc"] = _rate(col("n_ent_correct"), col("n_ent"))
        if "n_term" in d and col("n_term").sum() > 0:
            # exploratory: shared medical vocabulary (Eka-annotated terms that also occur in PriMock57)
            stats["term_acc"] = _rate(col("n_term_correct"), col("n_term"))
            row_terms = int(col("n_term").sum())
        row = {"set": name, "units": len(d), "groups": d["group_id"].nunique(), "hours": d["duration"].sum() / 3600,
               "ref_words": int(n_ref.sum()), "median_ref_words": float(np.median(n_ref)),
               "any_err_rate": float(d["any_err"].mean()), "severe_err_rate": float(d["severe_err"].mean()),
               "wer_nocollapse": float(d["errors_nocollapse"].sum() / d["n_ref_nocollapse"].sum()),
               "digit_acc_draft_definition": float(col("n_digit_correct").sum() / max(col("n_digit").sum(), 1)),
               "n_numbers": int(col("n_num").sum()), "n_negations": int(col("n_neg").sum()),
               "mean_conf": float(d["conf"].mean()), "empty_hyp_rate": float(d["empty_hyp"].mean()),
               "wordlevel_disagreement": float((d["wl_wrong"] != d["sub"] + d["ins"]).mean())}
        if "term_acc" in stats:
            row["n_terms"] = row_terms
        all_ix = np.arange(len(d))
        for k, f in stats.items():
            row[k] = f(all_ix)
            row[f"{k}_lo"], row[f"{k}_hi"] = bs.interval(f)
        rows.append(row)
    return pd.DataFrame(rows)


def entity_table(eka_test: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in ("drugs", "advices", "diagnostics", "clinical_findings", "misc_medical"):
        n, c = f"n_ent_{t}", f"n_ent_{t}_correct"
        if n not in eka_test:
            continue
        tot, ok = eka_test[n].fillna(0).sum(), eka_test[c].fillna(0).sum()
        rows.append({"entity_type": t, "entities": int(tot), "accuracy": ok / tot if tot else np.nan,
                     "error_rate": 1 - ok / tot if tot else np.nan})
    tot, ok = eka_test["n_ent"].fillna(0).sum(), eka_test["n_ent_correct"].fillna(0).sum()
    rows.append({"entity_type": "all", "entities": int(tot), "accuracy": ok / tot, "error_rate": 1 - ok / tot})
    for ctx, g in eka_test.groupby("recording_context"):
        acc = g["n_ent_correct"].sum() / g["n_ent"].sum()
        rows.append({"entity_type": f"context:{ctx}", "entities": int(g["n_ent"].sum()), "accuracy": acc,
                     "error_rate": 1 - acc})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ main tables

def _tuning_set(policy, eval_name: str, sets: dict) -> pd.DataFrame | None:
    if policy.tuned_on == "eka":
        return sets["eka_validation"]
    if eval_name.startswith("eka"):
        return None
    return sets["pm_window_recal"] if "window" in eval_name else sets["pm_turn_recal"]


def evaluate_label(df: pd.DataFrame, label: str, n_boot: int, seed: int, eval_sets=EVAL_SETS,
                   with_target: bool = True) -> dict:
    sets = _subsets(df)
    policies = {"window": build_policies(sets["eka_calibration"], sets["pm_window_recal"] if with_target else None, label, seed),
                "turn": build_policies(sets["eka_calibration"], sets["pm_turn_recal"] if with_target else None, label, seed)}
    ops, cal_rows, rank_rows, rel_rows = [], [], [], []
    scores: dict[str, dict[str, np.ndarray]] = {}
    for ev in eval_sets:
        d = sets[ev]
        y = d[label].to_numpy(int)
        errors, n_ref = d["errors"].to_numpy(float), d["n_ref"].to_numpy(float)
        extra = _extra(d)
        bs = M.ClusterBootstrap(d["group_id"].to_numpy(), n_boot, seed)
        scores[ev] = {}
        for pol in policies["turn" if "turn" in ev else "window"]:
            tune = _tuning_set(pol, ev, sets)
            if tune is None:
                continue
            r = pol.score(d)
            scores[ev][pol.name] = r
            rank_rows.append({"eval": ev, "policy": pol.name, "auroc": M.auroc(r, y), "aurc": M.aurc(r, y),
                              "aurc_wer": M.aurc(r, errors / np.maximum(n_ref, 1), n_ref),
                              **dict(zip(("auroc_lo", "auroc_hi"), bs.interval(lambda ix: M.auroc(r[ix], y[ix]))))})
            if pol.probabilistic or pol.name == "P1":
                p = r if pol.probabilistic else np.clip(r, 0, 1)
                row = {"eval": ev, "policy": pol.name, "probabilistic": pol.probabilistic, "n": len(d),
                       "base_rate": float(y.mean()), "mean_p": float(p.mean()),
                       "ece": M.ece(p, y), "brier": M.brier(p, y), "citl": M.calibration_in_the_large(p, y)}
                for k, f in (("ece", M.ece), ("brier", M.brier), ("citl", M.calibration_in_the_large)):
                    row[f"{k}_lo"], row[f"{k}_hi"] = bs.interval(lambda ix, f=f: f(p[ix], y[ix]))
                cal_rows.append(row)
                rel_rows.extend({"eval": ev, "policy": pol.name, **b} for b in M.reliability(p, y))

            r_tune = pol.score(tune)
            y_tune = tune[label].to_numpy(int)
            points = [(f"budget_{b:.2f}", ">=", M.budget_threshold(r_tune, b), b) for b in BUDGETS]
            # empirical risk-target rules are rank-based, so every policy gets them (Amendment 2)
            points += [(f"risk_{a:.2f}", ">", M.risk_target_threshold(r_tune, y_tune, a, n_min=RISK_N_MIN), a)
                       for a in RISK_TARGETS]
            points += [(f"riskcp_{a:.2f}", ">", M.risk_target_threshold(r_tune, y_tune, a, n_min=RISK_N_MIN, bound="cp"), a)
                       for a in RISK_TARGETS]
            if pol.name in ("P3", "P3-conf", "P4", "P4-refit"):
                points += [(f"plugin_{a:.2f}", ">", M.plugin_threshold(r, a), a) for a in RISK_TARGETS]
            if pol.name in ("P2", "P2-retuned", "P2-rawCR"):
                points.append(("whisper_rule", ">=", 1.0, None))
            for name, op_, thr, target in points:
                decide = (lambda ix, t=thr: r[ix] >= t) if op_ == ">=" else (lambda ix, t=thr: r[ix] > t)
                all_ix = np.arange(len(d))
                o = M.operating_point(decide(all_ix), y, errors, n_ref, extra, r if pol.probabilistic else None)
                row = {"eval": ev, "policy": pol.name, "point": name, "target": target, "threshold": thr,
                       "feasible": bool(np.isfinite(thr)),
                       "note": P4_NOTE if pol.name == "P4" and not name.startswith("plugin") else "", **o}
                for k in ("review_rate", "err_caught", "accepted_err_rate", "accepted_crit_err_rate"):
                    row[f"{k}_lo"], row[f"{k}_hi"] = bs.interval(
                        lambda ix, k=k, dec=decide: M.operating_point(dec(ix), y[ix], errors[ix], n_ref[ix],
                                                                      _take(extra, ix)).get(k, np.nan))
                ops.append(row)
        if with_target and not ev.startswith("eka"):
            p = crossfit_ceiling(d, label, seed)
            scores[ev]["ceiling"] = p
            rank_rows.append({"eval": ev, "policy": "ceiling", "auroc": M.auroc(p, y), "aurc": M.aurc(p, y),
                              "aurc_wer": M.aurc(p, errors / np.maximum(n_ref, 1), n_ref)})
            cal_rows.append({"eval": ev, "policy": "ceiling", "probabilistic": True, "n": len(d),
                             "base_rate": float(y.mean()), "mean_p": float(p.mean()), "ece": M.ece(p, y),
                             "brier": M.brier(p, y), "citl": M.calibration_in_the_large(p, y)})
        rank_rows.append({"eval": ev, "policy": "oracle", "aurc": M.aurc(y + 1e-9 * errors, y),
                          "aurc_wer": M.aurc(errors / np.maximum(n_ref, 1), errors / np.maximum(n_ref, 1), n_ref)})

    info = {name: {p.name: p.info for p in pols} for name, pols in policies.items()}
    return {"operating_points": pd.DataFrame(ops), "calibration": pd.DataFrame(cal_rows),
            "ranking": pd.DataFrame(rank_rows), "reliability": pd.DataFrame(rel_rows),
            "policy_info": info, "scores": scores, "policies": policies, "sets": sets}


def paired_differences(res: dict, label: str, n_boot: int, seed: int) -> pd.DataFrame:
    """P4 minus P3 on the same PriMock test windows, paired consultation bootstrap (H3a-c)."""
    sets = res["sets"]
    d = sets["pm_window_test"]
    y = d[label].to_numpy(int)
    errors, n_ref = d["errors"].to_numpy(float), d["n_ref"].to_numpy(float)
    by = {p.name: p for p in res["policies"]["window"]}
    p3, p4 = res["scores"]["pm_window_test"]["P3"], res["scores"]["pm_window_test"]["P4"]
    val, rc = sets["eka_validation"], sets["pm_window_recal"]
    bs = M.ClusterBootstrap(d["group_id"].to_numpy(), n_boot, seed)
    rows = []

    def add(hyp, metric, f3, f4):
        diff = lambda ix: f4(ix) - f3(ix)  # noqa: E731
        lo, hi = bs.interval(diff)
        a = np.arange(len(d))
        rows.append({"hypothesis": hyp, "metric": metric, "P3": f3(a), "P4": f4(a), "diff": diff(a),
                     "diff_lo": lo, "diff_hi": hi})

    add("H3a", "ece", lambda ix: M.ece(p3[ix], y[ix]), lambda ix: M.ece(p4[ix], y[ix]))
    add("H3a", "brier", lambda ix: M.brier(p3[ix], y[ix]), lambda ix: M.brier(p4[ix], y[ix]))
    add("H3a", "abs_citl", lambda ix: abs(M.calibration_in_the_large(p3[ix], y[ix])),
        lambda ix: abs(M.calibration_in_the_large(p4[ix], y[ix])))
    for a in RISK_TARGETS:
        t3 = M.risk_target_threshold(by["P3"].score(val), val[label].to_numpy(int), a, n_min=RISK_N_MIN)
        t4 = M.risk_target_threshold(by["P4"].score(rc), rc[label].to_numpy(int), a, n_min=RISK_N_MIN)
        for k in ("accepted_err_rate", "review_rate"):
            add("H3b", f"risk_{a:.2f}:{k}",
                lambda ix, k=k, t=t3: M.operating_point(p3[ix] > t, y[ix], errors[ix], n_ref[ix])[k],
                lambda ix, k=k, t=t4: M.operating_point(p4[ix] > t, y[ix], errors[ix], n_ref[ix])[k])
        # plug-in: thresholds recomputed from each resample's own (unlabelled) probabilities
        for k in ("accepted_err_rate", "review_rate"):
            add("H3c", f"plugin_{a:.2f}:{k}",
                lambda ix, k=k: M.operating_point(p3[ix] > M.plugin_threshold(p3[ix], a), y[ix], errors[ix], n_ref[ix])[k],
                lambda ix, k=k: M.operating_point(p4[ix] > M.plugin_threshold(p4[ix], a), y[ix], errors[ix], n_ref[ix])[k])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- recalibration size

def learning_curve(df: pd.DataFrame, label: str, seed: int, ks=(2, 5, 10, 20), draws: int = 50) -> pd.DataFrame:
    """Redraw k recalibration consultations; recalibrate P3 on them; score on the other consultations."""
    sets = _subsets(df)
    cal, val = sets["eka_calibration"], sets["eka_validation"]
    p3m = fit_logreg(cal[FEATURES].to_numpy(float), cal[label].to_numpy(int), cal["group_id"].to_numpy(), seed)
    win = df[(df["dataset"] == "primock57") & (df["subset"] == "window")]
    p_val = p3m.predict_proba(val[FEATURES].to_numpy(float))[:, 1]
    p_all = p3m.predict_proba(win[FEATURES].to_numpy(float))[:, 1]
    y_all = win[label].to_numpy(int)
    errors, n_ref = win["errors"].to_numpy(float), win["n_ref"].to_numpy(float)
    cons = np.array(sorted(win["consultation"].unique()))
    rng = np.random.default_rng(seed)
    rows = []
    for k in ks:
        for i in range(draws):
            rc = win["consultation"].isin(set(rng.choice(cons, size=k, replace=False))).to_numpy()
            te = ~rc
            fits = {"P4": Platt().fit(p_all[rc], y_all[rc]), "P4-int": Platt(intercept_only=True).fit(p_all[rc], y_all[rc])}
            row = {"k": k, "draw": i, "n_recal_windows": int(rc.sum()), "platt_fell_back": fits["P4"].fell_back,
                   "ece_P3": M.ece(p_all[te], y_all[te]), "citl_P3": M.calibration_in_the_large(p_all[te], y_all[te])}
            probs = {"P3": p_all, **{n: f(p_all) for n, f in fits.items()}}
            for n in ("P4", "P4-int"):
                row[f"ece_{n}"] = M.ece(probs[n][te], y_all[te])
                row[f"citl_{n}"] = M.calibration_in_the_large(probs[n][te], y_all[te])
            for a in RISK_TARGETS:
                t3 = M.risk_target_threshold(p_val, val[label].to_numpy(int), a, n_min=RISK_N_MIN)
                t4 = M.risk_target_threshold(probs["P4"][rc], y_all[rc], a, n_min=RISK_N_MIN)
                for n, p, t in (("P3", p_all, t3), ("P4", probs["P4"], t4)):
                    o = M.operating_point(p[te] > t, y_all[te], errors[te], n_ref[te])
                    row[f"acc_err_{n}_{a:.2f}"], row[f"review_{n}_{a:.2f}"] = o["accepted_err_rate"], o["review_rate"]
                    row[f"infeasible_{n}_{a:.2f}"] = bool(np.isneginf(t))
                for n in ("P3", "P4", "P4-int"):
                    p = probs[n][te]
                    o = M.operating_point(p > M.plugin_threshold(p, a), y_all[te], errors[te], n_ref[te])
                    row[f"plugin_acc_err_{n}_{a:.2f}"] = o["accepted_err_rate"]
                    row[f"plugin_review_{n}_{a:.2f}"] = o["review_rate"]
            rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- length controls

def length_table(res: dict, label: str, n_boot: int, seed: int) -> pd.DataFrame:
    """Calibration of P3 / P3-conf within reference-length bins (Amendment 1.1)."""
    sets = res["sets"]
    val = sets["eka_validation"]
    by = {p.name: p for p in res["policies"]["window"]}
    t10 = {n: M.budget_threshold(by[n].score(val), MAIN_BUDGET) for n in ("P1", "P3", "P3-conf")}
    rows = []
    bins = [(f"{lo}-{hi}" if hi < 10_000 else f">{lo - 1}", lo, hi) for lo, hi in LENGTH_BINS]
    bins.append((f"matched {MATCHED_BIN[0]}-{MATCHED_BIN[1]}", *MATCHED_BIN))
    for ev in EVAL_SETS:
        d = sets[ev]
        n_ref = d["n_ref"].to_numpy()
        for name, lo, hi in bins:
            m = (n_ref >= lo) & (n_ref <= hi)
            if ev == "eka_test" and name.startswith("matched"):
                m &= (d["recording_context"] == "narration_sentence").to_numpy()
            if m.sum() == 0:
                continue
            sub = d[m]
            y = sub[label].to_numpy(int)
            row = {"eval": ev, "bin": name, "n": int(m.sum()), "groups": sub["group_id"].nunique(),
                   "base_rate": float(y.mean()), "crit_err_rate": float(sub["crit_err"].mean())}
            bs = M.ClusterBootstrap(sub["group_id"].to_numpy(), n_boot, seed) if m.sum() >= 20 else None
            for pol in ("P1", "P3", "P3-conf"):
                r = res["scores"][ev][pol][m]
                o = M.operating_point(r >= t10[pol], y, sub["errors"].to_numpy(float), sub["n_ref"].to_numpy(float))
                row[f"{pol}_review_rate_at_eka10"] = o["review_rate"]
                row[f"{pol}_accepted_err_at_eka10"] = o["accepted_err_rate"]
                if pol == "P1":
                    continue
                row[f"{pol}_mean_p"] = float(r.mean())
                row[f"{pol}_citl"] = M.calibration_in_the_large(r, y)
                row[f"{pol}_ece"] = M.ece(r, y)
                if bs is not None:
                    row[f"{pol}_citl_lo"], row[f"{pol}_citl_hi"] = bs.interval(
                        lambda ix, r=r, y=y: M.calibration_in_the_large(r[ix], y[ix]))
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- word level (H4)

def word_confidence(model: str, df: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = score.words_path(model)
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame()
    words = pd.read_parquet(path)
    sets = _subsets(df)
    summary, bins = [], []
    for name in EVAL_SETS:
        u = sets[name][["unit_id", "group_id"]]
        w = words.merge(u, on="unit_id")
        p, c = w["probability"].to_numpy(float), w["correct"].to_numpy(float)
        crit = w["critical"].to_numpy(bool)
        wrong = c == 0
        bs = M.ClusterBootstrap(w["group_id"].to_numpy(), n_boot, seed)
        stats = {
            "word_ece": lambda ix: M.ece(1 - p[ix], 1 - c[ix]),
            "word_citl": lambda ix: M.calibration_in_the_large(1 - p[ix], 1 - c[ix]),
            "wrong_with_p_ge_0.9": lambda ix: float((p[ix][wrong[ix]] >= 0.9).mean()) if wrong[ix].any() else np.nan,
            "wrong_with_p_ge_0.5": lambda ix: float((p[ix][wrong[ix]] >= 0.5).mean()) if wrong[ix].any() else np.nan,
            "err_rate_among_p_ge_0.9": lambda ix: float(1 - c[ix][p[ix] >= 0.9].mean()) if (p[ix] >= 0.9).any() else np.nan,
            "critical_wrong_with_p_ge_0.9": lambda ix: (float((p[ix][wrong[ix] & crit[ix]] >= 0.9).mean())
                                                        if (wrong[ix] & crit[ix]).any() else np.nan),
        }
        row = {"set": name, "words": len(w), "word_acc": float(c.mean()), "mean_p": float(p.mean()),
               "critical_words": int(crit.sum()), "critical_wrong": int((crit & wrong).sum())}
        a = np.arange(len(w))
        for k, f in stats.items():
            row[k] = f(a)
            row[f"{k}_lo"], row[f"{k}_hi"] = bs.interval(f)
        summary.append(row)
        bins.extend({"set": name, **b} for b in M.reliability(1 - p, 1 - c, 10))
    return pd.DataFrame(summary), pd.DataFrame(bins)


# ------------------------------------------------------------------------------ deployment view

def deployment_view(res: dict, label: str) -> pd.DataFrame:
    """Per-consultation consequences of each policy on PriMock57 test windows (Amendment 1.6)."""
    d = res["sets"]["pm_window_test"]
    ops = res["operating_points"]
    rows = []
    y = d[label].to_numpy(int)
    crit_miss = d["n_crit_missed"].to_numpy(float) + d["n_crit_false"].to_numpy(float)
    for pol in ("P1", "P3", "P1-retuned", "P4"):
        r = res["scores"]["pm_window_test"].get(pol)
        if r is None:
            continue
        for point in ("budget_0.05", "budget_0.10", "risk_0.20", "plugin_0.20"):
            o = ops[(ops["eval"] == "pm_window_test") & (ops["policy"] == pol) & (ops["point"] == point)]
            if o.empty:
                continue
            thr = o["threshold"].iloc[0]
            review = r >= thr if point.startswith("budget") else r > thr
            g = pd.DataFrame({"c": d["consultation"].to_numpy(), "rev": review, "dur": d["duration"].to_numpy(),
                              "acc_err": (~review) & (y == 1), "acc_crit": np.where(~review, crit_miss, 0)}).groupby("c")
            per = pd.DataFrame({"review_min": g.apply(lambda x: x.loc[x["rev"], "dur"].sum() / 60),
                                "audio_min": g["dur"].sum() / 60,
                                "acc_err_windows": g["acc_err"].sum(), "acc_crit_tokens": g["acc_crit"].sum()})
            rows.append({"policy": pol, "point": point, "consultations": len(per),
                         "review_min_per_consult": per["review_min"].mean(),
                         "audio_min_per_consult": per["audio_min"].mean(),
                         "acc_err_windows_per_consult": per["acc_err_windows"].mean(),
                         "consult_with_acc_err": float((per["acc_err_windows"] > 0).mean()),
                         "acc_crit_errors_per_consult": per["acc_crit_tokens"].mean(),
                         "consult_with_acc_crit_error": float((per["acc_crit_tokens"] > 0).mean())})
    return pd.DataFrame(rows)


def drift_alarm(res: dict, feature_shift_df: pd.DataFrame) -> pd.DataFrame:
    """Label-free alarm (Amendment 1.6): does anything that needs no target labels flag the shift?"""
    ops = res["operating_points"]
    sets = res["sets"]
    rows = []
    for pol in ("P1", "P3"):
        e = ops[(ops["eval"] == "eka_test") & (ops["policy"] == pol) & (ops["point"] == "budget_0.10")].iloc[0]
        for ev in ("pm_window_test", "pm_turn_test"):
            t = ops[(ops["eval"] == ev) & (ops["policy"] == pol) & (ops["point"] == "budget_0.10")].iloc[0]
            fired = not (e["review_rate_lo"] <= t["review_rate"] <= e["review_rate_hi"])
            rows.append({"signal": f"{pol} review share at Eka 10% threshold", "eval": ev, "value": t["review_rate"],
                         "reference": f"Eka test {e['review_rate']:.3f} [{e['review_rate_lo']:.3f}, {e['review_rate_hi']:.3f}]",
                         "fired": fired})
    for ev in ("pm_window_test", "pm_turn_test"):
        s = feature_shift_df[(feature_shift_df["set"] == ev) & (feature_shift_df["feature"] == "conf")]["shift_sd"].iloc[0]
        rows.append({"signal": "mean confidence shift (Eka SDs)", "eval": ev, "value": s, "reference": "|shift| > 0.5",
                     "fired": abs(s) > 0.5})
        p3_eka = res["scores"]["eka_test"]["P3"]
        p3_t = res["scores"][ev]["P3"]
        p3_policy = next(p for p in res["policies"]["window"] if p.name == "P3")
        sd = float(np.std(p3_policy.score(sets["eka_calibration"])))
        shift = (p3_t.mean() - p3_eka.mean()) / sd
        rows.append({"signal": "mean P3 risk shift (Eka SDs)", "eval": ev, "value": shift, "reference": "|shift| > 0.5",
                     "fired": abs(shift) > 0.5})
        ks = ks_2samp(sets["eka_validation"]["conf"], sets[ev]["conf"])
        rows.append({"signal": "KS test on confidence vs Eka validation", "eval": ev, "value": ks.statistic,
                     "reference": f"p = {ks.pvalue:.2g}", "fired": ks.pvalue < 0.01})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------- descriptives

def feature_shift(df: pd.DataFrame) -> pd.DataFrame:
    """Mean of every feature per set, and the shift from Eka calibration in calibration-set SDs."""
    sets = _subsets(df)
    ref = sets["eka_calibration"][FEATURES + ["conf"]]
    mu, sd = ref.mean(), ref.std().replace(0, np.nan)
    rows = []
    for name in ("eka_calibration", "eka_test", "pm_window_test", "pm_turn_test"):
        m = sets[name][FEATURES + ["conf"]].mean()
        for f in FEATURES + ["conf"]:
            rows.append({"set": name, "feature": f, "mean": m[f], "shift_sd": (m[f] - mu[f]) / sd[f]})
    return pd.DataFrame(rows)


def conf_bins(df: pd.DataFrame, label: str, edges=(0, .5, .6, .7, .8, .85, .9, .95, 1.0001)) -> pd.DataFrame:
    """Observed error rate within bins of raw confidence, per evaluation set."""
    sets = _subsets(df)
    rows = []
    for name in EVAL_SETS:
        d = sets[name]
        g = d.groupby(pd.cut(d["conf"], edges, right=False), observed=True)[label].agg(["mean", "size"])
        for iv, r in g.iterrows():
            rows.append({"set": name, "conf_lo": iv.left, "conf_hi": min(iv.right, 1.0), "err_rate": r["mean"], "n": int(r["size"])})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ confirmatory hypotheses

def hypothesis_tests(res: dict, df: pd.DataFrame, model: str, n_boot: int, seed: int, alpha_target: float = 0.20) -> pd.DataFrame:
    """The confirmatory family of Amendment 2 with one-sided bootstrap p-values and Holm adjustment.

    All on PriMock57 test windows with label err, policies P3 (frozen) and P4 (recalibrated).
    """
    sets = res["sets"]
    by = {p.name: p for p in res["policies"]["window"]}
    w, e = sets["pm_window_test"], sets["eka_test"]
    yw, ye = w["err"].to_numpy(int), e["err"].to_numpy(int)
    p3w, p4w, p3e = res["scores"]["pm_window_test"]["P3"], res["scores"]["pm_window_test"]["P4"], res["scores"]["eka_test"]["P3"]
    bw = M.ClusterBootstrap(w["group_id"].to_numpy(), n_boot, seed)
    be = M.ClusterBootstrap(e["group_id"].to_numpy(), n_boot, seed + 1)
    val, rc = sets["eka_validation"], sets["pm_window_recal"]
    t10 = M.budget_threshold(by["P3"].score(val), MAIN_BUDGET)
    t3 = M.risk_target_threshold(by["P3"].score(val), val["err"].to_numpy(int), alpha_target, n_min=RISK_N_MIN)
    t4 = M.risk_target_threshold(by["P4"].score(rc), rc["err"].to_numpy(int), alpha_target, n_min=RISK_N_MIN)

    def miss(p, y, ix, t):  # share of erroneous units auto-accepted
        yy = y[ix]
        return float(((p[ix] < t) & (yy == 1)).sum() / max(yy.sum(), 1))

    def gap(p, y, ix, t):  # |realised accepted-error - target|
        acc = p[ix] <= t
        return abs(float(y[ix][acc].mean()) - alpha_target) if acc.any() else np.nan

    def plugin_gap(p, y, ix):
        pp = p[ix]
        acc = pp <= M.plugin_threshold(pp, alpha_target)
        return abs(float(y[ix][acc].mean()) - alpha_target) if acc.any() else np.nan

    rows = []
    a = np.arange(len(w))
    rows.append(("H1", "P3 calibration-in-the-large on windows < 0 (overconfident)", "less",
                 M.calibration_in_the_large(p3w, yw), bw.replicates(lambda ix: M.calibration_in_the_large(p3w[ix], yw[ix]))))
    rw = bw.replicates(lambda ix: miss(p3w, yw, ix, t10))
    re_ = be.replicates(lambda ix: miss(p3e, ye, ix, t10))
    rows.append(("H2", "P3 share of erroneous units auto-accepted at the Eka 10% threshold: windows minus Eka test > 0",
                 "greater", miss(p3w, yw, a, t10) - miss(p3e, ye, np.arange(len(e)), t10), rw - re_))
    rows.append(("H3a", "ECE of P4 minus P3 on windows < 0", "less", M.ece(p4w, yw) - M.ece(p3w, yw),
                 bw.replicates(lambda ix: M.ece(p4w[ix], yw[ix]) - M.ece(p3w[ix], yw[ix]))))
    rows.append(("H3b", f"|accepted error - {alpha_target:.2f}| at the empirical risk target: P4 minus P3 < 0", "less",
                 gap(p4w, yw, a, t4) - gap(p3w, yw, a, t3),
                 bw.replicates(lambda ix: gap(p4w, yw, ix, t4) - gap(p3w, yw, ix, t3))))
    rows.append(("H3c", f"|accepted error - {alpha_target:.2f}| with the label-free plug-in rule: P4 minus P3 < 0", "less",
                 plugin_gap(p4w, yw, a) - plugin_gap(p3w, yw, a),
                 bw.replicates(lambda ix: plugin_gap(p4w, yw, ix) - plugin_gap(p3w, yw, ix))))
    if score.words_path(model).exists():
        words = pd.read_parquet(score.words_path(model), columns=["unit_id", "probability", "correct"])
        ww = words.merge(w[["unit_id", "group_id"]], on="unit_id")
        we = words.merge(e[["unit_id", "group_id"]], on="unit_id")
        pw, cw = ww["probability"].to_numpy(float), ww["correct"].to_numpy(float)
        pe, ce = we["probability"].to_numpy(float), we["correct"].to_numpy(float)
        bww = M.ClusterBootstrap(ww["group_id"].to_numpy(), n_boot, seed + 2)
        bwe = M.ClusterBootstrap(we["group_id"].to_numpy(), n_boot, seed + 3)
        rows.append(("H4", "word-level ECE: windows minus Eka test > 0", "greater",
                     M.ece(1 - pw, 1 - cw) - M.ece(1 - pe, 1 - ce),
                     bww.replicates(lambda ix: M.ece(1 - pw[ix], 1 - cw[ix]))
                     - bwe.replicates(lambda ix: M.ece(1 - pe[ix], 1 - ce[ix]))))
    out = []
    for hyp, desc, direction, est, reps in rows:
        fin = reps[np.isfinite(reps)]
        testable = len(fin) >= 0.95 * len(reps) and np.isfinite(est)
        out.append({"hypothesis": hyp, "statement": desc, "estimate": est,
                    "ci_lo": float(np.quantile(fin, 0.025)) if len(fin) else np.nan,
                    "ci_hi": float(np.quantile(fin, 0.975)) if len(fin) else np.nan,
                    # a statistic undefined in >5% of replicates (e.g. an infeasible risk target) is not tested
                    "p_one_sided": M.one_sided_p(reps, direction) if testable else np.nan,
                    "n_boot": int(len(fin)), "testable": bool(testable)})
    t = pd.DataFrame(out)
    t["p_holm"] = M.holm(t["p_one_sided"].fillna(1.0).tolist())
    t.loc[~t["testable"], "p_holm"] = np.nan
    t["supported"] = t["p_holm"] < 0.05
    return t


def null_baseline(res: dict, model: str, draws: int, seed: int) -> pd.DataFrame:
    """What H1/H2 would show if Whisper's word probabilities were perfectly calibrated everywhere.

    Each hypothesis word is drawn wrong with probability 1 - p (its own Whisper probability), a
    simulated label err* = (wrong words / hypothesis words > 0.10) is formed (empty output = 1), P3 is
    refitted on Eka calibration with err* (P3's own C), and the H1/H2 statistics are recomputed. A gap
    that appears here comes from how unit length and the label interact, not from confidence failing
    to transfer. Caveat: the simulation has no deletions, so it is a guide, not an exact null.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    path = score.words_path(model)
    if not path.exists():
        return pd.DataFrame()
    words = pd.read_parquet(path, columns=["unit_id", "probability"])
    sets = res["sets"]
    names = ("eka_calibration", "eka_validation", "eka_test", "pm_window_test")
    units = pd.concat([sets[n].assign(_set=n) for n in names], ignore_index=True)
    code = pd.Series(np.arange(len(units)), index=units["unit_id"].to_numpy())
    w = words[words["unit_id"].isin(code.index)]
    widx = code.loc[w["unit_id"].to_numpy()].to_numpy()
    wp = w["probability"].to_numpy(float)
    n_hyp = np.bincount(widx, minlength=len(units))
    X = units[FEATURES].to_numpy(float)
    is_ = {n: (units["_set"] == n).to_numpy() for n in names}
    c_fixed = res["policy_info"]["window"]["P3"]["C"]
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(draws):
        wrong = np.bincount(widx, weights=(rng.random(len(wp)) > wp).astype(float), minlength=len(units))
        y = np.where(n_hyp == 0, 1, (wrong / np.maximum(n_hyp, 1)) > score.LABELS["err"]).astype(int)
        m = make_pipeline(StandardScaler(), LogisticRegression(C=c_fixed, max_iter=5000))
        m.fit(X[is_["eka_calibration"]], y[is_["eka_calibration"]])
        p = m.predict_proba(X)[:, 1]
        t10 = M.budget_threshold(p[is_["eka_validation"]], MAIN_BUDGET)

        def miss(mask):
            yy = y[mask]
            return float(((p[mask] < t10) & (yy == 1)).sum() / max(yy.sum(), 1))

        win, eka = is_["pm_window_test"], is_["eka_test"]
        rows.append({"draw": r, "citl_windows": M.calibration_in_the_large(p[win], y[win]),
                     "ece_windows_minus_eka": M.ece(p[win], y[win]) - M.ece(p[eka], y[eka]),
                     "miss_windows_minus_eka": miss(win) - miss(eka), "base_rate_windows": float(y[win].mean())})
    return pd.DataFrame(rows)


def feature_support(res: dict, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label-free: how far PriMock features sit outside Eka's support, and how that moves P3's logit."""
    from sklearn.model_selection import GroupKFold

    from asrshift.features import CONFIDENCE_FEATURES

    sets = res["sets"]
    cal = sets["eka_calibration"]
    lo, hi = cal[FEATURES].quantile(0.01), cal[FEATURES].quantile(0.99)
    rows = []
    for pol_name, feats in (("P3", FEATURES), ("P3-conf", CONFIDENCE_FEATURES)):
        m = fit_logreg(cal[feats].to_numpy(float), cal["err"].to_numpy(int), cal["group_id"].to_numpy(), seed)
        scaler, lr = m[0], m[-1]
        z = {n: (sets[n][feats].to_numpy(float) - scaler.mean_) / scaler.scale_
             for n in ("eka_test", "pm_window_test", "pm_turn_test")}
        for j, f in enumerate(feats):
            row = {"model": pol_name, "feature": f, "coef_std": float(lr.coef_[0, j])}
            for n in ("pm_window_test", "pm_turn_test"):
                d = float(z[n][:, j].mean() - z["eka_test"][:, j].mean())
                row[f"z_shift_{n}"] = d
                row[f"logit_shift_{n}"] = float(lr.coef_[0, j] * d)
                if pol_name == "P3":
                    v = sets[n][f].to_numpy(float)
                    row[f"outside_eka_1_99_{n}"] = float(((v < lo[f]) | (v > hi[f])).mean())
            rows.append(row)
    dom = []
    pm = pd.concat([sets["pm_window_recal"], sets["pm_window_test"]])
    both = pd.concat([cal.assign(_d=0), pm.assign(_d=1)])
    for name, feats in (("all features", FEATURES), ("confidence features", CONFIDENCE_FEATURES)):
        X, yd, g = both[feats].to_numpy(float), both["_d"].to_numpy(int), both["group_id"].to_numpy()
        oof = np.zeros(len(both))
        for tr, te in GroupKFold(n_splits=5).split(X, yd, g):
            oof[te] = fit_logreg(X[tr], yd[tr], g[tr], seed).predict_proba(X[te])[:, 1]
        dom.append({"features": name, "domain_auroc": M.auroc(oof, yd)})
    return pd.DataFrame(rows), pd.DataFrame(dom)


def devpool_robustness(res: dict, seed: int) -> pd.DataFrame:
    """10% budget thresholds tuned on Eka calibration + validation instead of validation alone.

    Calibration clips get out-of-fold P3 scores (grouped 5-fold), so no clip is scored by a model
    trained on it. Checks that the transfer result does not hinge on the lumpy validation split.
    """
    from sklearn.model_selection import GroupKFold

    sets = res["sets"]
    cal, val = sets["eka_calibration"], sets["eka_validation"]
    by = {p.name: p for p in res["policies"]["window"]}
    X, y, g = cal[FEATURES].to_numpy(float), cal["err"].to_numpy(int), cal["group_id"].to_numpy()
    oof = np.zeros(len(cal))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        oof[te] = fit_logreg(X[tr], y[tr], g[tr], seed).predict_proba(X[te])[:, 1]
    pools = {"P1": np.r_[1 - cal["conf"].to_numpy(float), 1 - val["conf"].to_numpy(float)],
             "P3": np.r_[oof, by["P3"].score(val)]}
    rows = []
    for pol, pool in pools.items():
        for tuning, t in (("validation", M.budget_threshold(by[pol].score(val), MAIN_BUDGET)),
                          ("calibration+validation", M.budget_threshold(pool, MAIN_BUDGET))):
            for ev in ("eka_test", "pm_window_test", "pm_turn_test"):
                d = sets[ev]
                o = M.operating_point(res["scores"][ev][pol] >= t, d["err"].to_numpy(int), d["errors"].to_numpy(float),
                                      d["n_ref"].to_numpy(float))
                rows.append({"policy": pol, "tuned_on": tuning, "threshold": t, "eval": ev, **o})
    return pd.DataFrame(rows)


def sensitivity(df: pd.DataFrame, name: str, keep: pd.Series, n_boot: int, seed: int) -> pd.DataFrame:
    """Key H1-H3 numbers on a subset of PriMock windows (recalibration and test alike)."""
    r = evaluate_label(df[keep], "err", n_boot, seed, eval_sets=("pm_window_test",))
    cal = r["calibration"].set_index("policy")
    ops = r["operating_points"]
    rows = []
    for pol in ("P3", "P3-conf", "P4"):
        row = {"analysis": name, "windows_test": int(len(r["sets"]["pm_window_test"])), "policy": pol,
               "ece": cal.loc[pol, "ece"], "citl": cal.loc[pol, "citl"]}
        for point in ("budget_0.10", "risk_0.20", "plugin_0.20"):
            o = ops[(ops["policy"] == pol) & (ops["point"] == point)]
            if len(o):
                row[f"{point}:review_rate"] = o["review_rate"].iloc[0]
                row[f"{point}:accepted_err_rate"] = o["accepted_err_rate"].iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------- export

def export_policies(df: pd.DataFrame, label: str, seed: int, model: str) -> None:
    """Save P1/P3/P4 and their thresholds as JSON for asrshift.review (demo app, notebook)."""
    from asrshift import review

    sets = _subsets(df)
    cal, val = sets["eka_calibration"], sets["eka_validation"]
    p3 = fit_logreg(cal[FEATURES].to_numpy(float), cal[label].to_numpy(int), cal["group_id"].to_numpy(), seed)

    def prob(d):
        return p3.predict_proba(d[FEATURES].to_numpy(float))[:, 1]

    def points(r_tune, y_tune, probabilistic):
        out = {f"budget_{b:.2f}": M.budget_threshold(r_tune, b) for b in BUDGETS}
        if probabilistic:
            # review when p > tau; stored as a ">=" threshold just above tau
            out.update({f"risk_{a:.2f}": float(np.nextafter(
                M.risk_target_threshold(r_tune, y_tune, a, n_min=RISK_N_MIN), np.inf)) for a in RISK_TARGETS})
        return out

    platts, p4_thresholds = {}, {}
    for key in ("window", "turn"):
        rc = sets[f"pm_{key}_recal"]
        platts[key] = Platt().fit(prob(rc), rc[label].to_numpy(int))
        p4_thresholds[key] = points(platts[key](prob(rc)), rc[label].to_numpy(int), True)
    thresholds = {"P1": points(1.0 - val["conf"].to_numpy(float), None, False),
                  "P3": points(prob(val), val[label].to_numpy(int), True), "P4": p4_thresholds}
    review.export(review.policy_path(model), FEATURES, p3, platts, thresholds, label)


# ------------------------------------------------------------------------------------------- run

def _write(tables: dict, out) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        if isinstance(t, pd.DataFrame) and not t.empty:
            t.to_csv(out / f"{name}.csv", index=False, float_format="%.5g")


def run(cfg: dict, model: str | None = None, n_boot: int | None = None) -> dict:
    model = model or cfg["asr"]["model"]
    n_boot = n_boot or cfg.get("evaluation", {}).get("n_boot", 2000)
    small_boot = max(200, n_boot // 10)
    seed = cfg["seed"]
    df = score.load(model)
    tag = model.replace("/", "_")
    out = RESULTS_DIR / "tables" / tag
    sets = _subsets(df)
    fs = feature_shift(df)
    words, word_bins = word_confidence(model, df, n_boot, seed)
    _write({"asr_summary": asr_summary(sets, n_boot, seed), "eka_entities": entity_table(sets["eka_test"]),
            "feature_shift": fs, "word_confidence": words, "word_reliability": word_bins}, out)

    summary = {"model": model, "n_boot": n_boot, "labels": {}}
    # primary label: everything
    res = evaluate_label(df, "err", n_boot, seed)
    _write({k: res[k] for k in ("operating_points", "calibration", "ranking", "reliability")} | {
        "paired": paired_differences(res, "err", n_boot, seed),
        "conf_bins": conf_bins(df, "err"),
        "length_strata": length_table(res, "err", small_boot, seed),
        "learning_curve": learning_curve(df, "err", seed),
        "deployment": deployment_view(res, "err"),
        "drift_alarm": drift_alarm(res, fs),
    }, out / "err")
    scores = [pd.DataFrame({"unit_id": sets[ev]["unit_id"].to_numpy(), **s}).assign(eval=ev) for ev, s in res["scores"].items()]
    pd.concat(scores).to_csv(out / "err" / "unit_scores.csv.gz", index=False, float_format="%.6g")
    summary["labels"]["err"] = {"policy_info": res["policy_info"]}
    contrib, domain = feature_support(res, seed)
    _write({"hypotheses": hypothesis_tests(res, df, model, n_boot, seed),
            "null_baseline": null_baseline(res, model, draws=100, seed=seed),
            "feature_contributions": contrib, "domain_classifier": domain,
            "devpool_robustness": devpool_robustness(res, seed)}, out / "err")

    # sensitivity analyses on PriMock windows (Amendments 1.5 and 2)
    is_win = df["subset"] == "window"
    _write({"sensitivity": pd.concat([
        sensitivity(df, "all windows (primary)", pd.Series(True, index=df.index), small_boot, seed),
        sensitivity(df, "no window longer than 30.6 s", ~(is_win & (df["duration"] > LONG_WINDOW_S)), small_boot, seed),
        sensitivity(df, "no overlapping speech in window", ~(is_win & (df["overlap_s"].fillna(0) > 0)), small_boot, seed),
    ], ignore_index=True)}, out / "err")

    # secondary labels (smaller bootstrap)
    for label in ("any_err", "severe_err", "crit_err", "err_c3"):
        r2 = evaluate_label(df, label, small_boot, seed)
        _write({k: r2[k] for k in ("operating_points", "calibration", "ranking", "reliability")}
               | {"conf_bins": conf_bins(df, label)}, out / label)
        summary["labels"][label] = {"policy_info": r2["policy_info"]}
    # Eka entity label, on clips that contain an entity
    eka_ent = df[(df["dataset"] == "eka") & (df["n_ent"].fillna(0) > 0)].assign(ent_err=lambda x: x["ent_err"].astype(int))
    r3 = evaluate_label(pd.concat([eka_ent, df[df["dataset"] != "eka"]]), "ent_err", small_boot, seed,
                        eval_sets=("eka_test",), with_target=False)
    _write({k: r3[k] for k in ("operating_points", "calibration", "ranking")}, out / "ent_err")
    export_policies(df, score.PRIMARY_LABEL, seed, model)
    (RESULTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics" / f"{tag}.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"tables written to {out}")
    return summary
