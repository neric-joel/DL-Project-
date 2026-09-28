"""Figures, drawn only from the CSV tables in results/tables/ (no raw data needed).

Colour follows the entity across every figure: Eka = blue, PriMock57 windows = orange,
PriMock57 turns = aqua; P1 raw confidence = violet, P2 rules = yellow, P3 Eka-calibrated = magenta,
P4 recalibrated = green. Palettes were checked with a colour-vision-deficiency validator; series
also differ by marker or line style, and every figure has a legend or direct labels.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from asrshift.paths import FIGURES_DIR, RESULTS_DIR  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DOMAIN = {
    "eka_test": ("Eka test (short medical clips)", "#2a78d6", "o"),
    "pm_turn_test": ("PriMock57 turns (one speaker)", "#1baf7a", "s"),
    "pm_window_test": ("PriMock57 windows (conversation)", "#eb6834", "D"),
}
POLICY = {
    "P1": ("P1 raw confidence", "#4a3aa7", "-"),
    "P2": ("P2 no-speech + repetition rules", "#eda100", "-"),
    "P3": ("P3 Eka-calibrated (frozen)", "#e87ba4", "-"),
    "P4": ("P4 recalibrated on PriMock57", "#008300", "-"),
    "P4-refit": ("P4 refitted on PriMock57", "#008300", "--"),
    "ceiling": ("In-domain ceiling (not deployable)", "#8a8985", "--"),
    "oracle": ("Perfect ranking", INK2, ":"),
}


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
        "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6.5,
    })


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def fig_confidence_vs_error(tables, units: pd.DataFrame | None):
    cb = pd.read_csv(tables / "err" / "conf_bins.csv")
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for key, (lab, col, mk) in DOMAIN.items():
        d = cb[(cb["set"] == key) & (cb["n"] >= 15)]
        x = (d["conf_lo"] + d["conf_hi"]) / 2
        lo, hi = _wilson(d["err_rate"] * d["n"], d["n"])
        ax.fill_between(x, lo, hi, color=col, alpha=0.12, linewidth=0)
        ax.plot(x, d["err_rate"], color=col, marker=mk, label=lab)
    ax.set_ylabel("Share of transcripts with WER > 10%")
    ax.set_ylim(0, 1)
    ax.set_title("Same confidence, different error rates")
    ax.legend(loc="upper right")
    if units is not None:
        bins = np.linspace(0, 1, 41)
        for key, (lab, col, _) in DOMAIN.items():
            ax2.hist(units.loc[units["eval"] == key, "conf"], bins=bins, density=True, histtype="step",
                     color=col, linewidth=2)
        ax2.set_ylabel("Density")
    ax2.set_xlabel("Whisper sequence confidence  exp(mean token log-prob)")
    ax2.set_xlim(0.3, 1.0)
    _save(fig, "fig1_confidence_vs_error")


def fig_reliability(tables):
    rel = pd.read_csv(tables / "err" / "reliability.csv")
    cal = pd.read_csv(tables / "err" / "calibration.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    panels = [("P3", "P3 calibrated on Eka, applied unchanged", ["eka_test", "pm_turn_test", "pm_window_test"]),
              ("P4", "P4 recalibrated on 10 PriMock57 consultations", ["pm_turn_test", "pm_window_test"])]
    for ax, (pol, title, sets) in zip(axes, panels):
        ax.plot([0, 1], [0, 1], color=INK2, linestyle=":", linewidth=1.2)
        for key in sets:
            lab, col, mk = DOMAIN[key]
            d = rel[(rel["policy"] == pol) & (rel["eval"] == key) & (rel["n"] >= 10)]
            c = cal[(cal["policy"] == pol) & (cal["eval"] == key)]
            e = f"  ECE {c['ece'].iloc[0]:.3f}" if len(c) else ""
            ax.plot(d["mean_p"], d["frac_err"], color=col, marker=mk, label=lab.split(" (")[0] + e)
        ax.set_title(title)
        ax.set_xlabel("Predicted P(transcript has WER > 10%)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=9)
        # x = predicted risk, y = observed error: above the diagonal the model under-predicts risk
        ax.text(0.97, 0.03, "above the dotted line: more errors than predicted\n(overconfident)",
                transform=ax.transAxes, ha="right", va="bottom", color=INK2, fontsize=8.5)
    axes[0].set_ylabel("Observed share with WER > 10%")
    _save(fig, "fig2_reliability")


def _risk_coverage(r, y):
    order = np.argsort(r, kind="stable")
    ys = y[order]
    cov = np.arange(1, len(ys) + 1) / len(ys)
    return cov, np.cumsum(ys) / np.arange(1, len(ys) + 1)


def fig_risk_coverage(tables, units: pd.DataFrame | None):
    if units is None:
        return
    ops = pd.read_csv(tables / "err" / "operating_points.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
    for ax, key in zip(axes, ["eka_test", "pm_window_test"]):
        d = units[units["eval"] == key]
        y = d["err"].to_numpy(float)
        for pol in ["P1", "P2", "P3", "P4-refit", "ceiling"]:
            if pol not in d or d[pol].isna().all():
                continue
            lab, col, ls = POLICY[pol]
            cov, risk = _risk_coverage(d[pol].to_numpy(float), y)
            ax.plot(cov, risk, color=col, linestyle=ls, label=lab if pol != "P3" else "P3 / P4 (same ranking)")
        cov, risk = _risk_coverage(y + np.random.default_rng(0).random(len(y)) * 1e-6, y)
        ax.plot(cov, risk, color=POLICY["oracle"][1], linestyle=":", label=POLICY["oracle"][0])
        ax.axhline(y.mean(), color=INK2, linewidth=0.8)
        ax.text(0.02, y.mean() + 0.01, f"accept everything: {y.mean():.0%} erroneous", color=INK2, fontsize=8.5)
        op = ops[(ops["eval"] == key) & (ops["policy"] == "P3") & (ops["point"] == "budget_0.10")]
        if len(op):
            o = op.iloc[0]
            ax.plot(1 - o["review_rate"], o["accepted_err_rate"], marker="o", markersize=9, color=POLICY["P3"][1],
                    markeredgecolor=INK, zorder=5)
            ax.annotate(f"Eka-tuned 10% budget:\nreviews {o['review_rate']:.0%}, accepted {o['accepted_err_rate']:.0%} erroneous",
                        (1 - o["review_rate"], o["accepted_err_rate"]), xytext=(-150, -45), textcoords="offset points",
                        fontsize=8.5, color=INK, arrowprops={"arrowstyle": "-", "color": INK2})
        ax.set_title(DOMAIN[key][0])
        ax.set_xlabel("Coverage (share accepted automatically)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Share of accepted transcripts with WER > 10%")
    axes[1].legend(loc="upper left", fontsize=8.5)
    _save(fig, "fig3_risk_coverage")


def fig_threshold_transfer(tables):
    ops = pd.read_csv(tables / "err" / "operating_points.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    ax = axes[0]
    budgets = [0.01, 0.05, 0.10, 0.20, 0.30]
    ax.plot([0, 0.32], [0, 0.32], color=INK2, linestyle=":", linewidth=1.2)
    for key in ["eka_test", "pm_turn_test", "pm_window_test"]:
        lab, col, mk = DOMAIN[key]
        d = ops[(ops["eval"] == key) & (ops["policy"] == "P3") & ops["point"].str.startswith("budget")].sort_values("target")
        ax.errorbar(d["target"], d["review_rate"], yerr=[d["review_rate"] - d["review_rate_lo"], d["review_rate_hi"] - d["review_rate"]],
                    color=col, marker=mk, capsize=3, label=lab.split(" (")[0])
    ax.set_xlabel("Review budget set on Eka validation")
    ax.set_ylabel("Share actually sent to review")
    ax.set_title("Frozen thresholds: promised vs actual review load")
    ax.legend(loc="upper left", fontsize=9)
    ax = axes[1]
    ax.plot([0.05, 0.35], [0.05, 0.35], color=INK2, linestyle=":", linewidth=1.2)
    for pol, key, off in [("P3", "eka_test", -0.008), ("P3", "pm_window_test", 0.0), ("P4", "pm_window_test", 0.008)]:
        d = ops[(ops["eval"] == key) & (ops["policy"] == pol) & ops["point"].str.startswith("risk")].sort_values("target")
        lab, col, _ = POLICY[pol]
        mk = DOMAIN[key][2]
        ax.errorbar(d["target"] + off, d["accepted_err_rate"],
                    yerr=[d["accepted_err_rate"] - d["accepted_err_rate_lo"], d["accepted_err_rate_hi"] - d["accepted_err_rate"]],
                    color=col, marker=mk, capsize=3, linestyle="none" if key == "eka_test" else "-",
                    label=f"{pol} on {DOMAIN[key][0].split(' (')[0]}")
    ax.set_xlabel("Target error rate among accepted transcripts")
    ax.set_ylabel("Realised error rate among accepted")
    ax.set_title("Risk targets: promise kept only after recalibration?")
    ax.legend(loc="upper left", fontsize=9)
    _save(fig, "fig4_threshold_transfer")


def fig_learning_curve(tables):
    p = tables / "err" / "learning_curve.csv"
    if not p.exists():
        return
    lc = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (m3, m4, ylab) in zip(axes, [("ece_P3", "ece_P4", "ECE on held-out consultations"),
                                          ("acc_err_P3_0.20", "acc_err_P4_0.20", "Error rate among accepted (target 20%)")]):
        g = lc.groupby("k")
        for m, pol in ((m3, "P3"), (m4, "P4")):
            q = g[m].quantile([0.25, 0.5, 0.75]).unstack()
            lab, col, _ = POLICY[pol]
            ax.fill_between(q.index, q[0.25], q[0.75], color=col, alpha=0.15, linewidth=0)
            ax.plot(q.index, q[0.5], color=col, marker="o", label=lab)
        if "acc_err" in m3:
            ax.axhline(0.20, color=INK2, linestyle=":", linewidth=1.2)
        ax.set_xlabel("Consultations used for recalibration")
        ax.set_ylabel(ylab)
        ax.set_xticks(sorted(lc["k"].unique()))
    axes[0].legend(fontsize=9)
    axes[0].set_title("How much target data does recalibration need?")
    _save(fig, "fig5_recalibration_data")


def fig_word_reliability(tables):
    wc = pd.read_csv(tables / "word_calibration.csv")
    bins = wc[wc["bin_lo"].notna()]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.plot([0, 1], [0, 1], color=INK2, linestyle=":", linewidth=1.2)
    for key, (lab, col, mk) in DOMAIN.items():
        d = bins[(bins["set"] == key) & (bins["n"] >= 30)]
        s = wc[(wc["set"] == key) & wc["bin_lo"].isna()]
        e = f"  ECE {s['ece'].iloc[0]:.3f}" if len(s) else ""
        ax.plot(d["mean_p_err"], d["frac_err"], color=col, marker=mk, label=lab.split(" (")[0] + e)
    ax.set_xlabel("1 − Whisper word probability")
    ax.set_ylabel("Observed word error rate")
    ax.set_title("Word-level confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, "fig6_word_reliability")


def fig_feature_shift(tables):
    fs = pd.read_csv(tables / "feature_shift.csv")
    feats = fs[fs["set"] == "pm_window_test"].sort_values("shift_sd")["feature"].tolist()
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    ypos = {f: i for i, f in enumerate(feats)}
    for key in ["eka_test", "pm_turn_test", "pm_window_test"]:
        lab, col, mk = DOMAIN[key]
        d = fs[fs["set"] == key]
        ax.plot(d["shift_sd"], d["feature"].map(ypos), linestyle="none", marker=mk, color=col, label=lab.split(" (")[0])
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.set_xlabel("Mean shift from Eka calibration set (in its standard deviations)")
    ax.set_title("Which confidence features move under shift")
    ax.legend(fontsize=9, loc="lower right")
    _save(fig, "fig7_feature_shift")


def fig_asr_overview(tables):
    a = pd.read_csv(tables / "asr_summary.csv").set_index("set")
    keys = ["eka_test", "pm_turn_test", "pm_window_test"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    for i, key in enumerate(keys):
        lab, col, _ = DOMAIN[key]
        r = a.loc[key]
        ax.barh(i, r["wer"], color=col, height=0.55)
        ax.errorbar(r["wer"], i, xerr=[[r["wer"] - r["wer_lo"]], [r["wer_hi"] - r["wer"]]], color=INK, capsize=3)
        ax.text(r["wer_hi"] + 0.01, i, f"{r['wer']:.1%}  (sub {r['sub_rate']:.1%} · del {r['del_rate']:.1%} · ins {r['ins_rate']:.1%})",
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([DOMAIN[k][0].split(" (")[0] for k in keys])
    ax.set_xlim(0, max(a.loc[keys, "wer_hi"]) * 2.1)
    ax.set_xlabel("Word error rate")
    ax.set_title("Whisper-small word error rate")
    ax.invert_yaxis()
    ax = axes[1]
    metrics = [("err_rate", "Transcripts with WER > 10%"), ("number_acc", "Numbers recognised"),
               ("negation_acc", "Negations recognised")]
    for i, key in enumerate(keys):
        lab, col, mk = DOMAIN[key]
        r = a.loc[key]
        for j, (m, _) in enumerate(metrics):
            if pd.isna(r.get(m)):
                continue
            ax.errorbar(r[m], j + (i - 1) * 0.18, xerr=[[r[m] - r[f"{m}_lo"]], [r[f"{m}_hi"] - r[m]]], color=col, marker=mk,
                        capsize=3, linestyle="none", label=lab.split(" (")[0] if j == 0 else None)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[1] for m in metrics])
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_title("Transcript-level and critical-token rates")
    ax.legend(fontsize=8.5, loc="lower left")
    _save(fig, "fig0_asr_overview")


def run(cfg: dict | None = None, model: str | None = None):
    model = (model or (cfg or {}).get("asr", {}).get("model") or "small").replace("/", "_")
    tables = RESULTS_DIR / "tables" / model
    _style()
    units = None
    us = tables / "err" / "unit_scores.csv.gz"
    uf = RESULTS_DIR / "units" / f"units_{model}.csv.gz"
    if us.exists() and uf.exists():
        sc = pd.read_csv(us)
        u = pd.read_csv(uf, usecols=["unit_id", "conf", "err"])
        units = sc.merge(u, on="unit_id")
    fig_asr_overview(tables)
    fig_confidence_vs_error(tables, units)
    fig_reliability(tables)
    fig_risk_coverage(tables, units)
    fig_threshold_transfer(tables)
    fig_learning_curve(tables)
    fig_word_reliability(tables)
    fig_feature_shift(tables)
    print(f"figures written to {FIGURES_DIR}")
