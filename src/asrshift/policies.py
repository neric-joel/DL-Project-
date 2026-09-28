"""Risk models and the review policies compared in the study.

| Policy | Risk score | Fitted on | Thresholds tuned on |
|---|---|---|---|
| P1 raw confidence | 1 - exp(mean token log-prob) | - | Eka validation |
| P2 Whisper rules | max(no_speech/0.6, compression/2.4) | - | Eka validation |
| P3 Eka-calibrated (frozen) | logistic regression, all features | Eka calibration | Eka validation |
| P4 PriMock-recalibrated | Platt refit of P3's logit | PriMock recalibration | PriMock recalibration |

Variants: ``P3-conf`` (confidence features only), ``P1-retuned`` / ``P2-retuned`` (same score,
thresholds re-tuned on PriMock recalibration), ``P4-refit`` (logistic regression refitted from
scratch on PriMock recalibration), and ``ceiling`` (logistic regression cross-fitted on the
evaluation set itself by consultation; not deployable, an upper reference).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from asrshift.features import CONFIDENCE_FEATURES, FEATURES

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
EPS = 1e-6


def _logloss(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class ConstantModel:
    """Stand-in when a fitting set has one class only: predicts that set's error rate everywhere."""

    chosen_C_ = None

    def __init__(self, rate: float):
        self.rate = float(np.clip(rate, EPS, 1 - EPS))

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.rate), np.full(len(X), self.rate)])


def fit_logreg(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, n_folds: int = 5):
    """Standardised L2 logistic regression; C chosen by grouped CV on log loss."""
    if len(np.unique(y)) < 2:
        return ConstantModel(float(np.mean(y)))
    n_folds = min(n_folds, len(np.unique(groups)))
    best_c, best_loss = C_GRID[0], np.inf
    if n_folds >= 2:
        for c in C_GRID:
            losses = []
            for tr, va in GroupKFold(n_splits=n_folds).split(X, y, groups):
                if len(np.unique(y[tr])) < 2:
                    continue
                m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
                m.fit(X[tr], y[tr])
                losses.append(_logloss(y[va], m.predict_proba(X[va])[:, 1]) * len(va))
            loss = sum(losses) / len(y) if losses else np.inf
            if loss < best_loss:
                best_c, best_loss = c, loss
    model = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=5000))
    model.fit(X, y)
    model.chosen_C_ = best_c
    return model


@dataclass
class Platt:
    """p' = sigmoid(a * logit(p) + b), fitted by (nearly unregularised) logistic regression.

    ``intercept_only`` fixes a = 1 and fits only the shift b (one parameter, for very small
    recalibration sets). A two-parameter fit whose slope comes out <= 0 would reverse the ranking,
    so it falls back to the intercept-only fit and records ``fell_back``.
    """

    a: float = 1.0
    b: float = 0.0
    intercept_only: bool = False
    fell_back: bool = False

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Platt":
        z = logit(np.clip(p, EPS, 1 - EPS))
        y = np.asarray(y, int)
        if not self.intercept_only and len(np.unique(y)) == 2:
            lr = LogisticRegression(C=1e4, max_iter=5000).fit(z.reshape(-1, 1), y)
            a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
            if a > 0:
                self.a, self.b = a, b
                return self
            self.fell_back = True
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(lambda b: _logloss(y, expit(z + b)), bounds=(-10, 10), method="bounded")
        self.a, self.b = 1.0, float(res.x)
        return self

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return expit(self.a * logit(np.clip(p, EPS, 1 - EPS)) + self.b)


@dataclass
class Policy:
    name: str
    label: str
    probabilistic: bool
    tuned_on: str                      # "eka" or "primock"
    score: callable = field(repr=False)  # DataFrame -> risk (probability if probabilistic)
    info: dict = field(default_factory=dict)


def raw_risk(df: pd.DataFrame) -> np.ndarray:
    return 1.0 - df["conf"].to_numpy(float)


def rules_risk(df: pd.DataFrame) -> np.ndarray:
    """P2 (Amendment 2): no-speech probability plus repetition signals that do not grow with length.

    risk >= 1 when no_speech > 0.6, the compression ratio exceeds Whisper's 2.4 limit, or at least one
    word trigram in five repeats (a looping transcript). Raw compression ratio rises with text
    length, so below 2.4 it is not used. Ties (most units score < 1 on every arm) are broken by
    raw confidence, so budgets rank the rest of the queue sensibly.
    """
    nsp = df["nsp_max"].to_numpy(float) / 0.6
    cr = df["cr_max"].to_numpy(float)
    cr_arm = np.where(cr > 2.4, cr / 2.4, 0.0)
    rep = df["rep3"].to_numpy(float) / 0.2
    tie = 1e-3 * (1.0 - df["conf"].to_numpy(float))
    return np.maximum.reduce([nsp, cr_arm, rep]) + tie


def rules_risk_raw_cr(df: pd.DataFrame) -> np.ndarray:
    """P2 as first registered: max(no_speech / 0.6, compression ratio / 2.4)."""
    return np.maximum(df["nsp_max"].to_numpy(float) / 0.6, df["cr_max"].to_numpy(float) / 2.4)


def build_policies(eka_cal: pd.DataFrame, pm_recal: pd.DataFrame | None, label: str, seed: int) -> list[Policy]:
    """Fit every policy. ``pm_recal`` holds one PriMock unit type (windows or turns); with ``None``
    only the source-domain policies (P1, P2, P3, P3-conf) are built."""
    y_cal = eka_cal[label].to_numpy(int)
    g_cal = eka_cal["group_id"].to_numpy()
    full = fit_logreg(eka_cal[FEATURES].to_numpy(float), y_cal, g_cal, seed)
    conf_only = fit_logreg(eka_cal[CONFIDENCE_FEATURES].to_numpy(float), y_cal, g_cal, seed)

    def p3(df):
        return full.predict_proba(df[FEATURES].to_numpy(float))[:, 1]

    def p3c(df):
        return conf_only.predict_proba(df[CONFIDENCE_FEATURES].to_numpy(float))[:, 1]

    coef = dict(zip(FEATURES, full[-1].coef_[0].round(4).tolist())) if not isinstance(full, ConstantModel) else {}
    source = [
        Policy("P1", "Raw confidence", False, "eka", raw_risk),
        Policy("P2", "No-speech + repetition rules", False, "eka", rules_risk),
        Policy("P2-rawCR", "Rules with raw compression ratio (as first registered)", False, "eka", rules_risk_raw_cr),
        Policy("P3", "Eka-calibrated (frozen)", True, "eka", p3, {"C": full.chosen_C_, "coef_standardised": coef}),
        Policy("P3-conf", "Eka-calibrated, confidence features only", True, "eka", p3c, {"C": conf_only.chosen_C_}),
    ]
    if pm_recal is None:
        return source

    y_rc = pm_recal[label].to_numpy(int)
    platt = Platt().fit(p3(pm_recal), y_rc)
    refit = fit_logreg(pm_recal[FEATURES].to_numpy(float), y_rc, pm_recal["group_id"].to_numpy(), seed)

    def p4(df):
        return platt(p3(df))

    def p4f(df):
        return refit.predict_proba(df[FEATURES].to_numpy(float))[:, 1]

    return source + [
        Policy("P1-retuned", "Raw confidence, thresholds re-tuned on PriMock", False, "primock", raw_risk),
        Policy("P2-retuned", "Rules, thresholds re-tuned on PriMock", False, "primock", rules_risk),
        Policy("P4", "Recalibrated on PriMock (Platt)", True, "primock", p4, {"a": platt.a, "b": platt.b}),
        Policy("P4-refit", "Refitted on PriMock", True, "primock", p4f, {"C": refit.chosen_C_}),
    ]


def crossfit_ceiling(df: pd.DataFrame, label: str, seed: int, n_folds: int = 5) -> np.ndarray:
    """Out-of-fold P(err) from a logistic regression trained on the evaluation set itself."""
    X, y, g = df[FEATURES].to_numpy(float), df[label].to_numpy(int), df["group_id"].to_numpy()
    out = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=min(n_folds, len(np.unique(g)))).split(X, y, g):
        m = fit_logreg(X[tr], y[tr], g[tr], seed)
        out[te] = m.predict_proba(X[te])[:, 1]
    return out
