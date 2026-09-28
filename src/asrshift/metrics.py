"""Calibration, ranking and selective-review metrics, and the cluster bootstrap.

Conventions: ``p`` is a predicted probability that the transcript is erroneous, ``y`` the 0/1 label,
``r`` any risk score where larger means riskier. A policy sends a unit to review when ``r >= t``.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score

N_BINS = 15


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in np.unique(bins):
        m = bins == b
        total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(p))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def calibration_in_the_large(p: np.ndarray, y: np.ndarray) -> float:
    """Mean predicted risk minus observed error rate. Negative means overconfident."""
    return float(np.mean(p) - np.mean(y))


def reliability(p: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            out.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "frac_err": float(y[m].mean())})
    return out


def auroc(r: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, r))


def risk_coverage(r: np.ndarray, loss: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Accept units in order of increasing risk. Returns (coverage, selective risk) at every step.

    With ``weights`` (e.g. reference word counts) the selective risk is a weighted mean, so passing
    per-unit error counts as ``loss * weights`` gives pooled WER of the accepted set.
    """
    order = np.argsort(np.asarray(r, float), kind="stable")
    loss = np.asarray(loss, float)[order]
    w = np.ones_like(loss) if weights is None else np.asarray(weights, float)[order]
    cum_w = np.cumsum(w)
    coverage = np.arange(1, len(loss) + 1) / len(loss)
    return coverage, np.cumsum(loss * w) / cum_w


def aurc(r: np.ndarray, loss: np.ndarray, weights: np.ndarray | None = None) -> float:
    return float(np.mean(risk_coverage(r, loss, weights)[1]))


def budget_threshold(r_tune: np.ndarray, budget: float) -> float:
    """Risk threshold that sends a share ``budget`` of the tuning set to review."""
    return float(np.quantile(np.asarray(r_tune, float), 1.0 - budget, method="higher"))


def risk_target_threshold(p_tune: np.ndarray, y_tune: np.ndarray, alpha: float, n_min: int = 1,
                          bound: str | None = None, level: float = 0.95) -> float:
    """Threshold t for "accept if score <= t" that keeps the accepted-error rate at or below alpha.

    * ``bound=None``: the largest t whose *empirical* accepted-error rate on the tuning set is <= alpha,
      among cut points that accept at least ``n_min`` units. Optimistic on small tuning sets (the
      winner's curse of picking the best-looking cut).
    * ``bound="cp"``: fixed-sequence testing with a one-sided Clopper-Pearson upper bound. Walk the cut
      points from the most confident unit outwards (starting at ``n_min`` accepted) and stop at the
      first one whose upper bound exceeds alpha. The guarantee holds with probability ``level``.

    Returns -inf if no cut point qualifies, meaning every unit is sent to review.
    """
    p, y = np.asarray(p_tune, float), np.asarray(y_tune, float)
    order = np.argsort(p, kind="stable")
    ps, ys = p[order], y[order]
    n = np.arange(1, len(ys) + 1)
    k = np.cumsum(ys)
    # only thresholds between distinct values are valid cut points
    valid = np.r_[ps[1:] != ps[:-1], True] & (n >= n_min)
    if bound is None:
        ok = valid & (k / n <= alpha + 1e-12)  # tolerance for floating-point sums
        return float(ps[np.nonzero(ok)[0].max()]) if ok.any() else float("-inf")
    from scipy.stats import beta

    upper = np.where(k < n, beta.ppf(level, k + 1, n - k), 1.0)
    best = float("-inf")
    for i in np.nonzero(valid)[0]:
        if upper[i] > alpha:
            break
        best = float(ps[i])
    return best


def plugin_threshold(p_stream: np.ndarray, alpha: float) -> float:
    """Largest tau such that the *predicted* error rate of the accepted set, mean(p | p <= tau), is <= alpha.

    Uses only the model's probabilities, not labels, so it can be computed on the unlabelled stream of
    transcripts a deployed system sees. It is only as good as the calibration of ``p``.
    """
    return risk_target_threshold(p_stream, p_stream, alpha)


def _ratio(num, den, mask):
    d = den[mask].sum()
    return float(num[mask].sum() / d) if d else float("nan")


def operating_point(review: np.ndarray, y: np.ndarray, errors: np.ndarray, n_ref: np.ndarray,
                    extra: dict | None = None, p: np.ndarray | None = None) -> dict:
    """Outcomes of one review decision vector.

    ``extra`` may hold per-unit count arrays (sub, del, ins, n_num, n_num_correct, n_neg,
    n_neg_correct, n_ent, n_ent_correct, crit_err) for residual clinical error among accepted units.
    ``p`` (a policy's probabilities) adds the error rate the policy *predicts* for the accepted set.
    """
    review = np.asarray(review, bool)
    y = np.asarray(y, float)
    acc = ~review
    n_err = y.sum()
    out = {
        "review_rate": float(review.mean()),
        "err_caught": float((y * review).sum() / n_err) if n_err else float("nan"),
        "accepted_err_rate": float(y[acc].mean()) if acc.any() else float("nan"),
        "accepted_wer": float(errors[acc].sum() / n_ref[acc].sum()) if acc.any() else float("nan"),
        "word_errors_caught": float(errors[review].sum() / errors.sum()) if errors.sum() else float("nan"),
        "n_accepted_errors": float((y * acc).sum()),
    }
    if p is not None:
        out["pred_accepted_err"] = float(np.mean(p[acc])) if acc.any() else float("nan")
    if extra:
        for k in ("sub", "del", "ins"):
            if k in extra:
                out[f"accepted_{k}_rate"] = _ratio(extra[k], n_ref, acc)
        for kind in ("num", "neg", "ent", "term"):
            if f"n_{kind}" in extra:
                out[f"accepted_{kind}_err"] = 1.0 - _ratio(extra[f"n_{kind}_correct"], extra[f"n_{kind}"], acc)
        if "crit_err" in extra:
            c = np.asarray(extra["crit_err"], float)
            out["accepted_crit_err_rate"] = float(c[acc].mean()) if acc.any() else float("nan")
            out["crit_caught"] = float((c * review).sum() / c.sum()) if c.sum() else float("nan")
    return out


class ClusterBootstrap:
    """Resample whole clusters (speakers, sessions, consultations) with replacement."""

    def __init__(self, groups: np.ndarray, n_boot: int, seed: int):
        _, codes = np.unique(np.asarray(groups), return_inverse=True)
        self.members = [np.nonzero(codes == g)[0] for g in range(codes.max() + 1)]
        rng = np.random.default_rng(seed)
        self.draws = [rng.integers(0, len(self.members), len(self.members)) for _ in range(n_boot)]

    def indices(self):
        for d in self.draws:
            yield np.concatenate([self.members[g] for g in d])

    def replicates(self, stat: Callable[[np.ndarray], float]) -> np.ndarray:
        """The statistic on every bootstrap replicate (non-finite values kept as NaN)."""
        return np.array([stat(ix) for ix in self.indices()], dtype=float)

    def interval(self, stat: Callable[[np.ndarray], float], level: float = 0.95) -> tuple[float, float]:
        vals = self.replicates(stat)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return float("nan"), float("nan")
        a = (1 - level) / 2
        return float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def holm(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def one_sided_p(replicates: np.ndarray, direction: str) -> float:
    """Bootstrap p-value for H: statistic < 0 (direction="less") or > 0 ("greater").

    The share of replicates on the wrong side of zero, with a +1 correction so it is never exactly 0.
    """
    r = replicates[np.isfinite(replicates)]
    wrong = (r >= 0).sum() if direction == "less" else (r <= 0).sum()
    return float((wrong + 1) / (len(r) + 1))
