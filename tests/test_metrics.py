import numpy as np
import pytest

from asrshift import metrics as M


def test_ece_zero_for_perfectly_calibrated_bins():
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(900)], np.r_[np.ones(500), np.zeros(500)],
                        np.r_[np.ones(900), np.zeros(100)]])
    assert M.ece(p, y) == pytest.approx(0.0, abs=1e-9)
    assert M.calibration_in_the_large(p, y) == pytest.approx(0.0, abs=1e-9)


def test_overconfidence_is_negative_citl():
    p = np.full(100, 0.2)
    y = np.r_[np.ones(60), np.zeros(40)]
    assert M.calibration_in_the_large(p, y) == pytest.approx(-0.4)
    assert M.ece(p, y) == pytest.approx(0.4)
    assert M.brier(p, y) == pytest.approx(0.6 * 0.64 + 0.4 * 0.04)


def test_budget_threshold_reviews_the_budget():
    r = np.random.default_rng(0).random(10_000)
    t = M.budget_threshold(r, 0.10)
    assert (r >= t).mean() == pytest.approx(0.10, abs=0.002)


def test_risk_target_threshold():
    p = np.array([0.05, 0.1, 0.2, 0.3, 0.6, 0.9])
    y = np.array([0, 0, 1, 0, 1, 1])
    t = M.risk_target_threshold(p, y, 0.25)
    accepted = p <= t
    assert y[accepted].mean() <= 0.25 and t == pytest.approx(0.3)
    assert M.risk_target_threshold(p, np.ones(6), 0.1) == float("-inf")


def test_risk_target_min_size_and_cp_bound():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p * 0.5).astype(int)
    t_emp = M.risk_target_threshold(p, y, 0.10, n_min=30)
    t_cp = M.risk_target_threshold(p, y, 0.10, n_min=30, bound="cp")
    assert (p <= t_emp).sum() >= 30 and t_cp < t_emp  # the bound is more conservative
    assert y[p <= t_cp].mean() <= 0.10
    assert M.risk_target_threshold(p[:20], y[:20], 0.5, n_min=30) == float("-inf")


def test_thresholds_are_equivariant_under_increasing_maps():
    # a monotone recalibration (e.g. Platt with a > 0) cannot change which units a rank rule picks
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    f = lambda x: 1 / (1 + np.exp(-(2.0 * np.log(x / (1 - x)) + 0.7)))  # noqa: E731
    for rule in (lambda s: M.budget_threshold(s, 0.1), lambda s: M.risk_target_threshold(s, y, 0.2, n_min=30)):
        assert ((p >= rule(p)) == (f(p) >= rule(f(p)))).all()


def test_risk_target_respects_ties():
    p = np.array([0.1, 0.1, 0.5])
    y = np.array([0, 1, 1])
    # accepting only the first 0.1 is impossible: both 0.1 units go together (error rate 0.5)
    assert M.risk_target_threshold(p, y, 0.3) == float("-inf")


def test_plugin_threshold_uses_predictions_only():
    p = np.array([0.05, 0.1, 0.15, 0.4, 0.8])
    t = M.plugin_threshold(p, 0.10)
    assert t == pytest.approx(0.15)  # mean(0.05, 0.1, 0.15) = 0.10
    assert p[p <= t].mean() <= 0.10 + 1e-12


def test_operating_point_residuals():
    review = np.array([1, 0, 0], bool)
    extra = {"n_num": np.array([1.0, 2.0, 0.0]), "n_num_correct": np.array([0.0, 1.0, 0.0]),
             "crit_err": np.array([1, 1, 0]), "del": np.array([3.0, 1.0, 0.0])}
    op = M.operating_point(review, np.array([1, 1, 0]), np.array([3.0, 1.0, 0.0]), np.array([10.0, 10.0, 10.0]),
                           extra=extra, p=np.array([0.9, 0.2, 0.1]))
    assert op["accepted_num_err"] == pytest.approx(0.5)
    assert op["accepted_crit_err_rate"] == pytest.approx(0.5) and op["crit_caught"] == pytest.approx(0.5)
    assert op["accepted_del_rate"] == pytest.approx(1 / 20) and op["pred_accepted_err"] == pytest.approx(0.15)


def test_operating_point():
    review = np.array([1, 0, 0, 1, 0], bool)
    y = np.array([1, 1, 0, 0, 0])
    errors = np.array([5.0, 2.0, 0.0, 0.0, 1.0])
    n_ref = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
    op = M.operating_point(review, y, errors, n_ref)
    assert op["review_rate"] == pytest.approx(0.4)
    assert op["err_caught"] == pytest.approx(0.5)
    assert op["accepted_err_rate"] == pytest.approx(1 / 3)
    assert op["accepted_wer"] == pytest.approx(3 / 30)
    assert op["word_errors_caught"] == pytest.approx(5 / 8)


def test_risk_coverage_and_aurc():
    y = np.array([0, 0, 1, 1])
    cov, risk = M.risk_coverage(np.array([0.1, 0.2, 0.8, 0.9]), y)
    assert list(risk) == [0, 0, pytest.approx(1 / 3), 0.5]
    assert M.aurc(np.array([0.9, 0.8, 0.2, 0.1]), y) > M.aurc(np.array([0.1, 0.2, 0.8, 0.9]), y)


def test_holm_and_one_sided_p():
    assert M.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert M.one_sided_p(np.array([-1.0, -2.0, -0.5, 0.2]), "less") == pytest.approx(2 / 5)
    assert M.one_sided_p(np.array([1.0, 2.0]), "greater") == pytest.approx(1 / 3)


def test_cluster_bootstrap_resamples_whole_groups():
    groups = np.array(["a", "a", "b", "b", "c"])
    bs = M.ClusterBootstrap(groups, n_boot=50, seed=0)
    for ix in bs.indices():
        picked = groups[ix]
        for g in set(picked):
            assert (picked == g).sum() % (groups == g).sum() == 0
    lo, hi = bs.interval(lambda ix: float(len(ix)))
    assert lo <= hi


def test_auroc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(1)
    for _ in range(20):
        y = rng.integers(0, 2, 300)
        r = np.round(rng.random(300), 1)  # many ties
        assert M.auroc(r, y) == pytest.approx(roc_auc_score(y, r), abs=1e-12)
    assert np.isnan(M.auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_headline_matches_operating_point():
    rng = np.random.default_rng(2)
    review, y = rng.random(200) < 0.3, rng.integers(0, 2, 200)
    crit = rng.integers(0, 2, 200).astype(float)
    op = M.operating_point(review, y, np.ones(200), np.ones(200), {"crit_err": crit})
    assert M.headline(review, y.astype(float), crit) == tuple(op[k] for k in M.HEADLINE)


def test_bootstrap_intervals_match_interval_with_and_without_cache():
    groups = np.repeat(np.arange(15), 4)
    x = np.random.default_rng(3).random(len(groups))
    stats = (lambda ix: float(x[ix].mean()), lambda ix: float(np.median(x[ix])))
    cached = M.ClusterBootstrap(groups, n_boot=100, seed=0)
    plain = M.ClusterBootstrap(groups, n_boot=100, seed=0)
    plain._cacheable = False
    joint = cached.intervals(lambda ix: tuple(f(ix) for f in stats))
    assert joint == [plain.interval(f) for f in stats] == [cached.interval(f) for f in stats]
