"""End-to-end run of the evaluation stack on synthetic units (no data or GPU needed)."""

import numpy as np
import pandas as pd
import pytest

from asrshift import evaluate as E
from asrshift.features import FEATURES


def synthetic_units(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    def unit(dataset, subset, split, group, shift, n_words):
        conf = float(np.clip(rng.beta(6, 2) - shift, 0.05, 0.99))
        n_ref = int(max(1, n_words + rng.integers(-2, 3)))
        # true per-word error grows as confidence falls; the target domain is worse than conf suggests
        p_word = np.clip(0.02 + 0.5 * (1 - conf) + shift, 0, 1)
        errors = int(rng.binomial(n_ref, p_word))
        sub_, del_ = errors // 2, errors - errors // 2
        n_num, n_neg = int(rng.integers(0, 3)), int(rng.integers(0, 2))
        f = {k: rng.normal() for k in FEATURES}
        f.update(mean_logprob=np.log(conf), word_p_mean=conf, conf=conf, empty_hyp=0.0, nsp_max=rng.random() * 0.3,
                 cr_max=1.2 + rng.random(), log_words=np.log1p(n_ref), n_segments=1 + n_words // 30)
        rows.append({
            "unit_id": f"{subset}_{len(rows)}", "dataset": dataset, "subset": subset, "split": split,
            "group_id": group, "consultation": group, "duration": n_words / 2.5, "recording_context": subset,
            "n_ref": n_ref, "errors": errors, "sub": sub_, "del": del_, "ins": 0, "wer": errors / n_ref,
            "errors_nocollapse": errors, "n_ref_nocollapse": n_ref,
            "n_num": n_num, "n_num_correct": n_num - int(errors > 3 and n_num > 0), "n_num_hyp": n_num, "n_num_false": 0,
            "n_neg": n_neg, "n_neg_correct": n_neg, "n_neg_hyp": n_neg, "n_neg_false": 0,
            "n_digit": n_num, "n_digit_correct": n_num, "n_crit_missed": int(errors > 3 and n_num > 0), "n_crit_false": 0,
            "n_ent": 1 if dataset == "eka" else np.nan, "n_ent_correct": float(errors == 0) if dataset == "eka" else np.nan,
            "wl_wrong": sub_, "wl_tokens": n_ref, "overlap_s": float(rng.random() < 0.3), "wild_excess": 0, **f,
        })

    for split, n_groups in (("calibration", 25), ("validation", 10), ("test", 15)):
        for g in range(n_groups):
            for _ in range(20):
                unit("eka", "narration_sentence", split, f"spk_{split}_{g}", 0.0, int(rng.choice([4, 30])))
    for split, n_groups in (("recalibration", 10), ("test", 30)):
        for g in range(n_groups):
            for _ in range(15):
                unit("primock57", "window", split, f"c_{split}_{g}", 0.08, 60)
                unit("primock57", "turn_doctor", split, f"c_{split}_{g}", 0.05, 12)
    df = pd.DataFrame(rows)
    df["err"] = (df["wer"] > 0.10).astype(int)
    df["any_err"] = (df["wer"] > 0).astype(int)
    df["severe_err"] = (df["wer"] > 0.30).astype(int)
    df["crit_err"] = ((df["n_crit_missed"] + df["n_crit_false"]) > 0).astype(int)
    df["err_c3"] = df["err"]
    df["ent_err"] = np.where(df["n_ent"] > 0, (df["n_ent_correct"] < df["n_ent"]).astype(float), np.nan)
    return df


@pytest.fixture(scope="module")
def result():
    df = synthetic_units()
    return df, E.evaluate_label(df, "err", n_boot=30, seed=1)


def test_every_policy_and_set_is_evaluated(result):
    _, res = result
    ops = res["operating_points"]
    assert set(ops["eval"]) == set(E.EVAL_SETS)
    win = ops[ops["eval"] == "pm_window_test"]
    assert {"P1", "P2", "P3", "P3-conf", "P1-retuned", "P2-retuned", "P4", "P4-refit"} <= set(win["policy"])
    assert "P4" not in set(ops[ops["eval"] == "eka_test"]["policy"])
    assert {"accepted_crit_err_rate", "accepted_num_err", "pred_accepted_err"} <= set(ops.columns)
    assert (win[win["policy"] == "P4"]["note"].str.len() > 0).any()


def test_budget_thresholds_hold_on_source_test(result):
    _, res = result
    ops = res["operating_points"]
    row = ops[(ops["eval"] == "eka_test") & (ops["policy"] == "P1") & (ops["point"] == "budget_0.10")].iloc[0]
    assert 0.03 < row["review_rate"] < 0.25


def test_platt_keeps_ranking_and_moves_probabilities(result):
    _, res = result
    rank = res["ranking"].set_index(["eval", "policy"])
    assert rank.loc[("pm_window_test", "P3"), "auroc"] == pytest.approx(rank.loc[("pm_window_test", "P4"), "auroc"])
    cal = res["calibration"].set_index(["eval", "policy"])
    # the synthetic target is worse than its confidence suggests: frozen P3 underpredicts risk
    assert cal.loc[("pm_window_test", "P3"), "citl"] < 0
    assert abs(cal.loc[("pm_window_test", "P4"), "citl"]) < abs(cal.loc[("pm_window_test", "P3"), "citl"])


def test_secondary_analyses_run(result):
    df, res = result
    paired = E.paired_differences(res, "err", n_boot=20, seed=1)
    assert {"H3a", "H3b", "H3c"} == set(paired["hypothesis"])
    lt = E.length_table(res, "err", n_boot=20, seed=1)
    assert lt["bin"].str.startswith("matched").any()
    lc = E.learning_curve(df, "err", seed=1, ks=(2, 5), draws=3)
    assert len(lc) == 6 and {"ece_P4", "ece_P4-int", "plugin_acc_err_P4_0.20"} <= set(lc.columns)
    dep = E.deployment_view(res, "err")
    assert (dep["consultations"] == 30).all()
    alarm = E.drift_alarm(res, E.feature_shift(df))
    assert alarm["fired"].dtype == bool


def test_confirmatory_family_and_diagnostics(result):
    df, res = result
    h = E.hypothesis_tests(res, df, model="synthetic-model-without-words", n_boot=40, seed=1)
    assert list(h["hypothesis"]) == ["H1", "H2", "H3a", "H3b", "H3c"]
    t = h[h["testable"]]
    assert (t["p_holm"] >= t["p_one_sided"]).all() and t["p_holm"].between(0, 1).all()
    # the synthetic target is built to be overconfident, so H1 should be clearly supported
    assert h.set_index("hypothesis").loc["H1", "estimate"] < 0
    contrib, dom = E.feature_support(res, seed=1)
    assert {"P3", "P3-conf"} == set(contrib["model"]) and dom["domain_auroc"].between(0, 1).all()
    rob = E.devpool_robustness(res, seed=1)
    assert set(rob["tuned_on"]) == {"validation", "calibration+validation"}
    s = E.sensitivity(df, "no overlap", df["overlap_s"] > 0, n_boot=10, seed=1)
    assert s["windows_test"].iloc[0] < (res["sets"]["pm_window_test"].shape[0])


def test_entity_label_eka_only():
    df = synthetic_units(1)
    eka = df[df["dataset"] == "eka"].assign(ent_err=lambda x: x["ent_err"].astype(int))
    r = E.evaluate_label(eka, "ent_err", n_boot=10, seed=1, eval_sets=("eka_test",), with_target=False)
    assert set(r["operating_points"]["policy"]) == {"P1", "P2", "P2-rawCR", "P3", "P3-conf"}
