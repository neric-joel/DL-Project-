import random

import jiwer
import pytest

from asrshift.align import align
from asrshift.textnorm import WILDCARD


def test_matches_jiwer_edit_count():
    rng = random.Random(0)
    vocab = list("abcdef")
    for _ in range(2000):
        ref = [rng.choice(vocab) for _ in range(rng.randint(1, 12))]
        hyp = [rng.choice(vocab) for _ in range(rng.randint(0, 12))]
        a = align(ref, hyp)
        o = jiwer.process_words(" ".join(ref), " ".join(hyp))
        assert a.errors == o.substitutions + o.deletions + o.insertions
        assert a.hits + a.substitutions + a.deletions == len(ref)
        assert len(a.hyp_status) == len(hyp) and "" not in a.hyp_status


def test_perfect_and_empty():
    a = align(["take", "500", "mg"], ["take", "500", "mg"])
    assert a.wer == 0 and a.hits == 3
    a = align(["take", "500", "mg"], [])
    assert a.deletions == 3 and a.wer == 1
    assert align([], ["x"]).n_ref == 0


def test_substitution_is_reported_on_both_sides():
    a = align(["500", "mg"], ["50", "mg"])
    assert a.ref_status == ["S", "C"] and a.hyp_status == ["S", "C"] and a.ref_to_hyp == [0, 1]


@pytest.mark.parametrize("hyp", [[], ["x"], ["x", "y", "z"]])
def test_wildcard_absorbs_any_run(hyp):
    a = align(["i", "have", WILDCARD, "pain"], ["i", "have", *hyp, "pain"])
    assert a.errors == 0 and a.n_ref == 3 and a.absorbed == len(hyp)
    assert a.ref_status[2] == "W"


def test_wildcard_does_not_hide_errors_elsewhere():
    a = align(["no", "chest", "pain", WILDCARD], ["chest", "pain", "mumble"])
    assert a.deletions == 1 and a.ref_status[0] == "D" and a.wer == pytest.approx(1 / 3)


def test_wildcard_runs_are_recorded():
    a = align(["a", WILDCARD, "b", WILDCARD], ["a", "x", "b", "p", "q", "r", "s", "t"])
    assert a.wild_runs == [1, 5] and a.wild_excess(3) == 2 and a.errors == 0


def test_wildcard_prefers_matching_real_words():
    # "pain" should align to the reference word, not be swallowed by the wildcard
    a = align([WILDCARD, "pain"], ["uh", "pain"])
    assert a.ref_status == ["W", "C"] and a.hyp_status == ["W", "C"]
