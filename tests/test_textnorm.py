from asrshift.textnorm import (
    WILDCARD, collapse_repeats, has_unintelligible, reference_tokens, scoring_tokens, strip_tags, tokens,
)


def test_whisper_normaliser_basics():
    assert tokens("Hello, Doctor!  Um, I've got a HEADACHE.") == ["hello", "doctor", "i", "have", "got", "a", "headache"]


def test_units_are_canonical_on_both_sides():
    assert tokens("Azithromycin five hundred milligrams") == ["azithromycin", "500", "mg"]
    assert tokens("azithromycin 500 mg") == ["azithromycin", "500", "mg"]
    assert tokens("5 millilitres") == ["5", "ml"]


def test_primock_tags():
    text = "<UNIN/> Sorry. <UNSURE>Hello how</UNSURE> are you <UNIN/> <UNIN/>"
    assert reference_tokens(text) == [WILDCARD, "sorry", "hello", "how", "are", "you", WILDCARD]
    assert has_unintelligible(text)
    assert strip_tags(text) == "Sorry. Hello how are you"


def test_collapse_repeats():
    toks = "it is a bit it is a bit not very clear".split()
    assert collapse_repeats(toks)[0] == "it is a bit not very clear".split()
    kept, idx = collapse_repeats("no no allergies".split())
    assert kept == ["no", "allergies"] and idx == [0, 2]
    assert collapse_repeats(["a", WILDCARD, WILDCARD, "a"])[0] == ["a", WILDCARD, WILDCARD, "a"]


def test_inaudible_speech_is_a_wildcard_and_unknown_tags_fail_loudly():
    assert reference_tokens("a <INAUDIBLE_SPEECH/> b") == ["a", WILDCARD, "b"]
    assert has_unintelligible("So I <INAUDIBLE_SPEECH/>")
    import pytest

    with pytest.raises(ValueError):
        reference_tokens("a <NEWTAG b")
    with pytest.raises(ValueError):
        reference_tokens("a <LAUGH/> b")  # well-formed but unknown: must not be dropped silently


def test_numbers_are_not_fused_across_sentences():
    assert tokens("I am 25. 3 days ago it started.") == ["i", "am", "25", "3", "days", "ago", "it", "started"]
    assert tokens("Take 12.5 mg") == ["take", "12.5", "mg"]
    assert scoring_tokens("and I'm twenty six.\nTwenty six, OK.", reference=True) == ["and", "i", "am", "26", "okay"]
    assert scoring_tokens("Seven.\nSeven.", reference=True) == ["7"]
    assert tokens("two, three days") == ["2", "3", "days"]
    assert tokens("Dolo 650 three times a day") == ["dolo", "650", "3", "times", "a", "day"]
    assert tokens("twenty six") == ["26"] and tokens("1,000 patients") == ["1000", "patients"]


def test_spelling_variants_are_merged_on_both_sides():
    assert tokens("OK, alright.") == tokens("Okay, all right.") == ["okay", "all", "right"]
    assert tokens("O.K. the B.P. is fine") == ["okay", "the", "bp", "is", "fine"]


def test_repetition_loops_are_not_collapsed():
    loop = ["thank", "you"] * 6
    kept, _ = collapse_repeats(loop)
    assert kept == loop  # a loop removing more than 6 tokens is a hallucination, kept as errors
    assert collapse_repeats(["thank", "you", "thank", "you"])[0] == ["thank", "you"]


def test_scoring_tokens_symmetric_on_restarts():
    ref = scoring_tokens("Is that affecting your um, your mood?", reference=True)
    hyp = scoring_tokens("Is that affecting your mood?")
    assert ref == hyp
