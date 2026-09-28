import json

import pytest

from asrshift.align import align
from asrshift.score import _critical, entity_scores, is_number, score_unit
from asrshift.textnorm import scoring_tokens


def _crit(ref_text, hyp_text):
    ref = scoring_tokens(ref_text, reference=True)
    hyp = scoring_tokens(hyp_text)
    a = align(ref, hyp)
    return _critical(ref, a.ref_status, hyp, a.hyp_status)


@pytest.mark.parametrize("text,expected", [
    ("take 1 tablet", True),
    ("take one tablet", True),
    ("the one on the left", False),
    ("twice a day", True),
    ("once daily", True),
    ("first of all", False),
    ("wait a second", False),
    ("500 mg", True),
])
def test_number_definition(text, expected):
    toks = scoring_tokens(text)
    assert any(is_number(toks, i) for i in range(len(toks))) == expected


def test_dose_error_is_a_miss_and_a_false_alarm():
    c = _crit("azithromycin 500 mg twice a day", "azithromycin 50 mg twice a day")
    assert c["n_num"] == 2 and c["n_num_correct"] == 1
    assert c["n_num_false"] == 1 and c["n_crit_missed"] == 1 and c["n_crit_false"] == 1


def test_hallucinated_negation_is_a_false_alarm():
    c = _crit("I have chest pain", "I do not have chest pain")
    assert c["n_neg"] == 0 and c["n_neg_false"] == 1


def test_missed_negation():
    c = _crit("no chest pain", "chest pain")
    assert c["n_neg"] == 1 and c["n_neg_correct"] == 0 and c["n_crit_missed"] == 1


def test_twice_to_once_is_caught():
    c = _crit("take it twice a day", "take it once a day")
    assert c["n_crit_missed"] == 1 and c["n_crit_false"] == 1


def test_entity_scoring_uses_alignment():
    ents = json.dumps([["Dolo 650", "drugs", [[0, 8]]], ["thrice a day", "advices", [[16, 28]]]])
    ref = scoring_tokens("Dolo 650 tablet thrice a day", reference=True)
    hyp = scoring_tokens("Dolo 65 tablet thrice a day")
    a = align(ref, hyp)
    s = entity_scores(ents, ref, a.ref_status, hyp)
    assert s["n_ent"] == 2 and s["n_ent_correct"] == 1
    assert s["n_ent_drugs_correct"] == 0 and s["n_ent_advices_correct"] == 1


def test_entity_mapped_to_annotated_occurrence_and_deduplicated():
    text = "fever today and no fever yesterday"
    # annotated span is the second "fever" (offset 19); listed twice, counted once
    ents = json.dumps([["fever", "clinical_findings", [[19, 24]]], ["fever", "clinical_findings", [[19, 24]]]])
    ref = scoring_tokens(text, reference=True)
    hyp = scoring_tokens("fever today and no favour yesterday")
    a = align(ref, hyp)
    s = entity_scores(ents, ref, a.ref_status, hyp, text)
    assert s["n_ent"] == 1 and s["n_ent_correct"] == 0


def test_term_matcher_longest_match_and_scoring():
    import pandas as pd

    from asrshift.terms import Matcher, term_scores

    lex = pd.DataFrame({"term": ["chest pain", "pain", "blood test"],
                        "type": ["clinical_findings", "clinical_findings", "diagnostics"]})
    toks = "i have chest pain and some pain after the blood test".split()
    assert [h[2] for h in Matcher(lex).find(toks)] == ["chest pain", "pain", "blood test"]
    status = ["C"] * len(toks)
    status[3] = "S"  # "pain" in "chest pain" misrecognised
    s = term_scores(toks, status, lex)
    assert s["n_term"] == 3 and s["n_term_correct"] == 2 and s["n_term_clinical_findings_correct"] == 1


def test_score_unit_end_to_end():
    row = {"reference": "No fever. <UNIN/> Take paracetamol 500 mg.", "dataset": "primock57"}
    rec = {"text": " No fever, um, take paracetamol 500 mg.", "audio_s": 4.0,
           "segments": [{"avg_logprob": -0.1, "n_tokens": 9, "no_speech_prob": 0.0, "compression_ratio": 1.0,
                         "temperature": 0.0}],
           "words": [{"word": w, "probability": 0.9} for w in [" No", " fever,", " um,", " take", " paracetamol", " 500", " mg."]]}
    out, words = score_unit(row, rec)
    assert out["errors"] == 0 and out["n_ref"] == 6 and out["wer"] == 0
    assert out["n_neg_correct"] == 1 and out["n_num_correct"] == 1
    assert all(s == "C" or s == "W" for _, _, s, _ in words)
