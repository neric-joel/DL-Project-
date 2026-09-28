"""Medical-term lexicon for PriMock57 (exploratory; the proposal's optional lexicon step).

PriMock57 has no entity annotations. We reuse Eka's human annotations as the lexicon: every
annotated entity (drugs, advice, diagnostics, clinical findings) is normalised with the scoring
normaliser, and pure numbers/doses and generic words are dropped (numbers are scored separately as
critical tokens). Occurrences of these terms in PriMock57 references are then found by greedy
longest-match and scored like Eka entities: a term is right if all its words are aligned as correct.
Because the vocabulary is the same on both sides, term accuracy can be compared across domains.

A deterministic lexicon was chosen over an LLM tagger: a small local model (qwen3.5:2b via Ollama)
tried first copied the prompt's examples and missed most findings, so it would have needed far more
auditing than the time allowed. ``audit_sample.csv`` is a random sample of matches for a manual check.

    asrshift terms
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

import pandas as pd

from asrshift import data_eka
from asrshift.data_primock import load_utterances
from asrshift.paths import RESULTS_DIR
from asrshift.textnorm import collapse_repeats, scoring_tokens, tokens

TERMS_DIR = RESULTS_DIR / "terms"
LEXICON = TERMS_DIR / "lexicon.csv"
TYPES = ("drugs", "advices", "diagnostics", "clinical_findings")
# Annotated spans that are too generic to count as a medical term when they stand alone.
GENERIC = frozenset({
    "tablet", "tablets", "capsule", "capsules", "medicine", "medicines", "medication", "drug", "drugs",
    "test", "tests", "day", "days", "week", "weeks", "month", "months", "daily", "once", "twice", "time",
    "times", "take", "use", "doctor", "patient", "history", "normal", "left", "right", "mg", "ml", "dose",
    # single words that are usually not clinical in conversation ("come back", "the rest of", "a report")
    "back", "rest", "report", "appearance", "symptoms", "cold", "skin", "heart", "blood", "chest", "diet",
    "one tablet", "medications", "sugar", "weight", "smoke", "radiating to", "exercise", "non", "signal",
})


def _is_candidate(toks: list[str]) -> bool:
    if not toks or " ".join(toks) in GENERIC:
        return False
    return any(t.isalpha() and len(t) >= 3 and t not in GENERIC for t in toks)


def build_lexicon() -> pd.DataFrame:
    meta = data_eka.load_metadata()
    kinds: dict[str, Counter] = defaultdict(Counter)
    for raw in meta["medical_entities"]:
        for ent in data_eka.parse_entities(raw):
            if ent["type"] not in TYPES:
                continue
            toks = collapse_repeats(tokens(ent["text"]))[0]
            if _is_candidate(toks):
                kinds[" ".join(toks)][ent["type"]] += 1
    rows = [{"term": t, "type": c.most_common(1)[0][0], "eka_count": sum(c.values())} for t, c in kinds.items()]
    return pd.DataFrame(rows)


class Matcher:
    """Greedy longest-match of lexicon terms in a token list (non-overlapping)."""

    def __init__(self, lexicon: pd.DataFrame):
        self.by_first: dict[str, list[tuple[list[str], str]]] = defaultdict(list)
        for term, ttype in zip(lexicon["term"], lexicon["type"]):
            t = term.split()
            self.by_first[t[0]].append((t, ttype))
        for v in self.by_first.values():
            v.sort(key=lambda x: -len(x[0]))

    def find(self, toks: list[str]) -> list[tuple[int, int, str, str]]:
        out, i = [], 0
        while i < len(toks):
            for t, ttype in self.by_first.get(toks[i], ()):
                if toks[i:i + len(t)] == t:
                    out.append((i, i + len(t), " ".join(t), ttype))
                    i += len(t)
                    break
            else:
                i += 1
        return out


def build(seed: int = 598) -> pd.DataFrame:
    lex = build_lexicon()
    m = Matcher(lex)
    utts = load_utterances()
    hits = []
    for u in utts.itertuples(index=False):
        toks = scoring_tokens(u.text, reference=True)
        for s, e, term, ttype in m.find(toks):
            hits.append({"consultation": u.consultation, "role": u.role, "term": term, "type": ttype,
                         "context": " ".join(toks[max(0, s - 6): e + 6])})
    hits = pd.DataFrame(hits)
    counts = hits.groupby("term").size().rename("primock_count")
    lex = lex.merge(counts, on="term", how="inner").sort_values(["type", "primock_count"], ascending=[True, False])
    TERMS_DIR.mkdir(parents=True, exist_ok=True)
    lex.to_csv(LEXICON, index=False)
    rng = random.Random(seed)
    sample = hits.iloc[sorted(rng.sample(range(len(hits)), min(100, len(hits))))].copy()
    sample["is_medical_term (y/n)"] = ""
    sample["type_correct (y/n)"] = ""
    sample["checked_by"] = ""
    sample.to_csv(TERMS_DIR / "audit_sample.csv", index=False)
    (TERMS_DIR / "lexicon_meta.json").write_text(json.dumps({
        "source": "Eka entity annotations (all splits), normalised; generic words and pure numbers removed",
        "terms_in_eka": int(len(build_lexicon())), "terms_found_in_primock57": int(len(lex)),
        "occurrences_in_primock57": int(len(hits))}, indent=2))
    return lex


def load() -> pd.DataFrame | None:
    return pd.read_csv(LEXICON) if LEXICON.exists() else None


_MATCHERS: dict[int, Matcher] = {}


def term_scores(ref: list[str], ref_status: list[str], lexicon: pd.DataFrame | None) -> dict:
    """Lexicon-term occurrences in one reference, and how many are recognised."""
    out = {"n_term": 0, "n_term_correct": 0}
    if lexicon is None:
        return out
    m = _MATCHERS.setdefault(id(lexicon), Matcher(lexicon))
    for s, e, _term, ttype in m.find(ref):
        ok = all(st == "C" for st in ref_status[s:e])
        out["n_term"] += 1
        out["n_term_correct"] += int(ok)
        out[f"n_term_{ttype}"] = out.get(f"n_term_{ttype}", 0) + 1
        out[f"n_term_{ttype}_correct"] = out.get(f"n_term_{ttype}_correct", 0) + int(ok)
    return out
