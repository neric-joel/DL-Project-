"""Text normalisation shared by every scorer.

All scoring uses Whisper's English normaliser (lower case, punctuation and filler words removed,
spelled-out numbers turned into digits, British spellings mapped to American), preceded by a few
symmetric spelling merges ("OK"/"O.K." -> "okay", "alright" -> "all right", dotted acronyms joined)
and followed by one spelling per dosage unit. PriMock57 references carry three transcriber tags:

* ``<UNSURE>words</UNSURE>``: the transcriber was unsure but gave the words, so they are kept.
* ``<UNIN/>`` and ``<INAUDIBLE_SPEECH/>``: speech with no reference words. Each becomes a wildcard
  token that may absorb hypothesis words at zero cost (see ``asrshift.align``), so the model is
  neither rewarded nor penalised for what it writes over audio a human could not transcribe. How many
  words each wildcard absorbed is recorded, so an upper-bound score that caps absorption can be
  reported as a sensitivity analysis.
"""

from __future__ import annotations

import re
from functools import lru_cache

WILDCARD = "<*>"

_UNIN = re.compile(r"<\s*(?:UNIN|INAUDIBLE_SPEECH)\s*/?\s*>", re.IGNORECASE)
_OTHER_TAGS = re.compile(r"</?\s*[A-Za-z_]+\s*/?\s*>")
_ACRONYM = re.compile(r"\b((?:[A-Za-z]\.){2,})")
_OK = re.compile(r"\b(?:ok|okey)\b", re.IGNORECASE)
_ALRIGHT = re.compile(r"\balright\b", re.IGNORECASE)


def _variants(text: str) -> str:
    """Merge spellings that mean the same word. Applied to both sides, so it cannot hide an error."""
    text = _ACRONYM.sub(lambda m: m.group(1).replace(".", ""), text)
    text = _OK.sub("okay", text)
    return _ALRIGHT.sub("all right", text)


@lru_cache(maxsize=1)
def _normalizer():
    from whisper_normalizer.english import EnglishTextNormalizer

    return EnglishTextNormalizer()


# Dosage units are written both ways in references and hypotheses ("500 mg" / "500 milligrams").
# Mapping them to one form is symmetric, so it can only remove spurious errors.
_UNITS = {
    "milligram": "mg", "milligrams": "mg", "mgs": "mg",
    "microgram": "mcg", "micrograms": "mcg", "mcgs": "mcg",
    "milliliter": "ml", "milliliters": "ml", "millilitre": "ml", "millilitres": "ml", "mls": "ml",
    "kilogram": "kg", "kilograms": "kg", "kgs": "kg",
}


def normalize(text: str) -> str:
    """Whisper English normalisation of plain text (no tags), plus dosage-unit canonicalisation."""
    if not text:
        return ""
    words = _normalizer()(_variants(text)).split()
    return " ".join(_UNITS.get(w, w) for w in words)


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def strip_tags(text: str) -> str:
    """Remove all transcriber tags, keeping the words inside them. ``<UNIN/>`` is dropped."""
    return re.sub(r"\s+", " ", _OTHER_TAGS.sub(" ", _UNIN.sub(" ", text))).strip()


def reference_tokens(text: str) -> list[str]:
    """Normalised reference tokens, with ``WILDCARD`` wherever the reference had ``<UNIN/>``.

    Consecutive wildcards collapse into one; a wildcard absorbs any run of hypothesis words.
    """
    pieces = _UNIN.split(text)
    out: list[str] = []
    for i, piece in enumerate(pieces):
        if i > 0 and (not out or out[-1] != WILDCARD):
            out.append(WILDCARD)
        cleaned = _OTHER_TAGS.sub(" ", piece)
        if "<" in cleaned:
            raise ValueError(f"unhandled transcriber tag in reference: {cleaned!r}")
        out.extend(tokens(cleaned))
    return out


def has_unintelligible(text: str) -> bool:
    return bool(_UNIN.search(text))


def _protected(toks: list[str], max_n: int, max_removed: int) -> list[bool]:
    """Mark runs of repeats too long to be a speaker's restart (e.g. a Whisper repetition loop)."""
    prot = [False] * len(toks)
    for n in range(1, max_n + 1):
        i = 0
        while i + n <= len(toks):
            gram = toks[i:i + n]
            if WILDCARD in gram:
                i += 1
                continue
            k = 1
            while toks[i + k * n: i + (k + 1) * n] == gram:
                k += 1
            if k >= 2 and (k - 1) * n > max_removed:
                for j in range(i, i + k * n):
                    prot[j] = True
                i += k * n
            else:
                i += 1
    return prot


def collapse_repeats(toks: list[str], max_n: int = 4, max_removed: int = 6) -> tuple[list[str], list[int]]:
    """Drop immediate repetitions of 1..max_n tokens ("it's a bit it's a bit" -> "it's a bit").

    Returns the kept tokens and their indices in the input. Wildcards are never collapsed, and a
    repeat may not span a wildcard. Applied to reference and hypothesis alike, it removes the
    verbatim-transcript disfluencies (restarts, stutters) that a clean-text ASR model never writes.
    Runs whose repeats would remove more than ``max_removed`` tokens are left alone: those are not
    restarts but repetition loops (a known Whisper hallucination), and they must stay as errors.
    """
    prot = _protected(toks, max_n, max_removed)
    out: list[str] = []
    idx: list[int] = []
    for i, t in enumerate(toks):
        out.append(t)
        idx.append(i)
        changed = True
        while changed:
            changed = False
            for n in range(min(max_n, len(out) // 2), 0, -1):
                tail, prev = out[-n:], out[-2 * n:-n]
                if tail == prev and WILDCARD not in tail and not any(prot[j] for j in idx[-2 * n:]):
                    del out[-n:]
                    del idx[-n:]
                    changed = True
                    break
    return out, idx


def scoring_tokens(text: str, reference: bool = False) -> list[str]:
    """Tokens used for all scoring: normalised, units canonical, repeats collapsed."""
    toks = reference_tokens(text) if reference else tokens(strip_tags(text))
    return collapse_repeats(toks)[0]
