"""Confidence features from the decoder output of one unit.

Every feature is computed from what faster-whisper returns (segments and words), never from the
reference, and the same way in both domains. When the decoder returns nothing (faster-whisper drops
a segment whose no-speech probability is above 0.6 while its log-probability is below -1), the unit
gets the least confident value of every feature and ``empty_hyp = 1``.
"""

from __future__ import annotations

import math

import numpy as np

FEATURES = [
    "mean_logprob",       # token-weighted mean of segment average log-probability
    "min_seg_logprob",    # worst segment
    "word_p_mean",
    "word_p_min",
    "word_p_q10",
    "word_frac_low",      # share of words with probability < 0.5
    "nsp_max",            # largest no-speech probability
    "cr_max",             # largest gzip compression ratio (repetition)
    "rep3",               # share of repeated word trigrams
    "fallback",           # temperature fallback was used
    "log_words",
    "log_duration",
    "words_per_s",
    "n_segments",
    "empty_hyp",
]
# The subset used by the "confidence only" variant of the calibrated policy.
CONFIDENCE_FEATURES = [
    "mean_logprob", "min_seg_logprob", "word_p_mean", "word_p_min", "word_p_q10", "word_frac_low",
    "nsp_max", "cr_max", "rep3", "fallback", "empty_hyp",
]

_EMPTY = {
    "mean_logprob": -3.0, "min_seg_logprob": -3.0, "word_p_mean": 0.0, "word_p_min": 0.0,
    "word_p_q10": 0.0, "word_frac_low": 1.0, "nsp_max": 1.0, "cr_max": 0.0, "rep3": 0.0,
    "fallback": 1.0, "log_words": 0.0, "n_segments": 0.0, "words_per_s": 0.0, "empty_hyp": 1.0,
}


def _rep3(words: list[str]) -> float:
    w = [x.strip().lower() for x in words if x.strip()]
    if len(w) < 3:
        return 0.0
    grams = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 1.0 - len(set(grams)) / len(grams)


def unit_features(rec: dict) -> dict:
    segs = rec.get("segments") or []
    words = rec.get("words") or []
    duration = max(float(rec.get("audio_s") or 0.0), 0.05)
    out = {"log_duration": math.log(duration)}
    if not segs or not rec.get("text", "").strip():
        out.update(_EMPTY)
        out["conf"] = 0.0
        return out

    n_tok = np.array([max(s["n_tokens"], 1) for s in segs], dtype=float)
    lp = np.array([s["avg_logprob"] for s in segs], dtype=float)
    mean_lp = float((lp * n_tok).sum() / n_tok.sum())
    probs = np.array([w["probability"] for w in words], dtype=float) if words else np.array([math.exp(mean_lp)])
    out.update({
        "mean_logprob": mean_lp,
        "min_seg_logprob": float(lp.min()),
        "word_p_mean": float(probs.mean()),
        "word_p_min": float(probs.min()),
        "word_p_q10": float(np.quantile(probs, 0.10)),
        "word_frac_low": float((probs < 0.5).mean()),
        "nsp_max": float(max(s["no_speech_prob"] for s in segs)),
        "cr_max": float(max(s["compression_ratio"] for s in segs)),
        "rep3": _rep3([w["word"] for w in words]),
        "fallback": float(max(s["temperature"] for s in segs) > 0),
        "log_words": math.log1p(len(words)),
        "words_per_s": len(words) / duration,
        "n_segments": float(len(segs)),
        "empty_hyp": 0.0,
        "conf": math.exp(mean_lp),
    })
    return out
