"""Shared loader: committed-style unit table and per-token probabilities for the LoRA decode records."""
import json

import numpy as np
import pandas as pd

from asrshift import score, terms, verifier


def load_lora(jsonl: str):
    recs = {}
    for line in open(jsonl, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[r["unit_id"]] = r
    small = score.load("small")
    rows, probs, lex = [], {}, terms.load()
    for row in small[small.unit_id.isin(recs)].to_dict("records"):
        out, words = score.score_unit(row, recs[row["unit_id"]], lex)
        rows.append({**row, **out})
        probs[row["unit_id"]] = np.array([p for _, p, _, _ in words], float)
    return pd.DataFrame(rows), probs, recs


def install(u_lora: pd.DataFrame, probs: dict, second: str = "large-v3-turbo"):
    """Make verifier.unit_tables("lora", second, ...) read the LoRA records."""
    real = verifier._committed
    _, _, other_hyp, other_probs = real("small", second)

    def patched(model, sec):
        if model == "lora":
            return u_lora, probs, other_hyp, other_probs
        return real(model, sec)

    verifier._committed = patched
    return other_hyp, other_probs
