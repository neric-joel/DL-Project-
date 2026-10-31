"""Decode test windows with faster-whisper (base small or the LoRA model) and score them.

    python experiments/lora_pilot/decode.py --model small --name base
    python experiments/lora_pilot/decode.py --model experiments/lora_pilot/out/ct2_int8 --name lora
    python experiments/lora_pilot/decode.py --compare base lora

Both run int8 on CPU with the study's decoding settings (configs/default.yaml asr.decode), so the
comparison is like for like. Scoring uses the study's normaliser and wildcard alignment for WER, and
the verifier's clinical weights for weighted WER and critical errors.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from asrshift.audio import primock_audio
from asrshift.config import load_config
from asrshift.paths import RESULTS_DIR

OUT = Path(__file__).parent / "out"
# The first ten test consultations alphabetically, plus the two other consultations already on disk.
TEST = ["day1_consultation01", "day1_consultation03", "day1_consultation04", "day1_consultation05",
        "day1_consultation06", "day1_consultation07", "day1_consultation08", "day1_consultation09",
        "day1_consultation10", "day1_consultation11", "day2_consultation03", "day3_consultation06"]


def test_windows() -> pd.DataFrame:
    u = pd.read_csv(RESULTS_DIR / "units" / "units_small.csv.gz", low_memory=False)
    w = u[(u.subset == "window") & (u.split == "test") & u.consultation.isin(TEST)]
    return w.sort_values(["consultation", "start"]).reset_index(drop=True)


def decode(model_path: str, name: str) -> None:
    from faster_whisper import WhisperModel

    dec = dict(load_config()["asr"]["decode"])
    dec["temperature"] = tuple(dec["temperature"])
    model = WhisperModel(model_path, device="cpu", compute_type="int8")
    w = test_windows()
    rows, t0 = [], time.time()
    for k, r in enumerate(w.itertuples()):
        segs, _ = model.transcribe(primock_audio(f"{r.consultation}:mix", r.start, r.end), **dec)
        rows.append(dict(unit_id=r.unit_id, consultation=r.consultation, hypothesis=" ".join(s.text.strip() for s in segs)))
        if k % 25 == 0:
            print(f"{name}: {k}/{len(w)} windows, {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / f"hyp_{name}.csv", index=False)
    print(f"{name}: {len(w)} windows in {time.time() - t0:.0f} s", flush=True)


def score(name: str) -> pd.DataFrame:
    from asrshift.align import align
    from asrshift.textnorm import scoring_tokens
    from asrshift.verifier import Weights, categories, labels

    w8 = Weights.from_config(load_config())
    ref = test_windows().set_index("unit_id").reference
    h = pd.read_csv(OUT / f"hyp_{name}.csv").fillna("")
    rows = []
    for r in h.itertuples():
        rt, ht = scoring_tokens(ref[r.unit_id], reference=True), scoring_tokens(r.hypothesis)
        a = align(rt, ht)
        lab = labels(rt, ht, w8)
        rc = categories(rt, w8)
        rows.append(dict(unit_id=r.unit_id, consultation=r.consultation, n_ref=a.n_ref, errors=a.errors,
                         sub=a.substitutions, dele=a.deletions, ins=a.insertions,
                         w_err=float(lab["word_b"].sum() + lab["gap_b"].sum()), w_ref=lab["ref_weight"],
                         crit_err=int(lab["word_cr"].sum() + lab["gap_cr"].sum()),
                         crit_ref=sum(c == "crit" and s != "W" for c, s in zip(rc, a.ref_status))))
    return pd.DataFrame(rows)


def compare(a: str, b: str, n_boot: int = 1000, seed: int = 598) -> None:
    A, B = score(a), score(b)
    cols = ["n_ref", "errors", "sub", "dele", "ins", "w_err", "w_ref", "crit_err", "crit_ref"]
    ca, cb = A.groupby("consultation")[cols].sum(), B.groupby("consultation")[cols].sum()
    per = pd.DataFrame({
        "windows": A.groupby("consultation").size(),
        f"wer_{a}": ca.errors / ca.n_ref, f"wer_{b}": cb.errors / cb.n_ref,
        f"wwer_{a}": ca.w_err / ca.w_ref, f"wwer_{b}": cb.w_err / cb.w_ref,
        f"crit_{a}": ca.crit_err, f"crit_{b}": cb.crit_err,
    })
    print(per.to_string(float_format=lambda x: f"{x:.4f}"))
    per.to_csv(OUT / f"per_consultation_{a}_vs_{b}.csv")

    def pooled(c):
        return dict(wer=c.errors.sum() / c.n_ref.sum(), wwer=c.w_err.sum() / c.w_ref.sum(),
                    sub=c["sub"].sum() / c.n_ref.sum(), dele=c.dele.sum() / c.n_ref.sum(), ins=c.ins.sum() / c.n_ref.sum(),
                    crit_err=int(c.crit_err.sum()), crit_rate=c.crit_err.sum() / max(c.crit_ref.sum(), 1))

    pa, pb = pooled(ca), pooled(cb)
    rng = np.random.default_rng(seed)
    idx = np.array(ca.index)
    d_wer, d_wwer = [], []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        xa, xb = ca.loc[s], cb.loc[s]
        d_wer.append(xb.errors.sum() / xb.n_ref.sum() - xa.errors.sum() / xa.n_ref.sum())
        d_wwer.append(xb.w_err.sum() / xb.w_ref.sum() - xa.w_err.sum() / xa.w_ref.sum())
    q = lambda v: [round(float(np.quantile(v, t)), 4) for t in (0.025, 0.975)]
    print(f"\npooled over {len(idx)} consultations, {int(ca.n_ref.sum())} reference words")
    for k in ("wer", "wwer", "sub", "dele", "ins", "crit_rate", "crit_err"):
        print(f"  {k:<10}{a}: {pa[k]:.4f}   {b}: {pb[k]:.4f}")
    print(f"  WER difference {b} - {a}: {pb['wer'] - pa['wer']:+.4f}  95% CI {q(d_wer)}")
    print(f"  weighted WER difference {b} - {a}: {pb['wwer'] - pa['wwer']:+.4f}  95% CI {q(d_wwer)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--name")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        decode(args.model, args.name)


if __name__ == "__main__":
    main()
