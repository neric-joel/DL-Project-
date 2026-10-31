# LoRA fine-tuning: whisper-small pilot (CPU) and large-v3-turbo (Kaggle GPU)

Exploratory. Question: does LoRA adaptation on the 10 PriMock57 recalibration consultations lower WER
and clinically weighted WER on held-out test consultations? No test consultation is used in training.

## Setup

```bash
pip install -r experiments/lora_pilot/requirements.txt      # CPU torch, transformers, peft, accelerate
```

Audio needed in `data/raw/primock57/audio/`: the 10 recalibration consultations and the 12 test
consultations listed in `decode.py` (the first ten test consultations alphabetically, plus
day2_consultation03 and day3_consultation06).

* **Train:** `train.py`. whisper-small (HF), LoRA r=16, alpha=32, dropout 0.05 on q/k/v/out_proj in
  encoder and decoder (3.5 M trainable parameters, 1.4%). fp32 on CPU. The data is turns (own channel)
  and windows (mixed) of 8 recalibration consultations, at most 30 s long: 1,055 units, 2.06 h. The dev
  set is 64 units from the last two recalibration consultations (day5_consultation04 and
  day5_consultation07). AdamW lr 5e-4 with 20 warm-up steps, then linear decay. Micro-batches of 2 with
  2-step accumulation, so the effective batch is 4. Dev loss every 25 steps; the best adapter goes to
  `out/adapter_best`, and resumable state to `out/state.pt` every 10 steps.
* **Merge and convert:** `merge.py` merges the adapter and converts to CTranslate2 int8.
* **Decode and score:** `decode.py` uses faster-whisper int8 on CPU with the study's decoding settings,
  for both the base model and the LoRA model. WER comes from the study's normaliser and wildcard
  alignment. Weighted WER and critical errors use the verifier's weights (10/5/1/0.2). The paired
  bootstrap resamples consultations 1000 times.

```bash
python experiments/lora_pilot/train.py --batch 2 --accum 2 --max-steps 130 --eval-every 25 --patience 3 --resume
python experiments/lora_pilot/merge.py
python experiments/lora_pilot/decode.py --model small --name base
python experiments/lora_pilot/decode.py --model experiments/lora_pilot/out/ct2_int8 --name lora
python experiments/lora_pilot/decode.py --compare base lora
```

## Result (2026-10-29)

130 steps (about 0.5 epoch), 4 resumed runs (Claude Code stopped earlier runs when the laptop ran short
of memory; `--resume` picked up from the last checkpoint each time). Dev loss: 1.174 before training,
0.607 at step 25, 0.533 at 50, 0.529 at 75, 0.513 at 100, **0.502 at 125** (the adapter used), 0.502 at
130.

Pooled over the 277 test windows of 12 consultations (18,702 reference words), both models int8 on CPU:

| Metric | whisper-small | + LoRA | Difference (95% CI, consultation bootstrap) |
|---|---|---|---|
| WER | 15.36% | **11.60%** | −3.76 pts [−4.55, −2.88] |
| Weighted WER (10/5/1/0.2) | 16.98% | **13.99%** | −2.98 pts [−4.14, −1.82] |
| Substitutions | 5.27% | 5.17% | |
| Deletions | 8.58% | **4.74%** | |
| Insertions | 1.51% | 1.69% | |
| Critical errors (of 596) | 136 | **114** | |

WER fell in all 12 consultations and weighted WER in 11 (day1_consultation01 was flat, 17.75% →
17.84%). Almost all of the gain is fewer deletions: the adapted model stops dropping overlapping and
quiet speech, the main error source in the windows. Substitutions barely move, and insertions rise
slightly.

Caveats. This is a pilot: 12 of the 47 test consultations, one seed, and no tuning beyond the fixed
recipe above. The training consultations come from the same recording setup as the test set, which is
the recalibration scenario the study allows, not a new domain.

On the same 277 windows the committed large-v3-turbo transcripts score 13.39% WER and 13.92% weighted
WER: adapted small has lower WER (−1.79 pts, CI [−2.65, −0.95]) and ties on weighted WER (+0.07, CI
[−1.23, +1.39]). It drops far fewer words than turbo (4.74% vs 8.82%) but substitutes more (5.17% vs
3.36%) and invents more numbers and negations (2.19 vs 1.28 per 1,000 words, mostly a spurious "not" or
"no"). Semantic distance (all-MiniLM-L6-v2): base 0.092, LoRA 0.065, turbo 0.083.

## large-v3-turbo with LoRA on a Kaggle T4 (2026-10-30)

Same recipe on whisper-large-v3-turbo, trained on a Kaggle T4 with fp16 autocast and gradient
checkpointing (`train.py --base openai/whisper-large-v3-turbo --batch 4 --accum 2 --grad-ckpt`), merged
to CTranslate2 float16 and decoded through `asrshift.infer` with the study's settings. The Kaggle jobs
are written by `kaggle/build_kernel*.py` (each embeds train/merge/decode, clones this branch and runs
`asrshift download` / `prepare`); the scoring scripts are in `analysis/`. Still no test consultation in
training. All numbers are on all 47 PriMock57 test consultations unless stated, with consultation (Eka:
group) bootstrap CIs.

**PriMock57-only adapter** (8 training consultations; dev loss 0.40, 32 min on the T4):

| Windows, 47 consultations | turbo | turbo + LoRA | Difference (95% CI) |
|---|---|---|---|
| WER | 9.94% | **7.23%** | −2.71 pts [−3.21, −2.28] |
| Weighted WER | 10.94% | **8.67%** | −2.27 pts |
| Critical error rate | 14.7% | **11.5%** | |
| Number / negation recall | 86.7 / 88.8% | **90.7 / 92.8%** | |

Three seeds (598 / 599 / 600) give WER 7.23 / 7.29 / 7.51% and weighted WER 8.67 / 8.88 / 9.09%; every
seed's CI against turbo excludes 0 for WER, weighted WER, critical errors and negation recall. Turns
improve too (WER 8.94 → 6.49%). **But Eka gets worse** (WER 10.10 → 10.84%, critical error rate 9.3 →
12.5%, CIs exclude 0): the adapter specialises to PriMock57's audio.

**Mixed adapter: PriMock57 training consultations + Eka calibration** (`train.py --eka`; 6.0 h of audio,
dev set adds 64 Eka validation units, dev loss 0.31). This is the model we keep:

| | turbo | PriMock-only LoRA | **Mixed LoRA** |
|---|---|---|---|
| Eka test WER / weighted / critical | 10.1 / 10.0 / 9.3% | 10.8 / 11.6 / 12.5% | **8.3 / 7.6 / 4.9%** |
| PriMock windows WER / weighted / critical | 9.9 / 10.9 / 14.7% | **7.2 / 8.7 / 11.5%** | 7.5 / 9.2 / 12.2% |
| PriMock turns WER / weighted / critical | 8.9 / 10.4 / 13.4% | **6.5 / 8.8 / 11.7%** | 6.9 / 9.1 / 11.8% |
| Eka number F1 / negation F1 | 93.5 / 97.4 | 91.2 / 94.4 | **96.5 / 99.1** |

The mixed adapter fixes Eka (better than turbo on every Eka metric, CIs exclude 0) for a small PriMock57
cost against the PriMock-only adapter (+0.3 pts WER on windows), and still beats turbo everywhere.
Word-probability AUROC is unchanged (0.82-0.86), so the model's confidence still ranks its errors.

**Verifier.** The saved small verifier works on the fine-tuned transcripts. Retraining it on the
fine-tuned model's own errors did not help, twice (PriMock-only model, and out-of-fold transcripts from 4
models that each left 2 training consultations out, `train.py --holdout`): too little data, about 2,000
units. Pooling both, small's training transcripts plus the mixed model's held-out ones (5,782 units),
does help and is never worse: missing-word AUPRC 0.161 → 0.181 on windows and 0.168 → 0.213 on turns;
turns, 30 s of listening: critical errors left 3.47 → 3.02 per consultation (CI excludes 0).

**End to end**, errors left per consultation after 30 s of listening (start: whisper-small + verifier;
now: mixed turbo LoRA + pooled verifier): windows 129 → 94 weighted, 5.1 → 4.0 critical; turns 123 → 79
weighted, 4.5 → 2.8 critical (hybrid policy).

**Did not help (kept for the record):** combining the LoRA model with turbo word by word (worse than the
LoRA model alone); boosting critical words where the model and turbo disagree (about a third of the
critical errors are disagreements; the rest are errors both models share, and the verifier already uses
the disagreement); the audio-energy verifier features (`verifier.audio_features`, off by default).

**Files for the app.** `export_app.py` writes the mixed model's transcripts and the pooled verifier in the
committed-results format (`results/units/*_turbo-lora-mixed*`, `results/models/verifier_turbo-lora-mixed.*`,
`results/tables/turbo-lora-mixed/verifier/curves.json`); the 8 PriMock57 training consultations come from
the out-of-fold models, so no unit is scored by a model that trained on it (Eka calibration, the
mixed model's training data, is not decoded). The adapters and decoder records stay outside git
(adapter about 26 MB; records about 25 MB).

```bash
python experiments/lora_pilot/kaggle/build_kernel3.py .     # writes kaggle/turbo_lora_mixed and turbo_lora_folds
kaggle kernels push -p experiments/lora_pilot/kaggle/turbo_lora_mixed --accelerator NvidiaTeslaT4
python experiments/lora_pilot/analysis/turbo_final2.py PM_ONLY.jsonl MIXED.jsonl FOLDS.jsonl out_mixed
python experiments/lora_pilot/export_app.py MIXED_OOF.jsonl
```
