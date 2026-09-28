# When ASR Confidence Does Not Travel

**Selective review of Whisper transcripts under clinical speech domain shift.**
Course project for CSE 598 *Operationalizing Deep Learning: A Sociotechnical Perspective*,
Arizona State University, Fall 2026.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neric-joel/DL-Project-/blob/main/notebooks/reproduce_colab.ipynb)

<!-- RESULTS -->

## The question

Speech recognition is now used to draft clinical notes. The dangerous errors are the confident ones
(a dose, a drug name, a missed "not"), because nobody checks a transcript the system is sure about.
A **review policy** turns confidence signals that Whisper produces anyway into a risk score for each
transcript, sends the riskiest ones to a human, and accepts the rest automatically.

> Does a review policy calibrated on short medical speech stay reliable when it is applied, unchanged,
> to longer conversations between doctors and patients?

We calibrate on the **Eka Medical ASR Evaluation Dataset** (short English medical clips, many Indian
drug brand names) and deploy on **PriMock57** (57 simulated primary-care consultations, doctor and
patient). Whisper's weights are never changed.

## How it works

```
 audio ──► faster-whisper (small) ──► transcript + decoder statistics
                                           │
                     ┌─────────────────────┴──────────────────────┐
                     ▼                                             ▼
        confidence features (15)                     scoring against the human transcript
        log-prob, word probabilities,                WER, S/D/I, dose & negation tokens,
        no-speech, repetition, length ...            Eka entities  ──► label: WER > 10%
                     │                                             │
                     ▼                                             │
        review policy ──► risk ──► accept / send to human ◄────────┘  (evaluation only)
```

| Policy | Risk score | Fitted on | Thresholds tuned on |
|---|---|---|---|
| **P1** raw confidence | 1 − exp(mean token log-probability) | – | Eka validation |
| **P2** Whisper's rules | no-speech probability, compression ratio, repeated trigrams | – | Eka validation |
| **P3** Eka-calibrated, frozen | logistic regression on 15 decoder features | Eka calibration | Eka validation |
| **P4** recalibrated | P3 with a two-parameter (Platt) refit | 10 PriMock57 consultations | same 10 consultations |

Each policy is run at three kinds of operating point: a **review budget** (send the top 10% to a human),
an **empirical risk target** (keep the error rate among accepted transcripts at or below 20%, tuned on
labelled data), and a **plug-in rule** (the same target, computed from the model's own probabilities on
unlabelled incoming transcripts). The last one works only if the probabilities are calibrated, which is
exactly what domain shift breaks.

**Target units.** PriMock57 is cut two ways. *Windows* mix both channels into stretches of up to 30 s
cut only between utterances (4.8 speaker turns on average): the unit a chunked clinical scribe would
transcribe. *Turns* are single utterances from one speaker's channel, a control that is conversational
but short and single-speaker, like the Eka clips.

The full design, every definition, and the two amendments recorded before any PriMock57 transcript
was scored are in **[docs/PROTOCOL.md](docs/PROTOCOL.md)**. Data handling is in
[docs/DATA.md](docs/DATA.md).

## Reproducing the results

Tested on Windows 11 (RTX 3050 Laptop 4 GB) and Linux (Colab T4) with Python 3.12.

**1. From the committed per-unit results (minutes, no GPU, no downloads).**
Every table, figure and `results/SUMMARY.md` is rebuilt from `results/units/`.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps -e .
asrshift evaluate && asrshift figures && asrshift report
```

**2. The whole pipeline from raw audio (about 1 hour on a 4 GB GPU).**

```bash
pip install -r requirements-gpu.txt   # CUDA 12 cuBLAS for ctranslate2 (not needed on Colab)
asrshift download     # Eka parquet shards (HF Hub) + PriMock57 audio (GitHub LFS), ~2.1 GB, pinned revisions
asrshift prepare      # units, speaker-disjoint Eka split, consultation split
asrshift infer        # whisper-small on 11,213 units; resumable
asrshift score        # WER, critical tokens, entities, labels, features
asrshift evaluate     # policies, operating points, bootstrap, hypothesis tests (~15 min on CPU)
asrshift figures
asrshift report
```

`asrshift all` runs the same sequence. On a machine without a GPU, `asrshift infer --device cpu
--compute-type int8` works but takes several hours.

**3. Google Colab.** [notebooks/reproduce_colab.ipynb](notebooks/reproduce_colab.ipynb) runs route 1 in
a couple of minutes, and route 2 on a free T4.

**4. Docker (CPU).**

```bash
docker compose run --rm reproduce     # tests + tables + figures + summary from committed results
docker compose run --rm pipeline      # full pipeline on CPU (slow)
```

**Tests:** `pytest` (unit tests plus a synthetic end-to-end run of the evaluation; tests marked `data`
also check the real splits for leakage when the data is present).

## Demo: Transcript Review Console

A local web app over the results: pick the audio the system sees (Eka clips, PriMock57 turns or
windows), a policy and an operating point, and see how many transcripts it sends to review, how many
wrong ones it lets through, and why. Each transcript shows the audio, what Whisper wrote (shaded by word
confidence), a colour-coded diff against the human transcript, and the risk under every policy.
*Run Whisper on this audio now* re-decodes the clip live and scores it with the exported policies.

```bash
pip install -r requirements-app.txt
uvicorn app.server:app --port 8765     # then open http://127.0.0.1:8765
```

Playback and live decoding need the downloaded data; everything else works from `results/` alone.

## Repository layout

```
configs/default.yaml     every setting that shapes a reported number
src/asrshift/            download, prepare, infer, score, features, policies, metrics, evaluate, plots, report
app/                     Transcript Review Console (FastAPI + static page)
notebooks/               Colab notebook (generated by scripts/make_colab_notebook.py)
results/units/           per-unit and per-word scores (inputs to every table)
results/tables/small/    all result tables (CSV)
results/figures/         all figures
results/SUMMARY.md       key numbers, generated from the tables
results/splits/          unit-to-split assignments
docs/PROTOCOL.md         study protocol, deviations from the proposal, amendments
docs/DATA.md             datasets, units, splits, transcriber tags
tests/                   pytest suite
```

## Limitations

* **One model.** Only whisper-small was run. Larger Whisper models may be better calibrated.
* **Simulated consultations.** PriMock57 is role-played by clinicians and staff, recorded over video
  calls, and transcribed verbatim. It is not real clinical data, and its speakers, accents and recording
  channel differ from Eka's in several ways at once. The shift we measure is a realistic *bundle*, not one
  isolated factor. The turn-level control separates length and multi-speaker audio from the rest.
* **Oracle segmentation.** Windows are cut at the human-annotated utterance boundaries, which acts as a
  perfect voice-activity detector. A deployed system would cut less cleanly, so the target-domain error
  is probably understated.
* **Shared clinicians.** The 57 consultations were run by 7 clinicians, so the recalibration and test
  consultations share clinicians. Recalibration here means *same clinic*, not *new clinicians*.
* **Label depends on length.** "WER above 10%" means *any error* for a 3-word clip and *7 or more errors*
  for a 66-word window. The length-stratified analysis, the length-matched contrast and the null
  baseline in the protocol check how much of the effect this explains.
* **Medical terms on PriMock57** are matched with a lexicon extracted by a small local LLM from the human
  transcripts. It is exploratory and awaits a manual audit (`results/terms/audit_sample.csv`).

## Team

| Member | Responsibilities (from the proposal) |
|---|---|
| Neric Joel A | Project coordination; Eka preparation, session split and entity scoring |
| Amrish Sasikumar | Whisper inference with faster-whisper; confidence signals per segment |
| Nirmalraju Kangeyan | PriMock57 preparation and consultation split; WER, insertion and deletion scoring; lexicon matching and audit |
| Harshini Balamurugan Vadivazhagi | Calibration models and the four review policies, including PriMock57 recalibration |
| Ulagarchana Ulaga Narasimhan | Evaluation metrics, reliability diagrams, review budgets and risk–coverage results; figures and repository |

## Data and licences

Code: MIT (see [LICENSE](LICENSE)). The datasets are not included and keep their own licences:
Eka Medical ASR Evaluation Dataset (MIT, © Eka Care) and PriMock57 (CC BY 4.0, Papadopoulos Korfiatis
et al., 2022, *PriMock57: A Dataset of Primary Care Mock Consultations*, ACL). Whisper (Radford et al.,
2023) is run through [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
