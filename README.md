# When ASR Confidence Does Not Travel

Speech recognition (Whisper) is used to draft doctors' notes, and a person checks only the transcripts
the system is unsure about. We asked whether that "unsure" signal still works when you move from short
medical clips to real doctor-patient conversations, and then built a system that cuts how much a person
has to listen.

Course project for CSE 598 *Operationalizing Deep Learning: A Sociotechnical Perspective*, Arizona State
University, Fall 2026.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neric-joel/DL-Project-/blob/main/notebooks/reproduce_colab.ipynb)

## The setup

- **Model:** Whisper small, unchanged, on 11,195 recordings.
- **Tuned on:** Eka, short dictated medical clips (about 5 words each).
- **Tested on:** PriMock57, 57 doctor-patient consultations, cut into 30-second pieces (about 67 words
  and 5 speaker turns each).
- **Review rule:** score how risky each transcript is from Whisper's own confidence, send the riskiest to
  a person, accept the rest.

## What we found

1. **Whisper sounds more sure on conversations, but is not more right.** About half of the transcripts
   need a correction in both datasets, yet Whisper's confidence is higher on the conversations.
2. **A threshold tuned on clips breaks on conversations.** A setting that sends 10% of clips to a person
   sends only 0.3% of conversation pieces, and half of the pieces it accepts contain errors.
3. **The cause is long, overlapping speech, not single words.** Word by word, Whisper is, if anything,
   more right than it thinks. The trouble comes from long pieces where people talk over each other.
4. **Retuning on a few conversations fixes the numbers, at a cost.** After retuning on 10 consultations
   the promises hold again, but 96% of conversation pieces then go to a person.
5. **Serious words slip through.** A missed "not" or a wrong dose is not what a rule based on overall
   error rate looks for: an average consultation still has about 8 such errors that nobody checks.
6. **A bigger Whisper (large-v3-turbo) makes fewer mistakes**, but its thresholds still do not carry over.

Exact numbers and tests: [results/SUMMARY.md](results/SUMMARY.md) (Whisper small) and
[results/SUMMARY_large-v3-turbo.md](results/SUMMARY_large-v3-turbo.md).

**Advice for a team deploying a medical scribe:** do not reuse a review threshold from dictation on
conversations, watch for drift, retune on a few labelled consultations from your own clinic, and check
numbers, doses and negations separately.

## What we built next (exploratory)

These came after the main study ([protocol, amendment 7](docs/PROTOCOL.md)). Nothing was trained on the
test recordings, but the versions we kept were picked after comparing on them, so treat the numbers as
slightly optimistic.

- **A mistake spotter (the verifier).** Instead of whole transcripts, it flags single words that are
  probably wrong and places where words were probably dropped, so a person listens to short clips.
  Serious words (numbers, doses, "not") count most.
- **A fine-tuned Whisper.** Whisper large-v3-turbo fine-tuned with LoRA on 8 training consultations
  *and* the Eka training clips. Fine-tuning on conversations alone made it worse on Eka; adding both
  fixed that.

| 47 test consultations, 30-second pieces | Where we started (Whisper small) | Our system |
|---|---:|---:|
| Words wrong, per 100 | 11.6 | **7.5** |
| Words skipped, per 100 | 6.5 | **4.0** |
| Serious mistakes left after a person listens for 30 s (per consultation) | 5.2 | **4.0** |
| Serious mistakes caught by checking the riskiest 10% of words | 77% | **86%** |
| Words wrong per 100 on Eka clips | 13.7 | **8.3** |

Details, and the ideas that did not work: [experiments/lora_pilot/README.md](experiments/lora_pilot/README.md).

## The app: Second Listen

```bash
pip install -r requirements-app.txt
uvicorn app.server:app --port 8765        # then open http://127.0.0.1:8765
```

- **Watch:** a consultation plays with Whisper (transcribing live) and our system side by side, with
  the words to double-check marked.
- **Review:** go through the flagged words one by one, hear each clip, confirm or fix it, and see your
  score.
- **Our results** (`/results.html`): the overall numbers in plain words and charts.

Playing audio needs the downloaded data (step 2 below); everything else works from `results/`.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install --no-deps -e .
asrshift evaluate && asrshift figures && asrshift report      # 1. from the saved results, minutes, no GPU
asrshift all                                                  # 2. everything from raw audio, ~1 h on a GPU
pytest                                                        # tests
```

Also runs on [Google Colab](notebooks/reproduce_colab.ipynb) and in Docker (`docker compose run --rm
reproduce`). Route 1 gives the exact committed numbers; route 2 can differ slightly on other GPUs without
changing any conclusion.

## What is where

```
src/asrshift/            the pipeline: download, prepare, transcribe, score, evaluate, verifier
app/                     the Second Listen app
experiments/lora_pilot/  fine-tuning (exploratory)
results/                 all results, tables and figures
docs/PROTOCOL.md         every analysis choice, written before the results, plus amendments
docs/DATA.md             datasets and splits
configs/default.yaml     every setting behind a reported number
```

## Limitations

- PriMock57 consultations are role-played, not real patients.
- Pieces are cut at human-marked speaker boundaries, which real systems would do less cleanly, so real
  error rates are probably a bit higher.
- The same 7 clinicians appear in the tuning and test consultations.
- A scoring fix ("Oh" read as the number 0) came after the first results; both versions are reported
  (protocol, amendment 6).

## Team

| Member | Role |
|---|---|
| Neric Joel A | Coordination; Eka preparation, splits and entity scoring |
| Amrish Sasikumar | Whisper inference and confidence signals |
| Nirmalraju Kangeyan | PriMock57 preparation; WER scoring; medical-term matching |
| Harshini Balamurugan Vadivazhagi | Calibration models and the review policies; verifier, fine-tuning and the Second Listen app |
| Ulagarchana Ulaga Narasimhan | Evaluation metrics, reliability diagrams, review budgets; figures and repository |

## Data and licences

Code: MIT ([LICENSE](LICENSE)). The datasets are not included and keep their own licences: Eka Medical
ASR Evaluation Dataset (MIT, © Eka Care) and PriMock57 (CC BY 4.0, Papadopoulos Korfiatis et al., 2022).
Whisper (Radford et al., 2023) runs through [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
