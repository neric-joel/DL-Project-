# When ASR Confidence Does Not Travel

**Selective review of Whisper transcripts under clinical speech domain shift.**
CSE 598 *Operationalizing Deep Learning*, Arizona State University, Fall 2026.

> Status: pipeline and analysis plan are complete; the full results are being produced.
> The analysis plan in [docs/PROTOCOL.md](docs/PROTOCOL.md) was fixed before any target-domain
> transcript was scored.

## Question

Does an ASR review policy calibrated on short medical speech stay reliable when it is applied,
unchanged, to longer conversations between doctors and patients?

A review policy turns confidence signals that Whisper produces anyway into a risk score for each
transcript. The riskiest transcripts go to a human reviewer and the rest are accepted automatically.
We calibrate on the **Eka Medical ASR Evaluation Dataset** (short English medical clips) and test on
**PriMock57** (57 simulated primary-care consultations).

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-gpu.txt && pip install --no-deps -e .
asrshift all        # download -> prepare -> infer -> score -> evaluate -> figures
```

## Team

Neric Joel A · Amrish Sasikumar · Nirmalraju Kangeyan · Harshini Balamurugan Vadivazhagi ·
Ulagarchana Ulaga Narasimhan
