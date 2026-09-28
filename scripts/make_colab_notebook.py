"""Write notebooks/reproduce_colab.ipynb (kept as code so the notebook stays reviewable in diffs)."""

import json
from pathlib import Path

REPO = "https://github.com/neric-joel/DL-Project-"


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(keepends=True)}


cells = [
    md(f"""
# When ASR confidence does not travel: reproduce the results

This notebook reproduces the study in [{REPO}]({REPO}) on Google Colab.

* **Part A (about 2 minutes, CPU is fine):** rebuild every table and figure from the per-unit results
  committed in the repository.
* **Part B (about 1 hour on a T4 GPU):** run the whole pipeline from the raw audio: download both
  datasets, run Whisper-small on 11,213 units, score, evaluate, and draw the figures.

For Part B choose *Runtime → Change runtime type → T4 GPU* first.
"""),
    code(f"""
!git clone -q {REPO} asrshift
%cd asrshift
!pip install -q -e .
# make the package importable in this kernel without a restart (re-run this cell after a restart)
import sys; sys.path.insert(0, "/content/asrshift/src")
"""),
    md("""
## Part A: tables and figures from the committed per-unit results

Uses 500 bootstrap replicates to stay quick. Point estimates are identical to `results/`, and the
intervals and p-values differ slightly from the reported 2,000-replicate ones.
"""),
    code("""
!asrshift evaluate --n-boot 500
!asrshift figures
!asrshift report
"""),
    code("""
import pandas as pd
from IPython.display import Image, display

t = "results/tables/small"
asr = pd.read_csv(f"{t}/asr_summary.csv").set_index("set")
display(asr.loc[["eka_test", "pm_turn_test", "pm_window_test"],
                ["units", "hours", "wer", "err_rate", "number_acc", "negation_acc", "mean_conf"]].round(3))
cal = pd.read_csv(f"{t}/err/calibration.csv")
display(cal[cal.policy.isin(["P3", "P4"])][["eval", "policy", "base_rate", "mean_p", "ece", "brier", "citl"]].round(3))
ops = pd.read_csv(f"{t}/err/operating_points.csv")
display(ops[(ops.point == "budget_0.10") & ops.policy.isin(["P1", "P3", "P1-retuned", "P4"])]
        [["eval", "policy", "review_rate", "err_caught", "accepted_err_rate"]].round(3))
"""),
    code("""
for f in ["fig1_confidence_vs_error", "fig2_reliability", "fig3_risk_coverage", "fig4_threshold_transfer",
          "fig5_recalibration_data"]:
    display(Image(f"results/figures/{f}.png", width=900))
"""),
    md("""
## Part B: the full pipeline from raw audio (T4 GPU)

Downloads about 2.1 GB (Eka parquet shards from the Hugging Face Hub, PriMock57 audio from GitHub
LFS), builds the speaker-disjoint and consultation-level splits, runs faster-whisper, and rewrites
everything under `results/`. Each step is resumable if the runtime disconnects.
"""),
    code("""
!nvidia-smi --query-gpu=name,memory.total --format=csv
!asrshift download
!asrshift prepare
"""),
    code("""
# ~50 minutes on a T4, resumable. Part A's committed results are overwritten by the steps below.
!asrshift infer
"""),
    code("""
!asrshift score
!asrshift evaluate
!asrshift figures
!asrshift report
"""),
    md("""
## Part C (optional): a larger model

The proposal's stretch goal: does whisper-large-v3-turbo's confidence travel better? The repository
already contains this run (`results/SUMMARY_large-v3-turbo.md`, declared in protocol amendment 5). To
reproduce it from audio, allow about 2 to 3 hours on a T4. Each step is resumable.
"""),
    code("""
# pinned model revision and identical protocol, see configs/large-v3-turbo.yaml
!asrshift --config configs/large-v3-turbo.yaml infer
!asrshift --config configs/large-v3-turbo.yaml score
!asrshift --config configs/large-v3-turbo.yaml evaluate
!asrshift --config configs/large-v3-turbo.yaml report
"""),
    md("## One transcript, end to end"),
    code("""
import pandas as pd
from asrshift import audio, infer, prepare, review
from asrshift.config import load_config

cfg = load_config()
m = prepare.load()
row = m[(m.subset == "window") & (m.split == "test")].iloc[40].to_dict()
model, device, ctype = infer.load_model(cfg["asr"])
rec = infer.transcribe(model, audio.load_unit_audio(row), cfg["asr"]["decode"])
print("REFERENCE :", row["reference"][:400])
print("WHISPER   :", rec["text"][:400])
print(review.assess(rec, domain="window"))
"""),
]

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "reproduce_colab.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {out}")
