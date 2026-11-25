"""Print the headline numbers of the final system from results/tables/turbo-lora-mixed/overview.json.

usage (repo root): python scripts/key_results.py
The JSON is committed; `python experiments/lora_pilot/analysis/overview.py` rebuilds it first.
"""

from __future__ import annotations

import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "results" / "tables" / "turbo-lora-mixed" / "overview.json"
MODELS = ("start", "turbo", "final")   # whisper-small, off-the-shelf large-v3-turbo, our fine-tuned turbo


def main() -> None:
    d = json.loads(PATH.read_text(encoding="utf-8"))
    t = {m: d[m]["transcripts"] for m in MODELS}
    det = {m: d[m]["detection"]["pm_window_test"] for m in ("start", "final")}
    w = d["listening"]["window"]
    i = w["x_seconds"].index(30)

    rows = [  # (label, test set, key, decimals)
        ("WER %, PriMock57 windows (47 test consultations)", "pm_window_test", "wer", 2),
        ("Words skipped per 100, PriMock57 windows", "pm_window_test", "deletions", 2),
        ("WER %, Eka test", "eka_test", "wer", 2),
        ("Critical error rate %, Eka test", "eka_test", "critical_error_rate", 1),
    ]
    print(f"{'':52s}{'small':>8s}{'turbo':>8s}{'ours':>8s}")
    for label, s, key, dp in rows:
        print(f"{label:52s}" + "".join(f"{t[m][s][key]:8.{dp}f}" for m in MODELS))
    print()
    print("Critical wrong words in the transcript caught by flagging the riskiest 10% of words (windows):")
    print(f"  small + its verifier {det['start']['words']['top10']['critical_recall']:.1f}%, "
          f"ours {det['final']['words']['top10']['critical_recall']:.1f}% "
          f"(ours ranked by the model's own word probability instead: "
          f"{det['final']['whisper_confidence']['top10']['critical_recall']:.1f}%)")
    print("Critical errors left per consultation after 30 s of listening (windows, estimated):")
    print(f"  small + its verifier {w['critical']['start_flags'][i]:.1f}, ours {w['critical']['final_flags'][i]:.1f}")


if __name__ == "__main__":
    main()
