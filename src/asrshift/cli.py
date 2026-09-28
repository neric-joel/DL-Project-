"""Command-line entry point: ``asrshift <step>`` or ``python -m asrshift <step>``.

Steps, in order: download -> prepare -> infer -> score -> evaluate -> figures.
``asrshift all`` runs them in sequence; each step is resumable.
"""

from __future__ import annotations

import argparse
import sys

from asrshift.config import load_config


def _prepare(cfg, args):
    from asrshift import prepare

    df = prepare.build(cfg)
    print(prepare.summary(df).to_string())


def _infer(cfg, args):
    from asrshift import infer, prepare

    m = prepare.load()
    m = m[m["dropped_reason"].fillna("") == ""]
    if args.dataset:
        m = m[m["dataset"] == args.dataset]
    if args.subset:
        m = m[m["subset"].isin(args.subset)]
    if args.model:  # ad-hoc model; for a reported run use a config file with a pinned revision
        cfg["asr"]["model"] = args.model
        cfg["asr"]["model_repo"] = args.model
        cfg["asr"]["model_revision"] = None
    out = infer.run(m, cfg, limit=args.limit, device=args.device, compute_type=args.compute_type, shard=args.shard)
    print(f"wrote {out}")


def _score(cfg, args):
    from asrshift import score

    df = score.run(cfg, model=args.model)
    print(score.overview(df).to_string())


def _evaluate(cfg, args):
    from asrshift import evaluate

    evaluate.run(cfg, model=args.model, n_boot=args.n_boot)


def _figures(cfg, args):
    from asrshift import plots

    plots.run(cfg, model=args.model)


def _terms(cfg, args):
    from asrshift import terms

    lex = terms.build(seed=cfg["seed"])
    print(f"{len(lex)} lexicon terms occur in PriMock57; lexicon and audit sample in {terms.TERMS_DIR}")


def _report(cfg, args):
    from asrshift import report

    report.run(cfg, model=args.model)


def _download(cfg, args):
    from asrshift import download

    download.main(["--dataset", args.dataset or "all"] + (["--no-audio"] if args.no_audio else []))


STEPS = {"download": _download, "prepare": _prepare, "infer": _infer, "score": _score,
         "evaluate": _evaluate, "figures": _figures, "report": _report, "terms": _terms}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="asrshift", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=[*STEPS, "all"])
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--dataset", choices=["eka", "primock57"], default=None)
    p.add_argument("--subset", nargs="*", default=None, help="restrict inference to these subsets")
    p.add_argument("--model", default=None, help="Whisper model for infer/score/evaluate (default: config)")
    p.add_argument("--limit", type=int, default=None, help="infer: stop after this many new units (smoke test)")
    p.add_argument("--device", default=None, help="infer: cuda or cpu (default: config)")
    p.add_argument("--compute-type", default=None, help="infer: e.g. float16, int8_float16, int8")
    p.add_argument("--shard", default=None,
                   help="infer: write to a separate output file, so two processes can split the work")
    p.add_argument("--no-audio", action="store_true", help="download: PriMock57 transcripts only")
    p.add_argument("--n-boot", type=int, default=None, help="evaluate: bootstrap replicates (default: config)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    steps = ["download", "prepare", "infer", "score", "evaluate", "figures", "report"] if args.step == "all" else [args.step]
    for step in steps:
        print(f"== {step}", file=sys.stderr)
        STEPS[step](cfg, args)


if __name__ == "__main__":
    main()
