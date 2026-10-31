"""LoRA fine-tuning of Whisper (whisper-small by default) on the PriMock57 recalibration consultations.
Runs on CPU (the pilot) or on a CUDA GPU (fp16 autocast).

    python experiments/lora_pilot/train.py --measure 3          # time a few steps, then stop
    python experiments/lora_pilot/train.py --max-steps 600      # train, keeping the best adapter

Training units: turns (own channel) and windows (mixed) of the recalibration consultations, at most
30 s long. The last two recalibration consultations in configs/default.yaml are the dev set for
early stopping. No test consultation is read.

--eka adds the Eka calibration units to training and a sample of Eka validation units to the dev set
(needs `asrshift download` and `asrshift prepare`). --holdout leaves PriMock57 consultations out of
training, for out-of-fold transcripts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from asrshift.audio import load_unit_audio
from asrshift.config import load_config
from asrshift.paths import RESULTS_DIR
from asrshift.textnorm import strip_tags

OUT = Path(__file__).parent / "out"
BASE = "openai/whisper-small"
SEED = 598


def reference_text(ref: str) -> str:
    return " ".join(strip_tags(str(ref)).split())


def units(consultations: list[str]) -> pd.DataFrame:
    u = pd.read_csv(RESULTS_DIR / "units" / "units_small.csv.gz", low_memory=False)
    u = u[u.consultation.isin(consultations) & (u.duration <= 30.0)].copy()
    role = np.where(u.subset == "window", "mix", u.subset.str.replace("turn_", "", regex=False))
    u["audio_source"] = u.consultation + ":" + role
    u["text"] = u.reference.map(reference_text)
    return u[u.text.str.len() > 0].reset_index(drop=True)


def eka_units(split: str) -> pd.DataFrame:
    from asrshift import prepare
    m = prepare.load()
    u = m[(m.dataset == "eka") & (m.split == split) & (m.dropped_reason.fillna("") == "") & (m.duration <= 30.0)].copy()
    u["text"] = u.reference.map(reference_text)
    return u[u.text.str.len() > 0].reset_index(drop=True)


def load_audio(df: pd.DataFrame) -> list[np.ndarray]:
    return [load_unit_audio(r) for r in df.to_dict("records")]


def batches(n: int, size: int, rng: random.Random, shuffle: bool):
    idx = list(range(n))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, n, size):
        yield idx[i:i + size]


def collate(proc: WhisperProcessor, audio: list[np.ndarray], texts: list[str], start_id: int):
    feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    ids = [proc.tokenizer(t).input_ids for t in texts]
    L = max(map(len, ids))
    labels = torch.full((len(ids), L), -100, dtype=torch.long)
    for i, x in enumerate(ids):
        labels[i, :len(x)] = torch.tensor(x)
    if (labels[:, 0] == start_id).all():  # the model prepends the start token itself
        labels = labels[:, 1:]
    return feats, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimiser step (saves memory)")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--dev-n", type=int, default=64)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--measure", type=int, default=0, help="time this many steps and exit")
    ap.add_argument("--save-every", type=int, default=10, help="steps between resumable checkpoints")
    ap.add_argument("--seed", type=int, default=SEED, help="training seed (data order, LoRA init, dropout)")
    ap.add_argument("--base", default=BASE, help="Hugging Face model to adapt")
    ap.add_argument("--out", default=str(OUT), help="output folder (adapters, state, log)")
    ap.add_argument("--decoder-only", action="store_true",
                    help="LoRA on the decoder only: no backward pass through the encoder (much cheaper for turbo)")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (less memory, slower)")
    ap.add_argument("--eka", action="store_true", help="also train on Eka calibration (dev: + Eka validation sample)")
    ap.add_argument("--holdout", default="", help="comma-separated PriMock57 consultations left out of training")
    ap.add_argument("--resume", action="store_true",
                    help="continue from out/state.pt (saved every --save-every steps and at each dev evaluation)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(torch.get_num_threads())
    recal = load_config()["primock57"]["recalibration_consultations"]
    held = [c for c in args.holdout.split(",") if c]
    if set(held) - set(recal[:-2]):
        raise SystemExit(f"--holdout must name training consultations, got {held}")
    dev_c, train_c = recal[-2:], [c for c in recal[:-2] if c not in held]
    tr, dv = units(train_c), units(dev_c)
    dv = dv.sample(n=min(args.dev_n, len(dv)), random_state=SEED).reset_index(drop=True)
    cols = ["dataset", "audio_source", "start", "end", "duration", "text"]
    if args.eka:
        ev = eka_units("validation")
        tr = pd.concat([tr[cols], eka_units("calibration")[cols]], ignore_index=True)
        dv = pd.concat([dv[cols], ev.sample(n=min(args.dev_n, len(ev)), random_state=SEED)[cols]], ignore_index=True)
    print(f"train {len(tr)} units ({tr.duration.sum() / 3600:.2f} h) from {len(train_c)} consultations"
          f"{' + Eka calibration' if args.eka else ''}; dev {len(dv)} units from {dev_c}"
          f"{' + Eka validation' if args.eka else ''}; held out {held or 'none'}", flush=True)
    tr_audio, dv_audio = load_audio(tr), load_audio(dv)

    out = Path(args.out)
    proc = WhisperProcessor.from_pretrained(args.base, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.base)
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    start_id = model.config.decoder_start_token_id
    if args.grad_ckpt:
        # non-reentrant, so LoRA weights get gradients although input_features never require grad
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # a regex limits the adapters to decoder layers; the plain list matches every attention projection
    targets = (r".*decoder\.layers\.\d+\.(self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)"
               if args.decoder_only else ["q_proj", "k_proj", "v_proj", "out_proj"])
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=targets))
    model.print_trainable_parameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # fp16 autocast on a GPU (LoRA weights stay fp32); plain fp32 on CPU, as in the pilot
    amp = dict(device_type=device, dtype=torch.float16, enabled=device == "cuda")
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    warm = 20
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.05, 1 - s / max(args.max_steps, 1)))

    def dev_loss() -> float:
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for b in batches(len(dv), args.batch, rng, False):
                x, y = collate(proc, [dv_audio[i] for i in b], [dv.text[i] for i in b], start_id)
                with torch.autocast(**amp):
                    tot += model(input_features=x.to(device), labels=y.to(device)).loss.item() * len(b)
        model.train()
        return tot / len(dv)

    out.mkdir(parents=True, exist_ok=True)
    log = []
    best, bad, step, t0 = math.inf, 0, 0, time.time()
    state_file = out / "state.pt"
    if args.resume and state_file.exists():
        # trainable LoRA weights, optimiser and schedule from the last dev evaluation; the data order
        # is reshuffled, so a resumed run is not bit-identical to an uninterrupted one
        st = torch.load(state_file, weights_only=False, map_location=device)
        model.load_state_dict(st["lora"], strict=False)
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        step, best, bad, log = st["step"], st["best"], st["bad"], st["log"]
        rng.setstate(st["rng"])
        print(f"resumed at step {step}, best dev loss {best:.4f}", flush=True)
    elif not args.measure:
        best = dev_loss()
        log.append(dict(step=0, dev_loss=best, elapsed_s=round(time.time() - t0, 1)))
        print(f"step 0 dev loss {best:.4f}", flush=True)
        model.save_pretrained(out / "adapter_best")

    def save_state():
        # written to a temporary file and renamed, so a run killed mid-save keeps the previous checkpoint
        tmp = state_file.with_suffix(".tmp")
        torch.save(dict(lora={k: v for k, v in model.state_dict().items() if "lora_" in k},
                        opt=opt.state_dict(), sched=sched.state_dict(), step=step, best=best, bad=bad,
                        log=log, rng=rng.getstate()), tmp)
        os.replace(tmp, state_file)

    model.train()
    done, micro, ts = False, 0, time.time()
    while not done:
        for b in batches(len(tr), args.batch, rng, True):
            x, y = collate(proc, [tr_audio[i] for i in b], [tr.text[i] for i in b], start_id)
            with torch.autocast(**amp):
                loss = model(input_features=x.to(device), labels=y.to(device)).loss
            scaler.scale(loss / args.accum).backward()
            micro += 1
            if micro % args.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
            step += 1
            if step % 10 == 0 or args.measure:
                print(f"step {step} loss {loss.item():.4f} ({(time.time() - ts) / (1 if args.measure else 10):.1f} s/step)", flush=True)
                ts = time.time()
            if args.measure and step >= args.measure:
                print(f"measured {(time.time() - t0) / step:.1f} s/step incl. data", flush=True)
                return
            if step % args.save_every == 0 and step % args.eval_every and step < args.max_steps:
                save_state()   # between dev evaluations, so a killed run loses at most save_every steps
            if step % args.eval_every == 0 or step >= args.max_steps:
                dl = dev_loss()
                log.append(dict(step=step, train_loss=loss.item(), dev_loss=dl, elapsed_s=round(time.time() - t0, 1)))
                print(f"step {step} dev loss {dl:.4f} (best {best:.4f}) elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
                if dl < best:
                    best, bad = dl, 0
                    model.save_pretrained(out / "adapter_best")
                else:
                    bad += 1
                model.save_pretrained(out / "adapter_last")
                json.dump(log, open(out / "train_log.json", "w"), indent=1)
                save_state()
                if bad >= args.patience:
                    print("dev loss stopped improving", flush=True)
                    done = True
                    break
            if step >= args.max_steps:
                done = True
                break
    json.dump(log, open(out / "train_log.json", "w"), indent=1)
    print(f"done: {step} steps, best dev loss {best:.4f}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
