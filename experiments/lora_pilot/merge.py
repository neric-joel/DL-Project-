"""Merge the best LoRA adapter into its base model (whisper-small by default) and convert it to CTranslate2 int8 for faster-whisper.

    python experiments/lora_pilot/merge.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ctranslate2.converters import TransformersConverter
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizerFast

OUT = Path(__file__).parent / "out"
BASE = "openai/whisper-small"


class Converter(TransformersConverter):
    # ctranslate2 4.8 passes from_pretrained(dtype=...), which transformers 4.46 (pinned for peft)
    # does not accept; it still calls the argument torch_dtype
    def load_model(self, model_class, model_name_or_path, **kwargs):
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return model_class.from_pretrained(model_name_or_path, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--quantization", default="int8", help="int8 for CPU, float16 for GPU")
    args = ap.parse_args()
    out, base_name = Path(args.out), args.base
    merged_dir, ct2_dir = out / "merged", out / f"ct2_{args.quantization}"
    base = WhisperForConditionalGeneration.from_pretrained(base_name)
    model = PeftModel.from_pretrained(base, out / "adapter_best").merge_and_unload()
    model.save_pretrained(merged_dir)
    WhisperProcessor.from_pretrained(base_name, language="en", task="transcribe").save_pretrained(merged_dir)
    # tokenizer.json is what faster-whisper reads; the slow tokenizer files alone do not carry it
    WhisperTokenizerFast.from_pretrained(base_name).backend_tokenizer.save(str(merged_dir / "tokenizer.json"))
    Converter(str(merged_dir), copy_files=["tokenizer.json", "preprocessor_config.json"]).convert(
        str(ct2_dir), quantization=args.quantization, force=True)
    print("wrote", ct2_dir)


if __name__ == "__main__":
    main()
