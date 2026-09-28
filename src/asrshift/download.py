"""Download the two datasets at pinned revisions.

Eka Medical ASR Evaluation Dataset (MIT) comes from the Hugging Face Hub as parquet shards.
PriMock57 (CC BY 4.0) comes from GitHub; its audio is stored in Git LFS and is fetched through
the media endpoint, so git-lfs does not need to be installed.

Downloads are resumable: a file that already exists with the expected size is skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

from asrshift.paths import RAW_DIR

EKA_REPO = "ekacare/eka-medical-asr-evaluation-dataset"
EKA_REVISION = "433e66037c8aebb8f396b938624a5ab6afc4ca36"
EKA_EN_SHARDS = [f"en/test-{i:05d}.parquet" for i in range(8)]

PRIMOCK_REPO = "babylonhealth/primock57"
PRIMOCK_REVISION = "cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5"

CHUNK = 1 << 20


def _get(url: str, dest: Path, retries: int = 5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                # With Content-Encoding (gzip) the header is the compressed size, so skip the check.
                encoded = r.headers.get("Content-Encoding", "identity") != "identity"
                expected = 0 if encoded else int(r.headers.get("Content-Length", 0))
                if dest.exists() and expected and dest.stat().st_size == expected:
                    return
                with open(tmp, "wb") as f:
                    for block in r.iter_content(CHUNK):
                        f.write(block)
            if expected and tmp.stat().st_size != expected:
                raise OSError(f"size mismatch for {url}: {tmp.stat().st_size} != {expected}")
            tmp.replace(dest)
            return
        except (requests.RequestException, OSError) as exc:
            if attempt == retries:
                raise
            wait = 2 ** attempt
            print(f"  retry {attempt}/{retries} for {dest.name} in {wait}s ({exc})", file=sys.stderr)
            time.sleep(wait)


def download_eka(root: Path) -> None:
    out = root / "eka"
    base = f"https://huggingface.co/datasets/{EKA_REPO}/resolve/{EKA_REVISION}"
    for shard in EKA_EN_SHARDS:
        dest = out / shard
        if dest.exists() and dest.stat().st_size > 0:
            print(f"eka: {shard} present")
            continue
        print(f"eka: downloading {shard}")
        _get(f"{base}/{shard}", dest)


def _primock_listing(folder: str) -> list[str]:
    url = f"https://api.github.com/repos/{PRIMOCK_REPO}/contents/{folder}?ref={PRIMOCK_REVISION}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return [item["name"] for item in r.json() if item["type"] == "file"]


def download_primock(root: Path, with_audio: bool = True) -> None:
    out = root / "primock57"
    raw = f"https://raw.githubusercontent.com/{PRIMOCK_REPO}/{PRIMOCK_REVISION}"
    media = f"https://media.githubusercontent.com/media/{PRIMOCK_REPO}/{PRIMOCK_REVISION}"
    for folder in ("transcripts", "notes"):
        for name in _primock_listing(folder):
            dest = out / folder / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            _get(f"{raw}/{folder}/{name}", dest)
        print(f"primock57: {folder} done")
    if not with_audio:
        return
    wavs = [n for n in _primock_listing("audio") if n.endswith(".wav")]
    for i, name in enumerate(sorted(wavs), 1):
        dest = out / "audio" / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            continue
        print(f"primock57: audio {i}/{len(wavs)} {name}")
        _get(f"{media}/audio/{name}", dest)
    print("primock57: audio done")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["eka", "primock57", "all"], default="all")
    p.add_argument("--no-audio", action="store_true", help="PriMock57 transcripts and notes only")
    p.add_argument("--root", type=Path, default=RAW_DIR)
    args = p.parse_args(argv)
    if args.dataset in ("eka", "all"):
        download_eka(args.root)
    if args.dataset in ("primock57", "all"):
        download_primock(args.root, with_audio=not args.no_audio)


if __name__ == "__main__":
    main()
