"""Make the pip-installed CUDA 12 libraries visible on Windows.

ctranslate2's Windows wheels load ``cublas64_12.dll`` lazily. When CUDA comes from the
``nvidia-cublas-cu12`` wheel (``pip install -e .[gpu]``), its ``bin`` folder has to be on the DLL
search path, and on ``PATH``, before the first GPU call. On Linux and in Colab this is a no-op.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_HANDLES: list = []


def nvidia_dll_dirs() -> list[Path]:
    dirs = []
    for base in map(Path, sys.path):
        root = base / "nvidia"
        if root.is_dir():
            dirs.extend(p for p in root.glob("*/bin") if p.is_dir())
    return sorted(set(dirs))


def setup_cuda_dlls() -> list[str]:
    if os.name != "nt":
        return []
    added = []
    for d in nvidia_dll_dirs():
        try:
            _HANDLES.append(os.add_dll_directory(str(d)))
        except OSError:
            continue
        if str(d) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
        added.append(str(d))
    return added
