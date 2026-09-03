#!/usr/bin/env python
"""Patch a SalUn-style harness (BARRIER/Classification or gmum/semu
Classification, both derived from Unlearn-Saliency) so that the shared
ESC-keys-compatible AllCNN is registered as arch "allcnn".

Idempotent: copies allcnn_esc.py into <repo>/models/ and appends the two
registration lines to <repo>/models/__init__.py if missing.

Usage:
    python patch_models_init.py <repo>/Classification/models
"""

import shutil
import sys
from pathlib import Path

PATCH_LINES = (
    "from .allcnn_esc import AllCNN\n"
    'model_dict["allcnn"] = AllCNN\n'
)


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)

    models_dir = Path(sys.argv[1]).resolve()
    if not models_dir.is_dir():
        print(f"[error] {models_dir} is not a directory")
        sys.exit(1)

    # 1. copy the AllCNN module next to the other models
    src = Path(__file__).resolve().parent / "allcnn_esc.py"
    dst = models_dir / "allcnn_esc.py"
    shutil.copyfile(src, dst)
    print(f"[patch] copied {src} -> {dst}")

    # 2. register "allcnn" in the model dict (idempotent)
    init = models_dir / "__init__.py"
    text = init.read_text()
    if "allcnn" not in text:
        with init.open("a") as f:
            f.write("\n" + PATCH_LINES)
        print(f"[patch] registered 'allcnn' in {init}")
    else:
        print(f"[patch] {init} already contains 'allcnn', nothing to do")

    # verification (text-level; importing the harness needs torch/torchvision)
    if 'model_dict["allcnn"] = AllCNN' in init.read_text():
        print(f"[patch] OK: 'allcnn' registered in {init}")
    else:
        print("[patch] WARNING: registration lines missing after patch")


if __name__ == "__main__":
    main()