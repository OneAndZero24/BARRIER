"""Remain accuracy (TA) of airplane-unlearned DDPM runs, RN34 @ native 32x32.

Same metric as DDPM/pipeline.py::classifier_eval_remaining, but WITHOUT the
Resize((224,224)) upscaling -- class-conditional samples (500/class) are fed
to the ResNet-34 classifier at their native 32x32 resolution.

Also reports forget-side numbers for context: acc_forgotten = fraction of
class-<label_to_forget> samples still classified as the forgotten class.
"""

import argparse
import csv
import fnmatch
import os
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, transform=None):
        self.transform = transform
        path = pathlib.Path(img_folder)
        self.files = sorted(
            f for ext in IMAGE_EXTENSIONS for f in path.glob(f"*.{ext}")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        assert img.size == (32, 32), f"unexpected size {img.size} in {self.files[i]}"
        if self.transform:
            img = self.transform(img)
        return img


def get_loader(path, batch_size, workers, img_size=32):
    # img_size=32: no Resize, native 32x32 straight through the classifier.
    # img_size=224: same Resize((224, 224)) as pipeline.py / SEMU.
    resize = [] if img_size == 32 else [transforms.Resize((img_size, img_size))]
    transform = transforms.Compose(
        resize + [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    ds = ImagePathDataset(path, transform=transform)
    return ds, DataLoader(
        ds, batch_size=batch_size, num_workers=workers, pin_memory=torch.cuda.is_available()
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", type=str, required=True)
    p.add_argument("--classifier_ckpt", type=str, required=True)
    p.add_argument("--label_to_forget", type=int, default=0)
    p.add_argument("--n_classes", type=int, default=10)
    p.add_argument("-b", "--batch-size", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out_csv", type=str, default="remain_acc_32.csv")
    p.add_argument("--img_size", type=int, default=32, help="32 (native) or 224")
    p.add_argument(
        "--only",
        type=str,
        default=None,
        help="glob to select run dirs (e.g. '*lam5.0_s0_*')",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, args.n_classes)
    model.load_state_dict(torch.load(args.classifier_ckpt, map_location="cpu"))
    model = model.to(device).eval()

    run_dirs = sorted(
        d
        for d in pathlib.Path(args.runs_dir).iterdir()
        if d.is_dir()
        and (d / "class_samples").is_dir()
        and (args.only is None or fnmatch.fnmatch(d.name, args.only))
    )
    print(f"[eval_remain_32] device={device} runs_with_samples={len(run_dirs)}", flush=True)

    rows = []
    for rd in run_dirs:
        name = rd.name
        per_class = {}
        for cls_idx in range(args.n_classes):
            if cls_idx == args.label_to_forget:
                continue
            cls_dir = rd / "class_samples" / str(cls_idx)
            if not cls_dir.is_dir():
                continue
            ds, loader = get_loader(str(cls_dir), args.batch_size, args.workers, args.img_size)
            if len(ds) == 0:
                continue
            correct = total = 0
            with torch.no_grad():
                for data in loader:
                    preds = model(data.to(device, non_blocking=True)).argmax(dim=-1)
                    correct += (preds == cls_idx).sum().item()
                    total += data.size(0)
            per_class[cls_idx] = correct / max(total, 1)

        # forgotten-class side: fraction still predicted as the forgotten class
        fdir = rd / "class_samples" / str(args.label_to_forget)
        acc_forget = ""
        if fdir.is_dir():
            ds, loader = get_loader(str(fdir), args.batch_size, args.workers, args.img_size)
            if len(ds):
                hits = tot = 0
                with torch.no_grad():
                    for data in loader:
                        preds = model(data.to(device, non_blocking=True)).argmax(dim=-1)
                        hits += (preds == args.label_to_forget).sum().item()
                        tot += data.size(0)
                acc_forget = hits / max(tot, 1)

        ta = float(np.mean(list(per_class.values()))) if per_class else float("nan")
        row = {
            "run": name,
            "remain_acc_32": ta,
            "acc_forgotten_32": acc_forget if acc_forget != "" else float("nan"),
            **{f"acc_{k}": v for k, v in per_class.items()},
        }
        rows.append(row)
        print(f"[eval_remain_32] {name}  TA@{args.img_size}={ta:.4f}  acc_forgotten={acc_forget}", flush=True)

    fieldnames = ["run", "remain_acc_32", "acc_forgotten_32"] + [
        f"acc_{i}" for i in range(10) if i != args.label_to_forget
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[eval_remain_32] wrote {args.out_csv}", flush=True)

    # summary: mean TA per lambda (avg over seeds)
    try:
        import re

        by_lam = {}
        for r in rows:
            m = re.search(r"lam([0-9.]+)_s(\d+)_", r["run"])
            if m:
                by_lam.setdefault(m.group(1), []).append(r["remain_acc_32"])
        print(f"\n=== Remain accuracy @{args.img_size}x{args.img_size} (mean over seeds) ===")
        for lam in sorted(by_lam, key=float):
            v = np.array(by_lam[lam], dtype=float)
            print(f"lambda={lam:<6} TA@{args.img_size} = {v.mean():.4f} +/- {v.std():.4f}  (n={len(v)})")
    except Exception:
        pass


if __name__ == "__main__":
    main()
