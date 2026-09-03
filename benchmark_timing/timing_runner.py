#!/usr/bin/env python
"""
Per-method timing runner for the BARRIER unlearning benchmark.

Runs inside a SalUn-style harness (BARRIER/Classification OR gmum/semu
Classification - both derive from Unlearn-Saliency) and executes exactly the
unlearning code of that harness for a single method:

  --method salun   masked Random-Label unlearning (SalUn):
                   loader/model split identical to main_random.py, then
                   unlearn/RL.py for --unlearn_epochs epochs.

  --method semu    SEMU (unlearn/own_SVD.py): SVD transform of conv/linear
                   layers (own/transform_model.py), one train_iter epoch
                   per --unlearn_epochs.

No evaluation/MIA is run - this process only times the unlearning itself.
All epochs are three because the benchmark measures per-epoch walltime and
VRAM; the driver parses the printed markers.

Usage (from the harness dir, after `conda activate <env>`):
    python timing_runner.py --method semu --arch allcnn --dataset cifar10 \
        --data <data_dir> --model_path <ckpt.pth> --save_dir <tmp_out> \
        --class_to_replace 4 --batch_size 256 --seed 2 \
        --unlearn_epochs 3 --unlearn_lr 1e-5 --mask_path <salun_mask.pth> \
        --explained_variance_ratio 0.95 --use_projection_grad
"""

import argparse
import copy
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import arg_parser
import utils
import unlearn


def build_args() -> argparse.Namespace:
    args = arg_parser.parse_args()
    # sanity overrides for the timing benchmark
    args.workers = args.num_workers
    if not args.save_dir:
        args.save_dir = "./timing_out"
    os.makedirs(args.save_dir, exist_ok=True)
    return args


def split_loaders(args, marked_loader):
    """Identical to main_random.py/main_forget.py's forget/retain split."""
    def replace_loader_dataset(dataset, batch_size=args.batch_size, seed=1, shuffle=True):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=0,
            pin_memory=True, shuffle=shuffle,
        )

    seed = args.seed
    forget_dataset = copy.deepcopy(marked_loader.dataset)
    try:
        marked = forget_dataset.targets < 0
        forget_dataset.data = forget_dataset.data[marked]
        forget_dataset.targets = -forget_dataset.targets[marked] - 1
    except Exception:
        marked = forget_dataset.targets < 0
        forget_dataset.imgs = forget_dataset.imgs[marked]
        forget_dataset.targets = -forget_dataset.targets[marked] - 1
    forget_loader = replace_loader_dataset(forget_dataset, seed=seed, shuffle=True)

    retain_dataset = copy.deepcopy(marked_loader.dataset)
    try:
        marked = retain_dataset.targets >= 0
        retain_dataset.data = retain_dataset.data[marked]
        retain_dataset.targets = retain_dataset.targets[marked]
    except Exception:
        marked = retain_dataset.targets >= 0
        retain_dataset.imgs = retain_dataset.imgs[marked]
        retain_dataset.targets = retain_dataset.targets[marked]
    retain_loader = replace_loader_dataset(retain_dataset, seed=seed, shuffle=True)

    return retain_loader, forget_loader, retain_dataset


def load_pretrained(model, args):
    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=False)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif hasattr(ckpt, "state_dict"):
        ckpt = ckpt.state_dict()
    model.load_state_dict(ckpt, strict=False)
    model.cuda()


def run_salun(args):
    if not args.mask_path:
        raise SystemExit("salun needs --mask_path (from generate_mask.py)")

    torch.manual_seed(args.seed)
    (model, _, _, _, marked_loader) = utils.setup_model_dataset(args)
    load_pretrained(model, args)

    mask = torch.load(args.mask_path, map_location="cpu", weights_only=False)

    retain_loader, forget_loader, _ = split_loaders(args, marked_loader)
    loaders = OrderedDict(retain=retain_loader, forget=forget_loader, val=None, test=None)
    criterion = nn.CrossEntropyLoss()

    unlearn_fn = unlearn.get_unlearn_method("RL")

    for e in range(args.unlearn_epochs):
        t0 = time.time()
        unlearn_fn(loaders, model, criterion, args, mask)
        dt = time.time() - t0
        print(f"SALUN_EPOCH_SECONDS {dt:.4f}")
        sys.stdout.flush()


def run_semu(args):
    from unlearn.own.impl import set_requires_grad  # noqa: F401  (used by SEMU impl)
    from unlearn.own.transform_model import transform_model
    from unlearn.own_SVD import OwnSVD

    torch.manual_seed(args.seed)
    (model, _, _, _, marked_loader) = utils.setup_model_dataset(args)
    load_pretrained(model, args)

    retain_loader, forget_loader, _ = split_loaders(args, marked_loader)
    loaders = OrderedDict(retain=retain_loader, forget=forget_loader, val=None, test=None)
    criterion = nn.CrossEntropyLoss()

    # --- SEMU setup phase: SVD-based layer transform (not part of an epoch) ---
    t0 = time.time()
    transform_model(
        model,
        forget_loader,
        criterion,
        ["linear", "conv2d"],
        getattr(args, "explained_variance_ratio", None),
        use_projection_grad=getattr(args, "use_projection_grad", False),
    )
    set_requires_grad(model, changed_layers_class=["customlinear", "customconv2d"])
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"SEMU_TRANSFORM_SECONDS {time.time() - t0:.4f}")
    sys.stdout.flush()

    optimizer = torch.optim.SGD(
        params, lr=args.unlearn_lr, momentum=args.momentum, weight_decay=args.weight_decay,
    )

    for e in range(args.unlearn_epochs):
        t0 = time.time()
        OwnSVD.train_iter(loaders, model, criterion, optimizer, e, args)
        dt = time.time() - t0
        print(f"SEMU_EPOCH_SECONDS {dt:.4f}")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser("timing_runner")
    parser.add_argument("--method", required=True, choices=["salun", "semu"])
    args, _ = parser.parse_known_args()
    harness_args = build_args()

    seed = getattr(harness_args, "seed", 2)
    utils.setup_seed(seed)
    np.random.seed(seed)

    if args.method == "salun":
        run_salun(harness_args)
    else:
        run_semu(harness_args)


if __name__ == "__main__":
    main()