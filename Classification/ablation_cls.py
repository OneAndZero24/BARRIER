#!/usr/bin/env python3
"""
Headless ResNet-18 (CIFAR-10) runner for the BARRIER mechanism + design-choice
ablations (backbone B of the writeup).

Settings (paper protocol):
    classwise : forget one class (airplane = 0), final FC target layer
    random    : forget a random 10% (4500 samples), final FC target layer
k = 32, SGD lr = 1e-3, 10 epochs, RL objective, no remain-set loss.

The flags mirror DDPM/ablation_regions.py, so Experiments 3-9 run identically
on both backbones.  No wandb.  One row per run goes to
<results_dir>/ablations.csv; Experiments 1-2 are analysed offline by
ablation_mechanism.py from the saved pca_info.pth + run_meta.json artifacts.

Usage:
    python ablation_cls.py --config configs/pipeline_classwise.yaml \
        --experiment exp3 --lambda 10 --uniform_margin --seed 0
    python ablation_cls.py --summarize --results_dir <results_dir>
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ablation_common import torch_load  # noqa: E402
import torch.nn as nn
import yaml

_CLS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_CLS_DIR, "..", ".."))  # repo root (InTAct, ablation_common)
sys.path.insert(0, _CLS_DIR)  # Classification last -> highest priority

from InTAct.intact import (  # noqa: E402
    UnlearnIntervalProtection,
    classification_forward_fn,
    make_region_boxes,
    percentile_alpha_to_quantiles,
)
from pipeline import build_args, build_data_loaders  # noqa: E402
from ablation_common import (  # noqa: E402
    FIXED_LAMBDA,
    aggregate_rows,
    safe_run_suffix,
    summarize_ablations,
    write_sidecar,
)

import utils  # noqa: E402
import evaluation  # noqa: E402
from trainer import validate  # noqa: E402

log = logging.getLogger(__name__)

REGION_MODES = ("two_corner", "two_random", "boxes_4", "boxes_8", "slabs_2k", "env_box")
ABLATION_INTERVAL_MODES = ("full", "width_only", "centre_only", "off")
LAMBDA_SWEEP = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)

ANALYTIC_MATVECS = {
    "two_corner": 4,
    "two_random": 4,
    "boxes_4": 8,
    "boxes_8": 16,
    "slabs_2k": lambda k: 8 * k,
    "env_box": 2,
}
INTERVAL_MATVECS = {"full": 1.0, "width_only": 0.5, "centre_only": 0.5, "off": 0.0}


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def microbench_protection(protection, model, device, warmup=20, iters=200):
    cuda = device.type == "cuda"
    for _ in range(warmup):
        protection.compute_protection_loss(model, device)
    if cuda:
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        protection.compute_protection_loss(model, device)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times = np.asarray(times)
    return float(times.mean()), float(times.std())


def evaluate_all_headless(model, data_loaders, args, device):
    """UA / RA / TA via the shared trainer.validate; MIA via SVC (SalUn
    convention).  Returns a flat metrics dict."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    metrics = {}
    for name, loader in data_loaders.items():
        if loader is None:
            continue
        utils.dataset_convert_to_test(loader.dataset, args)
        acc = validate(loader, model, criterion, args)
        metrics[f"acc/{name}"] = acc
        log.info(f"  {name} accuracy: {acc:.2f}%")

    metrics["UA"] = 100.0 - metrics.get("acc/forget", float("nan"))
    metrics["RA"] = metrics.get("acc/retain", float("nan"))
    metrics["TA"] = metrics.get("acc/test", float("nan"))

    if not args.skip_mia:
        try:
            test_loader = data_loaders["test"]
            test_len = len(test_loader.dataset)

            utils.dataset_convert_to_test(data_loaders["retain"].dataset, args)
            shadow_train = torch.utils.data.Subset(
                data_loaders["retain"].dataset, list(range(test_len)))
            shadow_train_loader = torch.utils.data.DataLoader(
                shadow_train, batch_size=args.batch_size, shuffle=False)
            utils.dataset_convert_to_test(data_loaders["forget"].dataset, args)
            utils.dataset_convert_to_test(test_loader.dataset, args)

            mia = evaluation.SVC_MIA(
                shadow_train=shadow_train_loader,
                shadow_test=test_loader,
                target_train=None,
                target_test=data_loaders["forget"],
                model=model,
            )
            metrics["MIA"] = mia.get("confidence", float("nan")) * 100.0
            log.info(f"  MIA: {metrics['MIA']:.4f}")
        except Exception as exc:
            log.warning(f"SVC_MIA failed ({exc}); MIA=NaN")
            metrics["MIA"] = float("nan")
    return metrics


def _ident(args):
    """Collision-proof run identity (all flags that change loss/metrics)."""
    return {
        "backbone": "resnet18",
        "setting": args.setting,
        "experiment": args.experiment,
        "region_mode": args.region_mode,
        "interval_mode": args.interval_mode,
        "alpha": args.alpha,
        "sign_flip_frac": args.sign_flip_frac,
        "include_db": int(args.include_db),
        "uniform_margin": int(args.uniform_margin),
        "include_mean": int(not args.no_include_mean),
        "include_res": int(not args.no_include_res),
        "lambda": args.lam,
        "seed": args.seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region_mode", choices=REGION_MODES, default="two_corner")
    parser.add_argument("--interval_mode", choices=ABLATION_INTERVAL_MODES,
                        default="full")
    parser.add_argument("--alpha", type=int, choices=[1, 5, 10], default=5)
    parser.add_argument("--sign_flip_frac", type=float, default=0.0)
    parser.add_argument("--include_db", action="store_true")
    parser.add_argument("--uniform_margin", action="store_true")
    parser.add_argument("--no_include_mean", action="store_true")
    parser.add_argument("--no_include_res", action="store_true")
    parser.add_argument("--experiment", default="exp5", choices=[
        "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp9", "grid"])
    parser.add_argument("--lambda", dest="lam", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config",
                        default="configs/pipeline_classwise.yaml")
    parser.add_argument("--results_dir", default="/shared/results/common/miksa/intact/Cls/r")
    parser.add_argument("--microbench_warmup", type=int, default=20)
    parser.add_argument("--microbench_iters", type=int, default=200)
    parser.add_argument("--skip_mia", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.summarize:
        aggregate_rows(args.results_dir)
        summarize_ablations(args.results_dir)
        return

    t0_run = time.time()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    utils.setup_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = load_config(args.config)
    setting = cfg["pipeline"]["setting"]
    args.setting = "classwise" if setting == "classifier_classwise" else "random"
    if args.lam is None:
        args.lam = FIXED_LAMBDA.get(("resnet18", "classwise" if setting == "classifier_classwise" else "random"))
    args.lower_p, args.upper_p = percentile_alpha_to_quantiles(args.alpha)

    cfg["pipeline"]["seed"] = seed
    cfg["unlearn"]["method"] = "intact"
    cfg["intact"]["lambda_interval"] = args.lam
    cfg["intact"]["region_mode"] = args.region_mode
    cfg["intact"]["normalize_region"] = True
    cfg["intact"]["reduced_dim"] = 32
    cfg["intact"]["use_actual_bounds"] = True
    cfg["intact"]["interval_mode"] = args.interval_mode
    cfg["intact"]["lower_percentile"] = args.lower_p
    cfg["intact"]["upper_percentile"] = args.upper_p
    cfg["intact"]["include_db"] = args.include_db
    cfg["intact"]["include_mean"] = not args.no_include_mean
    cfg["intact"]["include_res"] = not args.no_include_res
    cfg["intact"]["sign_flip_frac"] = args.sign_flip_frac
    cfg["intact"]["sign_flip_seed"] = seed
    cfg["intact"]["uniform_margin"] = args.uniform_margin

    # ---- model + data (same as pipeline.main, headless) --------------------
    fargs = build_args(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.results_dir, exist_ok=True)

    model, _, val_loader, test_loader, marked_loader = utils.setup_model_dataset(fargs)
    model = model.to(device)

    ckpt = torch_load(fargs.model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    log.info(f"Loaded pretrained model from {fargs.model_path}")

    global forget_loader, retain_loader
    data_loaders, forget_dataset, retain_dataset = build_data_loaders(
        fargs, marked_loader, val_loader, test_loader)
    forget_loader = data_loaders["forget"]
    retain_loader = data_loaders["retain"]

    # ---- protection setup (timed) + microbenchmark -------------------------
    protection = UnlearnIntervalProtection(
        targets=["fc"],                      # paper protocol: final FC only
        lambda_interval=args.lam,
        lower_percentile=args.lower_p,
        upper_percentile=args.upper_p,
        reduced_dim=32,
        infinity_scale=cfg["intact"].get("infinity_scale", 20.0),
        use_actual_bounds=True,
        normalize_protection=True,
        region_mode=args.region_mode,
        normalize_region=True,
        region_random_seed=0,
        interval_mode=args.interval_mode,
        include_db=args.include_db,
        include_mean=not args.no_include_mean,
        include_res=not args.no_include_res,
        sign_flip_frac=args.sign_flip_frac,
        sign_flip_seed=seed,
        uniform_margin=args.uniform_margin,
    )
    t0 = time.time()
    protection.setup_protection(
        model, forget_loader, device,
        remain_dataloader=retain_loader,
        forward_fn=classification_forward_fn,
    )
    setup_wall = time.time() - t0
    pl_ms, pl_ms_std = microbench_protection(
        protection, model, device,
        warmup=args.microbench_warmup, iters=args.microbench_iters,
    )
    log.info(f"setup {setup_wall:.2f}s  protection microbench "
             f"{pl_ms:.3f} +- {pl_ms_std:.3f} ms/call")

    protection.freeze_non_target_params(model)
    trainable = protection.get_trainable_params(model)
    tparams = sum(p.numel() for p in trainable)

    criterion = nn.CrossEntropyLoss()
    n_classes = cfg["data"].get("num_classes", 10)
    base_method = cfg["intact"].get("base_method", "rl")
    n_epochs = cfg["unlearn"].get("unlearn_epochs", 10)
    optimizer = torch.optim.SGD(
        trainable, lr=1e-3, momentum=0.9, weight_decay=5e-4)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.train()
    t0 = time.time()
    for epoch in range(n_epochs):
        for x, y in forget_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            if base_method == "rl":
                rand_t = torch.randint(0, n_classes, y.shape, device=device)
                base_loss = criterion(outputs, rand_t)
            else:
                base_loss = -criterion(outputs, y)
            protect_loss = protection.compute_protection_loss(model, device)
            loss = base_loss + protect_loss
            loss.backward()
            optimizer.step()
        log.info(f"epoch {epoch}: base={base_loss.item():.4f} "
                 f"protect={protect_loss.item():.4f}")
    train_wall = time.time() - t0
    peak_mem = (torch.cuda.max_memory_allocated() / 1e6
                if torch.cuda.is_available() else 0.0)
    model.eval()
    log.info(f"setup {setup_wall:.2f}s  train ({n_epochs} epochs) "
             f"{train_wall:.2f}s  peak {peak_mem:.0f}MB")

    # ---- run metadata + artifacts for Experiments 1-2 ----------------------
    run_suffix = (
        f"{safe_run_suffix(_ident(args))}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_dir = os.path.join(args.results_dir, "runs", run_suffix)
    os.makedirs(run_dir, exist_ok=True)
    torch.save(protection.pca_info, os.path.join(run_dir, "pca_info.pth"))
    torch.save(model.state_dict(), os.path.join(run_dir, "ckpt.pth"))
    with open(os.path.join(run_dir, "run_meta.json"), "w") as f:
        json.dump({
            "backbone": "resnet18",
            "setting": "classwise" if setting == "classifier_classwise" else "random",
            "config": os.path.abspath(args.config),
            "seed": seed,
            "experiment": args.experiment,
            "region_mode": args.region_mode,
            "interval_mode": args.interval_mode,
            "alpha": args.alpha,
            "sign_flip_frac": args.sign_flip_frac,
            "include_db": args.include_db,
            "uniform_margin": args.uniform_margin,
            "include_mean": not args.no_include_mean,
            "include_res": not args.no_include_res,
            "lambda": args.lam,
            "base_ckpt": fargs.model_path,
            "data_dir": fargs.data,
            "results_dir": args.results_dir,
            "n_epochs": n_epochs,
        }, f, indent=2)

    # ---- evaluate --------------------------------------------------------
    metrics = evaluate_all_headless(model, data_loaders, fargs, device)

    k = protection.pca_info[0]["z_min"].numel() if protection.pca_info else 0
    boxes = (make_region_boxes(
        args.region_mode,
        protection.pca_info[0]["inf_low"], protection.pca_info[0]["z_min"],
        protection.pca_info[0]["z_max"], protection.pca_info[0]["inf_high"],
        protection.pca_info[0].get("region_side"),
    ) if protection.pca_info else [])
    terms = 2 * len(boxes) if args.interval_mode != "off" else 0
    if args.interval_mode in ("width_only", "centre_only"):
        terms //= 2
    analytic_matvecs = ANALYTIC_MATVECS[args.region_mode]
    if callable(analytic_matvecs):
        analytic_matvecs = analytic_matvecs(k)
    if isinstance(analytic_matvecs, int):
        analytic_matvecs = int(
            analytic_matvecs * INTERVAL_MATVECS[args.interval_mode])

    row = {
        "experiment": args.experiment,
        "backbone": "resnet18",
        "setting": "classwise" if setting == "classifier_classwise" else "random",
        "region_mode": args.region_mode,
        "interval_mode": args.interval_mode,
        "alpha": args.alpha,
        "sign_flip_frac": args.sign_flip_frac,
        "include_db": int(args.include_db),
        "uniform_margin": int(args.uniform_margin),
        "include_mean": int(not args.no_include_mean),
        "include_res": int(not args.no_include_res),
        "lambda": args.lam,
        "fixed_lambda": int(args.lam == FIXED_LAMBDA.get(
            ("resnet18", "classwise" if setting == "classifier_classwise" else "random"))),
        "seed": seed,
        "terms": terms,
        "analytic_matvecs": analytic_matvecs,
        "ua": metrics.get("UA", float("nan")),
        "ra": metrics.get("RA", float("nan")),
        "ta": metrics.get("TA", float("nan")),
        "fid": "", "fid_backend": "none",
        "mia": metrics.get("MIA", float("nan")),
        "prot_loss_ms": pl_ms, "prot_loss_ms_std": pl_ms_std,
        "setup_wall_s": setup_wall, "train_wall_s": train_wall,
        "total_wall_s": time.time() - t0_run, "peak_mem_mb": peak_mem,
        "n_iters": n_epochs, "k": k, "tparams": tparams,
        "run_dir": run_dir,
    }
    write_sidecar(run_dir, row, diagnostics=None)

    log.info(f"run complete: {args.experiment} {args.region_mode}/"
             f"{args.interval_mode} lam={args.lam} seed={seed} "
             f"UA={metrics.get('UA'):.3f} RA={metrics.get('RA'):.3f} "
             f"TA={metrics.get('TA'):.3f} MIA={metrics.get('MIA', float('nan')):.3f} "
             f"wall={time.time()-t0_run:.1f}s")


if __name__ == "__main__":
    main()