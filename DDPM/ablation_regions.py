#!/usr/bin/env python3
"""
Ablation over protected-region constructions for BARRIER, on DDPM class
unlearning (CIFAR-10, forget class "airplane").

Region constructions, all built from the same two-corner primitive
T(l, u) = ||dWp @ l - dWn @ u||^2 + ||dWp @ u - dWn @ l||^2:

    A. two_corner  (current baseline):  T(inf_low, z_min) + T(z_max, inf_high)
    B. two_random  (placement control):  two boxes of the same shape but with a
       random per-coordinate side pattern (fixed seed, stored in pca_info)
    D. boxes_4     (4 boxes, predefined):  coordinates split into 2 consecutive
       groups; one box per group per side (2m boxes with m = 2)
    E. boxes_8     (8 boxes, predefined):  same construction with m = 4 groups
    C. slabs_2k    (exact complement):   sum over the 2k coordinate slabs whose
       union is exactly the complement of the forget box inside the envelope

The m-group family interpolates between two_corner (m=1) and slabs_2k (m=k).
Everything else is held at the paper's configuration: target layers = QKV
attention projections + class-embedding MLP, k = 32, Adam, lr = 1e-4, 3000
steps, Random-Label (RL) unlearning objective, no remain-set loss (the
protection term substitutes for it).  delta_b is kept OUT of the interval terms
in all variants.

Usage:
    # single run
    python ablation_regions.py --region_mode slabs_2k --lambda 5 --seed 0 \
        --config configs/pipeline_fulleval.yaml

    # aggregate a grid of completed runs into results/regions_table.tex
    python ablation_regions.py --summarize --results_dir ./results

The runner appends one row to <results_dir>/regions.csv and per-layer
diagnostics to <results_dir>/diagnostics.csv.
"""

import argparse
import csv
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import setup_cache  # noqa: E402  -- must precede torch/HF imports

from datasets import data_transform, get_forget_dataset  # noqa: E402
from functions import cycle, dict2namespace, get_optimizer  # noqa: E402
from functions.losses import loss_registry_conditional  # noqa: E402
from InTAct.intact import (  # noqa: E402
    UnlearnIntervalProtection,
    ddpm_forward_fn,
    make_region_boxes,
    INTERVAL_MODES,
    percentile_alpha_to_quantiles,
)
from models.diffusion import Conditional_Model  # noqa: E402
from runners.diffusion import Diffusion  # noqa: E402

from ablation_common import (  # noqa: E402
    FIXED_LAMBDA,
    REGIONS_CSV_COLS,
    DIAGNOSTICS_CSV_COLS,
    aggregate_rows,
    safe_run_suffix,
    summarize_ablations,
    write_sidecar,
)

log = logging.getLogger(__name__)

REGION_MODES = ("two_corner", "two_random", "boxes_4", "boxes_8", "slabs_2k", "env_box")
ABLATION_INTERVAL_MODES = ("full", "width_only", "centre_only", "off")
LAMBDA_SWEEP = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
ALPHAS = (1, 5, 10)

# Analytic number of [M, k] matvecs per layer per (region, interval) mode.
ANALYTIC_MATVECS = {
    "two_corner": 4,
    "two_random": 4,
    "boxes_4": 8,
    "boxes_8": 16,
    "slabs_2k": lambda k: 8 * k,
    "env_box": 2,
}

# interval_mode affects the matvec count: width_only / centre_only halve it
# (one term per box instead of two).
INTERVAL_MATVECS = {"full": 1.0, "width_only": 0.5, "centre_only": 0.5, "off": 0.0}

REGIONS_CSV_COLS = REGIONS_CSV_COLS  # re-exported from ablation_common
DIAGNOSTICS_CSV_COLS = DIAGNOSTICS_CSV_COLS  # re-exported from ablation_common

# Diagnostic tags: A=two_corner, B=two_random, C=slabs_2k, D=boxes_4, E=boxes_8,
# F=env_box
DIAG_TAGS = {"two_corner": "A", "two_random": "B",
             "slabs_2k": "C", "boxes_4": "D", "boxes_8": "E", "env_box": "F"}
DIAG_NAMES = ("A", "B", "C", "D", "E", "F")


# ============================================================================
# Config building (mirrors pipeline.py but headless / no wandb)
# ============================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_runner_config(cfg, results_dir, run_suffix):
    """Same as pipeline.build_runner_config but routes outputs under
    <results_dir>/runs/<run_suffix>."""
    model_cfg = cfg["model_config"]
    with open(model_cfg, "r") as f:
        runner_cfg = yaml.safe_load(f)

    uc = cfg["unlearn"]
    if "training" not in runner_cfg:
        runner_cfg["training"] = {}
    runner_cfg["training"]["batch_size"] = uc.get(
        "batch_size", runner_cfg["training"].get("batch_size", 128)
    )
    runner_cfg["training"]["n_iters"] = uc.get(
        "n_iters", runner_cfg["training"].get("n_iters", 3000)
    )

    tc = cfg.get("training", {})
    if tc:
        for k, v in tc.items():
            runner_cfg["training"][k] = v

    ic = cfg.get("intact", {})
    if ic:
        for k, v in ic.items():
            runner_cfg["training"][k] = v

    if "optim" not in runner_cfg:
        runner_cfg["optim"] = {}
    runner_cfg["optim"]["lr"] = uc.get("lr", runner_cfg["optim"].get("lr", 1e-4))

    config = dict2namespace(runner_cfg)

    config.exp_root_dir = os.path.join(results_dir, "runs", run_suffix)
    config.log_dir = os.path.join(config.exp_root_dir, "logs")
    config.ckpt_dir = os.path.join(config.exp_root_dir, "ckpts")
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.ckpt_dir, exist_ok=True)

    return config


def build_runner_args(cfg, runner_config, seed):
    uc = cfg["unlearn"]
    args = argparse.Namespace()
    args.config = cfg["model_config"]
    args.ckpt_folder = cfg["paths"]["pretrained_ckpt_folder"]
    args.mode = uc.get("mode", "intact")
    args.label_to_forget = uc.get("label_to_forget", 0)
    args.seed = seed
    args.sample_type = cfg.get("sample_type", "generalized")
    args.skip_type = cfg.get("skip_type", "uniform")
    args.timesteps = cfg.get("timesteps", 1000)
    args.eta = cfg.get("eta", 1.0)
    args.cond_scale = cfg.get("cond_scale", 2.0)
    args.sequence = False
    args.alpha = uc.get("alpha", 0.0)
    args.mask_path = None
    args.method = uc.get("method", "rl")
    args.uc = True
    args.negative_guidance = uc.get("negative_guidance", 7.5)
    args.mask_ratio = 0.5
    args.sparse = False
    args.n_samples_per_class = cfg.get("evaluate", {}).get("n_samples_per_class", 500)
    args.classes_to_generate = None
    return args


# ============================================================================
# Protection-loss microbenchmark
# ============================================================================

def microbench_protection(protection, model, device, warmup=20, iters=200):
    """Isolate compute_protection_loss: warmup + timed calls, CUDA-synced.
    Returns (mean_ms, std_ms)."""
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


# ============================================================================
# Diagnostics (computed once per layer per run, logged to diagnostics.csv)
# ============================================================================

def collect_remain_projections(protection, model, remain_loader, device,
                               data_transform_fn, betas, num_timesteps):
    """Project remain activations onto each target layer's U_forget basis."""
    pca_components = {
        info["layer_name"]: {
            "mu": info["mu"].to(device),
            "U_forget": info["U_forget"].to(device),
        }
        for info in protection.pca_info
    }
    projected = protection._collect_activations(
        model,
        list(pca_components.keys()),
        remain_loader,
        device,
        forward_fn=ddpm_forward_fn,
        data_transform_fn=data_transform_fn,
        betas=betas,
        num_timesteps=num_timesteps,
        pca_components=pca_components,
    )
    return {ln: t.cpu() for ln, t in projected.items()}


def penalty_direction(v, boxes):
    """T-sum over the variant's boxes for a single activation-space direction
    v in R^k (dWp = relu(v), dWn = relu(-v))."""
    dWp, dWn = np.maximum(v, 0.0), np.maximum(-v, 0.0)
    total = 0.0
    for l, u in boxes:
        total += (np.dot(dWp, l) - np.dot(dWn, u)) ** 2
        total += (np.dot(dWp, u) - np.dot(dWn, l)) ** 2
    return total


def _variant_boxes(info):
    """The region variants' box lists as numpy (l, u) pairs for layer `info`.
    Keys: A=two_corner, B=two_random, C=slabs_2k, D=boxes_4, E=boxes_8,
    F=env_box."""
    inf_low = info["inf_low"].numpy()
    z_min = info["z_min"].numpy()
    z_max = info["z_max"].numpy()
    inf_high = info["inf_high"].numpy()
    k = inf_low.shape[0]
    side = info.get("region_side")
    s = side.numpy().astype(bool) if side is not None else None

    boxA = [(inf_low, z_min), (z_max, inf_high)]
    boxB = boxA
    if s is not None:
        lA = np.where(s, z_max, inf_low)
        uA = np.where(s, inf_high, z_min)
        lB = np.where(s, inf_low, z_max)
        uB = np.where(s, z_min, inf_high)
        boxA = [(lA, uA), (lB, uB)]
        boxB = [(lB, uB), (lA, uA)]
    boxC = []
    for j in range(k):
        l = inf_low.copy()
        u = inf_high.copy()
        u[j] = z_min[j]
        boxC.append((l, u))
        l = inf_low.copy()
        u = inf_high.copy()
        l[j] = z_max[j]
        boxC.append((l, u))

    def groups(m):
        boxes = []
        m = max(1, min(m, k))
        for g in range(m):
            j0 = (g * k) // m
            j1 = ((g + 1) * k) // m
            if j1 <= j0:
                continue
            l = inf_low.copy()
            u = inf_high.copy()
            u[j0:j1] = z_min[j0:j1]
            boxes.append((l, u))
            l = inf_low.copy()
            u = inf_high.copy()
            l[j0:j1] = z_max[j0:j1]
            boxes.append((l, u))
        return boxes

    boxD = groups(2)
    boxE = groups(4)
    boxF = [(inf_low, inf_high)]
    return {"A": boxA, "B": boxB, "C": boxC, "D": boxD, "E": boxE, "F": boxF}


def _max_drift_box(v, l, u):
    pos = v >= 0
    upper = float(np.abs((v * np.where(pos, u, l)).sum()))
    lower = float(np.abs((v * np.where(pos, l, u)).sum()))
    return max(upper, lower)


def _max_drift_set(v, boxes):
    return max(_max_drift_box(v, l, u) for l, u in boxes)


def layer_diagnostics(info, z_remain, diag_dirs=10000):
    """Per-layer diagnostic quantities for all three variants.  z_remain may be
    None (then the activation-fraction columns are NaN)."""
    k = info["z_min"].numel()
    inf_low, z_min, z_max, inf_high = (
        info["inf_low"].numpy(), info["z_min"].numpy(),
        info["z_max"].numpy(), info["inf_high"].numpy(),
    )
    boxes = _variant_boxes(info)

    # --- envelope asymmetry ---
    c_lo = (inf_low + z_min) / 2.0
    c_hi = (z_max + inf_high) / 2.0
    c_lo_norm = float(np.linalg.norm(c_lo))
    c_hi_norm = float(np.linalg.norm(c_hi))
    env_asym = c_hi_norm / c_lo_norm if c_lo_norm > 0 else float("nan")

    # --- fraction of remain activations inside each variant's protected set ---
    frac_A = frac_B = frac_C = frac_D = frac_E = frac_F = float("nan")
    if z_remain is not None:
        zr = z_remain.numpy()
        fracs = {}
        for name in DIAG_NAMES:
            inside = np.zeros(zr.shape[0], dtype=bool)
            for l, u in boxes[name]:
                inside |= np.all((zr >= l) & (zr <= u), axis=1)
            fracs[name] = float(inside.mean())
        frac_A, frac_B, frac_C, frac_D, frac_E, frac_F = (
            fracs["A"], fracs["B"], fracs["C"], fracs["D"], fracs["E"], fracs["F"])

    # --- penalty anisotropy over random unit directions ---
    rng = np.random.default_rng(0)
    V = rng.standard_normal((diag_dirs, k))
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12

    aniso = {}
    vstar = {}
    for name in DIAG_NAMES:
        P = np.array([penalty_direction(v, boxes[name]) for v in V])
        pmin = P.min()
        aniso[name] = float(P.max() / pmin) if pmin > 0 else float("inf")
        vstar[name] = V[int(np.argmin(P))]

    drift = {
        name: (
            _max_drift_set(vstar[name], boxes[name]),
            _max_drift_box(vstar[name], inf_low, inf_high),
        )
        for name in DIAG_NAMES
    }

    row = {"k": k,
           "c_lo_norm": c_lo_norm, "c_hi_norm": c_hi_norm, "env_asym_ratio": env_asym,
           "frac_remain_in_A": frac_A, "frac_remain_in_B": frac_B,
           "frac_remain_in_C": frac_C, "frac_remain_in_D": frac_D,
           "frac_remain_in_E": frac_E, "frac_remain_in_F": frac_F}
    for name in DIAG_NAMES:
        row[f"aniso_{name}"] = aniso[name]
        row[f"vstar_drift_protected_{name}"] = drift[name][0]
        row[f"vstar_drift_envelope_{name}"] = drift[name][1]
    return row


# ============================================================================
# Unlearning + evaluation
# ============================================================================

def setup_protection(protection, runner, model, forget_loader, remain_loader):
    """Timed activation collection + SVD + bounds (paper config)."""
    t0 = time.time()
    remain_for_bounds = remain_loader if protection.use_actual_bounds else None
    protection.setup_protection(
        model, forget_loader, runner.device,
        remain_dataloader=remain_for_bounds,
        forward_fn=ddpm_forward_fn,
        data_transform_fn=lambda x: data_transform(runner.config, x),
        betas=runner.betas,
        num_timesteps=runner.num_timesteps,
    )
    return time.time() - t0


def rl_unlearn_steps(
    model, protection, optimizer, criteria, forget_iter, label_to_forget,
    n_iters, num_timesteps, betas, device, config, grad_clip, log_freq=100,
):
    """RL forget loss + protection loss, no remain-set loss (paper config with
    alpha = 0).  Mirrors runners/diffusion.intact_unlearn's RL step with the
    remain forward omitted (weighted by 0 there, so it has no effect)."""
    model.train()
    n_classes = config.data.n_classes
    pseudo_label = (label_to_forget + 1) % n_classes
    for step in range(n_iters):
        forget_x, forget_c = next(forget_iter)
        n = forget_x.size(0)
        forget_x = forget_x.to(device)
        forget_c = forget_c.to(device)
        forget_x = data_transform(config, forget_x)
        e = torch.randn_like(forget_x)
        t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
        t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]

        a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        forget_x_noisy = forget_x * a.sqrt() + e * (1.0 - a).sqrt()
        output = model(forget_x_noisy, t.float(), forget_c, mode="train")
        pseudo_c = torch.full(forget_c.shape, pseudo_label, device=forget_c.device)
        pseudo = model(forget_x_noisy, t.float(), pseudo_c, mode="train").detach()
        forget_loss = criteria(pseudo, output)

        protection_loss = protection.compute_protection_loss(model, device)
        loss = forget_loss + protection_loss

        optimizer.zero_grad()
        loss.backward()
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        except Exception:
            pass
        optimizer.step()

        if (step + 1) % log_freq == 0:
            log.info(
                f"step: {step}, forget_loss: {forget_loss.item():.4f}, "
                f"protection_loss: {protection_loss.item():.4f}, "
                f"total_loss: {loss.item():.4f}"
            )


def sample_for_eval(runner_args, runner_config, mode, classes_to_generate,
                    n_samples_per_class):
    """Run sampling through the shared Diffusion runner (paper protocol)."""
    runner_args.mode = mode
    runner_args.classes_to_generate = classes_to_generate
    runner_args.n_samples_per_class = n_samples_per_class
    Diffusion(runner_args, runner_config).sample()


def compute_ua_ta_fid(args, runner_args, runner_config, device, clf_ckpt,
                      ref_dir, n_classes):
    """Reuse pipeline.py's evaluation (UA, TA via ResNet-34; FID via the TF
    Inception evaluator with torchmetrics fallback)."""
    from pipeline import (
        classifier_eval,
        classifier_eval_remaining,
        compute_fid_reference,
    )

    run_dir = runner_config.exp_root_dir
    class_samples_dir = os.path.join(run_dir, "class_samples")
    forget_dir = os.path.join(class_samples_dir, str(args.label_to_forget))

    ua = float("nan")
    if os.path.isdir(forget_dir):
        clf = classifier_eval(forget_dir, "cifar10", args.label_to_forget,
                              clf_ckpt, device)
        ua = 1.0 - clf.get("classifier/acc_forgotten", 0.0)

    ta = float("nan")
    if os.path.isdir(class_samples_dir):
        ta, _ = classifier_eval_remaining(
            class_samples_dir, args.label_to_forget, n_classes, clf_ckpt, device,
        )
        if ta is None:
            ta = float("nan")

    fid = float("nan")
    fid_backend = "none"
    fid_dir = os.path.join(
        run_dir,
        f"fid_samples_guidance_{runner_args.cond_scale}_excluded_class_{args.label_to_forget}",
    )
    if os.path.isdir(fid_dir):
        try:
            metrics = compute_fid_reference(ref_dir, fid_dir)
            fid = float(metrics["fid"])
            fid_backend = "tf-inception"
        except Exception as exc:  # pragma: no cover - cluster fallback
            log.warning(f"TF FID failed ({exc}); falling back to torchmetrics")
            try:
                from torchmetrics.image.fid import FrechetInceptionDistance
                import torchvision.transforms as T
                from PIL import Image

                def _chunked_load(paths, chunk=256):
                    """Stream the image stack in chunks: passing all 2048
                    images through Inception in one forward pass peaks at tens
                    of GB of CPU activations (observed OOM kill at 64 GB)."""
                    t = T.Compose([T.ToTensor()])
                    for i in range(0, len(paths), chunk):
                        imgs = torch.stack([t(Image.open(p).convert("RGB"))
                                            for p in paths[i:i + chunk]])
                        yield (imgs * 255).byte()

                n = 2048
                fidm = FrechetInceptionDistance(feature=64).to(device)
                ref_paths = sorted(Path(ref_dir).glob("*.png"))[:n]
                gen_paths = sorted(Path(fid_dir).glob("*.png"))[:n]
                for batch in _chunked_load(ref_paths):
                    fidm.update(batch, real=True)
                for batch in _chunked_load(gen_paths):
                    fidm.update(batch, real=False)
                fid = float(fidm.compute())
                fid_backend = "torchmetrics"
            except Exception as exc2:
                log.warning(f"torchmetrics FID also failed ({exc2}); FID=NaN")

    return ua, ta, fid, fid_backend


# ============================================================================
# Run bookkeeping (append rows to results/regions.csv / diagnostics.csv)
# ============================================================================

def _ensure_header(path, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(cols)


def append_regions_row_deprecated(results_dir, row):
    raise NotImplementedError(
        "direct CSV appends are disabled for parallel safety; run "
        "--summarize to aggregate runs/*/rows.json")


def append_diagnostics_rows_deprecated(results_dir, rows):
    raise NotImplementedError(
        "direct CSV appends are disabled for parallel safety; run "
        "--summarize to aggregate runs/*/rows.json")


# NOTE: per-run rows go to <run_dir>/rows.json (sidecar); the CSVs under
# results_dir are regenerated offline by aggregate_rows().


# ============================================================================
# Summarize -> results/regions_table.tex
# ============================================================================

def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and v != v):
        return "-"
    return f"{v:.{nd}f}"


def _mean_std(vals):
    vals = [v for v in vals if v == v and v is not None]
    if not vals:
        return float("nan"), float("nan")
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std())


def summarize(results_dir):
    """Aggregate regions.csv -> regions_table.tex (mean +- std over seeds).
    Reports the fixed-lambda=5 row and the best-lambda-per-variant row."""
    path = os.path.join(results_dir, "regions.csv")
    if not os.path.exists(path):
        log.error(f"No {path}; nothing to summarize")
        return

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    # deduplicate: keep the last row per (region_mode, lambda, seed)
    latest = {}
    for r in rows:
        key = (r["region_mode"], float(r["lambda"]), int(r["seed"]))
        latest[key] = r
    rows = list(latest.values())

    by_var_lam = {}
    for r in rows:
        by_var_lam.setdefault((r["region_mode"], float(r["lambda"])), []).append(r)

    def agg(entries, key):
        vals = [float(e.get(key, "nan")) for e in entries]
        return _mean_std(vals)

    def cell_composite(entries):
        """Composite of the cell MEAN over seeds (nan-safe)."""
        ua_m, _ = _mean_std([float(e.get("ua", "nan")) for e in entries])
        ta_m, _ = _mean_std([float(e.get("ta", "nan")) for e in entries])
        fid_m, _ = _mean_std([float(e.get("fid", "nan")) for e in entries])
        if ua_m != ua_m or ta_m != ta_m:
            return float("-inf")
        concrete = fid_m if fid_m == fid_m else 0.0
        return 1.5 * ua_m + ta_m - concrete / 3.0

    # Best-per-variant = the lambda whose CELL MEAN (over seeds) maximises the
    # composite — not the best single seed (which can lie below the mean).
    variant_best = {}
    for (var, lam), entries in by_var_lam.items():
        if not entries:
            continue
        score = cell_composite(entries)
        if var not in variant_best or score > variant_best[var][0]:
            variant_best[var] = (score, lam, entries)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Protected-region construction ablation on DDPM class "
                 r"unlearning (CIFAR-10, forget airplane). Mean $\pm$ std over 3 "
                 r"seeds. \texttt{fixed} = $\lambda_\mathrm{int}=5$; "
                 r"\texttt{best} = best $\lambda_\mathrm{int}$ per variant "
                 r"(composite $1.5\cdot\mathrm{UA} + \mathrm{TA} - \mathrm{FID}/3$).}")
    lines.append(r"\label{tab:region-ablation}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{l c c c c c c c}")
    lines.append(r"\toprule")
    lines.append(r"Variant & $\lambda_\mathrm{int}$ & terms & UA & TA & FID & "
                 r"prot-loss (ms/step) & total wall (s) \\")
    lines.append(r"\midrule")
    for var in ("two_corner", "two_random", "boxes_4", "boxes_8", "slabs_2k", "env_box"):
        entries5 = by_var_lam.get((var, 5.0), [])
        terms_txt = entries5[0]["terms"] if entries5 else "-"
        if entries5:
            ua_m, ua_s = agg(entries5, "ua")
            ta_m, ta_s = agg(entries5, "ta")
            fid_m, fid_s = agg(entries5, "fid")
            pl_m, pl_s = agg(entries5, "prot_loss_ms")
            tw_m, tw_s = agg(entries5, "total_wall_s")
            fixed_cells = (
                rf"{_fmt(ua_m)}$\pm${_fmt(ua_s)} & {_fmt(ta_m)}$\pm${_fmt(ta_s)} & "
                rf"{_fmt(fid_m)}$\pm${_fmt(fid_s)} & "
                rf"{_fmt(pl_m)}$\pm${_fmt(pl_s)} & {_fmt(tw_m)}$\pm${_fmt(tw_s)}"
            )
        else:
            fixed_cells = " & ".join(["-", "-", "-", "-", "-"])
        lines.append(
            rf"{var} (fixed) & 5 & {terms_txt} & {fixed_cells} \\"
        )
        if var in variant_best:
            _, lam, best_entries = variant_best[var]
            ua_m, ua_s = agg(best_entries, "ua")
            ta_m, ta_s = agg(best_entries, "ta")
            fid_m, fid_s = agg(best_entries, "fid")
            pl_m, pl_s = agg(best_entries, "prot_loss_ms")
            tw_m, tw_s = agg(best_entries, "total_wall_s")
            lam_txt = f"{lam:.1f}".rstrip("0").rstrip(".")
            lines.append(
                rf"{var} (best) & {lam_txt} & {best_entries[0]['terms']} & "
                rf"{_fmt(ua_m)}$\pm${_fmt(ua_s)} & {_fmt(ta_m)}$\pm${_fmt(ta_s)} & "
                rf"{_fmt(fid_m)}$\pm${_fmt(fid_s)} & "
                rf"{_fmt(pl_m)}$\pm${_fmt(pl_s)} & {_fmt(tw_m)}$\pm${_fmt(tw_s)} \\"
            )
        lines.append(r"\addlinespace")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    out = os.path.join(results_dir, "regions_table.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    lambdas = sorted({lam for (_, lam) in by_var_lam})
    log.info(f"Wrote {out} (aggregated {len(rows)} unique runs; "
             f"lambdas present: {lambdas})")

    # --- console view: mean +- std per (variant, lambda) -------------------
    print()
    hdr = f"{'variant':<12s} {'lam':>5s} {'n':>2s} {'UA':>14s} {'TA':>14s} {'FID':>14s} {'prot-ms':>12s} {'wall(s)':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for var in ("two_corner", "two_random", "boxes_4", "boxes_8", "slabs_2k", "env_box"):
        for lam in lambdas:
            entries = by_var_lam.get((var, lam), [])
            if not entries:
                continue
            ua_m, ua_s = agg(entries, "ua")
            ta_m, ta_s = agg(entries, "ta")
            fid_m, fid_s = agg(entries, "fid")
            pl_m, pl_s = agg(entries, "prot_loss_ms")
            tw_m, tw_s = agg(entries, "total_wall_s")
            lam_txt = f"{lam:.1f}".rstrip("0").rstrip(".")
            print(
                f"{var:<12s} {lam_txt:>5s} {len(entries):>2d} "
                f"{_fmt(ua_m)}+-{_fmt(ua_s):>7s} "
                f"{_fmt(ta_m)}+-{_fmt(ta_s):>7s} "
                f"{_fmt(fid_m)}+-{_fmt(fid_s):>7s} "
                f"{_fmt(pl_m,1)}+-{_fmt(pl_s,1):>5s} "
                f"{_fmt(tw_m,0):>10s}"
            )
    best_line = "best per variant (cell mean over seeds): " + ", ".join(
        f"{var}(λ={_fmt(variant_best[var][1],1).rstrip('0').rstrip('.')}) "
        f"UA={_fmt(agg(variant_best[var][2], 'ua')[0])} "
        f"TA={_fmt(agg(variant_best[var][2], 'ta')[0])} "
        f"FID={_fmt(agg(variant_best[var][2], 'fid')[0])}"
        for var in ("two_corner", "two_random", "boxes_4", "boxes_8", "slabs_2k", "env_box")
        if var in variant_best
    )
    print()
    print(best_line)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region_mode", choices=REGION_MODES, default="two_corner")
    parser.add_argument("--interval_mode", choices=ABLATION_INTERVAL_MODES,
                        default="full")
    parser.add_argument("--alpha", type=int, choices=[1, 5, 10], default=5,
                        help="Experiment 7: percentile alpha (1/99 .. 10/90 "
                             "quantiles for z_min/z_max)")
    parser.add_argument("--sign_flip_frac", type=float, default=0.0,
                        help="Experiment 6: fraction of U_forget rows whose "
                             "sign is flipped before the bounds are recomputed")
    parser.add_argument("--include_db", action="store_true",
                        help="Experiment 8: put delta_b in the interval corner "
                             "legs (InTAct Eqs. 35-36)")
    parser.add_argument("--uniform_margin", action="store_true",
                        help="Experiment 3: constant envelope margins = mean "
                             "of the per-coordinate margins")
    parser.add_argument("--include_mean", action="store_true",
                        help="Experiment 9: keep L_mean (default on; set "
                             "--no_include_mean to drop it)")
    parser.add_argument("--no_include_mean", action="store_true",
                        help="Experiment 9: drop L_mean")
    parser.add_argument("--include_res", action="store_true",
                        help="Experiment 9: keep L_res (default on; set "
                             "--no_include_res to drop it)")
    parser.add_argument("--no_include_res", action="store_true",
                        help="Experiment 9: drop L_res")
    parser.add_argument("--experiment", default="exp5", choices=[
        "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp9", "grid"],
        help="which experiment this run belongs to (labels the ablations.csv "
             "row and the summarise table)")
    parser.add_argument("--lambda", dest="lam", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", default="configs/pipeline_fulleval.yaml")
    parser.add_argument("--n_iters", type=int, default=3000)
    parser.add_argument("--label_to_forget", type=int, default=0)
    parser.add_argument("--results_dir", default="/shared/results/common/miksa/intact/DDPM/r")
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--microbench_warmup", type=int, default=20)
    parser.add_argument("--microbench_iters", type=int, default=200)
    parser.add_argument("--diag_dirs", type=int, default=10000)
    parser.add_argument("--skip_diagnostics", action="store_true")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.summarize:
        # regenerate the CSVs from per-run sidecars, then render tables
        aggregate_rows(args.results_dir)
        summarize(args.results_dir)
        summarize_ablations(args.results_dir)
        return

    if args.interval_mode == "full_decomp":
        raise ValueError("interval_mode='full_decomp' is the internal (F1) "
                         "identity route used by the tests, not an ablation mode")

    t0_run = time.time()

    # ---- fixed paper hyperparameters (not retuned) -------------------------
    args.lr = 1e-4
    args.method = "rl"
    args.alpha_remain = 0.0

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    lower_p, upper_p = percentile_alpha_to_quantiles(args.alpha)

    cfg = load_config(args.config)
    cfg["unlearn"]["lr"] = args.lr
    cfg["unlearn"]["n_iters"] = args.n_iters
    cfg["unlearn"]["method"] = args.method
    cfg["unlearn"]["alpha"] = args.alpha_remain
    cfg["unlearn"]["label_to_forget"] = args.label_to_forget
    cfg["pipeline"]["seed"] = seed

    ic = cfg.setdefault("intact", {})
    ic["lambda_interval"] = args.lam
    ic["region_mode"] = args.region_mode
    ic["normalize_region"] = True
    ic["region_random_seed"] = 0
    ic["reduced_dim"] = 32
    ic["use_actual_bounds"] = True
    ic["interval_mode"] = args.interval_mode
    ic["lower_percentile"] = lower_p
    ic["upper_percentile"] = upper_p
    ic["include_db"] = args.include_db
    ic["include_mean"] = not args.no_include_mean
    ic["include_res"] = not args.no_include_res
    ic["sign_flip_frac"] = args.sign_flip_frac
    ic["sign_flip_seed"] = seed
    ic["uniform_margin"] = args.uniform_margin

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    run_ident = {
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
        "seed": seed,
    }
    run_suffix = (
        f"{safe_run_suffix(run_ident)}"
        f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    runner_config = build_runner_config(cfg, results_dir, run_suffix)
    runner_args = build_runner_args(cfg, runner_config, seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = Diffusion(runner_args, runner_config)
    label = args.label_to_forget

    remain_loader, forget_loader = get_forget_dataset(
        runner_args, runner_config, label
    )
    forget_iter = cycle(forget_loader)

    # ---- model + protection (paper config) --------------------------------
    model = Conditional_Model(runner_config)
    states = torch.load(
        os.path.join(runner_args.ckpt_folder, "ckpts", "ckpt.pth"),
        map_location=device,
    )
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(states[0], strict=True)

    protection = UnlearnIntervalProtection(
        targets=runner_config.training.targets,
        lambda_interval=runner_config.training.lambda_interval,
        lower_percentile=runner_config.training.lower_percentile,
        upper_percentile=runner_config.training.upper_percentile,
        reduced_dim=runner_config.training.reduced_dim,
        infinity_scale=runner_config.training.infinity_scale,
        use_actual_bounds=runner_config.training.use_actual_bounds,
        normalize_protection=runner_config.training.normalize_protection,
        skip_svd=getattr(runner_config.training, "skip_svd", False),
        skip_interval=getattr(runner_config.training, "skip_interval", False),
        remove_top_directions=getattr(runner_config.training, "remove_top_directions", False),
        decomp_method=getattr(runner_config.training, "decomp_method", "svd"),
        region_mode=runner_config.training.region_mode,
        normalize_region=runner_config.training.normalize_region,
        region_random_seed=runner_config.training.region_random_seed,
        interval_mode=getattr(runner_config.training, "interval_mode", "full"),
        include_db=getattr(runner_config.training, "include_db", False),
        include_mean=getattr(runner_config.training, "include_mean", True),
        include_res=getattr(runner_config.training, "include_res", True),
        sign_flip_frac=getattr(runner_config.training, "sign_flip_frac", 0.0),
        sign_flip_seed=getattr(runner_config.training, "sign_flip_seed", 0),
        uniform_margin=getattr(runner_config.training, "uniform_margin", False),
    )
    # ---- setup / preprocessing wall time (activations + SVD + bounds) -----
    setup_wall = setup_protection(
        protection, runner, model, forget_loader, remain_loader
    )
    log.info(f"setup/preprocessing wall time: {setup_wall:.2f}s")

    # freeze AFTER setup: setup_protection populates target_layers; calling
    # freeze before it would mark 0 parameters -> empty optimizer (observed).
    protection.freeze_non_target_params(model)
    trainable_params = protection.get_trainable_params(model)
    if not trainable_params:
        raise RuntimeError(
            "no trainable parameters after freeze_non_target_params — "
            f"targets {runner_config.training.targets} did not match the model")
    tparams = sum(p.numel() for p in trainable_params)

    k = protection.pca_info[0]["z_min"].numel() if protection.pca_info else 0
    terms = 0
    analytic_matvecs = ANALYTIC_MATVECS[args.region_mode]
    if callable(analytic_matvecs):
        analytic_matvecs = analytic_matvecs(k)
    if protection.pca_info:
        boxes = make_region_boxes(
            args.region_mode,
            protection.pca_info[0]["inf_low"],
            protection.pca_info[0]["z_min"],
            protection.pca_info[0]["z_max"],
            protection.pca_info[0]["inf_high"],
            protection.pca_info[0].get("region_side"),
        )
        if args.interval_mode == "off":
            terms, analytic_matvecs = 0, 0
        else:
            terms = 2 * len(boxes)
            if args.interval_mode in ("width_only", "centre_only"):
                terms //= 2
            if isinstance(analytic_matvecs, int):
                analytic_matvecs = int(
                    analytic_matvecs * INTERVAL_MATVECS[args.interval_mode])

    # ---- diagnostics (once per layer) ------------------------------------
    diag_rows = []
    if not args.skip_diagnostics:
        zr = None
        try:
            zr = collect_remain_projections(
                protection, model, remain_loader, device,
                lambda x: data_transform(runner_config, x),
                runner.betas, runner.num_timesteps,
            )
        except Exception as exc:
            log.warning(f"remain-projection diagnostics failed ({exc}); "
                        f"fraction columns set to NaN")
        for info in protection.pca_info:
            diag = layer_diagnostics(
                info, zr.get(info["layer_name"]) if zr else None,
                diag_dirs=args.diag_dirs,
            )
            diag_rows.append({
                "region_mode": args.region_mode,
                "layer_name": info["layer_name"],
                "diag_dirs": args.diag_dirs,
                **diag,
            })

    # ---- protection-loss microbenchmark ----------------------------------
    pl_ms, pl_ms_std = microbench_protection(
        protection, model, device,
        warmup=args.microbench_warmup, iters=args.microbench_iters,
    )
    log.info(f"protection loss microbench: {pl_ms:.3f} +- {pl_ms_std:.3f} ms/call")

    # ---- training ---------------------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    optimizer = get_optimizer(runner_config, trainable_params)
    criteria = torch.nn.MSELoss()
    t0 = time.time()
    rl_unlearn_steps(
        model, protection, optimizer, criteria, forget_iter, label,
        args.n_iters, runner.num_timesteps, runner.betas, device,
        runner_config, runner_config.optim.grad_clip,
        log_freq=args.log_freq,
    )
    train_wall = time.time() - t0
    peak_mem = (
        torch.cuda.max_memory_allocated() / 1e6
        if torch.cuda.is_available() else 0.0
    )
    log.info(f"training wall time ({args.n_iters} steps): {train_wall:.2f}s")

    os.makedirs(runner_config.ckpt_dir, exist_ok=True)
    torch.save(
        [model.state_dict(), optimizer.state_dict(), args.n_iters - 1],
        os.path.join(runner_config.ckpt_dir, "ckpt.pth"),
    )

    # ---- artifacts for Experiments 1-2 (mechanism plot / kappa stats) ------
    torch.save(
        protection.pca_info,
        os.path.join(runner_config.exp_root_dir, "pca_info.pth"),
    )
    run_meta = {
        "backbone": "ddpm",
        "setting": "classwise",
        "config": os.path.abspath(args.config),
        "model_config": cfg["model_config"],
        "label_to_forget": args.label_to_forget,
        "n_iters": args.n_iters,
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
        "base_ckpt_folder": cfg["paths"]["pretrained_ckpt_folder"],
        "results_dir": results_dir,
        "k": k,
    }
    import json
    with open(os.path.join(runner_config.exp_root_dir, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    # ---- evaluation (UA / TA / FID) ---------------------------------------
    ua = ta = fid = float("nan")
    fid_backend = "none"
    if not args.skip_eval:
        runner_args.ckpt_folder = runner_config.exp_root_dir
        clf_samples = cfg.get("evaluate", {}).get(
            "classifier", {}).get("n_samples_per_class", 500)
        fid_samples = cfg.get("evaluate", {}).get(
            "fid", {}).get("n_samples_per_class", 5000)

        sample_for_eval(runner_args, runner_config, "sample_classes",
                        ",".join(str(i) for i in range(10)), clf_samples)
        sample_for_eval(runner_args, runner_config, "sample_fid",
                        f"x{label}", fid_samples)

        clf_ckpt = cfg["paths"].get("classifier_ckpt", "cifar10_resnet34.pth")
        ref_dir = cfg["paths"].get("ref_dataset_dir", "cifar10_without_label_0")
        ua, ta, fid, fid_backend = compute_ua_ta_fid(
            args, runner_args, runner_config, device, clf_ckpt, ref_dir,
            cfg["data"].get("n_classes", 10),
        )

    total_wall = time.time() - t0_run
    ab_row = {
        "experiment": args.experiment,
        "backbone": "ddpm",
        "setting": "classwise",
        "region_mode": args.region_mode,
        "interval_mode": args.interval_mode,
        "alpha": args.alpha,
        "sign_flip_frac": args.sign_flip_frac,
        "include_db": int(args.include_db),
        "uniform_margin": int(args.uniform_margin),
        "include_mean": int(not args.no_include_mean),
        "include_res": int(not args.no_include_res),
        "lambda": args.lam,
        "fixed_lambda": int(args.lam == FIXED_LAMBDA.get(("ddpm", "classwise"))),
        "seed": seed,
        "terms": terms,
        "analytic_matvecs": analytic_matvecs,
        "ua": ua, "ra": "", "ta": ta, "fid": fid, "fid_backend": fid_backend,
        "mia": "",
        "prot_loss_ms": pl_ms, "prot_loss_ms_std": pl_ms_std,
        "setup_wall_s": setup_wall, "train_wall_s": train_wall,
        "total_wall_s": total_wall, "peak_mem_mb": peak_mem,
        "n_iters": args.n_iters, "k": k, "tparams": tparams,
        "diag_dirs": args.diag_dirs,
        "run_dir": runner_config.exp_root_dir,
    }
    if not args.skip_diagnostics:
        for d in diag_rows:
            d["lambda"] = args.lam
            d["seed"] = seed
    write_sidecar(runner_config.exp_root_dir, ab_row, diag_rows)

    log.info(f"run complete: variant={args.region_mode} lambda={args.lam} "
             f"seed={seed} UA={ua:.3f} TA={ta:.3f} FID={fid:.3f} "
             f"wall={total_wall:.1f}s peak={peak_mem:.0f}MB")


if __name__ == "__main__":
    main()
