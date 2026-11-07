"""
Forward-pass-only overlap analysis (rebuttal): how much retain-set variance
lies in BARRIER's protected forget subspace U_k, and how many retain rows fall
inside the forget box H_f.  No training.

Everything BARRIER-specific is the repo's own code, unchanged:
  - model + data          : pipeline.build_args -> utils.setup_model_dataset ->
                            pipeline.build_data_loaders (same as ablation_cls.py)
  - activation hooks      : UnlearnIntervalProtection._collect_activations
                            (layer INPUT; Conv2d -> one row per spatial position)
  - PCA (U_k) + box H_f   : UnlearnIntervalProtection.setup_protection
                            (SVD of forget activations centred on the forget
                            mean mu; z_min/z_max = 5th/95th percentiles of the
                            projected forget coordinates)

Paper protocol (ablation_cls.py / pipeline_*.yaml): ResNet-18, k = 32, the
protected layer is the final `fc` only.  pipeline_*.yaml lists
["conv1x1", "fc"], but `conv1x1` is the helper that builds the downsample
convs (module names layerX.0.downsample.0), so it matches no module and the
protected set is exactly {fc}.  --contrast_layer adds one earlier, NOT
protected layer for contrast (protected=0 in the CSV).

Settings:
  classwise : pipeline_classwise.yaml   (forget class 0 = airplane)
  random    : pipeline_random_10pp.yaml (4500 random train samples; the
              selection depends on the seed, paper seeds 0/1/2)

Per (setting, seed, layer):
  overlap   = ||(A_r - mean(A_r)) U_k^T||^2 / ||A_r - mean(A_r)||^2
  rand_*    = same overlap for 10 random orthonormal d x k bases
  box_frac  = % of retain rows with (a_r - mu) U_k^T inside [z_min, z_max]
              on all k coordinates (BARRIER's projection / centring)
Sanity checks: overlap in [0, 1]; variance of the forget set captured by U_k
equals the explained-variance ratio of the SVD (evr_check vs evr_pca).

Usage (from Classification/, PYTHONPATH=<repo root>):
    python scripts/overlap_analysis.py --seeds 0 1 2 --out results/overlap.csv
"""

import argparse
import csv
import logging
import math
import os
import random
import sys

import numpy as np
import torch
import yaml

_CLS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_CLS_DIR))  # repo root (InTAct, ablation_common)
sys.path.insert(0, _CLS_DIR)  # Classification last -> highest priority

from ablation_common import torch_load  # noqa: E402
from InTAct.intact import (  # noqa: E402
    UnlearnIntervalProtection,
    classification_forward_fn,
    percentile_alpha_to_quantiles,
)
from pipeline import build_args, build_data_loaders  # noqa: E402
import utils  # noqa: E402

log = logging.getLogger("overlap")

COLUMNS = [
    "setting", "layer", "d", "k", "n_forget", "n_retain", "overlap",
    "rand_mean", "rand_std", "k_over_d", "box_frac",
    # extras
    "seed", "protected", "n_retain_rows_total", "evr_pca", "evr_check",
    "forget_box_frac",
]


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    utils.setup_seed(seed)


def make_protection(layer, k, alpha):
    lower_p, upper_p = percentile_alpha_to_quantiles(alpha)
    # Same constructor arguments as ablation_cls.py (lambda is irrelevant here).
    return UnlearnIntervalProtection(
        targets=[layer],
        lambda_interval=1.0,
        lower_percentile=lower_p,
        upper_percentile=upper_p,
        reduced_dim=k,
        infinity_scale=20.0,
        use_actual_bounds=True,
        normalize_protection=True,
        region_mode="two_corner",
        normalize_region=True,
        region_random_seed=0,
        interval_mode="full",
    )


def collect_raw(protection, model, layer_name, loader, device):
    acts = protection._collect_activations(
        model, [layer_name], loader, device,
        forward_fn=classification_forward_fn,
    )
    return acts[layer_name]["activations"].float()


def captured_variance(Xc, U):
    """||Xc U^T||^2 / ||Xc||^2 for row-orthonormal U (k x d)."""
    return float((Xc @ U.T).pow(2).sum() / Xc.pow(2).sum())


def random_bases_overlap(Xc, k, n_rand, seed):
    d = Xc.size(1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    vals = []
    for _ in range(n_rand):
        G = torch.randn(d, k, generator=gen, dtype=torch.float64)
        Q, _ = torch.linalg.qr(G)  # d x k, orthonormal columns
        vals.append(captured_variance(Xc, Q.T.to(Xc)))
    return float(np.mean(vals)), float(np.std(vals))


def analyse_layer(model, loaders, layer, protected, setting, seed, args, device):
    protection = make_protection(layer, args.k, args.alpha)

    # BARRIER setup (forget activations -> SVD -> z_min/z_max), exactly as in
    # the unlearning runs.  Re-seed so the augmented forget pass below sees the
    # same crops/flips as the one inside setup_protection.
    seed_all(seed)
    protection.setup_protection(
        model, loaders["forget"], device,
        remain_dataloader=loaders["retain"],
        forward_fn=classification_forward_fn,
    )
    info = protection.pca_info[0]
    layer_name = info["layer_name"]
    mu = info["mu"].double()
    U = info["U_forget"].double()  # k x d
    z_min, z_max = info["z_min"].double(), info["z_max"].double()
    S2_f = info["S_forget"].double().pow(2)
    S2_r = info["S_residual"].double().pow(2)
    evr_pca = float(S2_f.sum() / (S2_f.sum() + S2_r.sum()))
    k, d = U.shape

    seed_all(seed)
    A_f = collect_raw(protection, model, layer_name, loaders["forget"], device).double()
    n_forget = len(loaders["forget"].dataset)
    rows_per_sample = A_f.size(0) // n_forget
    evr_check = captured_variance(A_f - A_f.mean(0), U)
    Zf = (A_f - mu) @ U.T
    forget_box = float(((Zf >= z_min) & (Zf <= z_max)).all(1).double().mean())
    del A_f, Zf

    # Retain: enough samples for max_retain_rows rows, then exactly
    # max_retain_rows rows (fixed generator).
    retain_ds = loaders["retain"].dataset
    n_samp = min(len(retain_ds), math.ceil(args.max_retain_rows / rows_per_sample))
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(retain_ds), generator=gen)[:n_samp].tolist()
    sub_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(retain_ds, idx),
        batch_size=loaders["retain"].batch_size, shuffle=False, num_workers=0,
    )
    seed_all(seed)
    A_r = collect_raw(protection, model, layer_name, sub_loader, device).double()
    n_rows_total = A_r.size(0)
    if A_r.size(0) > args.max_retain_rows:
        keep = torch.randperm(A_r.size(0), generator=gen)[:args.max_retain_rows]
        A_r = A_r[keep]
    log.info(f"[{setting} s{seed} {layer_name}] forget rows={rows_per_sample * n_forget} "
             f"({n_forget} samples)  retain rows={A_r.size(0)} of {n_rows_total} "
             f"({n_samp} samples)")

    Xc = A_r - A_r.mean(0)
    overlap = captured_variance(Xc, U)
    rand_mean, rand_std = random_bases_overlap(Xc, k, args.n_rand, seed)
    Zr = (A_r - mu) @ U.T
    box_frac = 100.0 * float(((Zr >= z_min) & (Zr <= z_max)).all(1).double().mean())

    assert 0.0 <= overlap <= 1.0 + 1e-9, overlap
    if abs(evr_check - evr_pca) > 1e-3:
        log.warning(f"[{setting} s{seed} {layer_name}] forget captured variance "
                    f"{evr_check:.6f} != PCA EVR {evr_pca:.6f}")

    return {
        "setting": setting, "layer": layer_name, "d": d, "k": k,
        "n_forget": n_forget, "n_retain": A_r.size(0), "overlap": overlap,
        "rand_mean": rand_mean, "rand_std": rand_std, "k_over_d": k / d,
        "box_frac": box_frac, "seed": seed, "protected": int(protected),
        "n_retain_rows_total": n_rows_total, "evr_pca": evr_pca,
        "evr_check": evr_check, "forget_box_frac": 100.0 * forget_box,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--settings", nargs="+", default=[
        "classwise:configs/pipeline_classwise.yaml",
        "random:configs/pipeline_random_10pp.yaml"])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--layers", nargs="+", default=["fc"],
                   help="protected layers (paper: fc)")
    p.add_argument("--contrast_layer", default="layer4.0.conv1",
                   help="earlier, NOT protected layer reported for contrast "
                        "('' to skip)")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--alpha", type=int, default=5, choices=[1, 5, 10])
    p.add_argument("--max_retain_rows", type=int, default=10000)
    p.add_argument("--n_rand", type=int, default=10)
    p.add_argument("--out", default="results/overlap.csv")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    layers = [(l, True) for l in args.layers]
    if args.contrast_layer:
        layers.append((args.contrast_layer, False))

    rows = []
    for spec in args.settings:
        setting, cfg_path = spec.split(":", 1)
        for seed in args.seeds:
            seed_all(seed)
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            cfg["pipeline"]["seed"] = seed
            cfg["unlearn"]["method"] = "intact"
            fargs = build_args(cfg)

            model, _, val_loader, test_loader, marked_loader = \
                utils.setup_model_dataset(fargs)
            model = model.to(device)
            ckpt = torch_load(fargs.model_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt, strict=False)
            model.eval()
            loaders, _, _ = build_data_loaders(
                fargs, marked_loader, val_loader, test_loader)

            for layer, protected in layers:
                rows.append(analyse_layer(model, loaders, layer, protected,
                                          setting, seed, args, device))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    log.info(f"wrote {args.out}")

    print("\n| setting | seed | layer | protected | d | k | n_forget | n_retain "
          "| overlap | rand mean ± std | k/d | box_frac (%) | EVR pca / check |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['setting']} | {r['seed']} | {r['layer']} | "
              f"{'yes' if r['protected'] else 'no (contrast)'} | {r['d']} | {r['k']} | "
              f"{r['n_forget']} | {r['n_retain']} | {r['overlap']:.4f} | "
              f"{r['rand_mean']:.4f} ± {r['rand_std']:.4f} | {r['k_over_d']:.4f} | "
              f"{r['box_frac']:.2f} | {r['evr_pca']:.4f} / {r['evr_check']:.4f} |")


if __name__ == "__main__":
    main()
