"""
SVD Separation Scatter Plot for SD NSFW Activation Analysis
=============================================================
For each protected layer, finds the TWO SVD dimensions with the LOWEST IoU
(most separation) between forget (NSFW) and remain (non-NSFW) projected data,
and produces a 2D scatter plot.

Usage:
    cd SD
    python scripts/svd_separation_scatter.py \
        --device 0 --batch_size 4 \
        --forget_batches 50 --remain_batches 50 \
        --out_dir experiments/svd_separation
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "train-scripts"))

from InTAct.intact import UnlearnIntervalProtection
from dataset import setup_forget_nsfw_data, setup_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fractional_dataloader(dataloader, n_batches, seed=42):
    if n_batches is None or n_batches <= 0:
        return dataloader
    dataset = dataloader.dataset
    bs = dataloader.batch_size
    max_samples = min(len(dataset), n_batches * bs)
    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    sub = Subset(dataset, indices)
    return DataLoader(
        sub, batch_size=bs, shuffle=False,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=False,
    )


def sd_forward_fn(model, batch, device, prompts=None, **kwargs):
    images = batch
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], torch.Tensor):
        images, _labels = batch
    images = torch.stack([item for item in images]).to(device)
    n = images.size(0)
    txt = [prompts[0]] * n if prompts else [""] * n
    batch_dict = {"jpg": images.permute(0, 2, 3, 1), "txt": txt}
    with torch.no_grad():
        x, c = model.get_input(batch_dict, model.first_stage_key)
    t = torch.randint(0, model.num_timesteps, (n,), device=device).long()
    betas = model.betas.to(device) if hasattr(model, "betas") else None
    if betas is not None:
        e = torch.randn_like(x)
        a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x_noisy = x * a.sqrt() + e * (1.0 - a).sqrt()
    else:
        x_noisy = x
    model.model.diffusion_model(x_noisy, t.float(), context=c)


# ---------------------------------------------------------------------------
# IoU / separation metrics
# ---------------------------------------------------------------------------

def compute_1d_overlap(forget_vals, remain_vals, n_bins=80):
    """Compute IoU of 1D histograms for a single SVD dimension."""
    combined = torch.cat([forget_vals, remain_vals])
    lo, hi = combined.min().item(), combined.max().item()
    if hi <= lo:
        return 1.0
    bins = np.linspace(lo, hi, n_bins + 1)
    f_hist, _ = np.histogram(forget_vals.numpy(), bins=bins, density=True)
    r_hist, _ = np.histogram(remain_vals.numpy(), bins=bins, density=True)
    f_hist /= max(f_hist.sum(), 1e-12)
    r_hist /= max(r_hist.sum(), 1e-12)
    intersection = np.minimum(f_hist, r_hist).sum()
    union = np.maximum(f_hist, r_hist).sum()
    if union < 1e-12:
        return 1.0
    return float(intersection / union)


def compute_2d_iou(forget_2d, remain_2d, n_bins=50):
    """Compute IoU of 2D binned projections for a pair of SVD dimensions."""
    f = forget_2d.numpy()
    r = remain_2d.numpy()
    combined = np.concatenate([f, r], axis=0)
    x_lo, x_hi = combined[:, 0].min(), combined[:, 0].max()
    y_lo, y_hi = combined[:, 1].min(), combined[:, 1].max()
    if x_hi <= x_lo or y_hi <= y_lo:
        return 1.0
    bins_x = np.linspace(x_lo, x_hi, n_bins + 1)
    bins_y = np.linspace(y_lo, y_hi, n_bins + 1)
    f_hist, _, _ = np.histogram2d(f[:, 0], f[:, 1], bins=[bins_x, bins_y], density=True)
    r_hist, _, _ = np.histogram2d(r[:, 0], r[:, 1], bins=[bins_x, bins_y], density=True)
    f_hist /= max(f_hist.sum(), 1e-12)
    r_hist /= max(r_hist.sum(), 1e-12)
    intersection = np.minimum(f_hist, r_hist).sum()
    union = np.maximum(f_hist, r_hist).sum()
    if union < 1e-12:
        return 1.0
    return float(intersection / union)


def find_best_dims(forget_proj, remain_proj, pca_info, top_k_1d=10, n_bins_2d=50):
    """
    For each layer: compute 1D overlap per dim, take top_k_1d candidates,
    then search all pairs among them for lowest 2D IoU.

    Returns list of (layer_name, dim_a, dim_b, iou_2d, iou_1d_a, iou_1d_b)
    """
    results = []
    for entry in pca_info:
        name = entry["layer_name"]
        fproj = forget_proj.get(name)
        rproj = remain_proj.get(name)
        if fproj is None or rproj is None or fproj.size(1) < 2:
            continue

        k_dims = fproj.size(1)
        log.info("  %s: %d SVD dims, %d forget tokens, %d remain tokens",
                 name[-50:], k_dims, fproj.size(0), rproj.size(0))

        # 1D overlap per dimension
        overlaps_1d = []
        for d in range(k_dims):
            ov = compute_1d_overlap(fproj[:, d], rproj[:, d])
            overlaps_1d.append((d, ov))
        overlaps_1d.sort(key=lambda x: x[1])

        # Take top_k_1d candidates with LOWEST overlap (most separated)
        candidates = [d for d, _ in overlaps_1d[:top_k_1d]]
        log.info("    top-%d 1D candidate dims: %s", len(candidates), candidates)

        # 2D IoU for all pairs among candidates
        best_iou = 1.0
        best_pair = (0, 1)
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                di, dj = candidates[i], candidates[j]
                iou = compute_2d_iou(
                    fproj[:, [di, dj]], rproj[:, [di, dj]], n_bins=n_bins_2d,
                )
                if iou < best_iou:
                    best_iou = iou
                    best_pair = (di, dj)

        log.info("    best pair: dims (%d, %d)  IoU_2d=%.4f", best_pair[0], best_pair[1], best_iou)
        results.append({
            "layer_name": name,
            "dim_a": best_pair[0],
            "dim_b": best_pair[1],
            "iou_2d": best_iou,
            "iou_1d_a": overlaps_1d[best_pair[0]][1] if best_pair[0] < len(overlaps_1d) else 1.0,
            "iou_1d_b": overlaps_1d[best_pair[1]][1] if best_pair[1] < len(overlaps_1d) else 1.0,
            "all_1d_overlaps": [{"dim": d, "iou_1d": ov} for d, ov in overlaps_1d],
            "k_dims": k_dims,
        })

    return results


# ---------------------------------------------------------------------------
# Zone occupancy for the best pair of dims
# ---------------------------------------------------------------------------

ZONE_LABELS = ["inside_box", "neg_inf", "pos_inf", "outside"]

def compute_zone_fractions_pair(fproj_2d, rproj_2d, z_min_2d, z_max_2d, inf_low_2d, inf_high_2d):
    """
    Classify each 2D point into one of 4 zones relative to the forget box.

        inside_box  :  z_min <= val <= z_max  (for BOTH dimensions)
        neg_inf     :  in data range but at least one dim < z_min
        pos_inf     :  in data range but at least one dim > z_max
        outside     :  at least one dim outside inf_low/inf_high

    Returns dict with per-zone fraction (0-1) for forget and remain.
    """
    def classify_2d(points, z_min, z_max, inf_low, inf_high):
        inside = (
            (points[:, 0] >= z_min[0]) & (points[:, 0] <= z_max[0]) &
            (points[:, 1] >= z_min[1]) & (points[:, 1] <= z_max[1])
        )
        in_data = (
            (points[:, 0] >= inf_low[0]) & (points[:, 0] <= inf_high[0]) &
            (points[:, 1] >= inf_low[1]) & (points[:, 1] <= inf_high[1])
        )
        outside = ~in_data
        neg = in_data & ~inside & (
            (points[:, 0] < z_min[0]) | (points[:, 1] < z_min[1])
        )
        pos = in_data & ~inside & ~neg

        n = points.shape[0]
        return {
            "inside_box": inside.float().mean().item(),
            "neg_inf": neg.float().mean().item(),
            "pos_inf": pos.float().mean().item(),
            "outside": outside.float().mean().item(),
        }

    f_zones = classify_2d(fproj_2d, z_min_2d, z_max_2d, inf_low_2d, inf_high_2d)
    r_zones = classify_2d(rproj_2d, z_min_2d, z_max_2d, inf_low_2d, inf_high_2d)
    f_zones["margin"] = f_zones["neg_inf"] + f_zones["pos_inf"]
    r_zones["margin"] = r_zones["neg_inf"] + r_zones["pos_inf"]
    return {"forget": f_zones, "remain": r_zones}


def compute_all_zone_fractions(forget_proj, remain_proj, pca_info, best_dims):
    """Add zone_occupancy to each best_dims entry."""
    for entry, best in zip(pca_info, best_dims):
        name = entry["layer_name"]
        fproj = forget_proj[name]
        rproj = remain_proj[name]
        di, dj = best["dim_a"], best["dim_b"]

        z_min = entry["z_min"]
        z_max = entry["z_max"]
        inf_low = fproj.min(dim=0)[0]
        inf_high = fproj.max(dim=0)[0]

        f_2d = fproj[:, [di, dj]]
        r_2d = rproj[:, [di, dj]]

        zones = compute_zone_fractions_pair(
            f_2d, r_2d,
            torch.stack([z_min[di], z_min[dj]]),
            torch.stack([z_max[di], z_max[dj]]),
            torch.stack([inf_low[di], inf_low[dj]]),
            torch.stack([inf_high[di], inf_high[dj]]),
        )
        best["zone_occupancy"] = zones

    return best_dims


# ---------------------------------------------------------------------------
# Zone occupancy across ALL SVD dimensions (per-dim, then mean)
# ---------------------------------------------------------------------------

def compute_all_dim_zone_means(forget_proj, remain_proj, pca_info):
    """
    For each layer, compute per-dimension zone classification across ALL SVD
    dims, then aggregate means across layers.
    """
    per_layer_all = []
    all_f_ins, all_f_mar, all_f_out = [], [], []
    all_r_ins, all_r_mar, all_r_out = [], [], []

    for entry in pca_info:
        name = entry["layer_name"]
        fproj = forget_proj.get(name)
        rproj = remain_proj.get(name)
        if fproj is None or rproj is None:
            continue

        z_min = entry["z_min"]
        z_max = entry["z_max"]
        inf_low = fproj.min(dim=0)[0]
        inf_high = fproj.max(dim=0)[0]
        k = fproj.size(1)

        f_ins_dims, f_mar_dims, f_out_dims = [], [], []
        r_ins_dims, r_mar_dims, r_out_dims = [], [], []

        for d in range(k):
            fd = fproj[:, d]
            rd = rproj[:, d]
            il = inf_low[d]; ih = inf_high[d]
            zmn = z_min[d]; zmx = z_max[d]

            f_in = ((fd >= zmn) & (fd <= zmx)).float().mean().item()
            f_neg = ((fd >= il) & (fd < zmn)).float().mean().item()
            f_pos = ((fd > zmx) & (fd <= ih)).float().mean().item()
            f_out = ((fd < il) | (fd > ih)).float().mean().item()

            r_in = ((rd >= zmn) & (rd <= zmx)).float().mean().item()
            r_neg = ((rd >= il) & (rd < zmn)).float().mean().item()
            r_pos = ((rd > zmx) & (rd <= ih)).float().mean().item()
            r_out = ((rd < il) | (rd > ih)).float().mean().item()

            f_ins_dims.append(f_in)
            f_mar_dims.append(f_neg + f_pos)
            f_out_dims.append(f_out)
            r_ins_dims.append(r_in)
            r_mar_dims.append(r_neg + r_pos)
            r_out_dims.append(r_out)

        layer_means = {
            "layer_name": name,
            "k_dims": k,
            "forget_inside": float(np.mean(f_ins_dims)),
            "forget_margin": float(np.mean(f_mar_dims)),
            "forget_outside": float(np.mean(f_out_dims)),
            "remain_inside": float(np.mean(r_ins_dims)),
            "remain_margin": float(np.mean(r_mar_dims)),
            "remain_outside": float(np.mean(r_out_dims)),
        }
        per_layer_all.append(layer_means)

        all_f_ins.extend(f_ins_dims)
        all_f_mar.extend(f_mar_dims)
        all_f_out.extend(f_out_dims)
        all_r_ins.extend(r_ins_dims)
        all_r_mar.extend(r_mar_dims)
        all_r_out.extend(r_out_dims)

    global_means = {
        "n_layers": len(per_layer_all),
        "n_dims_total": len(all_f_ins),
        "forget_inside": float(np.mean(all_f_ins)),
        "forget_margin": float(np.mean(all_f_mar)),
        "forget_outside": float(np.mean(all_f_out)),
        "remain_inside": float(np.mean(all_r_ins)),
        "remain_margin": float(np.mean(all_r_mar)),
        "remain_outside": float(np.mean(all_r_out)),
    }

    log.info("\n" + "=" * 60)
    log.info("Zone occupancy across ALL %d SVD dims, %d layers:", len(all_f_ins) // len(all_f_mar) if all_f_mar else 0, len(per_layer_all))
    log.info("                 Forget    Remain")
    log.info("  inside_box      %5.1f%%    %5.1f%%", global_means["forget_inside"] * 100,
             global_means["remain_inside"] * 100)
    log.info("  margin (neg+pos)%5.1f%%    %5.1f%%", global_means["forget_margin"] * 100,
             global_means["remain_margin"] * 100)
    log.info("  outside         %5.1f%%    %5.1f%%", global_means["forget_outside"] * 100,
             global_means["remain_outside"] * 100)
    log.info("=" * 60)

    return {"global": global_means, "per_layer": per_layer_all}


def print_zone_summary(best_dims):
    """Print a clean table of zone occupancy per layer + mean across layers."""
    header = f"{'Layer':<45s} {'Data':>7s} {'inside_box':>10s} {'margin':>8s} {'outside':>8s}"
    sep = "-" * len(header)
    log.info("\n" + sep)
    log.info(header)
    log.info(sep)

    f_ins, f_mar, f_out = [], [], []
    r_ins, r_mar, r_out = [], [], []

    for best in best_dims:
        name = best["layer_name"][-44:]
        zones = best.get("zone_occupancy", {})
        for key in ["forget", "remain"]:
            z = zones.get(key, {})
            m = z.get("margin", z.get("neg_inf", 0) + z.get("pos_inf", 0))
            o = z.get("outside", 0)
            ins = z.get("inside_box", 0)
            log.info(
                f"{name:<45s} {key:>7s} "
                f"{ins:>10.2%} {m:>8.2%} "
                f"{o:>8.2%}"
            )
            if key == "forget":
                f_ins.append(ins); f_mar.append(m); f_out.append(o)
            else:
                r_ins.append(ins); r_mar.append(m); r_out.append(o)
        log.info(sep)

    log.info(f"{'MEAN (n=' + str(len(best_dims)) + ' layers)':<45s} {'':>7s} {'':>10s} {'':>8s} {'':>8s}")
    log.info(f"{'':<45s} {'Forget':>7s} {np.mean(f_ins):>10.1%} {np.mean(f_mar):>8.1%} {np.mean(f_out):>8.1%}")
    log.info(f"{'':<45s} {'Remain':>7s} {np.mean(r_ins):>10.1%} {np.mean(r_mar):>8.1%} {np.mean(r_out):>8.1%}")
    log.info(sep)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_separation_scatters(forget_proj, remain_proj, pca_info, best_dims, out_dir):
    """One scatter per layer using the two most-separated SVD dimensions — zoomed only, no titles."""
    for entry, best in zip(pca_info, best_dims):
        name = entry["layer_name"]
        fproj = forget_proj[name]
        rproj = remain_proj[name]
        di, dj = best["dim_a"], best["dim_b"]

        f = fproj[:, [di, dj]].numpy()
        r = rproj[:, [di, dj]].numpy()

        max_pts = 3000
        if f.shape[0] > max_pts:
            f = f[np.random.choice(f.shape[0], max_pts, replace=False)]
        if r.shape[0] > max_pts:
            r = r[np.random.choice(r.shape[0], max_pts, replace=False)]

        z_min = entry["z_min"]
        z_max = entry["z_max"]
        inf_low = fproj.min(dim=0)[0]
        inf_high = fproj.max(dim=0)[0]

        fig, ax = plt.subplots(figsize=(7, 6))

        ax.scatter(f[:, 0], f[:, 1], s=5, alpha=0.45, color="black", label="Forget (NSFW)")
        ax.scatter(r[:, 0], r[:, 1], s=5, alpha=0.45, color="#1f77b4", label="Remain (SFW)")
        ax.add_patch(Rectangle(
            (z_min[di].item(), z_min[dj].item()),
            z_max[di].item() - z_min[di].item(),
            z_max[dj].item() - z_min[dj].item(),
            fill=False, edgecolor="darkorange", lw=1.5, ls="--",
        ))
        ax.add_patch(Rectangle(
            (inf_low[di].item(), inf_low[dj].item()),
            inf_high[di].item() - inf_low[di].item(),
            inf_high[dj].item() - inf_low[dj].item(),
            fill=False, edgecolor="gray", lw=1.0, ls=":",
        ))
        ax.legend(loc="upper right", fontsize=8, markerscale=2)

        pad_x = (inf_high[di].item() - inf_low[di].item()) * 0.15
        pad_y = (inf_high[dj].item() - inf_low[dj].item()) * 0.15
        ax.set_xlim(inf_low[di].item() - pad_x, inf_high[di].item() + pad_x)
        ax.set_ylim(inf_low[dj].item() - pad_y, inf_high[dj].item() + pad_y)
        ax.set_xlabel(f"SVD dim {di + 1}")
        ax.set_ylabel(f"SVD dim {dj + 1}")

        plt.tight_layout()

        safe_name = name.replace(".", "_").replace("/", "_")[-80:]
        path = os.path.join(out_dir, f"svd_separation_{safe_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info("  Saved %s", path)


def plot_iou_rankings(best_dims, out_dir):
    """Horizontal bar chart of per-dim 1D IoU for each layer."""
    n_layers = len(best_dims)
    if n_layers == 0:
        return
    fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 5), squeeze=False)
    axes = axes[0]

    for ax_idx, best in enumerate(best_dims):
        ax = axes[ax_idx]
        all_overlaps = best["all_1d_overlaps"]
        dims = [d["dim"] + 1 for d in all_overlaps]
        ious = [d["iou_1d"] for d in all_overlaps]
        colors = []
        for d in all_overlaps:
            if d["dim"] == best["dim_a"] or d["dim"] == best["dim_b"]:
                colors.append("#d62728")
            else:
                colors.append("#7f7f7f")
        ax.barh(dims, ious, color=colors, height=0.7)
        ax.axvline(best["iou_2d"], color="#d62728", ls="--", lw=1.2, label=f"IoU_2d={best['iou_2d']:.3f}")
        ax.set_xlabel("1D IoU (lower = more separated)")
        ax.set_ylabel("SVD dimension")
        ax.set_title(f"{best['layer_name'][-40:]}")
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, "iou_rankings.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SVD separation scatter: best 2 dims per layer by lowest IoU"
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--config_path", type=str,
                        default="configs/stable-diffusion/v1-intact.yaml")
    parser.add_argument("--ckpt_path", type=str,
                        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--nsfw_data_path", type=str, default="data/nsfw")
    parser.add_argument("--not_nsfw_data_path", type=str, default="data/not-nsfw")

    parser.add_argument("--targets", type=str, nargs="+",
                        default=["attn2.to_q", "attn2.to_k", "attn2.to_v",
                                 "attn2.to_out.0"])
    parser.add_argument("--reduced_dim", type=int, default=32)
    parser.add_argument("--lower_percentile", type=float, default=0.05)
    parser.add_argument("--upper_percentile", type=float, default=0.95)

    parser.add_argument("--forget_batches", type=int, default=50)
    parser.add_argument("--remain_batches", type=int, default=50)
    parser.add_argument("--svd_batches", type=int, default=50)

    parser.add_argument("--use_actual_bounds", action="store_true", default=True,
                        help="Use remain data to compute actual bounds (default: True)")
    parser.add_argument("--no_actual_bounds", dest="use_actual_bounds", action="store_false",
                        help="Use forget-only bounds")
    parser.add_argument("--top_k_1d", type=int, default=10,
                        help="Number of best 1D dims to search for 2D pairs")
    parser.add_argument("--n_bins_2d", type=int, default=50,
                        help="Number of bins per axis for 2D IoU histogram")

    parser.add_argument("--out_dir", type=str, default="experiments/svd_separation")

    args = parser.parse_args()

    device = f"cuda:{args.device}"
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    log.info("Loading SD model...")
    model = setup_model(args.config_path, args.ckpt_path, device)
    model = model.to(device)
    if hasattr(model, "logvar"):
        model.logvar = model.logvar.to(device)

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    log.info("Loading NSFW forget + remain data...")
    forget_dl, remain_dl = setup_forget_nsfw_data(
        args.batch_size, args.image_size,
        nsfw_data_path=args.nsfw_data_path,
        not_nsfw_data_path=args.not_nsfw_data_path,
    )

    forget_svd_dl = make_fractional_dataloader(forget_dl, args.svd_batches)
    forget_collect_dl = make_fractional_dataloader(forget_dl, args.forget_batches)
    remain_collect_dl = make_fractional_dataloader(remain_dl, args.remain_batches)

    log.info("SVD batches: %d  |  Forget batches: %d  |  Remain batches: %d",
             len(forget_svd_dl), len(forget_collect_dl), len(remain_collect_dl))

    # ------------------------------------------------------------------
    # 3. InTAct setup_protection() — SVD on forget data
    # ------------------------------------------------------------------
    log.info("Setting up InTAct (SVD on forget data)...")
    protection = UnlearnIntervalProtection(
        targets=args.targets,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        reduced_dim=args.reduced_dim,
        infinity_scale=20.0,
        use_actual_bounds=args.use_actual_bounds,
    )

    word_nude = "a photo of a nude person"
    word_wear = "a photo of a person wearing clothes"
    descriptions = [word_nude, word_wear]

    def forward_fn(diffusion_model, batch, dev, **kwargs):
        return sd_forward_fn(model, batch, dev, prompts=descriptions, **kwargs)

    protection.setup_protection(
        model.model.diffusion_model,
        forget_svd_dl,
        device,
        remain_dataloader=remain_collect_dl if args.use_actual_bounds else None,
        forward_fn=forward_fn,
        betas=model.betas.to(device) if hasattr(model, "betas") else None,
        num_timesteps=model.num_timesteps if hasattr(model, "num_timesteps") else 1000,
    )

    if not protection.pca_info:
        log.error("No pca_info built. Check --targets patterns.")
        return

    target_layer_names = [e["layer_name"] for e in protection.pca_info]
    log.info("%d protected layers: %s", len(target_layer_names),
             [n[-40:] for n in target_layer_names])

    # ------------------------------------------------------------------
    # 4. Project forget + remain into SVD subspace
    # ------------------------------------------------------------------
    pca_components = {}
    for entry in protection.pca_info:
        pca_components[entry["layer_name"]] = {
            "mu": entry["mu"].to(device),
            "U_forget": entry["U_forget"].to(device),
            "layer_type": entry.get("layer_type", "Linear"),
        }

    diffusion_model = model.model.diffusion_model

    def forget_forward_fn(_m, batch, dev, **kw):
        return sd_forward_fn(model, batch, dev, prompts=[word_nude], **kw)

    log.info("Collecting & projecting FORGET activations...")
    forget_proj_raw = protection._collect_activations(
        diffusion_model, target_layer_names, forget_collect_dl, device,
        forward_fn=forget_forward_fn,
        betas=model.betas.to(device) if hasattr(model, "betas") else None,
        num_timesteps=model.num_timesteps if hasattr(model, "num_timesteps") else 1000,
        pca_components=pca_components,
    )
    forget_proj = dict(forget_proj_raw)

    def remain_forward_fn(_m, batch, dev, **kw):
        return sd_forward_fn(model, batch, dev, prompts=[word_wear], **kw)

    log.info("Collecting & projecting REMAIN activations...")
    remain_proj_raw = protection._collect_activations(
        diffusion_model, target_layer_names, remain_collect_dl, device,
        forward_fn=remain_forward_fn,
        betas=model.betas.to(device) if hasattr(model, "betas") else None,
        num_timesteps=model.num_timesteps if hasattr(model, "num_timesteps") else 1000,
        pca_components=pca_components,
    )
    remain_proj = dict(remain_proj_raw)

    # ------------------------------------------------------------------
    # 5. Find best 2 dims per layer (lowest 2D IoU)
    # ------------------------------------------------------------------
    log.info("Finding best separation dimensions per layer...")
    best_dims = find_best_dims(
        forget_proj, remain_proj, protection.pca_info,
        top_k_1d=args.top_k_1d, n_bins_2d=args.n_bins_2d,
    )

    # ------------------------------------------------------------------
    # 6. Zone occupancy for the best pair
    # ------------------------------------------------------------------
    log.info("Computing zone occupancy on best dim pair...")
    best_dims = compute_all_zone_fractions(
        forget_proj, remain_proj, protection.pca_info, best_dims,
    )
    print_zone_summary(best_dims)

    # ------------------------------------------------------------------
    # 7. Zone occupancy across ALL SVD dimensions
    # ------------------------------------------------------------------
    log.info("Computing zone occupancy across ALL SVD dimensions...")
    all_dim_means = compute_all_dim_zone_means(forget_proj, remain_proj, protection.pca_info)

    # Save to JSON
    f_ins = [b["zone_occupancy"]["forget"]["inside_box"] for b in best_dims]
    f_mar = [b["zone_occupancy"]["forget"]["margin"] for b in best_dims]
    f_out = [b["zone_occupancy"]["forget"]["outside"] for b in best_dims]
    r_ins = [b["zone_occupancy"]["remain"]["inside_box"] for b in best_dims]
    r_mar = [b["zone_occupancy"]["remain"]["margin"] for b in best_dims]
    r_out = [b["zone_occupancy"]["remain"]["outside"] for b in best_dims]
    ious = [b["iou_2d"] for b in best_dims]

    json_data = {
        "config": {
            "targets": args.targets,
            "reduced_dim": args.reduced_dim,
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "top_k_1d": args.top_k_1d,
            "n_bins_2d": args.n_bins_2d,
            "use_actual_bounds": args.use_actual_bounds,
        },
        "means": {
            "n_layers": len(best_dims),
            "forget_inside_box": float(np.mean(f_ins)),
            "forget_margin": float(np.mean(f_mar)),
            "forget_outside": float(np.mean(f_out)),
            "remain_inside_box": float(np.mean(r_ins)),
            "remain_margin": float(np.mean(r_mar)),
            "remain_outside": float(np.mean(r_out)),
            "iou_2d_mean": float(np.mean(ious)),
            "iou_2d_median": float(np.median(ious)),
        },
        "all_dims": all_dim_means,
        "per_layer": best_dims,
    }
    json_path = os.path.join(args.out_dir, "svd_separation.json")
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    log.info("Saved metrics -> %s", json_path)

    # ------------------------------------------------------------------
    # 6. Plots
    # ------------------------------------------------------------------
    log.info("Generating separation scatter plots...")
    plot_separation_scatters(forget_proj, remain_proj, protection.pca_info,
                             best_dims, args.out_dir)
    plot_iou_rankings(best_dims, args.out_dir)

    log.info("Done. All outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
