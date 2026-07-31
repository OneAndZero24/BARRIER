"""
Activation Space Occupancy Analysis for BARRIER NSFW Unlearning
================================================================

Maps where *forget* and *remain* data activations land in the SVD subspace
defined by BARRIER/InTAct — **without modifying any existing method code**.

Four occupancy zones (per SVD dimension), all estimated from DATA:
  1.  Below range    :  x <  inf_low       (outside observed forget data)
  2.  Negative space :  inf_low ≤ x < z_min   (below forget box, inside data range)
  3.  Inside box     :  z_min  ≤ x ≤ z_max   (the forget box — 5th/95th percentiles)
  4.  Positive space :  z_max  < x ≤ inf_high (above forget box, inside data range)
  5.  Above range    :  x >  inf_high      (outside observed forget data)

where inf_low/inf_high = min/max of projected forget data (observed data range),
and z_min/z_max = lower_percentile/upper_percentile from InTAct SVD.

Outputs:
  - activation_analysis.json     : per-layer zone fractions
  - density_histograms.png       : forget vs remain distributions + all bounds
  - zone_breakdown.png           : stacked-bar: fraction in each zone per layer
  - svd_scatter.png              : 2-D scatter with inner + outer box
  - per_dim_remain_zones.png     : per-dimension zone breakdown for remain

Usage:
    cd SD
    python scripts/activation_space_analysis.py \
        --device 0 --batch_size 4 \
        --forget_batches 50 --remain_batches 50 \
        --out_dir experiments/activation_space
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
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # InTAct
sys.path.insert(0, str(Path(__file__).parent.parent))          # SD root
sys.path.insert(0, str(Path(__file__).parent.parent / "train-scripts"))

from InTAct.intact import UnlearnIntervalProtection
from dataset import setup_forget_nsfw_data, setup_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers  (no InTAct reimplementation — uses existing UnlearnIntervalProtection)
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
    """Exact copy of intact_unlearn.sd_forward_fn — no modifications."""
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
# Zone occupancy (5-zone classification)
# ---------------------------------------------------------------------------

ZONE_NAMES = ["below_range", "negative_space", "inside_box",
              "positive_space", "above_range"]

def compute_zone_fractions(projected, z_min, z_max, inf_low, inf_high):
    """
    Returns both dimension-level counts AND per-token mean overlap fractions.

    Per-token overlap: for a token with k SVD dimensions, what fraction of
    those k dims fall into each zone?  Averaged across all tokens.

    This directly answers:
      - "Is forget in the forget box?" → mean_inside_box ≈ 0.90
      - "Is remain in negative space?"  → mean_negative_space (ideally high)
    """
    k = projected.size(1)
    n = projected.size(0)

    below  = (projected < inf_low.unsqueeze(0)).float()
    neg    = ((projected >= inf_low.unsqueeze(0)) & (projected < z_min.unsqueeze(0))).float()
    inside = ((projected >= z_min.unsqueeze(0))  & (projected <= z_max.unsqueeze(0))).float()
    pos    = ((projected > z_max.unsqueeze(0))   & (projected <= inf_high.unsqueeze(0))).float()
    above  = (projected > inf_high.unsqueeze(0)).float()

    # dim-level: total counts per SVD dimension
    dim_counts = {
        "below_range":     below.sum(dim=0).tolist(),
        "negative_space":  neg.sum(dim=0).tolist(),
        "inside_box":      inside.sum(dim=0).tolist(),
        "positive_space":  pos.sum(dim=0).tolist(),
        "above_range":     above.sum(dim=0).tolist(),
    }

    # per-token: fraction of its k dims in each zone, then mean across tokens
    # negative_space + positive_space = combined "negative space" outside box
    mean_overlap = {
        "below_range":     below.mean(dim=1).mean().item(),
        "negative_space":  neg.mean(dim=1).mean().item(),
        "inside_box":      inside.mean(dim=1).mean().item(),
        "positive_space":  pos.mean(dim=1).mean().item(),
        "above_range":     above.mean(dim=1).mean().item(),
        "negative_combined": (neg + pos).mean(dim=1).mean().item(),
        "outside_data":     (below + above).mean(dim=1).mean().item(),
    }

    return dim_counts, mean_overlap


def occupancy_summary(forget_proj, remain_proj, pca_info):
    """Per-layer zone breakdown for both forget and remain."""
    summary = {}
    for entry in pca_info:
        name = entry["layer_name"]
        fproj = forget_proj.get(name)
        rproj = remain_proj.get(name)
        if fproj is None or rproj is None:
            continue

        z_min = entry["z_min"]
        z_max = entry["z_max"]
        inf_low = fproj.min(dim=0)[0]    # data-estimated lower bound
        inf_high = fproj.max(dim=0)[0]   # data-estimated upper bound

        f_dim, f_overlap = compute_zone_fractions(fproj, z_min, z_max, inf_low, inf_high)
        r_dim, r_overlap = compute_zone_fractions(rproj, z_min, z_max, inf_low, inf_high)

        summary[name] = {
            "n_forget_tokens":  int(fproj.size(0)),
            "n_remain_tokens":  int(rproj.size(0)),
            "k_dims":           int(fproj.size(1)),
            "z_min": z_min.tolist(),
            "z_max": z_max.tolist(),
            "inf_low": inf_low.tolist(),
            "inf_high": inf_high.tolist(),
            "forget_dim_counts": f_dim,
            "remain_dim_counts": r_dim,
            "forget_mean_overlap": f_overlap,
            "remain_mean_overlap": r_overlap,
        }

        log.info(
            "%s  forget: inside=%.2f  neg=%.2f  out=%.2f  |  remain: inside=%.2f  neg=%.2f  out=%.2f",
            name,
            f_overlap["inside_box"], f_overlap["negative_combined"], f_overlap["outside_data"],
            r_overlap["inside_box"], r_overlap["negative_combined"], r_overlap["outside_data"],
        )
    return summary


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_density_histograms(forget_proj, remain_proj, pca_info, out_dir):
    """Forget vs remain density per SVD dim, with z_min/z_max + inf_low/inf_high."""
    n_layers = len(pca_info)
    if n_layers == 0:
        return
    k = min(5, pca_info[0]["z_min"].size(0))
    fig, axes = plt.subplots(n_layers, k, figsize=(4 * k, 4 * n_layers), squeeze=False)
    for li, entry in enumerate(pca_info):
        name = entry["layer_name"]
        z_min, z_max = entry["z_min"], entry["z_max"]
        fproj = forget_proj[name]
        rproj = remain_proj[name]
        inf_low = fproj.min(dim=0)[0]
        inf_high = fproj.max(dim=0)[0]
        for di in range(k):
            ax = axes[li, di]
            ax.hist(fproj[:, di].numpy(), bins=80, density=True,
                    alpha=0.5, color="red", label="Forget")
            ax.hist(rproj[:, di].numpy(), bins=80, density=True,
                    alpha=0.5, color="blue", label="Remain")
            ax.axvline(z_min[di].item(), color="darkorange", ls="--", lw=1.2)
            ax.axvline(z_max[di].item(), color="darkorange", ls="--", lw=1.2,
                       label="z_min/z_max")
            ax.axvline(inf_low[di].item(), color="gray", ls=":", lw=0.8)
            ax.axvline(inf_high[di].item(), color="gray", ls=":", lw=0.8,
                       label="inf_low/high")
            ax.set_title(f"{name[-40:]}\nSVD dim {di+1}")
            if di == 0 and li == 0:
                ax.legend(fontsize=5)
    plt.tight_layout()
    path = os.path.join(out_dir, "density_histograms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_zone_breakdown(summary, out_dir):
    """Stacked-bar: per-layer distribution of remain tokens across 5 zones (dim-level)."""
    layers = list(summary.keys())
    short = [l[-35:] for l in layers]
    n = len(layers)
    zone_colors = {
        "below_range":    "#d62728",   # red
        "negative_space": "#ff7f0e",   # orange
        "inside_box":     "#2ca02c",   # green
        "positive_space": "#1f77b4",   # blue
        "above_range":    "#9467bd",   # purple
    }
    zone_order = ["below_range", "negative_space", "inside_box",
                  "positive_space", "above_range"]

    fig, axes = plt.subplots(1, 2, figsize=(max(10, n * 0.7), 6),
                             sharey=True)

    for ax_idx, (label, key) in enumerate([("FORGET", "forget_dim_counts"),
                                            ("REMAIN", "remain_dim_counts")]):
        ax = axes[ax_idx]
        bottom = np.zeros(n)
        for zone in zone_order:
            vals = []
            for l in layers:
                counts = summary[l][key][zone]
                total = sum(summary[l][key][z] for z in ZONE_NAMES)
                vals.append(sum(counts) / max(total, 1))
            ax.barh(short, vals, left=bottom, color=zone_colors[zone],
                    label=zone.replace("_", " "), height=0.7)
            bottom += np.array(vals)
        ax.set_title(f"{label} — dim-level zone distribution")
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        if ax_idx == 1:
            ax.legend(fontsize=7, loc="lower right", ncol=2)

    plt.tight_layout()
    path = os.path.join(out_dir, "zone_breakdown.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_mean_overlap_bars(summary, out_dir):
    """Paired bars: per-token mean fraction of dims in each zone (inside / negative / outside)."""
    layers = list(summary.keys())
    short = [l[-35:] for l in layers]
    n = len(layers)

    categories = ["inside_box", "negative_combined", "outside_data"]
    cat_colors = {"inside_box": "#2ca02c", "negative_combined": "#ff7f0e",
                  "outside_data": "#d62728"}
    cat_labels = ["Inside box", "Negative space", "Outside data"]

    fig, axes = plt.subplots(1, 2, figsize=(max(12, n * 0.8), 6), sharey=True)

    for ax_idx, (label, key) in enumerate([("FORGET", "forget_mean_overlap"),
                                            ("REMAIN", "remain_mean_overlap")]):
        ax = axes[ax_idx]
        x = np.arange(n)
        w = 0.22
        for i, cat in enumerate(categories):
            vals = [summary[l][key][cat] for l in layers]
            ax.bar(x + (i - 1) * w, vals, w, color=cat_colors[cat],
                   label=cat_labels[i])
        ax.set_xticks(x)
        ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{label} — mean fraction of dims per token")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fraction of SVD dimensions")
        if ax_idx == 1:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "mean_overlap_bars.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


def plot_svd_scatter(forget_proj, remain_proj, pca_info, out_dir):
    """2-D scatter with inner box (z_min/z_max) and outer bounds (inf_low/inf_high)."""
    for entry in pca_info:
        name = entry["layer_name"]
        fproj = forget_proj.get(name)
        rproj = remain_proj.get(name)
        if fproj is None or rproj is None or fproj.size(1) < 2:
            continue
        z_min, z_max = entry["z_min"], entry["z_max"]
        inf_low = fproj.min(dim=0)[0]
        inf_high = fproj.max(dim=0)[0]

        f = fproj[:, :2].numpy()
        r = rproj[:, :2].numpy()
        max_pts = 2000
        if f.shape[0] > max_pts:
            f = f[np.random.choice(f.shape[0], max_pts, replace=False)]
        if r.shape[0] > max_pts:
            r = r[np.random.choice(r.shape[0], max_pts, replace=False)]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(f[:, 0], f[:, 1], s=2, alpha=0.4, color="red", label="Forget")
        ax.scatter(r[:, 0], r[:, 1], s=2, alpha=0.4, color="blue", label="Remain")
        # Inner box (forget box)
        from matplotlib.patches import Rectangle
        inner = Rectangle((z_min[0].item(), z_min[1].item()),
                          z_max[0].item() - z_min[0].item(),
                          z_max[1].item() - z_min[1].item(),
                          fill=False, edgecolor="darkorange", lw=1.5, ls="--")
        outer = Rectangle((inf_low[0].item(), inf_low[1].item()),
                          inf_high[0].item() - inf_low[0].item(),
                          inf_high[1].item() - inf_low[1].item(),
                          fill=False, edgecolor="gray", lw=1.0, ls=":")
        ax.add_patch(inner)
        ax.add_patch(outer)
        ax.legend(["Forget", "Remain", "z_min/z_max", "inf_low/high"],
                  loc="upper right", fontsize=8)
        ax.set_xlabel("SVD dim 1"); ax.set_ylabel("SVD dim 2")
        ax.set_title(f"Activation space (SVD dims 1-2)\n{name[-60:]}")
        plt.tight_layout()
        path = os.path.join(out_dir, "svd_scatter.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info("Saved %s", path)
        return
    log.warning("No layer with >=2 SVD dims for scatter plot")


def plot_per_dim_remain_zones(summary, out_dir):
    """Per SVD dimension: what fraction of remain tokens fall in each zone."""
    zone_order = ["below_range", "negative_space", "inside_box",
                  "positive_space", "above_range"]
    zone_colors = {"below_range": "#d62728", "negative_space": "#ff7f0e",
                   "inside_box": "#2ca02c", "positive_space": "#1f77b4",
                   "above_range": "#9467bd"}

    fig, axes = plt.subplots(1, len(zone_order), figsize=(16, 5),
                             sharey=True)
    for ax, zone in zip(axes, zone_order):
        points_x = []
        points_y = []
        for li, (name, s) in enumerate(summary.items()):
            counts = np.array(s["remain_dim_counts"][zone])
            n_tokens = s["n_remain_tokens"]
            if n_tokens == 0:
                continue
            fracs = counts / n_tokens
            for di in range(len(fracs)):
                points_x.append(di + li * 0.15)
                points_y.append(fracs[di])
        ax.scatter(points_x, points_y, s=10, alpha=0.6, color=zone_colors[zone])
        ax.set_title(zone.replace("_", " "), fontsize=9)
        ax.set_xlabel("dim index (offset)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Fraction of remain tokens")
    fig.suptitle("Per-dimension zone occupancy for REMAIN data", fontsize=12)
    plt.tight_layout()
    path = os.path.join(out_dir, "per_dim_remain_zones.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BARRIER activation space occupancy analysis (forget vs remain)"
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

    parser.add_argument("--out_dir", type=str, default="experiments/activation_space")

    args = parser.parse_args()

    device = f"cuda:{args.device}"
    os.makedirs(args.out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1.  Load model  (identical to intact_unlearn.py / pipeline.py)
    # ------------------------------------------------------------------
    log.info("Loading SD model...")
    model = setup_model(args.config_path, args.ckpt_path, device)
    model = model.to(device)
    if hasattr(model, "logvar"):
        model.logvar = model.logvar.to(device)

    # ------------------------------------------------------------------
    # 2.  Load data  (identical to intact_unlearn_nsfw)
    # ------------------------------------------------------------------
    log.info("Loading NSFW forget + remain data...")
    forget_dl, remain_dl = setup_forget_nsfw_data(
        args.batch_size, args.image_size,
        nsfw_data_path=args.nsfw_data_path,
        not_nsfw_data_path=args.not_nsfw_data_path,
    )

    forget_svd_dl    = make_fractional_dataloader(forget_dl, args.svd_batches)
    forget_collect_dl = make_fractional_dataloader(forget_dl, args.forget_batches)
    remain_collect_dl = make_fractional_dataloader(remain_dl, args.remain_batches)

    log.info("SVD batches:    %d", len(forget_svd_dl))
    log.info("Forget batches: %d", len(forget_collect_dl))
    log.info("Remain batches: %d", len(remain_collect_dl))

    # ------------------------------------------------------------------
    # 3.  InTAct setup_protection() — SVD on forget data → pca_info
    #     (uses the EXISTING code path, no modifications)
    # ------------------------------------------------------------------
    log.info("Setting up InTAct (SVD on forget data → forget box + subspace)...")
    protection = UnlearnIntervalProtection(
        targets=args.targets,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        reduced_dim=args.reduced_dim,
        infinity_scale=20.0,
        use_actual_bounds=False,
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
        remain_dataloader=None,
        forward_fn=forward_fn,
        betas=model.betas.to(device) if hasattr(model, "betas") else None,
        num_timesteps=model.num_timesteps if hasattr(model, "num_timesteps") else 1000,
    )

    if not protection.pca_info:
        log.error("No pca_info built. Check --targets patterns.")
        return

    target_layer_names = [e["layer_name"] for e in protection.pca_info]
    log.info("%d protected layers", len(target_layer_names))

    # ------------------------------------------------------------------
    # 4.  Project forget + remain into SVD subspace via
    #     protection._collect_activations() with pca_components
    # ------------------------------------------------------------------
    pca_components = {}
    for entry in protection.pca_info:
        pca_components[entry["layer_name"]] = {
            "mu": entry["mu"].to(device),
            "U_forget": entry["U_forget"].to(device),
            "layer_type": entry.get("layer_type", "Linear"),
        }

    diffusion_model = model.model.diffusion_model

    # forget → nude prompt
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

    # remain → clothed prompt
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
    # 5.  Compute 5-zone occupancy (bounds from data, not infinity_scale)
    # ------------------------------------------------------------------
    log.info("Computing 5-zone occupancy metrics...")
    summary = occupancy_summary(forget_proj, remain_proj, protection.pca_info)

    total_f = sum(s["n_forget_tokens"] for s in summary.values())
    total_r = sum(s["n_remain_tokens"] for s in summary.values())
    log.info("Total forget tokens: %d", total_f)
    log.info("Total remain tokens: %d", total_r)

    metrics = {
        "config": {
            "targets": args.targets,
            "reduced_dim": args.reduced_dim,
            "lower_percentile": args.lower_percentile,
            "upper_percentile": args.upper_percentile,
            "forget_batches": args.forget_batches,
            "remain_batches": args.remain_batches,
            "svd_batches": args.svd_batches,
        },
        "per_layer": summary,
    }
    json_path = os.path.join(args.out_dir, "activation_analysis.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log.info("Saved metrics → %s", json_path)

    # ------------------------------------------------------------------
    # 6.  Plots
    # ------------------------------------------------------------------
    log.info("Generating plots...")
    plot_density_histograms(forget_proj, remain_proj, protection.pca_info, args.out_dir)
    plot_zone_breakdown(summary, args.out_dir)
    plot_mean_overlap_bars(summary, args.out_dir)
    plot_svd_scatter(forget_proj, remain_proj, protection.pca_info, args.out_dir)
    plot_per_dim_remain_zones(summary, args.out_dir)

    log.info("Done. All outputs in %s", args.out_dir)


if __name__ == "__main__":
    main()
