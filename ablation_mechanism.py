#!/usr/bin/env python3
"""
Experiments 1-2: realised update vs per-direction cost (the mechanism plot) and
kappa statistics on real activations.  Post-hoc analysis of trained checkpoints
produced by DDPM/ablation_regions.py or Classification/ablation_cls.py (each run
dir holds ckpt.pth + pca_info.pth + run_meta.json).

Exp 1 (mechanism).  For each protected layer, per projected direction j:
    rho_j     = max(|inf_low_j|, |inf_high_j|)           (envelope radius)
    F_j       = (z_max_j - z_min_j) / 2                  (forget half-width)
    w_low_j   = z_min_j - inf_low_j                      (low margin)
    w_high_j  = inf_high_j - z_max_j                     (high margin)
    sigma_j   = top-k singular value of the forget SVD
    cost(j)   = 2(c_low,j^2 + c_high,j^2) + 0.25(w_low,j^2 + w_high,j^2)
    n_j       = || (delta_W @ U_forget.T)[:, j] ||_2     (realised update)
  Correlations of n_j against {rho, F, w_low, w_high, sigma, cost} (Spearman +
  log-log Pearson) plus the top-8 vs bottom-8 share of ||delta_f||_F^2.

Exp 2 (kappa).  Over the retain activations projected into the forget subspace:
    w_j     = min(z_min_j - inf_low_j, inf_high_j - z_max_j)
    kappa(z)= max_j |z_j| / w_j
  percentiles (median/90/95/99), margin bar plot, coverage stats, and the
  empirical bound kappa * sqrt(2*T) vs the actual ||delta_f @ z|| (T =
  sum_j cost(j); the fraction of points where the bound holds is reported as
  measured; the intended value is 1.0).

Deliverables (under <results_dir>):
    mechanism.csv   per layer, per direction
    kappa.csv       per layer
    plots/*.pdf     mechanism scatter (n_j vs rho_j), kappa histogram,
                    bound-vs-actual scatter, margin bar plot
    tables/mechanism_correlations_<backbone>.tex

Usage:
    python ablation_mechanism.py --run_dir <runs/<suffix>>
    python ablation_mechanism.py --results_dir <root> --backbone ddpm \
        --region_mode two_corner --max_runs 10
"""

import argparse
import csv
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

log = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:  # pragma: no cover
    HAVE_MPL = False
    log.warning("matplotlib not available; PDF plots will be skipped")

MECHANISM_COLS = ["backbone", "setting", "region_mode", "experiment", "run_dir",
                  "lambda", "seed", "layer_name", "j", "sigma_j", "F_j",
                  "w_low", "w_high", "rho_j", "cost_j", "n_j"]
KAPPA_COLS = ["backbone", "setting", "run_dir", "seed", "layer_name", "k",
              "w_j_json", "kappa_median", "kappa_p90", "kappa_p95", "kappa_p99",
              "frac_in_BloH", "frac_in_BloL_BloH", "coords_outside_pct",
              "frac_bound_holds", "frac_top8", "frac_bottom8", "n_retain"]


# ============================================================================
# Run loading
# ============================================================================

def load_run(run_dir):
    """Returns (meta, pca_info, trained_state_dict) for one run directory."""
    meta_path = os.path.join(run_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{meta_path} missing")
    with open(meta_path) as f:
        meta = json.load(f)
    pca_info = torch.load(os.path.join(run_dir, "pca_info.pth"),
                          map_location="cpu", weights_only=False)
    ckpt = torch.load(os.path.join(run_dir, "ckpt.pth"),
                      map_location="cpu", weights_only=False)
    # DDPM saves [state_dict, optimizer, iter]; ResNet-18 saves state_dict.
    trained = ckpt[0] if isinstance(ckpt, list) else ckpt
    return meta, pca_info, {k.replace("module.", "", 1): v for k, v in trained.items()}


def load_base_weights(meta):
    if meta["backbone"] == "ddpm":
        path = os.path.join(meta["base_ckpt_folder"], "ckpts", "ckpt.pth")
        state = torch.load(path, map_location="cpu", weights_only=False)
        base = state[0]
    else:
        path = meta["base_ckpt"]
        base = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(base, dict) and "state_dict" in base:
            base = base["state_dict"]
    return {k.replace("module.", "", 1): v for k, v in base.items()}


# ============================================================================
# Exp 1 helpers
# ============================================================================

def direction_measures(info):
    """Per-direction sigma, F, w_low, w_high, rho, cost (numpy arrays)."""
    il, zm, zx, ih = (info[k].numpy() for k in
                      ("inf_low", "z_min", "z_max", "inf_high"))
    sf = info["S_forget"].numpy() if info["S_forget"].numel() else np.full(il.shape, np.nan)
    c_low = (il + zm) / 2.0
    c_high = (zx + ih) / 2.0
    return {
        "sigma_j": sf,
        "F_j": (zx - zm) / 2.0,
        "w_low": zm - il,
        "w_high": ih - zx,
        "rho_j": np.maximum(np.abs(il), np.abs(ih)),
        "cost_j": 2.0 * (c_low ** 2 + c_high ** 2) + 0.25 * ((zm - il) ** 2 + (ih - zx) ** 2),
    }


def delta_f_for_layer(info, dw):
    """[M, k] projected weight update, same math as the loss's interval block."""
    import torch.nn.functional as F
    dwt = torch.tensor(dw, dtype=torch.float32)
    Uf = info["U_forget"].float()
    if info["layer_type"] == "Conv2d":
        resp = F.conv2d(Uf.view(Uf.size(0), -1, 1, 1), dwt)
        return resp.view(Uf.size(0), -1).T.numpy()
    return (dwt @ Uf.T).numpy()


def spearman(a, b):
    from scipy.stats import spearmanr
    res = spearmanr(a, b)
    stat = getattr(res, "statistic", None)
    if stat is None:
        stat = res[0]
    return float(stat)


def loglog_pearson(a, b):
    mask = (a > 0) & (b > 0)
    if mask.sum() < 3:
        return float("nan")
    la, lb = np.log10(a[mask]), np.log10(b[mask])
    return float(np.corrcoef(la, lb)[0, 1])


def frac_top_bottom(n_j, k=8):
    """Fraction of ||n||_2^2 carried by the largest / smallest k directions."""
    order = np.argsort(n_j)[::-1]
    tot = float((n_j ** 2).sum()) or 1e-30
    return (float((n_j[order[:k]] ** 2).sum()) / tot,
            float((n_j[order[-k:]] ** 2).sum()) / tot)


# ============================================================================
# Exp 2 helpers
# ============================================================================

def kappa_stats(info, z_remain):
    il, zm, zx, ih = (info[k].numpy() for k in
                      ("inf_low", "z_min", "z_max", "inf_high"))
    w = np.minimum(zm - il, ih - zx)
    w = np.where(w > 0, w, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.max(np.abs(z_remain) / w[None, :], axis=1)
    kappa = kappa[np.isfinite(kappa)]
    in_low = np.all((z_remain >= il[None, :]) & (z_remain <= zm[None, :]), axis=1)
    in_high = np.all((z_remain >= zx[None, :]) & (z_remain <= ih[None, :]), axis=1)
    outside = np.logical_or(z_remain < zm[None, :],
                            z_remain > zx[None, :]).sum(axis=1)
    return {"w_j": w, "kappa": kappa, "frac_low": float(in_low.mean()),
            "frac_inside": float((in_low | in_high).mean()),
            "outside_count": outside}


def empirical_bound_fraction(info, z_remain, delta_f):
    """kappa(z) * sqrt(2*T) vs ||delta_f @ z||; returns (fraction where the
    bound holds over finite points, T)."""
    m = direction_measures(info)
    T = float(m["cost_j"].sum())
    w = np.minimum(info["z_min"].numpy() - info["inf_low"].numpy(),
                   info["inf_high"].numpy() - info["z_max"].numpy())
    w = np.where(w > 0, w, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.max(np.abs(z_remain) / w[None, :], axis=1)
    actual = np.linalg.norm(delta_f @ z_remain.T, axis=0)
    ok = np.isfinite(kappa) & np.isfinite(actual)
    bound = kappa[ok] * np.sqrt(2.0 * T)
    return float((bound >= actual[ok]).mean()), T, actual[ok], bound


# ============================================================================
# Remain-activation collection (per backbone; heavy imports kept lazy)
# ============================================================================

def _add_backbone_path(backbone):
    repo = os.path.dirname(os.path.abspath(__file__))
    if backbone == "ddpm":
        p = os.path.join(repo, "DDPM")
        sys.path.insert(0, p)
    elif backbone == "resnet18":
        sys.path.insert(0, os.path.join(repo, "Classification"))


def remain_projections_cls(meta, info, device):
    from InTAct.intact import UnlearnIntervalProtection, classification_forward_fn
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "Classification"))
    import yaml
    import utils
    from pipeline import build_args, build_data_loaders

    with open(meta["config"]) as f:
        cfg = yaml.safe_load(f)
    cfg["pipeline"]["seed"] = int(meta["seed"])
    cfg["unlearn"]["method"] = "intact"
    fargs = build_args(cfg)
    model, _, val_loader, test_loader, marked_loader = utils.setup_model_dataset(fargs)
    model = model.to(device).eval()
    ckpt = torch.load(meta["base_ckpt"], map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)

    dl, _, _ = build_data_loaders(fargs, marked_loader, val_loader, test_loader)

    protection = UnlearnIntervalProtection(targets=["fc"], reduced_dim=32)
    protection._find_target_layers(model)
    components = {info["layer_name"]: {"mu": info["mu"], "U_forget": info["U_forget"]}}
    proj = protection._collect_activations(
        model, [info["layer_name"]], dl["retain"], device,
        forward_fn=classification_forward_fn, pca_components=components)
    return proj[info["layer_name"]].numpy()


def remain_projections_ddpm(meta, info, device):
    from InTAct.intact import UnlearnIntervalProtection, ddpm_forward_fn
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "DDPM"))
    import yaml
    from datasets import get_forget_dataset, data_transform
    from functions import dict2namespace
    from models.diffusion import Conditional_Model
    from runners.diffusion import Diffusion
    from collections import OrderedDict

    with open(meta["config"]) as f:
        cfg = yaml.safe_load(f)
    with open(cfg["model_config"]) as f:
        rc = dict2namespace(yaml.safe_load(f))
    rc.exp_root_dir = "/tmp"
    rc.log_dir = "/tmp"
    rc.ckpt_dir = "/tmp"
    rc.training["batch_size"] = cfg["unlearn"].get("batch_size", 128)
    rc.training["n_iters"] = cfg["unlearn"].get("n_iters", 3000)

    rargs = argparse.Namespace(
        config=cfg["model_config"], ckpt_folder=meta["base_ckpt_folder"],
        label_to_forget=cfg["unlearn"].get("label_to_forget", 0),
        seed=int(meta["seed"]), mode="intact", sample_type="generalized",
        skip_type="uniform", timesteps=1000, eta=1.0, cond_scale=2.0,
        sequence=False, uc=True, negative_guidance=7.5, mask_ratio=0.5,
        sparse=False, n_samples_per_class=500, classes_to_generate=None,
        alpha=0.0, method="rl", mask_path=None,
    )
    runner = Diffusion(rargs, rc)
    remain_loader, _ = get_forget_dataset(rargs, rc, rargs.label_to_forget)

    model = Conditional_Model(rc)
    base = load_base_weights(meta)
    model.load_state_dict(OrderedDict(
        (k, v) for k, v in base.items() if k in model.state_dict()), strict=True)
    model.to(device).eval()

    protection = UnlearnIntervalProtection(targets=rc.training.targets, reduced_dim=32)
    protection._find_target_layers(model)
    components = {info["layer_name"]: {"mu": info["mu"], "U_forget": info["U_forget"]}}
    proj = protection._collect_activations(
        model, [info["layer_name"]], remain_loader, device,
        forward_fn=ddpm_forward_fn,
        data_transform_fn=lambda x: data_transform(rc, x),
        betas=runner.betas, num_timesteps=runner.num_timesteps,
        pca_components=components)
    return proj[info["layer_name"]].numpy()


def _collect_remain(meta, info, device):
    _add_backbone_path(meta["backbone"])
    if meta["backbone"] == "ddpm":
        return remain_projections_ddpm(meta, info, device)
    return remain_projections_cls(meta, info, device)


# ============================================================================
# Run-level analysis
# ============================================================================

def analyze_run(run_dir, device, collect_remain=True):
    run_dir = os.path.abspath(run_dir)
    meta, pca_info, trained = load_run(run_dir)
    base = load_base_weights(meta)

    mech_rows = []
    kappa_rows = []
    plots = {"layers": [], "kappa_hist": [], "margins": [], "bound_vs_actual": []}

    for info in pca_info:
        layer_name = info["layer_name"]
        wkey = f"{layer_name}.weight"
        if wkey not in trained or wkey not in base:
            log.warning(f"skip layer {layer_name}: no matching weights "
                        f"(expected key {wkey})")
            continue
        dw = trained[wkey].float().numpy() - base[wkey].float().numpy()
        measures = direction_measures(info)
        delta_f = delta_f_for_layer(info, dw)
        n_j = np.linalg.norm(delta_f, axis=0)
        k = n_j.shape[0]

        for j in range(k):
            mech_rows.append({
                "backbone": meta.get("backbone", ""),
                "setting": meta.get("setting", ""),
                "region_mode": meta.get("region_mode", ""),
                "experiment": meta.get("experiment", ""),
                "run_dir": run_dir,
                "lambda": meta.get("lambda", ""),
                "seed": meta.get("seed", ""),
                "layer_name": layer_name,
                "j": j,
                "sigma_j": measures["sigma_j"][j],
                "F_j": measures["F_j"][j],
                "w_low": measures["w_low"][j],
                "w_high": measures["w_high"][j],
                "rho_j": measures["rho_j"][j],
                "cost_j": measures["cost_j"][j],
                "n_j": n_j[j],
            })

        n8 = min(8, k)
        ft, fb = frac_top_bottom(n_j, n8)
        corrs = {key: (spearman(n_j, np.asarray(measures[key], dtype=float)),
                       loglog_pearson(n_j, np.asarray(measures[key], dtype=float)))
                 for key in ("sigma_j", "F_j", "w_low", "w_high", "rho_j", "cost_j")}
        layer_plot = {"name": layer_name, "k": k, "n_j": n_j,
                      "rho_j": measures["rho_j"], "cost_j": measures["cost_j"],
                      "corrs": corrs, "tf": (ft, fb), "delta_f": delta_f}

        if collect_remain:
            try:
                z_remain = _collect_remain(meta, info, device)
                ks = kappa_stats(info, z_remain)
                frac_b, T, actual_ok, bound_ok = empirical_bound_fraction(
                    info, z_remain, delta_f)
                kappa_rows.append({
                    "backbone": meta.get("backbone", ""),
                    "setting": meta.get("setting", ""),
                    "run_dir": run_dir,
                    "seed": meta.get("seed", ""),
                    "layer_name": layer_name,
                    "k": k,
                    "w_j_json": json.dumps(
                        np.nan_to_num(ks["w_j"], nan=-1.0).tolist()),
                    "kappa_median": float(np.median(ks["kappa"])),
                    "kappa_p90": float(np.percentile(ks["kappa"], 90)),
                    "kappa_p95": float(np.percentile(ks["kappa"], 95)),
                    "kappa_p99": float(np.percentile(ks["kappa"], 99)),
                    "frac_in_BloH": ks["frac_low"],
                    "frac_in_BloL_BloH": ks["frac_inside"],
                    "coords_outside_pct": float((ks["outside_count"] > 0).mean() * 100.0),
                    "frac_bound_holds": frac_b,
                    "frac_top8": ft,
                    "frac_bottom8": fb,
                    "n_retain": len(ks["kappa"]),
                })
                layer_plot.update({"kappa": ks["kappa"], "T": T,
                                   "actual": actual_ok, "bound": bound_ok,
                                   "w_j": ks["w_j"]})
                plots["kappa_hist"].append((layer_name, ks["kappa"]))
                plots["margins"].append((layer_name, ks["w_j"]))
                plots["bound_vs_actual"].append(
                    (layer_name, actual_ok, bound_ok))
            except Exception as exc:
                log.warning(f"Exp2 collection failed for {layer_name}: {exc}")
        else:
            layer_plot["tf"] = (ft, fb)

        plots["layers"].append(layer_plot)

    return meta, mech_rows, kappa_rows, plots


# ============================================================================
# CSV / plots / tex
# ============================================================================

def _ensure_header(path, cols):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(cols)


def append_rows(path, cols, rows):
    if not rows:
        return
    _ensure_header(path, cols)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def render_plots(results_dir, plots):
    if not HAVE_MPL or not plots["layers"]:
        return
    out = os.path.join(results_dir, "plots")
    os.makedirs(out, exist_ok=True)
    layers = plots["layers"]

    # mechanism scatter: n_j vs rho_j, one panel per layer
    nl = len(layers)
    cols = min(nl, 4)
    rows_n = int(np.ceil(nl / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.5 * cols, 3.6 * rows_n),
                             squeeze=False)
    for ax, ly in zip(axes.ravel(), layers):
        ax.loglog(ly["rho_j"] + 1e-12, ly["n_j"] + 1e-12, ".", ms=4, alpha=0.7)
        ax.set_title(ly["name"], fontsize=8)
        ax.set_xlabel(r"$\rho_j$ (envelope radius)")
        ax.set_ylabel(r"$n_j$ (realised update)")
    for ax in axes.ravel()[nl:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "mechanism_nj_vs_rho.pdf"))
    plt.close(fig)

    # kappa histogram
    if plots["kappa_hist"]:
        ks = plots["kappa_hist"]
        fig, axes = plt.subplots(1, len(ks), figsize=(4.5 * len(ks), 3.4),
                                 squeeze=False)
        for ax, (name, kappa) in zip(axes.ravel(), ks):
            ax.hist(kappa, bins=40, log=True)
            for q in (50, 90, 95, 99):
                ax.axvline(np.percentile(kappa, q), color="r", lw=0.8, ls="--")
            ax.set_title(name, fontsize=8)
            ax.set_xlabel(r"$\kappa(z)=\max_j |z_j|/w_j$")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "kappa_histogram.pdf"))
        plt.close(fig)

    # margin bar plot
    if plots["margins"]:
        ms = plots["margins"]
        fig, axes = plt.subplots(1, len(ms), figsize=(4.5 * len(ms), 3.4),
                                 squeeze=False)
        for ax, (name, w) in zip(axes.ravel(), ms):
            ax.bar(np.arange(len(w)), w, width=1.0)
            ax.axhline(np.nanmean(w), color="r", lw=0.8, ls="--")
            ax.set_title(name, fontsize=8)
            ax.set_xlabel("direction j")
            ax.set_ylabel(r"$w_j$ (margin)")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "margins.pdf"))
        plt.close(fig)

    # bound vs actual scatter
    if plots["bound_vs_actual"]:
        bva = plots["bound_vs_actual"]
        fig, axes = plt.subplots(1, len(bva), figsize=(4.5 * len(bva), 3.6),
                                 squeeze=False)
        for ax, (name, actual, bound) in zip(axes.ravel(), bva):
            ok = (actual > 0) & (bound > 0)
            ax.loglog(np.maximum(actual[ok], 1e-30),
                      np.maximum(bound[ok], 1e-30), ".", ms=3, alpha=0.5)
            lo = max(np.min(actual[ok]) * 0.5, 1e-30) if ok.any() else 1e-6
            hi = max(np.max([actual.max() if actual.size else 1.0,
                             bound.max() if bound.size else 1.0]) * 2.0, 1.0)
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_title(name, fontsize=8)
            ax.set_xlabel(r"actual $\|\delta_f \cdot z\|$")
            ax.set_ylabel(r"$\kappa(z)\sqrt{2T}$")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "bound_vs_actual.pdf"))
        plt.close(fig)

    log.info(f"plots written to {out}")


def correlation_table(results_dir, backbone, corr_agg):
    if not corr_agg:
        return
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Spearman / log-log Pearson of the realised update "
             rf"$n_j$ against per-direction quantities, {backbone}. "
             rf"Mean over analysed runs. \label{{tab:mechanism-{backbone}}}}}",
             r"\small",
             r"\begin{tabular}{l " + "c " * 12 + r"}",
             r"\toprule",
             r"Layer & \multicolumn{2}{c}{$\rho_j$} & \multicolumn{2}{c}{$F_j$}"
             r" & \multicolumn{2}{c}{$\sigma_j$}"
             r" & \multicolumn{2}{c}{$w_{\mathrm{low}}$}"
             r" & \multicolumn{2}{c}{$w_{\mathrm{high}}$}"
             r" & \multicolumn{2}{c}{cost$(j)$} \\",
             r" & Sp & LL & Sp & LL & Sp & LL & Sp & LL & Sp & LL & Sp & LL \\",
             r"\midrule"]
    key_order = ("rho_j", "F_j", "sigma_j", "w_low", "w_high", "cost_j")
    for (bb, st, name), d in sorted(corr_agg.items()):
        cells = [name.replace("_", r"\_")]
        for k2 in key_order:
            sp, ll = d["corrs"][k2]
            cells += [f"{sp:.3f}" if sp == sp else "-",
                      f"{ll:.3f}" if ll == ll else "-"]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)
    out = os.path.join(results_dir, "tables", f"mechanism_correlations_{backbone}.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"Wrote {out}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=str, default=None,
                        help="single run directory to analyse")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="scan <results_dir>/runs/* (dedup by run_dir)")
    parser.add_argument("--backbone", choices=("ddpm", "resnet18"), default=None)
    parser.add_argument("--region_mode", type=str, default=None)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--max_runs", type=int, default=0)
    parser.add_argument("--no_remain", action="store_true",
                        help="skip Exp2 (no remain-activation collection)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    run_dirs = []
    results_dir = args.results_dir
    if args.run_dir:
        run_dirs = [args.run_dir]
        results_dir = results_dir or os.path.dirname(os.path.abspath(args.run_dir))
        if os.path.basename(results_dir) == "runs":
            results_dir = os.path.dirname(results_dir)
    elif args.results_dir:
        runs_root = os.path.join(args.results_dir, "runs")
        for name in sorted(os.listdir(runs_root)):
            d = os.path.join(runs_root, name)
            meta_path = os.path.join(d, "run_meta.json")
            if not os.path.isdir(d) or not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            if args.backbone and meta.get("backbone") != args.backbone:
                continue
            if args.region_mode and meta.get("region_mode") != args.region_mode:
                continue
            if args.experiment and meta.get("experiment") != args.experiment:
                continue
            run_dirs.append(d)
            if args.max_runs and len(run_dirs) >= args.max_runs:
                break
    else:
        parser.error("need --run_dir or --results_dir")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mech_rows, kappa_rows_all = [], []
    corr_agg = {}
    plots_all = {"layers": [], "kappa_hist": [], "margins": [], "bound_vs_actual": []}

    for i, rd in enumerate(run_dirs):
        log.info(f"[{i + 1}/{len(run_dirs)}] analysing {rd}")
        try:
            meta, rows, krows, plots = analyze_run(
                rd, device, collect_remain=not args.no_remain)
        except Exception as exc:
            log.error(f"analysis of {rd} failed: {exc}")
            continue
        mech_rows += rows
        kappa_rows_all += krows
        for ly in plots["layers"]:
            key = (meta.get("backbone"), meta.get("setting"), ly["name"])
            d = corr_agg.setdefault(key, {
                "name": ly["name"],
                "corrs": {k2: [0.0, 0.0] for k2 in ly["corrs"]},
                "n": 0})
            for k2, (sp, ll) in ly["corrs"].items():
                if sp == sp:
                    d["corrs"][k2][0] += sp
                if ll == ll:
                    d["corrs"][k2][1] += ll
            d["n"] += 1
        plots_all["layers"].extend(plots["layers"])
        plots_all["kappa_hist"].extend(plots["kappa_hist"])
        plots_all["margins"].extend(plots["margins"])
        plots_all["bound_vs_actual"].extend(plots["bound_vs_actual"])
        append_rows(os.path.join(results_dir, "mechanism.csv"),
                    MECHANISM_COLS, rows)
        append_rows(os.path.join(results_dir, "kappa.csv"),
                    KAPPA_COLS, krows)

    for d in corr_agg.values():
        n = max(d["n"], 1)
        for k2 in d["corrs"]:
            d["corrs"][k2] = tuple(v / n for v in d["corrs"][k2])

    if corr_agg:
        backbone = "/".join(sorted({k[0] for k in corr_agg}))
        correlation_table(results_dir, backbone, corr_agg)
        print("\n=== realised update n_j vs per-direction quantities "
              "(mean over runs) ===")
        print(f"{'layer':<30s} n  (Sp = Spearman, LL = log-log Pearson)")
        for (bb, st, name), d in sorted(corr_agg.items()):
            parts = []
            for k2 in ("rho_j", "F_j", "sigma_j", "w_low", "w_high", "cost_j"):
                sp, ll = d["corrs"][k2]
                parts.append(f"{k2} Sp{sp:+.3f} LL{ll:+.3f}"
                             if sp == sp and ll == ll else f"{k2} -")
            print(f"{name:<30s} {d['n']}  {' | '.join(parts)}")

        # top/bottom-8 per layer (mean over analysed runs)
        tb = {}
        for ly in plots_all["layers"]:
            tb.setdefault(ly["name"], []).append(ly["tf"])
        print("\ntop/bottom-8 share of ||delta_f||_F^2 (mean over runs):")
        for name, tfs in sorted(tb.items()):
            ft = np.mean([t[0] for t in tfs])
            fb = np.mean([t[1] for t in tfs])
            print(f"{name:<30s} top8 {ft:.3f}  bottom8 {fb:.3f}")

    if kappa_rows_all:
        print("\n=== kappa(z) over retain activations (median over runs) ===")
        agg2 = {}
        for r in kappa_rows_all:
            key = (r["backbone"], r["layer_name"])
            agg2.setdefault(key, []).append(r)
        for (bb, name), rs in sorted(agg2.items()):
            med = np.mean([r["kappa_median"] for r in rs])
            p99 = np.mean([r["kappa_p99"] for r in rs])
            fb_ = np.mean([r["frac_bound_holds"] for r in rs])
            frac_in = np.mean([r["frac_in_BloL_BloH"] for r in rs])
            print(f"{name:<30s} kappa med {med:.2f}  p99 {p99:.2f}  "
                  f"frac_in_box {frac_in:.4f}  bound_holds {fb_:.3f}")

    if plots_all["layers"]:
        render_plots(results_dir, plots_all)

    if not mech_rows and not kappa_rows_all:
        log.warning("no usable runs analysed")
    else:
        print(f"\nwrote {os.path.join(results_dir, 'mechanism.csv')} "
              f"({len(mech_rows)} direction-rows) and "
              f"{os.path.join(results_dir, 'kappa.csv')} "
              f"({len(kappa_rows_all)} layer-rows)")


if __name__ == "__main__":
    main()