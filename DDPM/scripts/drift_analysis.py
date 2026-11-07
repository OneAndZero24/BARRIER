"""
Forward-pass-only drift analysis (rebuttal) over the DDPM CIFAR-10 lambda sweep.
No training.

For every unlearned checkpoint and every protected layer, the layer OUTPUT of
the unlearned model is compared with the original (pre-unlearning) model on
one fixed evaluation batch:
    drift = mean over samples and timesteps of ||f_new(x) - f_orig(x)|| / ||f_orig(x)||
on retain-class inputs (retain_drift) and forget-class inputs (forget_drift).

Repo code reused unchanged:
  - config / data   : ablation_regions.build_runner_config / build_runner_args,
                      datasets.get_forget_dataset, datasets.data_transform
  - model loading   : Conditional_Model + DataParallel + states[0]
                      (as ablation_regions.py; config has ema=False, so
                      states[0] is exactly the model that was sampled)
  - noise schedule  : runners.diffusion.get_beta_schedule
  - protected layers: UnlearnIntervalProtection._find_target_layers with the
                      paper targets from configs/pipeline_fulleval.yaml
  - penalty         : UnlearnIntervalProtection.compute_protection_loss with
                      each run's own pca_info.pth, params_snapshot = original
                      weights, lambda_interval = 1 (UNWEIGHTED term, same
                      normalisation as training).  Components via the repo's
                      own switches: L_mean (include_mean), L_res / orthogonal
                      (include_res), interval low / high corner box
                      (make_region_boxes restricted to one box).

Fixed evaluation batch (built once from --seed, saved to --batch_path and
reused for every checkpoint): N retain-class and N forget-class training
images, timesteps --timesteps, one fixed Gaussian noise per (image, t),
x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps, class-conditional forward with
cond_drop_prob = 0 (deterministic).

Runs (both families in --runs_dir; new runs, e.g. lambda = 0, are picked up
automatically):
  ddpm_classwise_*   main ablation_regions.py sweep: ckpts/ckpt.pth,
                     pca_info.pth, rows.json["run"] (ua, ta, fid)
  lam*_seed*         fid-repro pipeline.py sweep: output/ckpts/<ts>/ckpt.pth
                     for the <ts> whose metrics.json matches rows.json; no
                     pca_info.pth -> penalty = NaN
RA = the pipeline's TA (retain-class accuracy of the samples).

Outputs (--out_dir): drift.csv, drift_per_run.csv, skipped.csv,
spearman.csv, drift_vs_lambda.png, retain_drift_vs_fid.png

Usage (from DDPM/, PYTHONPATH=<repo root>):
    python scripts/drift_analysis.py \
        --runs_dir /shared/results/common/miksa/intact/DDPM/lambda_sweep/runs \
        --out_dir results
"""

import argparse
import contextlib
import csv
import glob
import json
import logging
import os
import random
import re
import sys

import numpy as np
import torch
import yaml

_DDPM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(_DDPM_DIR))  # repo root (InTAct, ablation_common)
sys.path.insert(0, _DDPM_DIR)

import ablation_regions  # noqa: E402  (imports setup_cache first)
from ablation_regions import build_runner_args, build_runner_config  # noqa: E402
from datasets import data_transform, get_forget_dataset  # noqa: E402
from InTAct import intact as intact_mod  # noqa: E402
from InTAct.intact import (  # noqa: E402
    UnlearnIntervalProtection,
    percentile_alpha_to_quantiles,
)
from models.diffusion import Conditional_Model  # noqa: E402
from runners.diffusion import get_beta_schedule  # noqa: E402

log = logging.getLogger("drift")

PENALTY_PARTS = ["penalty", "pen_mean", "pen_orth", "pen_low", "pen_high"]
COLUMNS = (["lambda", "seed", "layer", "retain_drift", "forget_drift"]
           + PENALTY_PARTS + ["UA", "RA", "FID", "family", "run"])


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Run discovery
# =============================================================================

def discover_runs(runs_dir):
    """Returns (runs, skipped).  Each run: dict(lambda, seed, family, run,
    ckpt, pca, UA, RA, FID, meta)."""
    runs, skipped = [], []

    for d in sorted(glob.glob(os.path.join(runs_dir, "ddpm_classwise_*"))):
        name = os.path.basename(d)
        rows_p, ckpt = os.path.join(d, "rows.json"), os.path.join(d, "ckpts", "ckpt.pth")
        meta_p, pca = os.path.join(d, "run_meta.json"), os.path.join(d, "pca_info.pth")
        missing = [os.path.basename(x) for x in (rows_p, ckpt, meta_p) if not os.path.isfile(x)]
        if missing:
            skipped.append({"run": name, "reason": f"missing {', '.join(missing)}"})
            continue
        r = json.load(open(rows_p))["run"]
        meta = json.load(open(meta_p))
        if not _finite(r.get("ua"), r.get("ta"), r.get("fid")):
            skipped.append({"run": name, "reason": "UA/RA/FID missing or NaN in rows.json"})
            continue
        runs.append({
            "lambda": float(r["lambda"]), "seed": int(r["seed"]),
            "family": "classwise", "run": name, "ckpt": ckpt,
            "pca": pca if os.path.isfile(pca) else None,
            "UA": float(r["ua"]), "RA": float(r["ta"]), "FID": float(r["fid"]),
            "meta": meta,
        })

    for d in sorted(glob.glob(os.path.join(runs_dir, "lam*_seed*"))):
        name = os.path.basename(d)
        rows_p = os.path.join(d, "rows.json")
        if not os.path.isfile(rows_p):
            skipped.append({"run": name, "reason": "missing rows.json"})
            continue
        r = json.load(open(rows_p))
        if not _finite(r.get("ua"), r.get("ta"), r.get("fid")):
            skipped.append({"run": name, "reason": "UA/RA/FID missing or NaN in rows.json"})
            continue
        # A run dir can hold several attempts (output/<ts>/); take the
        # checkpoint of the attempt whose metrics.json produced rows.json.
        ckpt = None
        for mp in sorted(glob.glob(os.path.join(d, "output", "*", "metrics.json"))):
            ts = os.path.basename(os.path.dirname(mp))
            m = json.load(open(mp))
            cand = os.path.join(d, "output", "ckpts", ts, "ckpt.pth")
            if os.path.isfile(cand) and abs(float(m.get("FID", np.nan)) - float(r["fid"])) < 1e-6:
                ckpt = cand
        if ckpt is None:
            skipped.append({"run": name, "reason": "no checkpoint whose metrics.json matches rows.json"})
            continue
        runs.append({
            "lambda": float(r["lambda"]), "seed": int(r["seed"]),
            "family": "fid-repro", "run": name, "ckpt": ckpt, "pca": None,
            "UA": float(r["ua"]), "RA": float(r["ta"]), "FID": float(r["fid"]),
            "meta": None,
        })
    return runs, skipped


def _finite(*vals):
    try:
        return all(v not in (None, "") and np.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False


# =============================================================================
# Model / batch
# =============================================================================

def load_model(runner_config, ckpt_path, device):
    model = Conditional_Model(runner_config)
    states = torch.load(ckpt_path, map_location=device)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(states[0], strict=True)
    model.eval()
    return model


def build_eval_batch(runner_args, runner_config, label, n, timesteps, seed):
    seed_all(seed)
    remain_loader, forget_loader = get_forget_dataset(runner_args, runner_config, label)
    betas = get_beta_schedule(
        beta_schedule=runner_config.diffusion.beta_schedule,
        beta_start=runner_config.diffusion.beta_start,
        beta_end=runner_config.diffusion.beta_end,
        num_diffusion_timesteps=runner_config.diffusion.num_diffusion_timesteps,
    )
    abar = torch.from_numpy(1.0 - betas).float().cumprod(dim=0)
    gen = torch.Generator().manual_seed(seed)
    t = torch.tensor(timesteps, dtype=torch.long)

    out = {"timesteps": t, "seed": seed, "label_to_forget": label}
    for split, loader in (("retain", remain_loader), ("forget", forget_loader)):
        ds = loader.dataset
        idx = torch.randperm(len(ds), generator=gen)[:n].tolist()
        x0 = torch.stack([ds[i][0] for i in idx])
        c = torch.tensor([int(ds[i][1]) for i in idx], dtype=torch.long)
        x0 = data_transform(runner_config, x0)
        eps = torch.randn((len(t),) + tuple(x0.shape), generator=gen)
        a = abar.index_select(0, t).view(-1, 1, 1, 1, 1)
        out[split] = {
            "x_t": a.sqrt() * x0.unsqueeze(0) + (1.0 - a).sqrt() * eps,  # T x N x C x H x W
            "c": c, "idx": torch.tensor(idx),
        }
        log.info(f"eval batch {split}: N={len(idx)} classes="
                 f"{torch.bincount(c, minlength=10).tolist()}")
    return out


@torch.no_grad()
def layer_outputs(model, modules, x, t, c):
    """Forward one batch; returns {layer: per-sample flattened output}."""
    outs, hooks = {}, []
    for name, mod in modules.items():
        hooks.append(mod.register_forward_hook(
            lambda m, i, o, name=name: outs.__setitem__(name, o.detach().flatten(1).float())))
    try:
        model(x, t.float(), c, mode="train", cond_drop_prob=0.0)
    finally:
        for h in hooks:
            h.remove()
    return outs


@torch.no_grad()
def drift_for_split(orig, new, mods_o, mods_n, split, device, bs):
    """Mean over (t, sample) of ||f_new - f_orig|| / ||f_orig|| per layer."""
    x_t, c = split["x_t"], split["c"]
    T, N = x_t.shape[:2]
    sums = {name: 0.0 for name in mods_o}
    for ti in range(T):
        for s in range(0, N, bs):
            x = x_t[ti, s:s + bs].to(device)
            cc = c[s:s + bs].to(device)
            tt = torch.full((x.size(0),), int(TIMESTEPS[ti]), device=device)
            fo = layer_outputs(orig, mods_o, x, tt, cc)
            fn = layer_outputs(new, mods_n, x, tt, cc)
            for name in mods_o:
                rel = (fn[name] - fo[name]).norm(dim=1) / fo[name].norm(dim=1)
                sums[name] += float(rel.sum())
    return {name: v / (T * N) for name, v in sums.items()}


# =============================================================================
# Penalty (closed form in the weight update, repo code)
# =============================================================================

def make_protection(cfg, meta, **overrides):
    ic = cfg["intact"]
    alpha = meta.get("alpha", 5)
    lower_p, upper_p = percentile_alpha_to_quantiles(alpha)
    kw = dict(
        targets=ic["targets"],
        lambda_interval=1.0,  # unweighted
        lower_percentile=lower_p,
        upper_percentile=upper_p,
        reduced_dim=meta.get("k", 32),
        infinity_scale=ic.get("infinity_scale", 20.0),
        use_actual_bounds=True,
        normalize_protection=ic.get("normalize_protection", True),
        region_mode=meta.get("region_mode", "two_corner"),
        normalize_region=True,
        region_random_seed=0,
        interval_mode=meta.get("interval_mode", "full"),
        include_db=bool(meta.get("include_db", False)),
        include_mean=bool(meta.get("include_mean", True)),
        include_res=bool(meta.get("include_res", True)),
        sign_flip_frac=meta.get("sign_flip_frac", 0.0),
        sign_flip_seed=meta.get("seed", 0),
        uniform_margin=bool(meta.get("uniform_margin", False)),
    )
    kw.update(overrides)
    return UnlearnIntervalProtection(**kw)


@contextlib.contextmanager
def only_box(i):
    """Restrict make_region_boxes to box i (two_corner: 0 = low, 1 = high)."""
    orig = intact_mod.make_region_boxes
    intact_mod.make_region_boxes = lambda *a, **k: [orig(*a, **k)[i]]
    try:
        yield
    finally:
        intact_mod.make_region_boxes = orig


@torch.no_grad()
def penalty_terms(cfg, meta, pca_info, model_new, snapshot, device):
    def term(**overrides):
        prot = make_protection(cfg, meta, **overrides)
        prot._find_target_layers(model_new)
        prot.pca_info = pca_info
        prot.param_to_name = {p: n for n, p in model_new.named_parameters()}
        prot.params_snapshot = snapshot
        return float(prot.compute_protection_loss(model_new, device))

    total = term()
    pen_mean = term(include_res=False, interval_mode="off")
    pen_orth = term(include_mean=False, interval_mode="off")
    # With normalize_region the interval chunk is sum(4 corner terms) / (2 * #boxes);
    # restricted to one box it is (its 2 terms) / 2, so its share of the full
    # two-box term is half of that.
    with only_box(0):
        pen_low = 0.5 * term(include_mean=False, include_res=False)
    with only_box(1):
        pen_high = 0.5 * term(include_mean=False, include_res=False)
    parts = pen_mean + pen_orth + pen_low + pen_high
    if total > 0 and abs(parts - total) > 1e-4 * max(abs(total), 1e-12):
        log.warning(f"penalty components {parts:.6g} != total {total:.6g}")
    return {"penalty": total, "pen_mean": pen_mean, "pen_orth": pen_orth,
            "pen_low": pen_low, "pen_high": pen_high}


# =============================================================================
# Stats / plots
# =============================================================================

def spearman_table(per_run):
    from scipy.stats import spearmanr
    out = []
    for x, y in (("retain_drift", "FID"), ("retain_drift", "RA"),
                 ("retain_drift", "lambda"), ("forget_drift", "UA")):
        a = np.array([r[x] for r in per_run], dtype=float)
        b = np.array([r[y] for r in per_run], dtype=float)
        rho, p = spearmanr(a, b)
        out.append({"x": x, "y": y, "n": len(a), "rho": rho, "p": p})
    return out


def make_plots(per_run, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lams = sorted({r["lambda"] for r in per_run})
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for key, lab in (("retain_drift", "retain"), ("forget_drift", "forget")):
        m = [np.mean([r[key] for r in per_run if r["lambda"] == l]) for l in lams]
        s = [np.std([r[key] for r in per_run if r["lambda"] == l]) for l in lams]
        ax.errorbar(lams, m, yerr=s, marker="o", capsize=3, label=f"{lab} drift")
    # symlog keeps a lambda = 0 run on the axis
    ax.set_xscale("symlog", linthresh=min([l for l in lams if l > 0] or [0.01]))
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"mean $\|f_{new}-f_{orig}\| / \|f_{orig}\|$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "drift_vs_lambda.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    lam = np.array([r["lambda"] for r in per_run])
    sc = ax.scatter([r["retain_drift"] for r in per_run], [r["FID"] for r in per_run],
                    c=np.log10(np.maximum(lam, 1e-3)), cmap="viridis", s=24)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(r"$\log_{10}\lambda$  ($\lambda=0$ shown at $-3$)")
    ax.set_xlabel("retain drift (mean over protected layers)")
    ax.set_ylabel("FID")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "retain_drift_vs_fid.png"), dpi=200)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

TIMESTEPS = None


def main():
    global TIMESTEPS
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs_dir",
                   default="/shared/results/common/miksa/intact/DDPM/lambda_sweep/runs")
    p.add_argument("--config", default="configs/pipeline_fulleval.yaml")
    p.add_argument("--orig_ckpt_folder", default=None,
                   help="default: paths.pretrained_ckpt_folder of --config")
    p.add_argument("--label_to_forget", type=int, default=0)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--timesteps", type=int, nargs="+", default=[50, 250, 500, 750, 950])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_path", default="results/drift_eval_batch.pt")
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = yaml.safe_load(open(args.config))
    cfg["unlearn"]["label_to_forget"] = args.label_to_forget
    # build_runner_config creates <results_dir>/runs/<suffix>/{logs,ckpts};
    # point it at a scratch dir inside out_dir.
    runner_config = build_runner_config(cfg, os.path.join(args.out_dir, "_tmp"), "drift")
    runner_args = build_runner_args(cfg, runner_config, args.seed)
    orig_folder = args.orig_ckpt_folder or cfg["paths"]["pretrained_ckpt_folder"]
    orig_ckpt = os.path.join(orig_folder, "ckpts", "ckpt.pth")

    # ---- fixed evaluation batch (identical across runs / invocations) ------
    if os.path.isfile(args.batch_path):
        batch = torch.load(args.batch_path)
        log.info(f"loaded eval batch {args.batch_path}")
        if batch["timesteps"].tolist() != args.timesteps:
            raise ValueError("saved eval batch has different timesteps; "
                             "delete it or pass a different --batch_path")
    else:
        batch = build_eval_batch(runner_args, runner_config, args.label_to_forget,
                                 args.n, args.timesteps, args.seed)
        os.makedirs(os.path.dirname(os.path.abspath(args.batch_path)), exist_ok=True)
        torch.save(batch, args.batch_path)
        log.info(f"saved eval batch {args.batch_path}")
    TIMESTEPS = batch["timesteps"].tolist()

    orig = load_model(runner_config, orig_ckpt, device)
    finder = make_protection(cfg, {})
    finder._find_target_layers(orig)
    layer_names = list(finder.target_layers)
    mods_o = {n: dict(orig.module.named_modules())[n] for n in layer_names}
    log.info(f"{len(layer_names)} protected layers: {layer_names}")
    snapshot = {n: p.detach().clone() for n, p in orig.named_parameters()}

    # ---- sanity: original vs itself must give exactly 0 --------------------
    orig2 = load_model(runner_config, orig_ckpt, device)
    mods_o2 = {n: dict(orig2.module.named_modules())[n] for n in layer_names}
    sanity = drift_for_split(orig, orig2, mods_o, mods_o2, batch["retain"], device, args.bs)
    worst = max(abs(v) for v in sanity.values())
    log.info(f"sanity: max self-drift of original model = {worst:.3e}")
    if worst != 0.0:
        raise RuntimeError(f"self-drift of the original model is {worst}, expected 0")
    del orig2, mods_o2

    runs, skipped = discover_runs(args.runs_dir)
    log.info(f"{len(runs)} runs, {len(skipped)} skipped")
    for s in skipped:
        log.info(f"  skipped {s['run']}: {s['reason']}")

    rows, per_run = [], []
    for i, run in enumerate(runs):
        log.info(f"[{i + 1}/{len(runs)}] {run['run']}  lambda={run['lambda']} seed={run['seed']}")
        new = load_model(runner_config, run["ckpt"], device)
        mods_n = {n: dict(new.module.named_modules())[n] for n in layer_names}
        rd = drift_for_split(orig, new, mods_o, mods_n, batch["retain"], device, args.bs)
        fd = drift_for_split(orig, new, mods_o, mods_n, batch["forget"], device, args.bs)

        pen = {k: float("nan") for k in PENALTY_PARTS}
        if run["pca"] is not None:
            pca_info = torch.load(run["pca"], map_location="cpu")
            pen = penalty_terms(cfg, run["meta"], pca_info, new, snapshot, device)

        base = {"lambda": run["lambda"], "seed": run["seed"], **pen,
                "UA": run["UA"], "RA": run["RA"], "FID": run["FID"],
                "family": run["family"], "run": run["run"]}
        for name in layer_names:
            rows.append({**base, "layer": name,
                         "retain_drift": rd[name], "forget_drift": fd[name]})
        per_run.append({**base, "layer": "mean",
                        "retain_drift": float(np.mean(list(rd.values()))),
                        "forget_drift": float(np.mean(list(fd.values())))})
        del new

    def write(path, data, fields):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)

    write(os.path.join(args.out_dir, "drift.csv"), rows, COLUMNS)
    write(os.path.join(args.out_dir, "drift_per_run.csv"), per_run, COLUMNS)
    write(os.path.join(args.out_dir, "skipped.csv"), skipped, ["run", "reason"])

    sp = spearman_table(per_run)
    write(os.path.join(args.out_dir, "spearman.csv"), sp, ["x", "y", "n", "rho", "p"])
    print("\nSpearman over checkpoints (drift = mean over protected layers):")
    for r in sp:
        print(f"  {r['x']:>12} vs {r['y']:<6}  n={r['n']:3d}  rho={r['rho']:+.3f}  p={r['p']:.2e}")

    print("\nper-lambda (mean over seeds):")
    print("| lambda | n | retain drift | forget drift | penalty (unweighted) | UA | RA | FID |")
    print("|---|---|---|---|---|---|---|---|")
    for lam in sorted({r["lambda"] for r in per_run}):
        rr = [r for r in per_run if r["lambda"] == lam]
        f = lambda k: np.nanmean([r[k] for r in rr]) if any(np.isfinite(r[k]) for r in rr) else float("nan")
        print(f"| {lam:g} | {len(rr)} | {f('retain_drift'):.4f} | {f('forget_drift'):.4f} | "
              f"{f('penalty'):.4g} | {f('UA'):.3f} | {f('RA'):.4f} | {f('FID'):.2f} |")

    make_plots(per_run, args.out_dir)
    log.info(f"wrote drift.csv, drift_per_run.csv, skipped.csv, spearman.csv and plots to {args.out_dir}")


if __name__ == "__main__":
    main()
