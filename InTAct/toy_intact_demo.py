#!/usr/bin/env python3
"""
InTAct toy-model visualisation (nn.Linear case only).

End-to-end mirror of InTAct/intact.py on a toy MLP:

  1. synthetic 2-class Gaussian data in R^96 (class A has 2-dim "content"
     structure; forget = random half of class A, remain = other half of A +
     all of class B),
  2. pretrain a small MLP (fc1/fc2 ReLU hidden + fc3 head),
  3. InTAct setup on the pretrained net:
       forget activations -> centre (mu) -> SVD -> top-k U_forget + U_residual,
       z_min/z_max  = 5/95-percentile intervals of forget projections,
       inf_low/high = padded interval, optionally widened by the *actual*
       remain-set projections (use_actual_bounds),
  4. unlearning: GA (gradient ascent on forget CE) and RL (random labels on
     forget), each unprotected / protected with the InTAct loss term,
  5. trajectories of delta_W(t) and, for linear layers, the per-output-row
     coefficients  c_o(t) = delta_W U_forget^T  (what the interval term sees),
  6. figures:
       f01 unlearning curves (GA / RL x unprotected / protected variants),
       f02 protection-loss components during unlearning,
       f03 interval geometry in the top-2 subspace (data side),
       f04 EXACT 2D slice of the protection loss along the two top SVD
           directions with trajectories overlaid and the "hypercube faces"
           (forget-box and inf-box edge-drift = 1 boundaries) marked,
       f05 EXACT 3D surface of that slice with hypercube faces on floor and
           surface,
       f05_<M>_3d_interactive.html  interactive plotly version (drag to
           rotate; click the legend to toggle surface / trajectories /
           hypercube faces),
       f06 1D cuts through the loss along canonical probes (top directions,
           residual direction, mu direction, random) with per-term breakdown,
       f07 actual drift of activation coordinates vs the collected intervals.

Note: the key facts about the implementation being visualised:
  * z_min / z_max / inf_low / inf_high are CONSTANTS computed once at
    setup (forget set is never re-collected); what changes during unlearning
    is only delta_W, i.e. c(t) = delta_W U_forget^T -- the loss is a fixed,
    piecewise-quadratic function of the CURRENT weights.
  * with --pretrain 3000 (fully converged net) lambda=1 pins GA completely at
    the snapshot; with --pretrain 600 the equilibrium allows partial
    forgetting.  RL (random labels) forgets through the protection.
  * remain-tail outliers (--tail) widen the actual-bounds inf box beyond the
    padded one -- visible effect: over-protection on the forget side.

Usage: python toy_intact_demo.py [--quick] [--steps N] [--lr X] [--lam X]
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

sys_path = os.path.dirname(os.path.abspath(__file__))
if sys_path not in sys.path:
    sys.path.insert(0, sys_path)
from intact import UnlearnIntervalProtection, classification_forward_fn

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 130,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
})
OUT = os.path.join(sys_path, "toy_figs")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
D_IN = 96
HID = 48
N_FORGET, N_REMAIN_A, N_B = 800, 800, 1000
REDUCED_DIM = 8
TARGETS = ["fc1", "fc2", "fc3"]
STATS_LAYERS = ["fc1", "fc2"]   # drift diagnostics on these inputs


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def make_data(seed=0, d_sep=2.4, s1=1.4, s2=0.45, noise=0.3, tail=0.15):
    """
    Class A lives near a 2-dim 'content plane' spanned by qA1,qA2 with scales
    (s1,s2); class B likewise on its own plane.  Forget = the half of class A
    with positive content coordinate alongside qA1, remain = the other half
    (negative) plus all of class B.

    The half-plane split is crucial: the forget activation box and the remain
    box then sit on OPPOSITE sides of the top forget-SVD direction, so the
    interval term can distinguish "drift that hits forget" from "drift that
    would also hit remain".

    `tail` = fraction of remain-A samples pushed far into the negative content
    direction: these give the *remaining set* a box larger than the padded
    forget box, so that `use_actual_bounds=True` (collecting the true remain
    projections) leads to different inf_low/inf_high than the padding alone.
    """
    g = torch.Generator().manual_seed(seed)

    def plane(d):
        a = torch.randn(d, generator=g); b = torch.randn(d, generator=g)
        b = b - (a @ b) * a
        return a / a.norm(), b / b.norm()

    qA1, qA2 = plane(D_IN); qB1, qB2 = plane(D_IN)
    mA = torch.zeros(D_IN)
    mB = torch.zeros(D_IN)
    mB[:] = d_sep * torch.randn(D_IN, generator=g) / torch.randn(
        D_IN, generator=g).norm()

    def content(n, sign):
        t1 = s1 * torch.abs(torch.randn(n, generator=g)) * sign
        t2 = torch.randn(n, generator=g) * s2
        e = torch.randn(n, D_IN, generator=g) * noise
        return mA + t1[:, None] * qA1 + t2[:, None] * qA2 + e

    X_f = content(N_FORGET, +1.0)
    X_rA = content(N_REMAIN_A, -1.0)
    if tail > 0:
        n_out = int(tail * N_REMAIN_A)
        out_t1 = -(3.0 + 2.5 * torch.rand(n_out, generator=g)) * s1
        out_t2 = (torch.rand(n_out, generator=g) - 0.5) * s2 * 2
        out_e = torch.randn(n_out, D_IN, generator=g) * noise
        X_rA[:n_out] = (mA
                        + out_t1[:, None] * qA1
                        + out_t2[:, None] * qA2 + out_e)

    def b_class(n):
        t1 = torch.randn(n, generator=g) * s1
        t2 = torch.randn(n, generator=g) * s2
        e = torch.randn(n, D_IN, generator=g) * noise
        return mB + t1[:, None] * qB1 + t2[:, None] * qB2 + e

    B = b_class(N_B)
    yA = torch.zeros(N_FORGET + N_REMAIN_A, dtype=torch.long)
    yB = torch.ones(N_B, dtype=torch.long)
    X_r = torch.cat([X_rA, B]); y_r = torch.cat([yA[N_FORGET:], yB])
    return dict(X_f=X_f, y_f=yA[:N_FORGET], X_rA=X_rA,
                y_rA=yA[N_FORGET:], X_B=B, y_B=yB, X_r=X_r, y_r=y_r)


def make_loaders(data, bs=256):
    def dl(X, y):
        return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=False)
    return dict(forget=dl(data["X_f"], data["y_f"]),
                remain=dl(data["X_r"], data["y_r"]))


class ToyMLP(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(D_IN, HID)
        self.fc2 = nn.Linear(HID, HID)
        self.fc3 = nn.Linear(HID, 2)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)


def train_model(model, X, y, steps=3000, lr=3e-3, log_every=500):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.zeros(())
    for it in range(steps):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward(); opt.step()
        if (it + 1) % log_every == 0:
            acc = (model(X).argmax(1) == y).float().mean().item()
            print(f"    pretrain {it+1:5d} loss={loss.item():.4f} acc={acc:.3f}")
    model.eval()
    acc = (model(X).argmax(1) == y).float().mean().item()
    print(f"    pretrain done loss={loss.item():.4f} acc={acc:.3f}")


# ---------------------------------------------------------------------------
# protection setup helpers
# ---------------------------------------------------------------------------
def make_protection(targets, model, dls, device, use_actual_bounds, **kw):
    kw.setdefault("use_actual_bounds", use_actual_bounds)
    p = UnlearnIntervalProtection(targets=targets, **kw)
    p.setup_protection(
        model,
        forget_dataloader=dls["forget"],
        remain_dataloader=dls["remain"] if use_actual_bounds else None,
        device=device, forward_fn=classification_forward_fn)
    return p


def linear_terms(info, dW, db=None):
    """Pure-function replica of the nn.Linear branch of
    compute_protection_loss (intact.py ~lines 467-500), unscaled per-layer."""
    mu = info["mu"]; Uf = info["U_forget"]; Ur = info["U_residual"]
    Sr = info["S_residual"]
    z_min = info["z_min"]; z_max = info["z_max"]
    inf_low = info["inf_low"]; inf_high = info["inf_high"]
    if db is None:
        db = torch.zeros((), dtype=dW.dtype)
    t = {}
    t["mean_shift"] = (dW @ mu + db).pow(2).mean()
    if Ur.size(0) > 0:
        wi = dW @ Ur.T * Sr.unsqueeze(0)
        t["residual"] = torch.norm(wi, p="fro").pow(2) / wi.numel()
    else:
        t["residual"] = torch.zeros(())
    delta_f = dW @ Uf.T
    Wp, Wn = torch.relu(delta_f), torch.relu(-delta_f)
    dl1 = Wp @ inf_low - Wn @ z_min
    dl2 = Wp @ z_min - Wn @ inf_low
    dh1 = Wp @ z_max - Wn @ inf_high
    dh2 = Wp @ inf_high - Wn @ z_max
    t["drift_low1"] = dl1.pow(2).mean()
    t["drift_low2"] = dl2.pow(2).mean()
    t["drift_high1"] = dh1.pow(2).mean()
    t["drift_high2"] = dh2.pow(2).mean()
    t["interval"] = sum(t[k] for k in ("drift_low1", "drift_low2",
                                       "drift_high1", "drift_high2"))
    t["layer"] = t["mean_shift"] + t["residual"] + t["interval"]
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in t.items()}


def check_terms(prot, model, trials=6):
    torch.manual_seed(123)
    for _ in range(trials):
        with torch.no_grad():
            for mod in prot.target_layers.values():
                if isinstance(mod, nn.Linear):
                    mod.weight.add_(0.02 * torch.randn_like(mod.weight))
        real = prot.compute_protection_loss(model, "cpu") / prot.lambda_interval
        s, nl = 0.0, 0
        for info in prot.pca_info:
            ln = info["layer_name"]; mod = prot.target_layers[ln]
            if not isinstance(mod, nn.Linear):
                continue
            wname = prot.param_to_name[mod.weight]
            bname = prot.param_to_name[mod.bias]
            dW = mod.weight - prot.params_snapshot[wname]
            db = mod.bias - prot.params_snapshot[bname]
            s += linear_terms(info, dW, db)["layer"]
            nl += 1
        s = s / nl
        assert abs(s - real.item()) < 1e-4 * (abs(real.item()) + 1e-9), \
            f"term mismatch: {s} vs {real.item()}"
    print("    term-decomposition matches compute_protection_loss "
          "(random-perturbation check passed)")


# ---------------------------------------------------------------------------
# forward-activation snapshots for diagnostics
# ---------------------------------------------------------------------------
def forward_acts(model, sets, layer_names):
    """Return {layer: {setname: act [n, in]}} for the CURRENT model weights."""
    out = {n: {} for n in layer_names}
    model.eval()

    def collect_for(X):
        cache = {n: [] for n in layer_names}
        hooks = []

        def make(name):
            def hook(mod, inp, out_):
                x = inp[0].detach()
                if isinstance(mod, nn.Linear):
                    cache[name].append(x.reshape(-1, mod.in_features))
                else:
                    cache[name].append(x.reshape(x.size(0), -1))
            return hook

        for name, mod in model.named_modules():
            if name in layer_names:
                hooks.append(mod.register_forward_hook(make(name)))
        with torch.no_grad():
            model(X)
        for h in hooks:
            h.remove()
        return {n: torch.cat(v) for n, v in cache.items()}

    for sname in sets:
        tmp = collect_for(sets[sname])
        for n in layer_names:
            out[n][sname] = tmp[n]
    return out


def coords_top2(prot, layer_name, acts):
    info = next(i for i in prot.pca_info if i["layer_name"] == layer_name)
    U = info["U_forget"][:2]
    return (acts - info["mu"]) @ U.T


def set_stats(prot, layer_name, acts):
    """per-direction mean / 5 / 95 percentile coords along all top-k dirs."""
    info = next(i for i in prot.pca_info if i["layer_name"] == layer_name)
    U = info["U_forget"]
    Z = (acts - info["mu"]) @ U.T
    q = torch.quantile(Z, torch.tensor([0.05, 0.5, 0.95]), dim=0)
    return dict(mean=Z.mean(0), q05=q[0], q50=q[1], q95=q[2])


# ---------------------------------------------------------------------------
# unlearning
# ---------------------------------------------------------------------------
def run_unlearning(model, prot, apply, data, sets, method, steps, lr,
                   stats_every, y_rand_seed=99):
    """Mutates model weights in place. Returns trajectory record.
    prot is always given (used for diagnostics); if apply=False the
    protection loss is NOT added to the objective."""
    assert method in ("GA", "RL")
    if method == "GA":
        unl = lambda lg: F.cross_entropy(lg, data["y_f"])
        sign = -1.0
    else:
        g = torch.Generator().manual_seed(y_rand_seed)
        y_rand = torch.randint(0, 2, (len(data["y_f"]),), generator=g)
        unl = lambda lg: F.cross_entropy(lg, y_rand)
        sign = 1.0
    X_f = data["X_f"]

    sn = {n: p.detach().clone() for n, p in model.named_parameters()}
    rec = dict(step=[], forget_loss=[], forget_acc=[], remain_acc=[],
               remainA_acc=[], classB_acc=[], prot_loss=[], terms=[],
               c_hist=[], dw_norm=[], stats=[])
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def record(it, save_stats):
        model.eval()
        with torch.no_grad():
            lf = model(X_f)
            rec["forget_loss"].append(
                F.cross_entropy(lf, data["y_f"]).item())
            rec["forget_acc"].append(
                (lf.argmax(1) == data["y_f"]).float().mean().item())
            lr_ = model(data["X_r"])
            rec["remain_acc"].append(
                (lr_.argmax(1) == data["y_r"]).float().mean().item())
            lA = model(data["X_rA"])
            rec["remainA_acc"].append(
                (lA.argmax(1) == data["y_rA"]).float().mean().item())
            lB = model(data["X_B"])
            rec["classB_acc"].append(
                (lB.argmax(1) == data["y_B"]).float().mean().item())
        rec["step"].append(it)
        nrm = 0.0
        crows = {}
        for info in prot.pca_info:
            ln = info["layer_name"]; mod = prot.target_layers[ln]
            if not isinstance(mod, nn.Linear):
                continue
            wname = prot.param_to_name[mod.weight]
            bname = prot.param_to_name[mod.bias]
            dW = mod.weight - sn[wname]
            db = mod.bias - sn[bname]
            nrm += dW.pow(2).sum().item() + db.pow(2).sum().item()
            crows[ln] = (dW @ info["U_forget"].T).detach().clone()
        rec["dw_norm"].append(float(np.sqrt(nrm)))
        rec["c_hist"].append(crows)
        rec["prot_loss"].append(
            prot.compute_protection_loss(model, "cpu").item())
        tsum = {}
        for info in prot.pca_info:
            ln = info["layer_name"]; mod = prot.target_layers[ln]
            if not isinstance(mod, nn.Linear):
                continue
            wname = prot.param_to_name[mod.weight]
            bname = prot.param_to_name[mod.bias]
            t = linear_terms(info, mod.weight - sn[wname],
                             mod.bias - sn[bname])
            for k, v in t.items():
                tsum[k] = tsum.get(k, 0.0) + v
        nlay = len(list(prot.pca_info))
        rec["terms"].append({k: v / nlay for k, v in tsum.items()})
        if save_stats:
            acts = forward_acts(model, sets, STATS_LAYERS)
            for ln in STATS_LAYERS:
                if ln not in prot.target_layers:
                    continue
                for sname in sets:
                    st = set_stats(prot, ln, acts[ln][sname])
                    rec["stats"].append((it, ln, sname, st))
        model.train()

    model.train()
    for it in range(1, steps + 1):
        opt.zero_grad()
        logits = model(X_f)
        loss = sign * unl(logits)
        if apply:
            loss = loss + prot.compute_protection_loss(model, "cpu")
        loss.backward()
        opt.step()
        if it % stats_every == 0 or it == steps:
            record(it, save_stats=True)
    model.eval()
    return rec


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------
def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


CMAP = "viridis"

LINE_CFG = {"GA_none": dict(c="C0", ls="--", label="GA, no protection"),
            "GA_fg": dict(c="C1", ls="-.", label="GA, InTAct (padded bounds)"),
            "GA_fg+rem": dict(c="C1", ls="-", label="GA, InTAct (actual remain bounds)"),
            "RL_none": dict(c="C2", ls="--", label="RL, no protection"),
            "RL_fg": dict(c="C3", ls="-.", label="RL, InTAct (padded bounds)"),
            "RL_fg+rem": dict(c="C3", ls="-", label="RL, InTAct (actual remain bounds)")}


def fig_unlearning_curves(runs, acc0):
    fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.2))
    for key, r in runs.items():
        m = key.split("_")[0]
        st = ax[0, 0] if m == "GA" else ax[1, 0]
        sa = ax[0, 1] if m == "GA" else ax[1, 1]
        lc = LINE_CFG[key]
        st.plot(r["step"], r["forget_loss"], **lc)
        sa.plot(r["step"], np.array(r["remain_acc"]) * 100, **lc)
    for a, t in [(ax[0, 0], "GA: forget CE loss (up = forgetting)"),
                 (ax[1, 0], "RL: forget CE loss (up = forgetting)"),
                 (ax[0, 1], "GA: remain accuracy (%)"),
                 (ax[1, 1], "RL: remain accuracy (%)")]:
        a.set_title(t); a.grid(alpha=0.3); a.legend(loc="best")
    for m, row in [("GA", 0), ("RL", 1)]:
        ax[row, 1].axhline(acc0["remain"] * 100, color="k", ls=":", lw=1)
    fig.suptitle("Unlearning dynamics (dashed = unprotected)")
    savefig(fig, "f01_unlearning_curves.png")


def fig_protection_terms(runs):
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    keys = [k for k in runs if "fg+rem" in k]
    cols = ["mean_shift", "residual", "interval"]
    for k in keys:
        r = runs[k]
        m = k.split("_")[0]
        a = ax[0] if m == "GA" else ax[1]
        for c_ in cols:
            a.plot(r["step"], [t[c_] for t in r["terms"]],
                   label=c_, lw=1.4)
        ax[2].plot(r["step"], r["dw_norm"], label=m + " InTAct",
                   lw=1.5, ls="-" if m == "GA" else "--")
    for a in ax[:2]:
        a.set_xlabel("unlearning step"); a.set_yscale("log")
        a.grid(alpha=0.3); a.legend(); a.set_title(
            "InTAct loss components\n(mean shift | residual | interval)")
    ax[2].set_xlabel("unlearning step"); ax[2].set_yscale("log")
    for k in runs:
        ax[2].plot(runs[k]["step"], runs[k]["dw_norm"],
                   color=LINE_CFG[k]["c"], ls=LINE_CFG[k]["ls"], lw=1,
                   label=LINE_CFG[k]["label"])
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=0.3)
    ax[2].set_title(r"$\|\Delta W\|_F$ + $|\Delta b|$ (all layers)")
    fig.suptitle("What the protection term does during unlearning")
    savefig(fig, "f02_protection_terms.png")


def fig_interval_geometry(prot, model, data, sets, layer_names):
    acts = forward_acts(model, sets, layer_names)
    n = len(layer_names)
    fig, axs = plt.subplots(1, n, figsize=(4.2 * n + 1, 4.2))
    for i, ln in enumerate(layer_names):
        ax = axs[i]
        info = next(j for j in prot.pca_info if j["layer_name"] == ln)
        C = coords_top2(prot, ln, acts[ln]["forget"])
        ax.scatter(C[:, 0], C[:, 1], s=4, c="tab:red", alpha=0.4,
                   label="forget (A)", linewidths=0)
        C = coords_top2(prot, ln, acts[ln]["remainA"])
        ax.scatter(C[:, 0], C[:, 1], s=4, c="tab:green", alpha=0.35,
                   label="remain (A)", linewidths=0)
        C = coords_top2(prot, ln, acts[ln]["classB"])
        ax.scatter(C[:, 0], C[:, 1], s=4, c="tab:blue", alpha=0.3,
                   label="class B", linewidths=0)
        for lo, hi, col, ls, lab in [
                (info["z_min"], info["z_max"], "tab:red", "--",
                 "forget interval [z_min,z_max]"),
                (info["inf_low"], info["inf_high"], "tab:purple", "-.",
                 "[inf_low,inf_high]")]:
            ax.add_patch(Rectangle(
                (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
                fill=False, ec=col, ls=ls, lw=1.5, label=lab))
        ax.set_title(f"layer '{ln}' input, top-2 SVD dirs of forget set\n"
                     f"(eig share {float(info['S_residual'][0])**2:0.2f}, "
                     f"{float(info['S_residual'][1])**2:0.2f})")
        ax.set_xlabel("coord $z_1$"); ax.set_ylabel("coord $z_2$")
        ax.legend(loc="upper right", fontsize=6.5)
        ax.grid(alpha=0.2)
    fig.suptitle("Activation geometry + collected intervals "
                 "(perpendicular view: centered, projected on forget-SVD plane)")
    for ln in layer_names:
        info = next(j for j in prot.pca_info if j["layer_name"] == ln)
        nf = acts[ln]["forget"].shape[0]
        nr = acts[ln]["remainA"].shape[0]
        nb = acts[ln]["classB"].shape[0]
        print(f"  [f03] {ln}: forget box u1 [{info['z_min'][0]:+.2f},"
              f"{info['z_max'][0]:+.2f}]  u2 [{info['z_min'][1]:+.2f},"
              f"{info['z_max'][1]:+.2f}] |  inf box u1 [{info['inf_low'][0]:+.2f},"
              f"{info['inf_high'][0]:+.2f}]  u2 [{info['inf_low'][1]:+.2f},"
              f"{info['inf_high'][1]:+.2f}]  (samples {nf},{nr},{nb})")
    savefig(fig, "f03_interval_geometry.png")


# ---------------------------------------------------------------------------
# exact loss-landscape slicing (per protected linear layer)
# ---------------------------------------------------------------------------
class Landscape:
    """Exact evaluation of one linear layer's protection loss as a function of
    delta_W. All rows of delta_W share one vector v -> the value equals the
    mean per-output-row loss; the 'interval' terms live on delta_f = dW Uf^T,
    i.e. on the top-k coefficients c = v.Uf^T (k dims, the 'true' dimension
    of the interval part).

    On any plane spanned by two top-k directions u_a, u_b the residual term
    vanishes (u_a,u_b are orthogonal to all U_residual rows) and the formulas
    reduce to pure functions of (c_a, c_b) with the other k-2 coefficients
    zero -- which the vectorized `plane()` exploits.  `at()` evaluates at a
    single point via the exact batched dW matrix (used for validation).
    """

    def __init__(self, prot, layer_name, d_out):
        self.info = next(i for i in prot.pca_info
                         if i["layer_name"] == layer_name)
        self.d_out = d_out
        d = self.info["mu"].size(0)
        self.k = self.info["U_forget"].size(0)

    def rowmat(self, v):
        return v.to(torch.float32).expand(self.d_out, -1)

    # ------------------------------------------------------------------
    def plane(self, e1, e2, A1, A2, ng=201):
        """Vectorised EXACT evaluation over the plane v = a1 e1 + a2 e2.

        Returns (A, B, val, box) with
          val: dict of term arrays (interval, mean_shift, residual=0, layer)
          box: dict with z_ext / inf_ext  = max(|drift of the box EDGES|)
               over the plane.  'box edge drift = 1' is the boundary of the
               hypercube {c : box extremes pushed by one response unit}.
        """
        a1 = torch.linspace(-A1, A1, ng, dtype=torch.float32)
        a2 = torch.linspace(-A2, A2, ng, dtype=torch.float32)
        A, B = torch.meshgrid(a1, a2, indexing="ij")
        c = torch.zeros(ng, ng, self.k, dtype=torch.float32)
        c[:, :, 0] = A
        c[:, :, 1] = B
        inf = self.info
        z_min, z_max = inf["z_min"].float(), inf["z_max"].float()
        i_lo, i_hi = inf["inf_low"].float(), inf["inf_high"].float()
        Wp, Wn = torch.relu(c), torch.relu(-c)

        # one-sided edge drifts (per output row; identical rows -> same scalar)
        z_hi = (Wp * z_max).sum(-1) - (Wn * z_min).sum(-1)       # = Wp@z_max - Wn@z_min
        z_lo = (Wp * z_min).sum(-1) - (Wn * z_max).sum(-1)       # = Wp@z_min - Wn@z_max
        i_hi_ = (Wp * i_hi).sum(-1) - (Wn * i_lo).sum(-1)        # inf box, top edge
        i_lo_ = (Wp * i_lo).sum(-1) - (Wn * i_hi).sum(-1)        # inf box, bottom edge

        dl1 = (Wp * i_lo).sum(-1) - (Wn * z_min).sum(-1)
        dl2 = (Wp * z_min).sum(-1) - (Wn * i_lo).sum(-1)
        dh1 = (Wp * z_max).sum(-1) - (Wn * i_hi).sum(-1)
        dh2 = (Wp * i_hi).sum(-1) - (Wn * z_max).sum(-1)
        interval = dl1.pow(2) + dl2.pow(2) + dh1.pow(2) + dh2.pow(2)

        mu = inf["mu"].float()
        p1, p2 = (e1.float() @ mu).item(), (e2.float() @ mu).item()
        mean_shift = (A * p1 + B * p2).pow(2)

        val = dict(interval=interval.numpy(),
                   mean_shift=mean_shift.numpy(),
                   residual=np.zeros_like(interval.numpy()),
                   layer=(mean_shift + interval).numpy())
        box = dict(z_ext=torch.maximum(z_hi.abs(), z_lo.abs()).numpy(),
                   inf_ext=torch.maximum(i_hi_.abs(), i_lo_.abs()).numpy())
        return A.numpy(), B.numpy(), val, box

    def slice1d(self, v, A, ng=401):
        aa = torch.linspace(-A, A, ng, dtype=torch.float32)
        out = {k: [] for k in ("layer", "mean_shift", "residual", "interval")}
        for a in aa:
            t = linear_terms(self.info, self.rowmat(a * v))
            for k in out:
                out[k].append(t[k])
        return aa.numpy(), {k: np.array(v) for k, v in out.items()}


def contour_vertices(X, Y, Z, level):
    """Extract level-set polylines from a 2D grid (for 3D rendering)."""
    fig, ax = plt.subplots(figsize=(0.1, 0.1))
    cs = ax.contour(X, Y, Z, levels=[level])
    verts = [np.asarray(seg) for sl in cs.allsegs for seg in sl if len(seg)]
    plt.close(fig)
    return verts


def plane_surface_heights(land, verts, A, ng):
    """Exact interval-loss at a list of (a1,a2) vertices (fast vectorised)."""
    if verts is None or len(verts) == 0:
        return np.array([])
    V = np.array(verts).reshape(-1, 2)
    c = np.zeros((V.shape[0], land.k), dtype=np.float32)
    c[:, 0], c[:, 1] = V[:, 0], V[:, 1]
    Wp = np.maximum(c, 0.0)
    Wn = np.maximum(-c, 0.0)
    inf = land.info
    z_min, z_max = inf["z_min"].numpy(), inf["z_max"].numpy()
    i_lo, i_hi = inf["inf_low"].numpy(), inf["inf_high"].numpy()
    dl1 = Wp @ i_lo - Wn @ z_min
    dl2 = Wp @ z_min - Wn @ i_lo
    dh1 = Wp @ z_max - Wn @ i_hi
    dh2 = Wp @ i_hi - Wn @ z_max
    return dl1 ** 2 + dl2 ** 2 + dh1 ** 2 + dh2 ** 2


def trajectory_range(keys, runs, plane=True):
    cmax = 1e-6
    for k in keys:
        if k in runs:
            for ch in runs[k]["c_hist"]:
                c = ch["fc1"]
                cmax = max(cmax, c[:, :2].abs().max().item() if plane
                           else c.abs().max().item())
    return float(cmax)


def mean_row_range(keys, runs, plane=True):
    """max |c| over the MEAN output row along the plane (2 coords)."""
    cmax = 1e-6
    for k in keys:
        if k in runs:
            for ch in runs[k]["c_hist"]:
                c = ch["fc1"].mean(0)
                cmax = max(cmax, c[:2].abs().max().item() if plane
                           else c.abs().max().item())
    return float(cmax)


def landscape_A(runs, prot_keys, any_keys=None):
    """Axis half-range: covers protected rows with margin, and the unprotected
    MEAN-row excursion (so hypercube faces stay visible)."""
    A = 1.15 * trajectory_range(prot_keys, runs, plane=True)
    if any_keys:
        A = max(A, 1.3 * mean_row_range(
            [k for k in any_keys if "none" in k], runs, plane=True))
    return A


def fig_landscape_2d(land, runs, tag_pick):
    """Heatmap of sqrt(loss) on the (u1,u2) plane, exact, with the
    per-step mean row-coefficient trajectory overlaid, and the box-edge
    drift=1 'hypercube' boundaries marked."""
    e1 = land.info["U_forget"][0]
    e2 = land.info["U_forget"][1]
    prot_keys = [k for k in tag_pick if "none" not in k]
    A = landscape_A(runs, prot_keys, tag_pick)
    print(f"landscape axis half-range A = {A:.4g}")
    Ag, Bg, V, box = land.plane(e1, e2, A, A, ng=251)
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.6),
                            sharex=True, sharey=True)
    fields = [("interval", "interval terms only (drift_low/high, 4 terms)"),
              ("layer", "FULL protection loss (mean-shift + residual + interval)"),
              ("interval", "interval terms only (log10)")]
    for ax, (fld, title) in zip(axs, fields):
        Vv = V[fld]
        if "log" in title:
            m = ax.pcolormesh(Ag, Bg, np.log10(1 + Vv), cmap=CMAP)
            cb = fig.colorbar(m, ax=ax)
            cb.set_label(r"$\log_{10}(1+\ell)$")
        else:
            m = ax.pcolormesh(Ag, Bg, np.sqrt(Vv), cmap=CMAP)
            cb = fig.colorbar(m, ax=ax)
            cb.set_label(r"$\sqrt{\ell}$")
        ax.set_title(title)
        ax.set_xlabel(r"$c_1$ : weight drift along top SVD dir 1")
        ax.set_ylabel(r"$c_2$ : weight drift along top SVD dir 2")
        # hinge lines at c1=0 / c2=0 (the relu boundaries)
        ax.axhline(0, color="w", lw=0.6, alpha=0.5)
        ax.axvline(0, color="w", lw=0.6, alpha=0.5)
        # hypercube faces: boundaries where pushing a box EDGE by one response
        # unit reaches +-1  (contour of max |edge drift| = 1)
        zc = ax.contour(Ag, Bg, box["z_ext"], levels=[1.0], colors="red",
                        linestyles="--", linewidths=1.4)
        ic = ax.contour(Ag, Bg, box["inf_ext"], levels=[1.0],
                        colors="purple", linestyles="-.", linewidths=1.4)
        handles = [Line2D([0], [0], color="red", ls="--", lw=1.4,
                          label="forget box edge drift = 1"),
                   Line2D([0], [0], color="purple", ls="-.", lw=1.4,
                          label="inf/remain box edge drift = 1")]
        for key in tag_pick:
            r = runs[key]
            tr = np.array([ch["fc1"].mean(0).numpy() for ch in r["c_hist"]])
            lc = LINE_CFG[key]
            ax.plot(tr[:, 0], tr[:, 1], c=lc["c"], lw=2.2, ls=lc["ls"])
            handles.append(Line2D([0], [0], c=lc["c"], lw=2.2, ls=lc["ls"],
                                  label=lc["label"] + " (mean row)"))
        ax.legend(handles=handles, loc="upper left", fontsize=6.5)
    fig.suptitle(
        f"Exact slice of layer fc1 protection loss along the two top forget-SVD "
        f"directions\n(k={land.info['U_forget'].shape[0]} directions total; "
        f"other {land.info['U_forget'].shape[0]-2} held at 0; each output row identical)",
        fontsize=10)
    gmax = V["layer"].max()
    print(f"  [f04] heatmap half-range A={A:.4g}  "
          f"max(sqrt) interval={np.sqrt(V['interval'].max()):.3g}  "
          f"layer={np.sqrt(gmax):.3g}")
    for key in tag_pick:
        r = runs[key]
        tr = np.array([ch["fc1"].mean(0).numpy() for ch in r["c_hist"]])
        print(f"  [f04] {key:10s} mean-row c-range: "
              f"c1 [{tr[:, 0].min():+.3f},{tr[:, 0].max():+.3f}]  "
              f"c2 [{tr[:, 1].min():+.3f},{tr[:, 1].max():+.3f}]\n"
              f"        max|c| over rows: "
              f"{max(float(ch['fc1'][:, :2].abs().max()) for ch in r['c_hist']):.4g}")
    savefig(fig, "f04_landscape_2d.png")


def draw_box_faces_3d(ax, land, Ag, Bg, box, A, ng, proj_surface=True):
    """Draw the forget/inf box edge-drift = 1 boundaries (the visible faces of
    the k-dim hypercubes) on a 3D axis: dashed traces on the floor and solid
    traces pulled up onto the surface."""
    for key, col, style, label in [
            ("z_ext", "red", "--", "forget box edge drift = 1"),
            ("inf_ext", "purple", "-.", "inf/remain box edge drift = 1")]:
        verts = contour_vertices(Ag, Bg, box[key], 1.0)
        for k, v in enumerate(verts):
            ax.plot(v[:, 0], v[:, 1], np.zeros(len(v)), c=col, ls=style,
                    lw=1.0, alpha=0.7)
            if proj_surface:
                zs = np.log10(1 + plane_surface_heights(land, v, A, ng))
                ax.plot(v[:, 0], v[:, 1], zs, c=col, ls=style, lw=1.6,
                        label=label if k == 0 else None)


def fig_landscape_3d(land, runs):
    e1, e2 = land.info["U_forget"][0], land.info["U_forget"][1]
    A = landscape_A(runs, ["GA_fg+rem", "RL_fg+rem"],
                    ["GA_none", "RL_none"])
    ng = 121
    Ag, Bg, V, box = land.plane(e1, e2, A, A, ng=ng)
    fig = plt.figure(figsize=(13.5, 6.4))
    for i, key in enumerate(["GA_fg+rem", "RL_fg+rem"]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(Ag, Bg, np.log10(1 + V["interval"]), cmap=CMAP,
                        rstride=2, cstride=2, alpha=0.9)
        r = runs[key]
        tr = np.array([ch["fc1"].mean(0).numpy() for ch in r["c_hist"]])
        ax.plot(tr[:, 0], tr[:, 1], np.log10(1 + np.zeros(len(tr))),
                c="w", lw=2.4, label="mean row trajectory (floor)")
        draw_box_faces_3d(ax, land, Ag, Bg, box, A, ng)
        ax.set_title(key + ": mean row trajectory on the surface")
        ax.set_xlabel("$c_1$"); ax.set_ylabel("$c_2$")
        ax.set_zlabel(r"$\log_{10}(1+\ell)$")
        ax.view_init(elev=26, azim=-130)
        ax.legend(loc="upper left", fontsize=7)
    fig.suptitle("3D view of the exact interval-loss slice (fc1)\n"
                 "dashed/solid traces = hypercube faces (box-edge drift = 1)",
                 fontsize=10)
    savefig(fig, "f05_landscape_3d.png")


def fig_landscape_3d_interactive(land, runs):
    """Interactive (plotly) version of f05 with clickable layers: loss
    surface, trajectories and the two hypercube faces.  Saved as HTML."""
    import plotly.graph_objects as go

    e1, e2 = land.info["U_forget"][0], land.info["U_forget"][1]
    A = landscape_A(runs, ["GA_fg+rem", "RL_fg+rem"],
                    ["GA_none", "RL_none"])
    ng = 101
    Ag, Bg, V, box = land.plane(e1, e2, A, A, ng=ng)
    Z = np.log10(1 + V["interval"])

    zverts = contour_vertices(Ag, Bg, box["z_ext"], 1.0)
    iverts = contour_vertices(Ag, Bg, box["inf_ext"], 1.0)

    def vlines(verts, zfun, color, name, dash):
        trs = []
        for k, v in enumerate(verts):
            zs = zfun(v)
            trs.append(go.Scatter3d(
                x=v[:, 0], y=v[:, 1], z=zs, mode="lines",
                line=dict(color=color, width=4, dash=dash),
                name=name if k == 0 else None, showlegend=(k == 0),
                legendgroup=name))
        return trs

    for method, key in [("GA", "GA_fg+rem"), ("RL", "RL_fg+rem")]:
        fig = go.Figure()
        fig.add_trace(go.Surface(
            x=Ag, y=Bg, z=Z, colorscale="Viridis", opacity=0.92,
            name="interval loss (log10)", colorbar=dict(title=r"log10(1+l)")))

        # hypercube faces at the floor (z=0) and pulled onto the surface
        for trs in vlines(zverts, lambda v: np.zeros(len(v)),
                          "red", "forget box edge drift = 1", "dash"):
            fig.add_trace(trs)
        for trs in vlines(zverts, lambda v: np.log10(1 + plane_surface_heights(
                              land, v, A, ng)), "red", "forget box (on surface)", "solid"):
            fig.add_trace(trs)
        for trs in vlines(iverts, lambda v: np.zeros(len(v)),
                          "purple", "inf/remain box edge drift = 1", "dashdot"):
            fig.add_trace(trs)
        for trs in vlines(iverts, lambda v: np.log10(1 + plane_surface_heights(
                              land, v, A, ng)), "purple", "inf/remain box (on surface)", "solid"):
            fig.add_trace(trs)

        # trajectories: mean row for none / fg+rem
        for runkey, col, name in [(f"{method}_none", "#1f77b4",
                                   f"{method} unprotected (mean row)"),
                                  (f"{method}_fg+rem", "#ff7f0e",
                                   f"{method} InTAct (mean row)")]:
            r = runs[runkey]
            tr = np.array([ch["fc1"].mean(0).numpy() for ch in r["c_hist"]])
            fig.add_trace(go.Scatter3d(
                x=tr[:, 0], y=tr[:, 1], z=np.log10(1 + 0 * tr[:, 0]),
                mode="lines+markers", name=name,
                line=dict(color=col, width=5), marker=dict(size=3, color=col)))

        fig.update_layout(
            title=f"Exact interval-loss slice of fc1, {method} unlearning "
                  "(click legend to toggle layers)",
            scene=dict(xaxis_title="c1 (drift along top SVD dir 1)",
                       yaxis_title="c2 (drift along top SVD dir 2)",
                       zaxis_title="log10(1+interval loss)",
                       aspectmode="data",
                       camera=dict(eye=dict(x=1.7, y=1.7, z=1.3))),
            legend=dict(font=dict(size=11)),
            height=760, width=980)
        path = os.path.join(OUT, f"f05_{method}_3d_interactive.html")
        import plotly.io as pio
        pio.write_html(fig, path, include_plotlyjs="inline",
                       full_html=True)
        print("wrote", path)


def fig_landscape_1d(land, runs):
    info = land.info
    Uf = info["U_forget"]
    Ur = info["U_residual"]
    mu = info["mu"]
    rng = np.random.default_rng(7)
    probes = [("top dir 1 (u_1)", Uf[0]),
              ("top dir 2 (u_2)", Uf[1]),
              ("top dir 3 (u_3)", Uf[2]),
              ("residual dir (u^res_1)", Ur[0] if Ur.size(0) else None),
              ("mean dir  $\\hat\\mu$", mu / mu.norm()),
              ("random unit vector", torch.tensor(
                  rng.normal(size=mu.size(0))))]
    probes = [(n, v) for n, v in probes if v is not None]
    A = 1.4 * trajectory_range(["GA_fg+rem", "RL_fg+rem"], runs, plane=False)
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 7.6))
    for i, (name, v) in enumerate(probes):
        v = v / v.norm()
        ax = axs.flat[i]
        aa, V = land.slice1d(v, A=A)
        cols = dict(mean_shift="C0", residual="C1", interval="C2", layer="k")
        for kk, cc in cols.items():
            ax.plot(aa, V[kk], c=cc, lw=1.3,
                    label=kk if kk != "layer" else "total layer")
        ax.axvline(0, c="0.5", ls=":", lw=1)
        ax.set_title(name)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\alpha$  (weight drift: one output row moves along $v$)")
        if i == 0:
            ax.set_ylabel("loss term")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6)
    fig.suptitle(
        "1D cuts through the fc1 protection loss along canonical directions\n"
        "(top-k dirs: interval term; residual dirs: U_residual term; "
        "mu dir: mean-shift term. Kink at alpha=0 = the relu hinge in the code)")
    for (name, v), ax in zip(probes, axs.flat):
        v = v / v.norm()
        aa, V = land.slice1d(v, A=0.2 * A, ng=601)
        mid = A * 0.2
        kink = abs(V["interval"][np.argmin(np.abs(aa))] -
                   V["interval"][np.argmin(np.abs(aa)) + 1])
        print(f"  [f06] {name:20s} interval@alpha=0: {V['interval'][np.argmin(np.abs(aa))]:.2e} "
              f"(should be ~0), at +edge: {V['interval'][-1]:.2e}, "
              f"mean_shift@edge: {V['mean_shift'][-1]:.2e}")
    savefig(fig, "f06_landscape_1d.png")


def fig_activation_drift(prot, runs, sets):
    """For STATS_LAYERS inputs: mean coordinate drift along top dir during
    unlearning, relative to snapshot stats. Horizontal bands show the
    collected forget interval and remain 'safe' interval."""
    gakeys = [k for k in runs if k.startswith("GA") and "fg+rem" in k]
    gakeys.insert(0, "GA_none")
    dirs = [0, 1]
    layers = STATS_LAYERS
    fig, axs = plt.subplots(len(layers), len(dirs),
                            figsize=(4.2 * len(dirs) + 1.5, 3.4 * len(layers)))
    for li, ln in enumerate(layers):
        info = next(i for i in prot.pca_info if i["layer_name"] == ln)
        for di, d in enumerate(dirs):
            ax = axs[li, di]
            # collected intervals along direction d
            ax.axhspan(info["z_min"][d].item(), info["z_max"][d].item(),
                       color="tab:red", alpha=0.18)
            ax.axhspan(info["inf_low"][d].item(), info["inf_high"][d].item(),
                       color="tab:purple", alpha=0.15)
            ax.axhline(0, c="0.6", lw=0.8)
            for key in gakeys:
                r = runs[key]
                for sname, cc in [("forget", "tab:red"), ("remainA", "tab:green")]:
                    st = [(it, s) for it, ln_, sn, s in r["stats"]
                          if ln_ == ln and sn == sname]
                    st = sorted(st)
                    if not st:
                        continue
                    ts = [x[0] for x in st]
                    mm = [x[1]["mean"][d].item() for x in st]
                    q05 = [x[1]["q05"][d].item() for x in st]
                    q95 = [x[1]["q95"][d].item() for x in st]
                    if key == "GA_none":
                        stl = dict(c=cc, ls="--", lw=1.1, alpha=0.85)
                    else:
                        stl = dict(c=cc, ls="-", lw=1.6)
                    ax.plot(ts, mm, **stl,
                            label=f"{sname} {'unprotected' if 'none' in key else 'protected'} (mean)")
                    ax.fill_between(ts, q05, q95, color=cc, alpha=0.07)
            ax.set_title(f"layer '{ln}' input, SVD dir {d + 1}")
            ax.set_xlabel("unlearning step")
            ax.set_ylabel("mean activation coordinate\nrelative to snapshot")
            ax.grid(alpha=0.2)
            if li == 0 and di == 0:
                h, lb = ax.get_legend_handles_labels()
                ax.legend(handles=h[:4], labels=lb[:4], fontsize=6.5,
                          loc="upper left")
    for li, ln in enumerate(layers):
        info = next(i for i in prot.pca_info if i["layer_name"] == ln)
        tmp = []
        for key in gakeys:
            r = runs[key]
            for sname in ("forget", "remainA"):
                st = [(it, s) for it, ln_, sn, s in r["stats"]
                      if ln_ == ln and sn == sname]
                st = sorted(st)
                if st:
                    tmp.append((key, sname, st[-1][1]["mean"][0].item()))
        if tmp:
            print(f"  [f07] {ln} final mean coord along dir1:")
            for t in tmp:
                print(f"        {t[0]:10s} {t[1]:8s} {t[2]:+.3f}")
    fig.suptitle(
        "Actual drift of the mean activation coordinate (5-95% spread shaded) "
        "during GA\nred band = collected forget interval [z_min,z_max]; "
        "purple band = padded/remain interval [inf_low,inf_high]",
        fontsize=10)
    savefig(fig, "f07_activation_drift.png")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=10.0)
    ap.add_argument("--inf_scale", type=float, default=1.0)
    ap.add_argument("--dsep", type=float, default=2.4)
    ap.add_argument("--tail", type=float, default=0.15,
                    help="fraction of remain-A samples pushed far into the "
                         "negative content direction (extends the actual-bounds "
                         "interval beyond the padded one)")
    ap.add_argument("--pretrain", type=int, default=600,
                    help="pretrain steps: 600='soft' fit -> protected GA can "
                         "still forget partly; 3000='converged' -> lambda=1 "
                         "pins GA at the snapshot (interesting failure mode)")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = make_data(seed=args.seed, d_sep=args.dsep, tail=args.tail)
    sets = dict(forget=data["X_f"], remainA=data["X_rA"], classB=data["X_B"])
    dls = make_loaders(data)
    Xall = torch.cat([data["X_f"], data["X_r"]])
    yall = torch.cat([data["y_f"], data["y_r"]])

    model = ToyMLP(seed=args.seed)
    steps_pretrain = args.pretrain
    train_model(model, Xall, yall, steps=steps_pretrain)

    def accs():
        model.eval()
        with torch.no_grad():
            return dict(
                forget=(model(data["X_f"]).argmax(1) == data["y_f"]).float().mean().item(),
                remain=(model(data["X_r"]).argmax(1) == data["y_r"]).float().mean().item(),
                remainA=(model(data["X_rA"]).argmax(1) == data["y_rA"]).float().mean().item(),
                classB=(model(data["X_B"]).argmax(1) == data["y_B"]).float().mean().item())
    acc0 = accs()
    print("pretrain acc:", {k: round(v, 3) for k, v in acc0.items()})

    # --- InTAct setup -------------------------------------------------
    base = dict(reduced_dim=REDUCED_DIM,
                lower_percentile=0.05, upper_percentile=0.95,
                lambda_interval=args.lam, infinity_scale=args.inf_scale)
    prot_fg = make_protection(TARGETS, model, dls, "cpu",
                              use_actual_bounds=False, **base)
    prot_rem = make_protection(TARGETS, model, dls, "cpu",
                               use_actual_bounds=True, **base)
    for tag, p in [("padded-only", prot_fg), ("actual-remain", prot_rem)]:
        check_terms(p, model)
        i0 = p.pca_info[0]
        print(f"  {tag}: z_min[:3]={i0['z_min'][:3].tolist()}",
              f"z_max[:3]={i0['z_max'][:3].tolist()}")
        print(f"  {tag}: inf_low[:3]={i0['inf_low'][:3].tolist()}",
              f"inf_high[:3]={i0['inf_high'][:3].tolist()}")
    print(f"  remaining variance in U_residual fc1: "
          f"{prot_rem.pca_info[0]['S_residual'].pow(2).sum():.3f}")
    for pa, pb in zip(prot_fg.pca_info, prot_rem.pca_info):
        d_lo = (pa["inf_low"] - pb["inf_low"]).abs().max().item()
        d_hi = (pa["inf_high"] - pb["inf_high"]).abs().max().item()
        print(f"  inf-box |diff| padded vs actual-remain, layer "
              f"{pa['layer_name']}: low {d_lo:.3f}  high {d_hi:.3f}")

    # --- landscape handle for fc1 (exact, independent of runs) ---------
    land = Landscape(prot_rem, "fc1", HID)
    # validate vectorised plane() against the exact linear_terms replicas
    e1, e2 = land.info["U_forget"][0], land.info["U_forget"][1]
    rng = np.random.default_rng(3)
    tol = 1e-5
    for _ in range(6):
        a1, a2 = rng.normal(size=2)
        c = torch.tensor([a1, a2, *([0.0] * (land.k - 2))], dtype=torch.float32)
        t = linear_terms(land.info, land.rowmat(c @ land.info["U_forget"]))
        v = a1 * e1 + a2 * e2
        t2 = linear_terms(land.info, land.rowmat(v.float()))
        assert abs(t["layer"] - t2["layer"]) < tol * (1 + t2["layer"]), \
            "plane basis mismatch"
    Ag, Bg, V, box = land.plane(e1, e2, 0.1, 0.1, ng=3)
    a1, a2 = float(Ag[2, 0]), float(Bg[2, 0])
    t3 = linear_terms(land.info, land.rowmat((a1 * e1 + a2 * e2).float()))
    assert abs(V["layer"][2, 0] - t3["layer"]) < tol * (1 + t3["layer"])
    print("  vectorised plane() matches exact linear_terms (check passed)")

    # --- snapshot/restore machinery -----------------------------------
    snapshot = {n: p.detach().clone() for n, p in model.named_parameters()}

    def restore():
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(snapshot[n])

    # --- unlearning runs -----------------------------------------------
    runs = {}
    stats_every = 15
    for method in ("GA", "RL"):
        for tag, prot in [("none", prot_fg), ("fg", prot_fg),
                          ("fg+rem", prot_rem)]:
            key = f"{method}_{tag}"
            print(f"\n== run {key} ==")
            restore()
            runs[key] = run_unlearning(model, prot, tag != "none", data, sets,
                                       method, args.steps, args.lr, stats_every)
    restore()
    for key, r in runs.items():
        print(f"{key:10s} final: forget_acc="
              f"{r['forget_acc'][-1]:.3f} remainA_acc="
              f"{r['remainA_acc'][-1]:.3f} classB_acc="
              f"{r['classB_acc'][-1]:.3f} forget_loss="
              f"{r['forget_loss'][-1]:.3f} ||dW||={r['dw_norm'][-1]:.4f}")
    for key in ("GA_none", "GA_fg+rem", "RL_none", "RL_fg+rem"):
        r = runs[key]
        fa = [round(v, 2) for v in r["forget_acc"]]
        ra = [round(v, 2) for v in r["remainA_acc"]]
        print(f"  curve {key:10s} forget_acc: {fa}")
        print(f"  curve {key:10s} remainA_acc: {ra}")

    # --- figures -------------------------------------------------------
    fig_unlearning_curves(runs, acc0)
    fig_protection_terms(runs)
    fig_interval_geometry(prot_rem, model, data, sets,
                          layer_names=["fc1", "fc2"])
    fig_landscape_2d(land, runs, tag_pick=["GA_none", "GA_fg+rem",
                                           "RL_none", "RL_fg+rem"])
    fig_landscape_3d(land, runs)
    try:
        fig_landscape_3d_interactive(land, runs)
    except ImportError:
        print("  [f05-interactive] plotly not installed, skipping HTML export")
    fig_landscape_1d(land, runs)
    fig_activation_drift(prot_rem, runs, sets)

    print(f"\ntotal wall time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
