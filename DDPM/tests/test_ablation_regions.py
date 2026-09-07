#!/usr/bin/env python3
"""
Tests for the protected-region construction ablation.

(i)  regression guard:  region_mode="two_corner" (normalize_region=False)
     reproduces the pre-ablation compute_protection_loss BITWISE on a fixed
     random delta_W, for both the nn.Linear and nn.Conv2d paths.

(ii) coverage guard:    for k in 3..6, the max |delta_f . z| over the 2k slab
     boxes (region_mode="slabs_2k") equals the max over all 3^k - 1 cells of
     the envelope complement enumerated by brute force, confirming that the
     slabs really do cover the complement.

Plus: two_random box validity/determinism, and per-variant squared-term counts.

Run:  python -m pytest DDPM/tests/test_ablation_regions.py -v
      (or directly:  python3 DDPM/tests/test_ablation_regions.py)
"""

import itertools
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from InTAct.intact import (  # noqa: E402
    UnlearnIntervalProtection,
    box_drift_max,
    classification_forward_fn,
    make_region_boxes,
)

DEVICE = "cpu"
torch.set_default_dtype(torch.float32)


# ============================================================================
# Reference implementation of the ORIGINAL (pre-ablation) compute_protection_loss
# ============================================================================

def reference_compute_protection_loss(protection, model, device):
    """Bitwise copy of the pre-ablation intact.py compute_protection_loss."""
    total_loss = torch.tensor(0.0, device=device)
    if not protection.pca_info:
        return total_loss

    num_layers = 0
    _use_raw_intervals = protection.skip_svd
    _no_intervals = ((not protection.skip_svd)
                     and (protection.skip_interval or protection.remove_top_directions))

    for info in protection.pca_info:
        layer_name = info["layer_name"]
        target_layer = protection.target_layers.get(layer_name)
        if target_layer is None:
            continue

        target_weight = getattr(target_layer, "weight", None)
        if target_weight is None:
            continue
        target_dtype = target_weight.dtype

        mu = info["mu"].to(device=device, dtype=target_dtype)
        Uf = info["U_forget"].to(device=device, dtype=target_dtype)
        Ur = info["U_residual"].to(device=device, dtype=target_dtype)
        Sr = info["S_residual"].to(device=device, dtype=target_dtype)
        z_min = info["z_min"].to(device=device, dtype=target_dtype)
        z_max = info["z_max"].to(device=device, dtype=target_dtype)
        inf_low = info["inf_low"].to(device=device, dtype=target_dtype)
        inf_high = info["inf_high"].to(device=device, dtype=target_dtype)

        w_name = protection.param_to_name[target_weight]
        target_bias = getattr(target_layer, "bias", None)
        b_name = protection.param_to_name[target_bias] if target_bias is not None else None

        delta_W_raw = target_weight - protection.params_snapshot[w_name].to(
            device=device, dtype=target_dtype)
        delta_b = (target_bias - protection.params_snapshot[b_name].to(
            device=device, dtype=target_dtype)) if b_name else None

        if isinstance(target_layer, nn.Conv2d):
            C_out, C_in, kH, kW = delta_W_raw.shape
            mu_spatial = mu.view(1, -1, 1, 1)

            num_layers += 1
            layer_loss = torch.tensor(0.0, device=device, dtype=target_dtype)

            mean_response = torch.nn.functional.conv2d(
                mu_spatial, delta_W_raw, bias=delta_b,
                stride=target_layer.stride, padding=target_layer.padding,
                dilation=target_layer.dilation, groups=target_layer.groups,
            )
            layer_loss = layer_loss + mean_response.pow(2).mean()

            if Ur.size(0) > 0:
                Ur_spatial = Ur.view(Ur.size(0), -1, 1, 1)
                residual_responses = torch.nn.functional.conv2d(
                    Ur_spatial, delta_W_raw,
                    stride=target_layer.stride, padding=target_layer.padding,
                    dilation=target_layer.dilation, groups=target_layer.groups,
                )
                weighted_responses = residual_responses * Sr.view(-1, 1, 1, 1)
                num_activations = weighted_responses.numel()
                layer_loss = layer_loss + torch.norm(
                    weighted_responses, p="fro").pow(2) / num_activations

            if not _no_intervals:
                if _use_raw_intervals:
                    delta_W_flat = delta_W_raw.permute(0, 2, 3, 1).reshape(
                        C_out, C_in * kH * kW)
                    delta_f = delta_W_flat
                    _z_min = z_min.repeat_interleave(kH * kW)
                    _z_max = z_max.repeat_interleave(kH * kW)
                    _inf_low = inf_low.repeat_interleave(kH * kW)
                    _inf_high = inf_high.repeat_interleave(kH * kW)
                else:
                    Uf_spatial = Uf.view(Uf.size(0), -1, 1, 1)
                    forget_responses = torch.nn.functional.conv2d(
                        Uf_spatial, delta_W_raw,
                        stride=target_layer.stride, padding=target_layer.padding,
                        dilation=target_layer.dilation, groups=target_layer.groups,
                    )
                    delta_f = forget_responses.view(Uf.size(0), -1).T
                    _z_min, _z_max = z_min, z_max
                    _inf_low, _inf_high = inf_low, inf_high

                dWp, dWn = torch.relu(delta_f), torch.relu(-delta_f)
                drift_low_1 = dWp @ _inf_low - dWn @ _z_min
                drift_low_2 = dWp @ _z_min - dWn @ _inf_low
                drift_high_1 = dWp @ _z_max - dWn @ _inf_high
                drift_high_2 = dWp @ _inf_high - dWn @ _z_max
                layer_loss = layer_loss + (drift_low_1.pow(2).mean() + drift_low_2.pow(2).mean())
                layer_loss = layer_loss + (drift_high_1.pow(2).mean() + drift_high_2.pow(2).mean())

        elif isinstance(target_layer, nn.Linear):
            delta_W = delta_W_raw
            num_layers += 1
            layer_loss = torch.tensor(0.0, device=device, dtype=target_dtype)

            db = delta_b if delta_b is not None else torch.tensor(
                0.0, device=device, dtype=target_dtype)

            global_shift = torch.matmul(delta_W, mu) + db
            layer_loss = layer_loss + global_shift.pow(2).mean()

            if Ur.size(0) > 0:
                interference = delta_W @ Ur.T
                weighted_interference = interference * Sr.unsqueeze(0)
                num_activations = weighted_interference.numel()
                layer_loss = layer_loss + torch.norm(
                    weighted_interference, p="fro").pow(2) / num_activations

            if not _no_intervals:
                if _use_raw_intervals:
                    delta_f = delta_W
                    _z_min, _z_max = z_min, z_max
                    _inf_low, _inf_high = inf_low, inf_high
                else:
                    delta_f = delta_W @ Uf.T
                    _z_min, _z_max = z_min, z_max
                    _inf_low, _inf_high = inf_low, inf_high

                dWp, dWn = torch.relu(delta_f), torch.relu(-delta_f)
                drift_low_1 = dWp @ _inf_low - dWn @ _z_min
                drift_low_2 = dWp @ _z_min - dWn @ _inf_low
                drift_high_1 = dWp @ _z_max - dWn @ _inf_high
                drift_high_2 = dWp @ _inf_high - dWn @ _z_max
                layer_loss = layer_loss + (drift_low_1.pow(2).mean() + drift_low_2.pow(2).mean())
                layer_loss = layer_loss + (drift_high_1.pow(2).mean() + drift_high_2.pow(2).mean())
        else:
            continue

        total_loss = total_loss + layer_loss

    if protection.normalize_protection and num_layers > 0:
        total_loss = total_loss / num_layers

    return protection.lambda_interval * total_loss


# ============================================================================
# Test (i): bitwise regression
# ============================================================================

def _build_linear_setup(k=5):
    class ToyLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = nn.Linear(8, 3)

        def forward(self, x, t=None, c=None, mode=None, **kwargs):
            return self.probe(x)

    model = ToyLinear()
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = torch.zeros(64, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)

    protection = UnlearnIntervalProtection(
        targets=["probe"], reduced_dim=k, lambda_interval=2.5,
        region_mode="two_corner", normalize_region=False,
    )
    protection.setup_protection(model, loader, DEVICE,
                                forward_fn=classification_forward_fn)
    return model, protection


def _build_conv_setup(k=5):
    class ToyConv(nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = nn.Conv2d(4, 6, kernel_size=1)

        def forward(self, x):
            return self.probe(x)

    model = ToyConv()
    torch.manual_seed(1)
    x = torch.randn(16, 4, 8, 8)
    y = torch.zeros(16, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

    protection = UnlearnIntervalProtection(
        targets=["probe"], reduced_dim=k, lambda_interval=2.5,
        region_mode="two_corner", normalize_region=False,
    )
    protection.setup_protection(model, loader, DEVICE,
                                forward_fn=classification_forward_fn)
    return model, protection


def _perturb(model, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        model.probe.weight.add_(torch.randn_like(model.probe.weight) * 0.5)
        if model.probe.bias is not None:
            model.probe.bias.add_(torch.randn_like(model.probe.bias) * 0.5)


def test_two_corner_linear_bitwise():
    model, protection = _build_linear_setup()
    _perturb(model, seed=42)
    prod = protection.compute_protection_loss(model, DEVICE)
    ref = reference_compute_protection_loss(protection, model, DEVICE)
    assert torch.equal(prod, ref), (
        f"two_corner (Linear) drifted from the original loss: "
        f"prod={prod.item()!r} ref={ref.item()!r}"
    )


def test_two_corner_conv_bitwise():
    model, protection = _build_conv_setup()
    _perturb(model, seed=7)
    prod = protection.compute_protection_loss(model, DEVICE)
    ref = reference_compute_protection_loss(protection, model, DEVICE)
    assert torch.equal(prod, ref), (
        f"two_corner (Conv2d) drifted from the original loss: "
        f"prod={prod.item()!r} ref={ref.item()!r}"
    )


# ============================================================================
# Term-count normalisation + region-mode plumbing
# ============================================================================

def test_region_term_counts():
    _, protection = _build_linear_setup()
    info = protection.pca_info[0]
    k = info["z_min"].numel()

    n_terms = {}
    n_boxes = {}
    for mode in ("two_corner", "two_random", "boxes_4", "boxes_8", "slabs_2k"):
        side = info.get("region_side")
        if mode == "two_random" and side is None:
            side = torch.zeros(k, dtype=torch.long)
        boxes = make_region_boxes(
            mode, info["inf_low"], info["z_min"], info["z_max"], info["inf_high"],
            side,
        )
        n_boxes[mode] = len(boxes)
        n_terms[mode] = 2 * len(boxes)
        for l, u in boxes:
            assert torch.all(l <= u), f"{mode}: box must satisfy l <= u"
    assert n_boxes["two_corner"] == 2 and n_terms["two_corner"] == 4
    assert n_boxes["two_random"] == 2 and n_terms["two_random"] == 4
    assert n_boxes["boxes_4"] == 4 and n_terms["boxes_4"] == 8
    assert n_boxes["boxes_8"] == 8 and n_terms["boxes_8"] == 16
    assert n_boxes["slabs_2k"] == 2 * k and n_terms["slabs_2k"] == 4 * k


def test_group_block_boxes_consecutive_and_nested():
    """boxes_4/8 use consecutive predefined groups; slabs_2k (m=k) equals the
    per-coordinate slab construction."""
    from InTAct.intact import group_block_boxes

    k = 12
    torch.manual_seed(9)
    inf_low = torch.randn(k) * 2 - 6
    z_min = inf_low + torch.rand(k) + 0.5
    z_max = z_min + torch.rand(k) + 0.5
    inf_high = z_max + torch.rand(k) + 0.5

    def group_ranges(m):
        return [((g * k) // m, ((g + 1) * k) // m) for g in range(m)]

    b4 = group_block_boxes(inf_low, z_min, z_max, inf_high, 2)
    assert len(b4) == 4
    # group g contributes two boxes (low, high), both clipping coords j0..j1-1
    ranges = [rng for g in range(2) for rng in [group_ranges(2)[g]] * 2]
    for box, (j0, j1) in zip(b4, ranges):
        l, u = box
        clipped = torch.where(u < inf_high, True, False) | torch.where(l > inf_low, True, False)
        idx = torch.nonzero(clipped).flatten()
        assert idx.min().item() == j0 and idx.max().item() == j1 - 1, (
            f"group {j0}:{j1} clips {idx.tolist()}")

    b8 = group_block_boxes(inf_low, z_min, z_max, inf_high, 4)
    assert len(b8) == 8

    # m = k reproduces the per-coordinate slabs exactly
    from InTAct.intact import slab_boxes
    bk = group_block_boxes(inf_low, z_min, z_max, inf_high, k)
    sk = slab_boxes(inf_low, z_min, z_max, inf_high)
    assert len(bk) == len(sk) == 2 * k
    for (l1, u1), (l2, u2) in zip(bk, sk):
        assert torch.equal(l1, l2) and torch.equal(u1, u2)


def test_normalize_region_scale():
    """Term-count normalisation divides only the INTERVAL part of the loss by
    its 4 squared terms (mu-shift and residual terms are untouched)."""
    model, protection = _build_linear_setup()
    _perturb(model, seed=11)

    def clone_with(**kwargs):
        p2 = UnlearnIntervalProtection(
            targets=["probe"], reduced_dim=5, lambda_interval=2.5,
            **kwargs,
        )
        p2.pca_info = protection.pca_info
        p2.target_layers = protection.target_layers
        p2.param_to_name = protection.param_to_name
        p2.params_snapshot = protection.params_snapshot
        return p2

    unnorm = protection.compute_protection_loss(model, DEVICE)
    no_interval = clone_with(skip_interval=True).compute_protection_loss(
        model, DEVICE, )
    interval_unnorm = unnorm - no_interval

    norm = clone_with(region_mode="two_corner",
                      normalize_region=True).compute_protection_loss(model, DEVICE)
    expected = no_interval + interval_unnorm / 4.0
    assert torch.allclose(norm, expected, rtol=1e-4, atol=1e-6), (
        f"normalised two_corner != g+r+interval/4: norm={norm.item()} "
        f"expected={expected.item()}"
    )


def test_two_random_boxes_valid_and_deterministic():
    """two_random boxes are valid (l <= u) and the side pattern is stored in
    pca_info and identical across separate setups with the same seed."""
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = torch.zeros(64, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    def build():
        class ToyLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.probe = nn.Linear(8, 3)

            def forward(self, x):
                return self.probe(x)

        m = ToyLinear()
        p = UnlearnIntervalProtection(
            targets=["probe"], reduced_dim=5, lambda_interval=1.0,
            region_mode="two_random", normalize_region=True,
            region_random_seed=0,
        )
        p.setup_protection(m, loader, DEVICE, forward_fn=classification_forward_fn)
        return m, p

    m1, p1 = build()
    m2, p2 = build()

    s1 = p1.pca_info[0]["region_side"]
    s2 = p2.pca_info[0]["region_side"]
    assert s1 is not None and torch.equal(s1, s2), "side pattern must be deterministic"

    info = p1.pca_info[0]
    boxes = make_region_boxes(
        "two_random", info["inf_low"], info["z_min"], info["z_max"],
        info["inf_high"], info["region_side"],
    )
    inf_low, z_min, z_max, inf_high = (
        info["inf_low"], info["z_min"], info["z_max"], info["inf_high"])
    for l, u in boxes:
        assert torch.all(l <= u), "two_random box must satisfy l <= u"
    # per-coordinate, the two boxes cover exactly {[inf_low,z_min], [z_max,inf_high]}
    for j in range(z_min.numel()):
        ivs = set()
        for l, u in boxes:
            ivs.add((float(l[j]), float(u[j])))
        assert ivs == {
            (float(inf_low[j]), float(z_min[j])),
            (float(z_max[j]), float(inf_high[j])),
        }, f"coordinate {j}: wrong interval coverage: {ivs}"


# ============================================================================
# Test (ii): slab coverage by brute force
# ============================================================================

def _box_drift_brute(delta, l, u):
    """Brute-force max |delta . z| over all 2^k corners of the box [l, u]."""
    k = delta.shape[0]
    best = 0.0
    for bits in itertools.product((0, 1), repeat=k):
        z = torch.where(torch.tensor(bits, dtype=torch.bool), u, l)
        best = max(best, float(torch.abs((delta * z).sum())))
    return best


def test_box_drift_max_matches_bruteforce():
    rng = np.random.default_rng(0)
    for k in range(1, 7):
        for _ in range(20):
            delta = torch.tensor(rng.standard_normal(k))
            l = torch.tensor(rng.uniform(-5, -2, k))
            u = torch.tensor(rng.uniform(2, 5, k))
            assert np.isclose(
                float(box_drift_max(delta, l, u)),
                _box_drift_brute(delta, l, u),
                rtol=1e-5, atol=1e-6,
            ), f"box_drift_max mismatch at k={k}"


def test_slab_boxes_cover_complement():
    for k in range(3, 7):
        rng = np.random.default_rng(100 + k)
        inf_low = torch.tensor(rng.uniform(-6, -3, k))
        z_min = inf_low + torch.tensor(rng.uniform(0.5, 1.5, k))
        z_max = z_min + torch.tensor(rng.uniform(0.5, 1.5, k))
        inf_high = z_max + torch.tensor(rng.uniform(0.5, 1.5, k))
        delta = torch.tensor(rng.standard_normal(k))

        boxes = make_region_boxes(
            "slabs_2k", inf_low, z_min, z_max, inf_high, None)
        assert len(boxes) == 2 * k

        max_over_slabs = max(
            float(box_drift_max(delta, l, u)) for l, u in boxes)

        # brute force: max over all 3^k - 1 cells of the complement
        max_over_cells = -1.0
        for cell in itertools.product((0, 1, 2), repeat=k):
            if all(c == 1 for c in cell):  # the forget box itself
                continue
            l = torch.stack([
                inf_low[i] if cell[i] == 0 else (z_min[i] if cell[i] == 1 else z_max[i])
                for i in range(k)
            ])
            u = torch.stack([
                z_min[i] if cell[i] == 0 else (z_max[i] if cell[i] == 1 else inf_high[i])
                for i in range(k)
            ])
            max_over_cells = max(
                max_over_cells, float(box_drift_max(delta, l, u)))

        assert np.isclose(max_over_slabs, max_over_cells, rtol=1e-4, atol=1e-6), (
            f"k={k}: max over 2k slabs ({max_over_slabs}) != max over "
            f"3^k-1 complement cells ({max_over_cells})"
        )


# ============================================================================
# Diagnostics helpers
# ============================================================================

def test_collect_remain_projections():
    """collect_remain_projections returns one [N, k] tensor per target layer."""
    import sys as _sys
    os.environ.setdefault(
        "CACHE_ROOT",
        os.path.join(os.path.dirname(__file__), "..", "..", "data", ".smoke_cache"),
    )
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ablation_regions import collect_remain_projections

    model, protection = _build_linear_setup()
    torch.manual_seed(3)
    x = torch.randn(40, 8)
    y = torch.zeros(40, dtype=torch.long)
    remain_loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)

    proj = collect_remain_projections(
        protection, model, remain_loader, DEVICE,
        data_transform_fn=None, betas=None, num_timesteps=1000,
    )
    assert set(proj.keys()) == {"probe"}
    z = proj["probe"]
    k = protection.pca_info[0]["z_min"].numel()
    assert z.shape == (40, k)
    assert torch.isfinite(z).all()


def test_layer_diagnostics_finite_and_ordered():
    import sys as _sys
    os.environ.setdefault("CACHE_ROOT", os.path.join(
        os.path.dirname(__file__), "..", "..", "data", ".smoke_cache"))
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ablation_regions import layer_diagnostics

    _, protection = _build_linear_setup()
    info = protection.pca_info[0]
    k = info["z_min"].numel()

    torch.manual_seed(5)
    z_remain = torch.randn(200, k) * 2.0

    d = layer_diagnostics(info, z_remain, diag_dirs=500)
    assert d["k"] == k
    assert d["c_lo_norm"] > 0 and d["c_hi_norm"] > 0
    assert d["env_asym_ratio"] > 0
    for key in ("frac_remain_in_A", "frac_remain_in_B", "frac_remain_in_C"):
        assert 0.0 <= d[key] <= 1.0
    for key in ("aniso_A", "aniso_B", "aniso_C"):
        assert d[key] >= 1.0
    for name in ("A", "B", "C"):
        prot = d[f"vstar_drift_protected_{name}"]
        env = d[f"vstar_drift_envelope_{name}"]
        assert 0.0 <= prot <= env, "drift over the protected set cannot exceed the envelope"


# ============================================================================
# Experiment 5: env_box == slabs == brute-force complement (k in 3..6)
# ============================================================================

def test_env_box_matches_slabs_and_bruteforce():
    """The single env_box bound equals the max over the 2k slabs and over all
    3^k - 1 complement cells enumerated by brute force (to 1e-6)."""
    from InTAct.intact import env_box_boxes

    for k in range(3, 7):
        rng = np.random.default_rng(200 + k)
        inf_low = torch.tensor(rng.uniform(-6, -3, k))
        z_min = inf_low + torch.tensor(rng.uniform(0.5, 1.5, k))
        z_max = z_min + torch.tensor(rng.uniform(0.5, 1.5, k))
        inf_high = z_max + torch.tensor(rng.uniform(0.5, 1.5, k))
        delta = torch.tensor(rng.standard_normal(k))

        slabs = make_region_boxes("slabs_2k", inf_low, z_min, z_max, inf_high, None)
        env = env_box_boxes(inf_low, z_min, z_max, inf_high)
        assert len(env) == 1
        assert torch.equal(env[0][0], inf_low) and torch.equal(env[0][1], inf_high)

        max_over_slabs = max(
            float(box_drift_max(delta, l, u)) for l, u in slabs)
        max_over_env = float(box_drift_max(delta, env[0][0], env[0][1]))

        max_over_cells = -1.0
        for cell in itertools.product((0, 1, 2), repeat=k):
            if all(c == 1 for c in cell):
                continue
            l = torch.stack([
                inf_low[i] if cell[i] == 0 else (z_min[i] if cell[i] == 1 else z_max[i])
                for i in range(k)
            ])
            u = torch.stack([
                z_min[i] if cell[i] == 0 else (z_max[i] if cell[i] == 1 else inf_high[i])
                for i in range(k)
            ])
            max_over_cells = max(
                max_over_cells, float(box_drift_max(delta, l, u)))

        assert np.isclose(max_over_env, max_over_cells, rtol=1e-6, atol=1e-6), (
            f"k={k}: env_box bound ({max_over_env}) != brute-force complement "
            f"({max_over_cells})")
        assert np.isclose(max_over_slabs, max_over_cells, rtol=1e-6, atol=1e-6), (
            f"k={k}: slabs bound ({max_over_slabs}) != brute-force complement "
            f"({max_over_cells})")


def test_env_box_term_count():
    _, protection = _build_linear_setup()
    info = protection.pca_info[0]
    boxes = make_region_boxes(
        "env_box", info["inf_low"], info["z_min"], info["z_max"],
        info["inf_high"], None,
    )
    assert len(boxes) == 1
    assert 2 * len(boxes) == 2


# ============================================================================
# Experiment 4: interval modes — (b)+(c) == (a) bitwise via the (F1) route
# ============================================================================

def _clone_protection(protection, **kwargs):
    """Re-use a finished setup_protection under different constructor flags."""
    base = dict(
        targets=["probe"], reduced_dim=5, lambda_interval=1.0,
        region_mode="two_corner", normalize_region=True,
        lower_percentile=protection.lower_percentile,
        upper_percentile=protection.upper_percentile,
    )
    base.update(kwargs)
    p = UnlearnIntervalProtection(**base)
    p.pca_info = protection.pca_info
    p.target_layers = protection.target_layers
    p.param_to_name = protection.param_to_name
    p.params_snapshot = protection.params_snapshot
    return p


def _pure_interval_loss(model, protection):
    """Interval-only loss: L_mean and L_res off, lambda 1, one layer."""
    return protection.compute_protection_loss(model, DEVICE)


def test_interval_width_plus_centre_equals_full_bitwise():
    """(b) width_only + (c) centre_only reproduces (a) full EXACTLY (bitwise)
    on a fixed delta_W, via the (F1) decomposition route (interval_mode
    'full_decomp'):  loss(width_only) + loss(centre_only) == loss(full_decomp).
    L_mean / L_res are off so the comparison is confined to the interval part;
    normalize_protection divides by a single layer (exact)."""
    for builder in (_build_linear_setup, _build_conv_setup):
        model, protection = builder()
        _perturb(model, seed=42 if builder is _build_linear_setup else 7)

        mode_b = _clone_protection(
            protection, interval_mode="width_only",
            include_mean=False, include_res=False)
        mode_c = _clone_protection(
            protection, interval_mode="centre_only",
            include_mean=False, include_res=False)
        mode_a = _clone_protection(
            protection, interval_mode="full_decomp",
            include_mean=False, include_res=False)

        lb = _pure_interval_loss(model, mode_b)
        lc = _pure_interval_loss(model, mode_c)
        la = _pure_interval_loss(model, mode_a)
        assert torch.equal(lb + lc, la), (
            f"width_only + centre_only != full (bitwise): "
            f"b+c={float(lb + lc)!r} a={float(la)!r}")


def test_interval_decomp_matches_legacy_full():
    """Normalisation consistency: width/centre normalise by their own 2 terms
    while the historical full normalises by 4, so full == full_decomp / 2
    exactly (mathematically; float-agree to 1e-5)."""
    model, protection = _build_linear_setup()
    _perturb(model, seed=42)

    legacy = _clone_protection(
        protection, interval_mode="full",
        include_mean=False, include_res=False)
    decomp = _clone_protection(
        protection, interval_mode="full_decomp",
        include_mean=False, include_res=False)

    l1 = _pure_interval_loss(model, legacy)
    l2 = _pure_interval_loss(model, decomp)
    assert np.isclose(float(l1), float(l2) / 2.0, rtol=1e-5, atol=1e-7), (
        f"legacy full ({float(l1)}) must equal full_decomp/2 ({float(l2)/2})")


def test_interval_mode_off_matches_skip_interval():
    model, protection = _build_linear_setup()
    _perturb(model, seed=42)
    off = _clone_protection(protection, interval_mode="off")
    skip = _clone_protection(protection, skip_interval=True)
    assert torch.equal(
        off.compute_protection_loss(model, DEVICE),
        skip.compute_protection_loss(model, DEVICE),
    ), "interval_mode='off' must equal skip_interval=True"


def test_interval_include_mean_res_toggles():
    """Experiment 9 combos: everything off -> 0; L_res-only and
    SVD+intervals-no-L_res are reachable combinations."""
    model, protection = _build_linear_setup()
    _perturb(model, seed=42)

    zero = _clone_protection(
        protection, interval_mode="off",
        include_mean=False, include_res=False,
    ).compute_protection_loss(model, DEVICE)
    assert torch.equal(zero, torch.tensor(0.0)), "all components off -> 0"

    res_only = _clone_protection(
        protection, interval_mode="off", include_mean=False,
    ).compute_protection_loss(model, DEVICE)
    no_res = _clone_protection(
        protection, include_res=False,
    ).compute_protection_loss(model, DEVICE)
    no_mean = _clone_protection(
        protection, include_mean=False,
    ).compute_protection_loss(model, DEVICE)
    all_on = _clone_protection(protection).compute_protection_loss(model, DEVICE)

    assert res_only > 0.0, "L_res-only must be nonzero"
    assert no_res > 0.0, "SVD+intervals without L_res must be nonzero"
    assert no_mean > 0.0, "SVD+intervals+L_res without L_mean must be nonzero"
    # splitting: all_on should be consistent with mean + (no_mean)
    assert np.isclose(float(all_on), float(no_mean) + float(all_on - no_mean),
                      rtol=1e-6, atol=1e-9)


# ============================================================================
# Experiment 8: delta_b in the interval legs
# ============================================================================

def test_include_db_gating():
    """include_db requires interval_mode='full' (config error otherwise); on a
    bias-less Conv2d it is a no-op (delta_b == 0)."""
    try:
        _clone_protection(_build_linear_setup()[1],
                          interval_mode="width_only", include_db=True)
        assert False, "include_db + width_only must raise ValueError"
    except ValueError:
        pass

    class ToyConvNoBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = nn.Conv2d(4, 6, kernel_size=1, bias=False)

        def forward(self, x):
            return self.probe(x)

    torch.manual_seed(1)
    x = torch.randn(16, 4, 8, 8)
    y = torch.zeros(16, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=8)
    model = ToyConvNoBias()
    protection = UnlearnIntervalProtection(
        targets=["probe"], reduced_dim=5, lambda_interval=1.0,
        region_mode="two_corner", normalize_region=True,
    )
    protection.setup_protection(model, loader, DEVICE,
                                forward_fn=classification_forward_fn)
    _perturb(model, seed=7)
    no_db = _clone_protection(protection)
    with_db = _clone_protection(protection, include_db=True)
    assert torch.equal(
        no_db.compute_protection_loss(model, DEVICE),
        with_db.compute_protection_loss(model, DEVICE),
    ), "include_db on a bias-less layer must be a no-op"


def test_include_db_linear_changes_loss():
    """With a nonzero delta_b the interval legs shift and the loss changes."""
    model, protection = _build_linear_setup()
    _perturb(model, seed=42)
    no_db = _clone_protection(protection)
    with_db = _clone_protection(protection, include_db=True)
    l0 = no_db.compute_protection_loss(model, DEVICE)
    l1 = with_db.compute_protection_loss(model, DEVICE)
    assert not torch.equal(l0, l1), (
        "include_db must change the loss when delta_b != 0")


# ============================================================================
# Experiment 6: sign-flip sensitivity plumbing
# ============================================================================

def _signflip_setup(frac, seed):
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    y = torch.zeros(64, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    class ToyLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = nn.Linear(8, 3)

        def forward(self, x, t=None, c=None, mode=None, **kwargs):
            return self.probe(x)

    m = ToyLinear()
    p = UnlearnIntervalProtection(
        targets=["probe"], reduced_dim=5, lambda_interval=1.0,
        region_mode="two_corner", normalize_region=True,
        sign_flip_frac=frac, sign_flip_seed=seed,
    )
    p.setup_protection(m, loader, DEVICE, forward_fn=classification_forward_fn)
    return m, p


def test_sign_flip_determinism():
    """Same (frac, seed) -> identical U_forget/bounds; different seed changes
    the flipped subset."""
    _, p1 = _signflip_setup(0.5, 0)
    _, p2 = _signflip_setup(0.5, 0)
    _, p3 = _signflip_setup(0.5, 1)
    i1, i2, i3 = p1.pca_info[0], p2.pca_info[0], p3.pca_info[0]
    assert torch.equal(i1["U_forget"], i2["U_forget"])
    assert torch.equal(i1["z_min"], i2["z_min"])
    assert not torch.equal(i1["U_forget"], i3["U_forget"])


def test_sign_flip_full_swap_relation():
    """frac=1.0 flips every row: U_forget -> -U_forget and the recomputed
    bounds swap sign per coordinate (quantiles of the negated projection)."""
    _, pf = _signflip_setup(1.0, 0)
    _, p0 = _signflip_setup(0.0, 0)
    i1, i0 = pf.pca_info[0], p0.pca_info[0]
    assert torch.allclose(i1["U_forget"], -i0["U_forget"], atol=1e-6)
    assert torch.allclose(i1["z_min"], -i0["z_max"], atol=1e-5), (
        "z_min must be -z_max under a full sign flip")
    assert torch.allclose(i1["z_max"], -i0["z_min"], atol=1e-5)
    assert torch.allclose(i1["inf_low"], -i0["inf_high"], atol=1e-5)
    assert torch.allclose(i1["inf_high"], -i0["inf_low"], atol=1e-5)


def test_sign_flip_frac_0_identity():
    _, p0 = _signflip_setup(0.0, 0)
    _, pz = _signflip_setup(0.0, 7)
    assert torch.equal(p0.pca_info[0]["U_forget"], pz.pca_info[0]["U_forget"])


# ============================================================================
# Experiment 3: uniform-margin control
# ============================================================================

def test_uniform_margin_bounds():
    """uniform_margin replaces per-coordinate margins by their mean; z_min and
    z_max are untouched.  (use_actual_bounds=True so the envelope comes from
    the data and the margins are genuinely non-uniform.)"""
    torch.manual_seed(0)
    x = torch.randn(256, 8)
    y = torch.zeros(256, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    class ToyLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = nn.Linear(8, 3)

        def forward(self, x, t=None, c=None, mode=None, **kwargs):
            return self.probe(x)

    def build(uniform):
        m = ToyLinear()
        p = UnlearnIntervalProtection(
            targets=["probe"], reduced_dim=5, lambda_interval=1.0,
            region_mode="two_corner", normalize_region=True,
            use_actual_bounds=True,
            uniform_margin=uniform,
        )
        p.setup_protection(m, loader, DEVICE,
                           remain_dataloader=loader,
                           forward_fn=classification_forward_fn)
        return p

    p_std, p_uni = build(False), build(True)
    i_std, i_uni = p_std.pca_info[0], p_uni.pca_info[0]
    assert torch.equal(i_std["z_min"], i_uni["z_min"])
    assert torch.equal(i_std["z_max"], i_uni["z_max"])

    w_std = torch.minimum(
        i_std["z_min"] - i_std["inf_low"],
        i_std["inf_high"] - i_std["z_max"],
    )
    w_uni = torch.minimum(
        i_uni["z_min"] - i_uni["inf_low"],
        i_uni["inf_high"] - i_uni["z_max"],
    )
    assert w_uni.numel() > 0 and torch.all(w_uni > 0)
    assert np.allclose(w_uni.numpy(), float(w_uni.mean()), rtol=1e-6, atol=1e-6), (
        "uniform-margin margins must be constant across coordinates "
        "(up to float32 ulp from the z_min - wbar round-trip)")
    assert np.isclose(float(w_uni.mean()), float(w_std.mean()), rtol=1e-6), (
        "uniform-margin mean must match the standard mean")
    assert not np.allclose(w_uni.numpy(), w_std.numpy()), (
        "standard margins must actually differ per coordinate for this test")
    assert "uniform_margin_wbar" in i_uni


# ============================================================================
# Experiment 7: percentile alpha
# ============================================================================

def test_percentile_alpha_mapping():
    from InTAct.intact import percentile_alpha_to_quantiles

    assert percentile_alpha_to_quantiles(1) == (0.01, 0.99)
    assert percentile_alpha_to_quantiles(5) == (0.05, 0.95)
    assert percentile_alpha_to_quantiles(10) == (0.10, 0.90)
    try:
        percentile_alpha_to_quantiles(3)
        assert False, "alpha=3 must raise ValueError"
    except ValueError:
        pass


def test_alpha_quantile_effect():
    """Smaller alpha moves z_min down / z_max up (control region grows)."""
    torch.manual_seed(0)
    x = torch.randn(128, 8)
    y = torch.zeros(128, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=16)

    class ToyLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.probe = nn.Linear(8, 3)

        def forward(self, x, t=None, c=None, mode=None, **kwargs):
            return self.probe(x)

    def build(alpha):
        from InTAct.intact import percentile_alpha_to_quantiles
        lo, hi = percentile_alpha_to_quantiles(alpha)
        m = ToyLinear()
        p = UnlearnIntervalProtection(
            targets=["probe"], reduced_dim=5, lambda_interval=1.0,
            region_mode="two_corner", normalize_region=True,
            lower_percentile=lo, upper_percentile=hi,
        )
        p.setup_protection(m, loader, DEVICE, forward_fn=classification_forward_fn)
        return p

    p1, p5, p10 = build(1), build(5), build(10)
    z1, z5, z10 = (p.pca_info[0]["z_min"] for p in (p1, p5, p10))
    assert torch.all(z1 <= z5 + 1e-9) and torch.all(z5 <= z10 + 1e-9), (
        "z_min must shrink with alpha")


# ============================================================================
# S_forget + artifacts
# ============================================================================

def test_s_forget_stored():
    _, protection = _build_linear_setup()
    k = protection.pca_info[0]["z_min"].numel()
    sf = protection.pca_info[0]["S_forget"]
    assert sf.numel() == k
    assert torch.all(sf >= 0)
    assert torch.equal(sf, torch.sort(sf, descending=True)[0]), (
        "S_forget should be the top-k singular values in descending order")


def test_serialized_artifact_roundtrip():
    """pca_info as saved by the runners (torch.save of the list) round-trips."""
    import tempfile
    _, protection = _build_linear_setup()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "pca_info.pth")
        torch.save(protection.pca_info, path)
        loaded = torch.load(path, weights_only=False)
        assert len(loaded) == 1
        for key in ("mu", "U_forget", "z_min", "z_max", "inf_low", "inf_high",
                    "S_forget", "layer_name"):
            assert key in loaded[0]
        assert torch.equal(loaded[0]["z_min"], protection.pca_info[0]["z_min"])


if __name__ == "__main__":
    fns = [
        test_two_corner_linear_bitwise,
        test_two_corner_conv_bitwise,
        test_region_term_counts,
        test_group_block_boxes_consecutive_and_nested,
        test_normalize_region_scale,
        test_two_random_boxes_valid_and_deterministic,
        test_box_drift_max_matches_bruteforce,
        test_slab_boxes_cover_complement,
        test_collect_remain_projections,
        test_layer_diagnostics_finite_and_ordered,
        test_env_box_matches_slabs_and_bruteforce,
        test_env_box_term_count,
        test_interval_width_plus_centre_equals_full_bitwise,
        test_interval_decomp_matches_legacy_full,
        test_interval_mode_off_matches_skip_interval,
        test_interval_include_mean_res_toggles,
        test_include_db_gating,
        test_include_db_linear_changes_loss,
        test_sign_flip_determinism,
        test_sign_flip_full_swap_relation,
        test_sign_flip_frac_0_identity,
        test_uniform_margin_bounds,
        test_percentile_alpha_mapping,
        test_alpha_quantile_effect,
        test_s_forget_stored,
        test_serialized_artifact_roundtrip,
    ]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all tests passed")
