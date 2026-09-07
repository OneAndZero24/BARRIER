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
    for mode in ("two_corner", "two_random", "slabs_2k"):
        side = info.get("region_side")
        if mode == "two_random" and side is None:
            side = torch.zeros(k, dtype=torch.long)
        boxes = make_region_boxes(
            mode, info["inf_low"], info["z_min"], info["z_max"], info["inf_high"],
            side,
        )
        n_terms[mode] = 2 * len(boxes)
    assert n_terms["two_corner"] == 4
    assert n_terms["two_random"] == 4
    assert n_terms["slabs_2k"] == 4 * k


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


if __name__ == "__main__":
    fns = [
        test_two_corner_linear_bitwise,
        test_two_corner_conv_bitwise,
        test_region_term_counts,
        test_normalize_region_scale,
        test_two_random_boxes_valid_and_deterministic,
        test_box_drift_max_matches_bruteforce,
        test_slab_boxes_cover_complement,
        test_collect_remain_projections,
        test_layer_diagnostics_finite_and_ordered,
    ]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print("all tests passed")
