import logging
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable

log = logging.getLogger(__name__)


# ============================================================================
# Region-construction primitives (ablation over protected-region layouts)
# ============================================================================
#
# All variants are built from the same primitive: for a box with lower bound l
# and upper bound u, the two-corner term
#     T(l, u) = || dWp @ l - dWn @ u ||^2 + || dWp @ u - dWn @ l ||^2
# with dWp = relu(delta_f), dWn = relu(-delta_f).  A and B evaluate T over two
# boxes, C evaluates it over the 2k coordinate slabs whose union is exactly the
# complement of the forget box [z_min, z_max] inside the envelope
# [inf_low, inf_high].  boxes_4 / boxes_8 are the predefined m-group family
# (2m boxes): each box is the full envelope clipped low/high on one consecutive
# group of coordinates.  env_box is the single-box variant T(inf_low, inf_high):
# the worst case over the complement equals the worst case over the envelope.

REGION_MODES = ("two_corner", "two_random", "slabs_2k", "boxes_4", "boxes_8", "env_box")

# Experiment 7: alpha is reported as a percent (1/5/10) and maps to the
# lower/upper percentile fractions used for z_min / z_max.
def percentile_alpha_to_quantiles(alpha):
    """alpha in {1, 5, 10} -> (lower, upper) percentile fractions."""
    if alpha not in (1, 5, 10):
        raise ValueError(f"alpha must be one of (1, 5, 10), got {alpha}")
    return alpha / 100.0, 1.0 - alpha / 100.0

# Which part of Lemma 1's centre/width identity the interval term evaluates.
#   full        : the historical expression over the box corners (current loss)
#   width_only  : 0.5 * || |delta_f| @ w ||^2 per box        (Experiment 4b)
#   centre_only : 2 * || delta_f @ c ||^2 per box            (Experiment 4c)
#   full_decomp : width_only + centre_only via the exact (F1) route; used by
#                 the bitwise (b)+(c) == (a) unit test, not an ablation config
#   off         : no interval terms (L_mean / L_res only)
INTERVAL_MODES = ("full", "width_only", "centre_only", "full_decomp", "off")


def two_corner_boxes(inf_low, z_min, z_max, inf_high):
    """Current (baseline) layout: the two box shapes "all coordinates below
    z_min" and "all coordinates above z_max".  Returns a list of (l, u)."""
    return [(inf_low, z_min), (z_max, inf_high)]


def env_box_boxes(inf_low, z_min, z_max, inf_high):
    """Cheap exact-complement variant (Experiment 5): a single box spanning the
    whole envelope.  Every envelope vertex lies outside the control region
    [z_min, z_max], so the worst case over the complement equals the worst case
    over the envelope; two squared terms are enough."""
    return [(inf_low, inf_high)]


def random_boxes(inf_low, z_min, z_max, inf_high, side):
    """Control for placement: two boxes of the same shape as two_corner, but
    with a random side pattern s in {0,1}^k.  For each j the box keeps
    [inf_low_j, z_min_j] if s_j == 0 else [z_max_j, inf_high_j]; the second box
    takes the complementary sides."""
    s = side.to(device=inf_low.device).bool()
    lA = torch.where(s, z_max, inf_low)
    uA = torch.where(s, inf_high, z_min)
    lB = torch.where(s, inf_low, z_max)
    uB = torch.where(s, z_min, inf_high)
    return [(lA, uA), (lB, uB)]


def slab_boxes(inf_low, z_min, z_max, inf_high):
    """The exact complement: for each coordinate j and each side, a box that is
    the full envelope in all i != j and clipped on j.
      low_j :  l = inf_low,                    u = inf_high with u[j] = z_min[j]
      high_j:  l = inf_low with l[j] = z_max[j], u = inf_high
    The union of these 2k slabs equals the complement of the forget box inside
    the envelope."""
    boxes = []
    k = z_min.numel()
    for j in range(k):
        l = inf_low.clone()
        u = inf_high.clone()
        u[j] = z_min[j]
        boxes.append((l, u))
        l = inf_low.clone()
        u = inf_high.clone()
        l[j] = z_max[j]
        boxes.append((l, u))
    return boxes


def group_block_boxes(inf_low, z_min, z_max, inf_high, m):
    """Predefined m-group construction (2m boxes).  Split the k coordinates
    into m consecutive, non-overlapping groups; for each group g and side:
      low_j :  l = inf_low, u = inf_high with u[j] = z_min[j]   for j in group g
      high_j:  l = inf_low with l[j] = z_max[j], u = inf_high   for j in group g
    m = 1 reproduces the two_corner layout; m = k reproduces the 2k slabs."""
    boxes = []
    k = z_min.numel()
    m = max(1, min(m, k))
    for g in range(m):
        j0 = (g * k) // m
        j1 = ((g + 1) * k) // m
        if j1 <= j0:
            continue
        l = inf_low.clone()
        u = inf_high.clone()
        u[j0:j1] = z_min[j0:j1]
        boxes.append((l, u))
        l = inf_low.clone()
        u = inf_high.clone()
        l[j0:j1] = z_max[j0:j1]
        boxes.append((l, u))
    return boxes


def make_region_boxes(region_mode, inf_low, z_min, z_max, inf_high, side=None):
    """Build the (l, u) box list for the requested region construction."""
    if region_mode == "two_corner":
        return two_corner_boxes(inf_low, z_min, z_max, inf_high)
    if region_mode == "two_random":
        if side is None:
            raise ValueError("region_mode='two_random' requires a stored side pattern")
        return random_boxes(inf_low, z_min, z_max, inf_high, side)
    if region_mode == "slabs_2k":
        return slab_boxes(inf_low, z_min, z_max, inf_high)
    if region_mode == "boxes_4":
        return group_block_boxes(inf_low, z_min, z_max, inf_high, 2)
    if region_mode == "boxes_8":
        return group_block_boxes(inf_low, z_min, z_max, inf_high, 4)
    if region_mode == "env_box":
        return env_box_boxes(inf_low, z_min, z_max, inf_high)
    raise ValueError(f"Unknown region_mode {region_mode!r} (expected one of {REGION_MODES})")


def box_drift_max(delta_f, l, u):
    """Max |delta_f . z| over the box [l, u].  A linear response is maximised in
    absolute value at a vertex of the box, so the value is
        max( |sum_{d_i>=0} d_i u_i + sum_{d_i<0} d_i l_i|,
             |sum_{d_i>=0} d_i l_i + sum_{d_i<0} d_i u_i| )."""
    d = delta_f
    pos = d >= 0
    upper = torch.where(pos, d * u, d * l).sum()
    lower = torch.where(pos, d * l, d * u).sum()
    return torch.maximum(upper.abs(), lower.abs())


# ============================================================================
# Forward Functions for Different Model Types
# ============================================================================

def ddpm_forward_fn(model, batch, device, data_transform_fn=None, betas=None, num_timesteps=1000):
    """
    Forward function for DDPM models.
    Expects batch = (x, c) where x is images and c is class labels.
    """
    x, c = batch
    n = x.size(0)
    x = x.to(device)
    c = c.to(device)
    
    if data_transform_fn is not None:
        x = data_transform_fn(x)
    
    t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
    t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
    
    if betas is not None:
        e = torch.randn_like(x)
        a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x_noisy = x * a.sqrt() + e * (1.0 - a).sqrt()
    else:
        x_noisy = x
    
    model(x_noisy, t.float(), c, mode="train")


def classification_forward_fn(model, batch, device, **kwargs):
    """
    Forward function for classification models.
    Expects batch = (x, y) where x is images and y is labels.
    """
    x, y = batch[:2]
    x = x.to(device)
    model(x)


class UnlearnIntervalProtection:
    """
    InTAct (Interval-based Task Activation Consolidation) for machine unlearning.
    
    Protects model activations by constraining them to safe intervals during unlearning,
    preventing catastrophic forgetting on retain data.
    """
    def __init__(
        self,
        targets: List[str],
        lambda_interval: float = 10.0,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        reduced_dim: int = 32,
        infinity_scale: float = 20.0,
        use_actual_bounds: bool = False,
        normalize_protection: bool = True,  # Normalize protection loss by number of layers

        # --- Ablation / experiment flags ------------------------------------
        # skip_svd:   no SVD at all — protect on raw activation space (per-dim
        #             quantile intervals, no dimensionality reduction).
        # skip_interval:  keep SVD but skip U_forget interval protection
        #             (only penalise U_residual + mu drift).
        # remove_top_directions:  compute SVD, discard top-k U_forget entirely
        #             (empty), keep only U_residual + mu.  No intervals.
        #             Conceptually: "remove top-k subspace, keep the rest".
        # decomp_method:  "svd" (torch.linalg.svd) or "pca" (torch.linalg.eigh
        #             on the covariance matrix).  Only meaningful when
        #             skip_svd=False and remove_top_directions=False.
        skip_svd: bool = False,
        skip_interval: bool = False,
        remove_top_directions: bool = False,
        decomp_method: str = "svd",

        # --- Ablation of protected-region constructions ---------------------
        # region_mode:        which box layout the interval term is evaluated
        #                     over: "two_corner" (current), "two_random"
        #                     (placement control), "slabs_2k" (the exact
        #                     complement of the forget box).
        # normalize_region:   divide each variant's interval part by its number
        #                     of squared terms so variants are on a comparable
        #                     scale (A/B: 4, C: 4k).
        # region_random_seed: fixed seed drawing the random side pattern of
        #                     "two_random" (stored in pca_info -> identical
        #                     across steps and seeds-of-training).
        region_mode: str = "two_corner",
        normalize_region: bool = False,
        region_random_seed: int = 0,

        # --- Mechanism / design-choice experiments --------------------------
        # interval_mode:  which part of Lemma 1's identity the interval term
        #                 evaluates ("full" / "width_only" / "centre_only" /
        #                 "off"; "full_decomp" is the exact (F1) route used by
        #                 the bitwise (b)+(c)==(a) test — not an ablation flag).
        # include_db:     add delta_b to each of the four corner legs of the
        #                 interval terms (InTAct Eqs. 35-36; Experiment 8).
        #                 Valid with interval_mode "full" only, since the (F1)
        #                 centre/width identity does not hold with delta_b in
        #                 the legs.  L_mean keeps its delta_b term regardless.
        # include_mean:   L_mean = ||delta_W @ mu + delta_b||^2 on/off.
        # include_res:    L_res  = ||delta_W @ Ur.T @ Sr||^2 on/off.
        # sign_flip_frac: fraction of U_forget rows whose sign is flipped
        #                 BEFORE the bounds are recomputed from the data
        #                 (SVD sign-convention sensitivity; Experiment 6).
        # sign_flip_seed: which rows get flipped (deterministic per run).
        # uniform_margin: replace the per-coordinate envelope margins by their
        #                 mean while keeping z_min/z_max (Experiment 3).
        interval_mode: str = "full",
        include_db: bool = False,
        include_mean: bool = True,
        include_res: bool = True,
        sign_flip_frac: float = 0.0,
        sign_flip_seed: int = 0,
        uniform_margin: bool = False,
    ):
        self.targets = targets
        self.lambda_interval = lambda_interval
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.reduced_dim = reduced_dim
        self.infinity_scale = infinity_scale
        self.use_actual_bounds = use_actual_bounds
        self.normalize_protection = normalize_protection

        self.skip_svd = skip_svd
        self.skip_interval = skip_interval
        self.remove_top_directions = remove_top_directions
        self.decomp_method = decomp_method.lower()
        if self.decomp_method not in ("svd", "pca"):
            raise ValueError(f"decomp_method must be 'svd' or 'pca', got '{decomp_method}'")
        # Precedence: skip_svd > remove_top_directions > skip_interval
        if self.skip_svd and (self.skip_interval or self.remove_top_directions):
            log.warning("skip_svd=True overrides skip_interval/remove_top_directions.")
        if self.remove_top_directions and self.skip_interval:
            log.warning("remove_top_directions=True overrides skip_interval.")

        self.region_mode = region_mode
        if self.region_mode not in REGION_MODES:
            raise ValueError(
                f"region_mode must be one of {REGION_MODES}, got {self.region_mode!r}"
            )
        self.normalize_region = normalize_region
        self.region_random_seed = int(region_random_seed)

        self.interval_mode = interval_mode
        if self.interval_mode not in INTERVAL_MODES:
            raise ValueError(
                f"interval_mode must be one of {INTERVAL_MODES}, got {self.interval_mode!r}"
            )
        if include_db and self.interval_mode != "full":
            raise ValueError(
                "include_db requires interval_mode='full' (the (F1) centre/width "
                "identity does not hold with delta_b in the legs)"
            )
        if include_db:
            log.info("include_db=True: delta_b enters the interval corner legs "
                     "(InTAct Eqs. 35-36) as well as L_mean")
        self.include_db = include_db
        self.include_mean = include_mean
        self.include_res = include_res

        self.sign_flip_frac = float(sign_flip_frac)
        if not (0.0 <= self.sign_flip_frac <= 1.0):
            raise ValueError(f"sign_flip_frac must be in [0, 1], got {sign_flip_frac}")
        self.sign_flip_seed = int(sign_flip_seed)
        self.uniform_margin = bool(uniform_margin)

        self.pca_info: List[Dict] = []
        self.params_snapshot = {}  # Only target layer parameters
        self.target_layers: Dict[str, nn.Module] = {}  # Maps target_name -> target_module
        self.param_to_name: Dict[nn.Parameter, str] = {}  # Maps parameter -> name

    def setup_protection(self, model: nn.Module, forget_dataloader, device, remain_dataloader=None,
                        forward_fn: Callable = None, data_transform_fn=None, betas=None, num_timesteps=1000):
        """
        Populates: pca_info, params_snapshot
        
        Args:
            model: The model to protect
            forget_dataloader: DataLoader for forget data
            device: Device to run on
            remain_dataloader: Optional DataLoader for remain data (used with use_actual_bounds)
            forward_fn: Function to call model forward. Signature: forward_fn(model, batch, device, **kwargs)
                       If None, uses ddpm_forward_fn as default for backward compatibility.
            data_transform_fn: Optional transform for input data
            betas: Noise schedule betas (for diffusion models)
            num_timesteps: Number of diffusion timesteps
        """
        # Default to DDPM forward for backward compatibility
        if forward_fn is None:
            forward_fn = ddpm_forward_fn

        log.info("Setting up InTAct with Mean Reparametrization...")
        
        # Find target layers using named_modules
        target_names = self._find_target_layers(model)
        
        if not target_names:
            log.warning("No target layers found for protection")
            return
        
        log.info(f"Found {len(target_names)} target layers to collect inputs from: {target_names}")

        # 1. Collect input activations from target layers (forget data)
        acts_dict = self._collect_activations(
            model, target_names, forget_dataloader, device,
            forward_fn=forward_fn,
            data_transform_fn=data_transform_fn, betas=betas, num_timesteps=num_timesteps
        )
        
        # 2. Compute SVD on forget data and optionally collect projected remain data
        pca_components = {}  # Store mu and U_forget for each layer
        # Snapshot the items so per-layer tensors can be freed as they are
        # processed (keeps the SVD-phase RAM flat instead of holding all raw
        # activation buffers at once).
        acts_items = list(acts_dict.items())

        def _maybe_flip(U):
            """Sign-convention sensitivity (Experiment 6): flip a random subset
            of U_forget rows BEFORE the bounds are recomputed from the data, so
            z_min/z_max/inf_low/inf_high are consistent with the flipped basis."""
            if self.sign_flip_frac <= 0.0 or U.size(0) == 0:
                return U
            gen = torch.Generator(device="cpu").manual_seed(self.sign_flip_seed)
            mask = torch.rand(U.size(0), generator=gen) < self.sign_flip_frac
            flip = (1.0 - 2.0 * mask.to(device=U.device, dtype=U.dtype)).unsqueeze(1)
            return U * flip
        
        for layer_name, acts_info in acts_items:
            acts = acts_info['activations']
            layer_type = acts_info.get('layer_type', 'Linear')
            
            # PCA/SVD is not implemented for bf16 on CUDA, so upcast only the
            # statistics working tensor while leaving the collected activations
            # and model weights unchanged.
            acts_gpu = acts.to(device=device, dtype=torch.float32)

            if not torch.isfinite(acts_gpu).all():
                nonfinite_count = (~torch.isfinite(acts_gpu)).sum().item()
                log.warning(
                    f"Layer {layer_name}: replacing {nonfinite_count} non-finite activation values before PCA/SVD"
                )
                acts_gpu = torch.nan_to_num(acts_gpu, nan=0.0, posinf=0.0, neginf=0.0)
            
            mu = acts_gpu.mean(dim=0)
            Xc = acts_gpu - mu

            if self.skip_svd:
                if self.decomp_method == "pca":
                    decomp_tag = "pca"
                else:
                    decomp_tag = "svd"

                z_min = torch.quantile(Xc, self.lower_percentile, dim=0)
                z_max = torch.quantile(Xc, self.upper_percentile, dim=0)

                U_forget = torch.eye(mu.size(0), device=mu.device, dtype=mu.dtype)
                U_forget = _maybe_flip(U_forget)
                U_residual = torch.empty(0, mu.size(0), device=mu.device, dtype=mu.dtype)
                S_residual = torch.empty(0, device=mu.device, dtype=mu.dtype)
                S_forget = torch.empty(0, device=mu.device, dtype=mu.dtype)
                Z_forget = Xc  # raw centred data — not used further

                inf_low = z_min - self.infinity_scale
                inf_high = z_max + self.infinity_scale
            elif self.decomp_method == "pca":
                decomp_tag = "pca"
                C = Xc.T @ Xc
                eigenvalues, V = torch.linalg.eigh(C)
                eigenvalues = eigenvalues.flip(0)
                V = V.T.flip(0)

                k = min(self.reduced_dim, V.size(0))
                U_forget = _maybe_flip(V[:k])
                U_residual = V[k:]
                S_residual = eigenvalues[k:].clamp(min=0).sqrt()
                S_forget = eigenvalues[:k].clamp(min=0).sqrt()

                Z_forget = Xc @ U_forget.T
                z_min = torch.quantile(Z_forget, self.lower_percentile, dim=0)
                z_max = torch.quantile(Z_forget, self.upper_percentile, dim=0)

                inf_low = z_min - self.infinity_scale
                inf_high = z_max + self.infinity_scale
            else:
                decomp_tag = "svd"
                try:
                    _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        raise
                    log.warning(
                        f"Layer {layer_name}: torch.linalg.svd failed ({e}); "
                        f"falling back to eigh-based PCA."
                    )
                    decomp_tag = "pca"
                    C = Xc.T @ Xc
                    eigenvalues, V = torch.linalg.eigh(C)
                    eigenvalues = eigenvalues.flip(0)
                    V = V.T.flip(0)
                    S = eigenvalues.clamp(min=0).sqrt()
                    Vh = V

                k = min(self.reduced_dim, Vh.size(0))
                U_forget = _maybe_flip(Vh[:k])
                U_residual = Vh[k:]
                S_residual = S[k:]
                S_forget = S[:k]

                Z_forget = Xc @ U_forget.T
                z_min = torch.quantile(Z_forget, self.lower_percentile, dim=0)
                z_max = torch.quantile(Z_forget, self.upper_percentile, dim=0)

                inf_low = z_min - self.infinity_scale
                inf_high = z_max + self.infinity_scale

            # --- Post-processing: remove top-k SVD directions if requested ---
            # Discards U_forget (empty) and its intervals; only U_residual + mu
            # remain.  The loss will skip all interval terms.
            if self.remove_top_directions and not self.skip_svd:
                U_forget = torch.empty(0, mu.size(0), device=mu.device, dtype=mu.dtype)
                S_forget = torch.empty(0, device=mu.device, dtype=mu.dtype)
                z_min = torch.empty(0, device=mu.device, dtype=mu.dtype)
                z_max = torch.empty(0, device=mu.device, dtype=mu.dtype)
                inf_low = torch.empty(0, device=mu.device, dtype=mu.dtype)
                inf_high = torch.empty(0, device=mu.device, dtype=mu.dtype)
                decomp_tag = decomp_tag + "-rmtop"
            
            # Store PCA components for remain data projection
            pca_components[layer_name] = {
                'mu': mu,
                'U_forget': U_forget,
                'layer_type': layer_type
            }

            # Free GPU memory
            del acts_gpu, Xc

            # Override inf_low/inf_high with actual data range if requested
            if (self.use_actual_bounds and remain_dataloader is not None
                    and not self.skip_svd and not self.remove_top_directions):
                combined_min = Z_forget.min(dim=0)[0]
                combined_max = Z_forget.max(dim=0)[0]
                inf_low = combined_min
                inf_high = combined_max
            del Z_forget

            # Store PCA info (will update inf_low/inf_high after remain collection if needed)
            region_side = None
            if self.region_mode == "two_random" and z_min.numel() > 0:
                gen = torch.Generator(device="cpu").manual_seed(self.region_random_seed)
                region_side = torch.randint(
                    0, 2, (z_min.numel(),), dtype=torch.long, generator=gen
                )

            pca_entry = {
                "layer_name": layer_name,
                "mu": mu.detach().cpu(),
                "U_forget": U_forget.detach().cpu(),
                "U_residual": U_residual.detach().cpu(),
                "S_residual": S_residual.detach().cpu(),
                "S_forget": S_forget.detach().cpu(),
                "z_min": z_min.detach().cpu(),
                "z_max": z_max.detach().cpu(),
                "inf_low": inf_low.detach().cpu(),
                "inf_high": inf_high.detach().cpu(),
                "layer_type": layer_type,
                "decomp_tag": decomp_tag,
                "region_side": region_side,
            }
            self.pca_info.append(pca_entry)
            del acts_dict[layer_name]  # free raw activations as we go

        # Free the raw forget-activation buffers BEFORE projecting the remain
        # set: with DDPM-scale data they are ~16 GB for all target layers and
        # the remain projection accumulates a similar amount, which otherwise
        # doubles peak RAM (observed OOM kill at 32 GB cgroup).
        del acts_dict

        # 2b. Collect projected remain data and update bounds
        if self.use_actual_bounds and remain_dataloader is not None:
            log.info("Collecting and projecting remain data on-the-fly...")
            remain_projected = self._collect_activations(
                model, list(pca_components.keys()), remain_dataloader, device,
                forward_fn=forward_fn,
                data_transform_fn=data_transform_fn, betas=betas, num_timesteps=num_timesteps,
                pca_components=pca_components  # Enable projection mode
            )
            
            # Update inf_low/inf_high for each layer
            for pca_entry in self.pca_info:
                layer_name = pca_entry["layer_name"]
                if layer_name in remain_projected:
                    Z_remain = remain_projected[layer_name].to(device)
                    
                    # Update bounds to include remain data
                    inf_low = pca_entry["inf_low"].to(device)
                    inf_high = pca_entry["inf_high"].to(device)
                    
                    combined_min = torch.minimum(inf_low, Z_remain.min(dim=0)[0])
                    combined_max = torch.maximum(inf_high, Z_remain.max(dim=0)[0])
                    
                    pca_entry["inf_low"] = combined_min.cpu()
                    pca_entry["inf_high"] = combined_max.cpu()
                    
                    log.info(f"Layer {layer_name}: Updated bounds with {Z_remain.size(0)} projected remain samples")
                    del Z_remain, inf_low, inf_high, combined_min, combined_max
                    del remain_projected[layer_name]  # free per layer

        # 2c. Uniform-margin control (Experiment 3): replace the per-coordinate
        # envelope margins by their mean while keeping z_min/z_max untouched.
        if self.uniform_margin:
            for pca_entry in self.pca_info:
                if pca_entry["z_min"].numel() == 0:
                    continue
                w = torch.minimum(
                    pca_entry["z_min"] - pca_entry["inf_low"],
                    pca_entry["inf_high"] - pca_entry["z_max"],
                )
                wbar = float(w.mean().item())
                pca_entry["inf_low"] = (pca_entry["z_min"] - wbar).clone()
                pca_entry["inf_high"] = (pca_entry["z_max"] + wbar).clone()
                pca_entry["uniform_margin_wbar"] = wbar
                log.info(
                    f"Layer {pca_entry['layer_name']}: uniform-margin control "
                    f"wbar={wbar:.6f} (was per-coordinate in "
                    f"[{float(w.min())}, {float(w.max())}])"
                )

        # 3. Build param_to_name mapping and snapshot only target layer parameters
        self.param_to_name = {p: n for n, p in model.named_parameters()}
        
        target_params = set()
        for target in self.target_layers.values():
            if hasattr(target, 'weight') and target.weight is not None:
                target_params.add(target.weight)
            if hasattr(target, 'bias') and target.bias is not None:
                target_params.add(target.bias)
        
        # Store snapshots on CPU to save GPU memory
        self.params_snapshot = {
            n: p.detach().clone().cpu() 
            for n, p in model.named_parameters() 
            if p in target_params
        }
        log.info(f"Snapshotted {len(self.params_snapshot)} target layer parameters")

    def freeze_non_target_params(self, model: nn.Module):
        """
        Mark non-target parameters for exclusion from optimization.
        
        NOTE: This does NOT set requires_grad=False to avoid breaking gradient checkpointing.
        Instead, it just identifies target parameters. The caller should only pass target
        parameters to the optimizer using get_trainable_params().
        
        Call this after setup_protection() to prepare for training.
        """
        # Collect all parameters in target layers
        target_params = set()
        for target_layer in self.target_layers.values():
            if hasattr(target_layer, 'weight') and target_layer.weight is not None:
                target_params.add(target_layer.weight)
            if hasattr(target_layer, 'bias') and target_layer.bias is not None:
                target_params.add(target_layer.bias)
        
        self._target_params = target_params
        
        total_params = sum(1 for _ in model.named_parameters())
        trainable_count = len(target_params)
        log.info(f"Marked {trainable_count}/{total_params} parameters as trainable (rest will be excluded from optimizer)")
    
    def get_trainable_params(self, model: nn.Module):
        """
        Get list of trainable parameters to pass to optimizer.
        Call this after freeze_non_target_params().
        
        Returns:
            List of parameters that should be optimized.
        """
        if not hasattr(self, '_target_params'):
            log.warning("get_trainable_params called before freeze_non_target_params, returning all parameters")
            return list(model.parameters())
        
        return [p for p in model.parameters() if p in self._target_params]

    def _interval_chunks(self, delta_f, z_min, z_max, inf_low, inf_high, side,
                         db_vec, dtype, device):
        """Interval-term chunks in the exact evaluation order of the requested
        interval_mode.

        Every chunk is a scalar tensor; callers add them to the layer loss in
        order, which preserves the historical aggregation exactly:
          - "full" + normalize       : single chunk, torch.stack(all corner
            terms).sum() / (2 * len(boxes))        (bitwise historical identity)
          - "full" + no normalization: one chunk per box, the two corner terms
            pre-summed in order (bitwise historical identity)
          - "width_only"/"centre_only": one chunk per box (normalize: their
            stack-sum divided by the mode's own term count = 2)
          - "full_decomp": normalized route returns [Wnorm, Cnorm] so that
            loss(width_only) + loss(centre_only) == loss(full_decomp) bitwise
            with include_mean=False, include_res=False, lambda=1 (single layer)
        """
        boxes = make_region_boxes(
            self.region_mode, inf_low, z_min, z_max, inf_high, side)
        if not boxes:
            return []
        dWp, dWn = torch.relu(delta_f), torch.relu(-delta_f)

        if self.interval_mode == "full":
            flat = []
            pairs = []
            for l, u in boxes:
                leg1 = dWp @ l - dWn @ u
                leg2 = dWp @ u - dWn @ l
                if self.include_db:
                    if db_vec is None:
                        log.warning(
                            f"include_db=True on bias-less layer: delta_b == 0, "
                            f"interval legs unchanged"
                        )
                    else:
                        leg1 = leg1 + db_vec
                        leg2 = leg2 + db_vec
                t1 = leg1.pow(2).mean()
                t2 = leg2.pow(2).mean()
                flat.append(t1)
                flat.append(t2)
                pairs.append(t1 + t2)
            if self.normalize_region:
                return [torch.stack(flat).sum() / (2 * len(boxes))]
            return pairs

        # (F1) decomposition route (width_only / centre_only / full_decomp).
        # include_db is rejected at construction for these modes because the
        # centre/width identity does not hold with delta_b in the legs.
        width_terms = []
        centre_terms = []
        for l, u in boxes:
            c = (l + u) / 2.0
            w = u - l
            width_terms.append((0.5 * (torch.abs(delta_f) @ w).pow(2)).mean())
            centre_terms.append((2.0 * (delta_f @ c).pow(2)).mean())

        if self.interval_mode == "width_only":
            if self.normalize_region:
                return [torch.stack(width_terms).sum() / len(width_terms)]
            return width_terms
        if self.interval_mode == "centre_only":
            if self.normalize_region:
                return [torch.stack(centre_terms).sum() / len(centre_terms)]
            return centre_terms
        # full_decomp: exact identity route used by the (b)+(c)==(a) test
        if self.normalize_region:
            return [
                torch.stack(width_terms).sum() / len(width_terms),
                torch.stack(centre_terms).sum() / len(centre_terms),
            ]
        return width_terms + centre_terms

    def compute_protection_loss(self, model: nn.Module, device) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=device)
        if not self.pca_info: return total_loss

        num_layers = 0
        # Three mutually-exclusive interval modes:
        #   skip_svd          → raw-dim intervals  (z_min has same dim as delta_W)
        #   skip_interval or remove_top_directions or interval_mode=="off"
        #                   → NO intervals at all
        #   otherwise         → projected intervals (U_forget-based)
        # When skip_svd is True it takes precedence: raw intervals are used
        # regardless of skip_interval / remove_top_directions.
        _use_raw_intervals = self.skip_svd
        _no_intervals = (not self.skip_svd) and (
            self.skip_interval or self.remove_top_directions or self.interval_mode == "off"
        )

        for info in self.pca_info:
            layer_name = info["layer_name"]
            target_layer = self.target_layers.get(layer_name)
            if target_layer is None:
                log.warning(f"Target layer {layer_name} not found, skipping protection loss computation.")
                continue

            target_weight = getattr(target_layer, 'weight', None)
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

            w_name = self.param_to_name[target_weight]
            target_bias = getattr(target_layer, 'bias', None)
            b_name = self.param_to_name[target_bias] if target_bias is not None else None

            delta_W_raw = target_weight - self.params_snapshot[w_name].to(device=device, dtype=target_dtype)
            delta_b = (target_bias - self.params_snapshot[b_name].to(device=device, dtype=target_dtype)) if b_name else None

            if isinstance(target_layer, nn.Conv2d):
                C_out, C_in, kH, kW = delta_W_raw.shape
                mu_spatial = mu.view(1, -1, 1, 1)

                num_layers += 1
                layer_loss = torch.tensor(0.0, device=device, dtype=target_dtype)

                if self.include_mean:
                    mean_response = torch.nn.functional.conv2d(
                        mu_spatial, delta_W_raw,
                        bias=delta_b,
                        stride=target_layer.stride,
                        padding=target_layer.padding,
                        dilation=target_layer.dilation,
                        groups=target_layer.groups,
                    )
                    layer_loss = layer_loss + mean_response.pow(2).mean()

                if self.include_res and Ur.size(0) > 0:
                    Ur_spatial = Ur.view(Ur.size(0), -1, 1, 1)
                    residual_responses = torch.nn.functional.conv2d(
                        Ur_spatial, delta_W_raw,
                        stride=target_layer.stride,
                        padding=target_layer.padding,
                        dilation=target_layer.dilation,
                        groups=target_layer.groups,
                    )
                    weighted_responses = residual_responses * Sr.view(-1, 1, 1, 1)
                    num_activations = weighted_responses.numel()
                    layer_loss = layer_loss + torch.norm(weighted_responses, p='fro').pow(2) / num_activations

                # --- Interval protection ---
                if not _no_intervals:
                    if _use_raw_intervals:
                        delta_W_flat = delta_W_raw.permute(0, 2, 3, 1).reshape(C_out, C_in * kH * kW)
                        delta_f = delta_W_flat
                        _z_min = z_min.repeat_interleave(kH * kW)
                        _z_max = z_max.repeat_interleave(kH * kW)
                        _inf_low = inf_low.repeat_interleave(kH * kW)
                        _inf_high = inf_high.repeat_interleave(kH * kW)
                    else:
                        Uf_spatial = Uf.view(Uf.size(0), -1, 1, 1)
                        forget_responses = torch.nn.functional.conv2d(
                            Uf_spatial, delta_W_raw,
                            stride=target_layer.stride,
                            padding=target_layer.padding,
                            dilation=target_layer.dilation,
                            groups=target_layer.groups,
                        )
                        delta_f = forget_responses.view(Uf.size(0), -1).T
                        _z_min, _z_max = z_min, z_max
                        _inf_low, _inf_high = inf_low, inf_high

                    for chunk in self._interval_chunks(
                        delta_f, _z_min, _z_max, _inf_low, _inf_high,
                        info.get("region_side"), delta_b, target_dtype, device,
                    ):
                        layer_loss = layer_loss + chunk

            elif isinstance(target_layer, nn.Linear):
                delta_W = delta_W_raw
                num_layers += 1
                layer_loss = torch.tensor(0.0, device=device, dtype=target_dtype)

                db = delta_b if delta_b is not None else torch.tensor(0.0, device=device, dtype=target_dtype)

                if self.include_mean:
                    global_shift = torch.matmul(delta_W, mu) + db
                    layer_loss = layer_loss + global_shift.pow(2).mean()

                if self.include_res and Ur.size(0) > 0:
                    interference = delta_W @ Ur.T
                    weighted_interference = interference * Sr.unsqueeze(0)
                    num_activations = weighted_interference.numel()
                    layer_loss = layer_loss + torch.norm(weighted_interference, p='fro').pow(2) / num_activations

                # --- Interval protection ---
                if not _no_intervals:
                    if _use_raw_intervals:
                        delta_f = delta_W
                        _z_min, _z_max = z_min, z_max
                        _inf_low, _inf_high = inf_low, inf_high
                    else:
                        delta_f = delta_W @ Uf.T
                        _z_min, _z_max = z_min, z_max
                        _inf_low, _inf_high = inf_low, inf_high

                    for chunk in self._interval_chunks(
                        delta_f, _z_min, _z_max, _inf_low, _inf_high,
                        info.get("region_side"), delta_b, target_dtype, device,
                    ):
                        layer_loss = layer_loss + chunk
            else:
                log.warning(f"Unknown layer type {type(target_layer)} for {layer_name}, skipping")
                continue

            total_loss = total_loss + layer_loss

        if self.normalize_protection and num_layers > 0:
            total_loss = total_loss / num_layers

        return self.lambda_interval * total_loss

    def _collect_activations(self, model, layer_names: List[str], dataloader, device,
                            forward_fn: Callable = None,
                            data_transform_fn=None, betas=None, num_timesteps=1000,
                            pca_components: Optional[Dict] = None):
        """
        Collect activations with optional on-the-fly projection.
        
        Args:
            forward_fn: Function to call model forward. Signature: forward_fn(model, batch, device, **kwargs)
            pca_components: If provided, project activations using {layer_name: {'mu': ..., 'U_forget': ...}}
                           Returns projected [N, reduced_dim] instead of full [N, feature_dim]
        """
        if forward_fn is None:
            forward_fn = ddpm_forward_fn
            
        model.eval()
        buf_dict = {name: [] for name in layer_names}
        layer_type_dict = {name: None for name in layer_names}
        hooks = []
        
        # Register hooks - either raw collection or projection
        if pca_components is None:
            # Standard hook: collect raw activations
            def make_hook(name):
                def hook(module, inp, out):
                    if len(inp) > 0 and inp[0] is not None:
                        input_tensor = inp[0]
                        
                        if layer_type_dict[name] is None:
                            layer_type_dict[name] = type(module).__name__
                        
                        # Handle different layer types
                        if isinstance(module, nn.Conv2d):
                            B, C, H, W = input_tensor.shape
                            reshaped = input_tensor.permute(0, 2, 3, 1).reshape(-1, C)
                        elif isinstance(module, nn.Linear):
                            # For Linear layers, reshape to [N, in_features] matching layer weight
                            # Handles both 2D [B, features] and 3D [B, seq, features] inputs
                            in_features = module.weight.shape[1]
                            reshaped = input_tensor.reshape(-1, in_features)
                        else:
                            # Fallback for other layer types
                            reshaped = input_tensor.view(input_tensor.size(0), -1)
                        buf_dict[name].append(reshaped.detach().cpu())
                return hook
        else:
            # Projection hook: project on GPU, store reduced representation
            def make_hook(name):
                mu = pca_components[name]['mu']
                U_forget = pca_components[name]['U_forget']
                
                def hook(module, inp, out):
                    if len(inp) > 0 and inp[0] is not None:
                        input_tensor = inp[0]
                        
                        # Handle different layer types
                        if isinstance(module, nn.Conv2d):
                            B, C, H, W = input_tensor.shape
                            reshaped = input_tensor.permute(0, 2, 3, 1).reshape(-1, C)
                        elif isinstance(module, nn.Linear):
                            # For Linear layers, reshape to [N, in_features] matching mu dimension
                            # Handles both 2D [B, features] and 3D [B, seq, features] inputs
                            in_features = mu.shape[0]
                            reshaped = input_tensor.reshape(-1, in_features)
                        else:
                            # Fallback for other layer types
                            reshaped = input_tensor.view(input_tensor.size(0), -1)
                        
                        # Project on GPU, then move to CPU
                        centered = reshaped - mu
                        projected = centered @ U_forget.T
                        buf_dict[name].append(projected.detach().cpu())
                return hook
        
        hooks = [layer_module.register_forward_hook(make_hook(layer_name)) 
                for layer_name, layer_module in self.target_layers.items() 
                if layer_name in layer_names]
        
        # Forward pass through all data using provided forward function
        with torch.no_grad():
            for batch in dataloader:
                forward_fn(model, batch, device, 
                          data_transform_fn=data_transform_fn, 
                          betas=betas, 
                          num_timesteps=num_timesteps)
        
        # Remove all hooks
        for h in hooks:
            h.remove()
        model.train()

        # Return results
        result = {}
        mode = "projected" if pca_components else "raw"
        log.info(f"Collected {mode} activations for layers: {list(buf_dict.keys())}, with counts: {[len(buf_dict[name]) for name in buf_dict]}")
        
        for name in buf_dict:
            if len(buf_dict[name]) > 0:
                activations = torch.cat(buf_dict[name], dim=0)
                
                if pca_components is None:
                    result[name] = {
                        'activations': activations,
                        'layer_type': layer_type_dict[name]
                    }
                    log.info(f"  Layer {name}: {result[name]['activations'].shape}, type={layer_type_dict[name]}")
                else:
                    result[name] = activations  # Already projected, just return tensor
                    log.info(f"  Projected layer {name}: {activations.shape}")
            else:
                log.warning(f"Skipping layer {name} - no activations collected")
        
        return result
    
    def _find_target_layers(self, model: nn.Module) -> List[str]:
        """
        Populates: target_layers
        """
        log.info("Finding target layers using named_modules...")
        target_names = []
        
        # Unwrap DataParallel
        base_model = model.module if isinstance(model, nn.DataParallel) else model
        
        for name, module in base_model.named_modules():
            # Check if module matches any pattern
            should_protect = False
            for pattern in self.targets:
                # Match by type name
                if type(module).__name__ == pattern:
                    should_protect = True
                    break
                # Match by substring in layer name
                if pattern.lower() in name.lower():
                    should_protect = True
                    break
            
            if should_protect:
                target_names.append(name)
                self.target_layers[name] = module
        
        log.info(f"Found {len(target_names)} target layers matching patterns: {self.targets}")
        
        return target_names