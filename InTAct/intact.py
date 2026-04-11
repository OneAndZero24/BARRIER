import logging
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable

log = logging.getLogger(__name__)


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
        svd_source: str = "covariance",  # covariance | full_activations
    ):
        self.targets = targets
        self.lambda_interval = lambda_interval
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.reduced_dim = reduced_dim
        self.infinity_scale = infinity_scale
        self.use_actual_bounds = use_actual_bounds
        self.normalize_protection = normalize_protection
        valid_svd_sources = {"covariance", "full_activations"}
        if svd_source not in valid_svd_sources:
            raise ValueError(f"svd_source must be one of {valid_svd_sources}, got: {svd_source}")
        self.svd_source = svd_source

        self.pca_info: List[Dict] = []
        self.params_snapshot = {}  # Only target layer parameters
        self.target_layers: Dict[str, nn.Module] = {}  # Maps target_name -> target_module
        self.param_to_name: Dict[nn.Parameter, str] = {}  # Maps parameter -> name

    def _reshape_hook_input(self, module: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
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
        return reshaped

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

        log.info(f"Setting up InTAct with Mean Reparametrization (svd_source={self.svd_source})...")

        self.pca_info = []
        self.params_snapshot = {}
        self.param_to_name = {}
        self.target_layers = {}
        
        # Find target layers using named_modules
        target_names = self._find_target_layers(model)
        
        if not target_names:
            log.warning("No target layers found for protection")
            return
        
        log.info(f"Found {len(target_names)} target layers to collect inputs from: {target_names}")

        layer_stats: Dict[str, Dict[str, torch.Tensor]] = {
            layer_name: {
                "layer_type": None,
                "running_sum": None,
                "count": 0,
                "mu": None,
                "cov": None,
                "U_forget": None,
                "U_residual": None,
                "S_residual": None,
                "full_forget_activations": [],
                "full_remain_activations": [],
                "projected_forget_activations": [],
                "projected_remain_activations": [],
                "z_min": None,
                "z_max": None,
                "inf_low": None,
                "inf_high": None,
            }
            for layer_name in target_names
        }

        def make_full_collection_update(name: str, bucket_key: str):
            def update_full_collection(reshaped_cpu: torch.Tensor):
                layer_stats[name][bucket_key].append(reshaped_cpu.detach().cpu())
            return update_full_collection

        def make_mean_update(name: str):
            def update_mean(reshaped_cpu: torch.Tensor):
                state = layer_stats[name]
                if state["running_sum"] is None:
                    state["running_sum"] = torch.zeros(reshaped_cpu.shape[1], device="cpu", dtype=torch.float32)
                state["running_sum"].add_(reshaped_cpu.sum(dim=0))
                state["count"] += int(reshaped_cpu.shape[0])
            return update_mean

        def make_cov_update(name: str):
            def update_cov(reshaped_cpu: torch.Tensor):
                state = layer_stats[name]
                mu = state["mu"]
                if mu is None:
                    raise RuntimeError(f"Mean must be computed before covariance for layer {name}")
                if state["cov"] is None:
                    dim = mu.shape[0]
                    state["cov"] = torch.zeros((dim, dim), device="cpu", dtype=torch.float32)

                # Keep the accumulation CPU-bound and avoid large temporary matrices.
                chunk_size = 16384 if reshaped_cpu.shape[1] <= 2048 else 4096
                for start in range(0, reshaped_cpu.shape[0], chunk_size):
                    chunk = reshaped_cpu[start:start + chunk_size]
                    x_centered = chunk - mu
                    state["cov"].add_(x_centered.T @ x_centered)
            return update_cov

        def make_projection_update(name: str, update_forget_bounds: bool, update_actual_bounds: bool):
            def update_projection(reshaped_cpu: torch.Tensor):
                state = layer_stats[name]
                mu = state["mu"]
                U_forget = state["U_forget"]
                if mu is None or U_forget is None:
                    raise RuntimeError(f"PCA basis must be computed before projection for layer {name}")

                projected = (reshaped_cpu - mu) @ U_forget.T
                if update_forget_bounds:
                    state["projected_forget_activations"].append(projected.detach().cpu())
                if update_actual_bounds:
                    state["projected_remain_activations"].append(projected.detach().cpu())
            return update_projection

        pca_components: Dict[str, Dict[str, torch.Tensor]] = {}

        if self.svd_source == "covariance":
            # 1. Pass 1: stream mean
            self._collect_activations(
                model, target_names, forget_dataloader, device,
                forward_fn=forward_fn,
                data_transform_fn=data_transform_fn, betas=betas, num_timesteps=num_timesteps,
                process_fns={name: make_mean_update(name) for name in target_names}
            )

            for layer_name in target_names:
                state = layer_stats[layer_name]
                if state["count"] == 0:
                    log.warning(f"Skipping layer {layer_name} - no activations collected during mean pass")
                    continue
                state["mu"] = state["running_sum"] / float(state["count"])

            # 2. Pass 2: stream covariance
            self._collect_activations(
                model, target_names, forget_dataloader, device,
                forward_fn=forward_fn,
                data_transform_fn=data_transform_fn, betas=betas, num_timesteps=num_timesteps,
                process_fns={name: make_cov_update(name) for name in target_names}
            )

            # 3. Final PCA components from covariance
            for layer_name in target_names:
                state = layer_stats[layer_name]
                if state["cov"] is None or state["mu"] is None:
                    log.warning(f"Skipping layer {layer_name} - insufficient statistics for PCA")
                    continue

                layer_type = state["layer_type"] or type(self.target_layers[layer_name]).__name__
                _, S, Vh = torch.linalg.svd(state["cov"], full_matrices=False)

                k = min(self.reduced_dim, Vh.size(0))
                U_forget = Vh[:k]
                U_residual = Vh[k:]
                # Recover the singular-value scale used by the original centered-data SVD.
                S_residual = torch.sqrt(torch.clamp(S[k:], min=0.0))

                state["U_forget"] = U_forget
                state["U_residual"] = U_residual
                state["S_residual"] = S_residual

                pca_components[layer_name] = {
                    "mu": state["mu"],
                    "U_forget": U_forget,
                    "layer_type": layer_type,
                }
        else:
            # Full-activation mode: materialize forget activations and run classic centered-data SVD.
            self._collect_activations(
                model, target_names, forget_dataloader, device,
                forward_fn=forward_fn,
                data_transform_fn=data_transform_fn, betas=betas, num_timesteps=num_timesteps,
                process_fns={
                    name: make_full_collection_update(name, "full_forget_activations")
                    for name in target_names
                }
            )

            for layer_name in target_names:
                state = layer_stats[layer_name]
                if len(state["full_forget_activations"]) == 0:
                    log.warning(f"Skipping layer {layer_name} - no activations collected in full_activations mode")
                    continue

                acts = torch.cat(state["full_forget_activations"], dim=0)

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
                centered = acts_gpu - mu
                _, S, Vh = torch.linalg.svd(centered, full_matrices=False)

                k = min(self.reduced_dim, Vh.size(0))
                U_forget = Vh[:k]
                U_residual = Vh[k:]
                S_residual = S[k:]

                mu_cpu = mu.detach().cpu()
                U_forget_cpu = U_forget.detach().cpu()
                U_residual_cpu = U_residual.detach().cpu()
                S_residual_cpu = S_residual.detach().cpu()

                state["mu"] = mu_cpu
                state["U_forget"] = U_forget_cpu
                state["U_residual"] = U_residual_cpu
                state["S_residual"] = S_residual_cpu

                pca_components[layer_name] = {
                    "mu": mu_cpu,
                    "U_forget": U_forget_cpu,
                    "layer_type": state["layer_type"] or type(self.target_layers[layer_name]).__name__,
                }

                # Reuse the already-materialized forget activations to produce forget projections.
                projected_forget = centered @ U_forget.T
                state["projected_forget_activations"].append(projected_forget.detach().cpu())
                del acts, acts_gpu, centered

        # 4. Collect projected forget activations (for percentiles).
        if self.svd_source == "covariance":
            self._collect_activations(
                model, list(pca_components.keys()), forget_dataloader, device,
                forward_fn=forward_fn,
                data_transform_fn=data_transform_fn, betas=betas, num_timesteps=num_timesteps,
                process_fns={name: make_projection_update(name, update_forget_bounds=True, update_actual_bounds=False)
                             for name in pca_components}
            )

        # 5. Optionally stream remain activations to tighten the actual bounds
        if self.use_actual_bounds and remain_dataloader is not None:
            log.info("Collecting and projecting remain data on-the-fly...")
            if self.svd_source == "covariance":
                self._collect_activations(
                    model, list(pca_components.keys()), remain_dataloader, device,
                    forward_fn=forward_fn,
                    data_transform_fn=data_transform_fn, betas=betas, num_timesteps=num_timesteps,
                    process_fns={name: make_projection_update(name, update_forget_bounds=False, update_actual_bounds=True)
                                 for name in pca_components}
                )
            else:
                self._collect_activations(
                    model, list(pca_components.keys()), remain_dataloader, device,
                    forward_fn=forward_fn,
                    data_transform_fn=data_transform_fn, betas=betas, num_timesteps=num_timesteps,
                    process_fns={
                        name: make_full_collection_update(name, "full_remain_activations")
                        for name in pca_components
                    }
                )

                for layer_name in pca_components:
                    state = layer_stats[layer_name]
                    if len(state["full_remain_activations"]) == 0:
                        continue
                    remain_acts = torch.cat(state["full_remain_activations"], dim=0)
                    projected_remain = (remain_acts - state["mu"]) @ state["U_forget"].T
                    state["projected_remain_activations"].append(projected_remain.detach().cpu())
                    del remain_acts

        # 6. Materialize the final PCA info structure used by the protection loss
        for layer_name in pca_components:
            state = layer_stats[layer_name]
            if len(state["projected_forget_activations"]) == 0:
                log.warning(f"Skipping layer {layer_name} - no projected activations collected")
                continue

            projected_forget = torch.cat(state["projected_forget_activations"], dim=0)
            z_min = torch.quantile(projected_forget, self.lower_percentile, dim=0)
            z_max = torch.quantile(projected_forget, self.upper_percentile, dim=0)

            if self.use_actual_bounds and remain_dataloader is not None:
                projected_for_bounds = [projected_forget]
                if len(state["projected_remain_activations"]) > 0:
                    projected_for_bounds.append(torch.cat(state["projected_remain_activations"], dim=0))
                stacked_bounds = torch.cat(projected_for_bounds, dim=0)
                inf_low = stacked_bounds.min(dim=0)[0]
                inf_high = stacked_bounds.max(dim=0)[0]
            else:
                inf_low = z_min - self.infinity_scale
                inf_high = z_max + self.infinity_scale

            pca_entry = {
                "layer_name": layer_name,
                "mu": state["mu"].detach().cpu(),
                "U_forget": state["U_forget"].detach().cpu(),
                "U_residual": state["U_residual"].detach().cpu(),
                "S_residual": state["S_residual"].detach().cpu(),
                "z_min": z_min.detach().cpu(),
                "z_max": z_max.detach().cpu(),
                "inf_low": inf_low.detach().cpu(),
                "inf_high": inf_high.detach().cpu(),
                "layer_type": state["layer_type"] or type(self.target_layers[layer_name]).__name__,
            }
            self.pca_info.append(pca_entry)

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

    def compute_protection_loss(self, model: nn.Module, device) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=device)
        if not self.pca_info: return total_loss
        
        num_layers = 0

        for info in self.pca_info:
            layer_name = info["layer_name"]
            target_layer = self.target_layers.get(layer_name)
            if target_layer is None: 
                log.warning(f"Target layer {layer_name} not found, skipping protection loss computation.")
                continue

            target_dtype = target_layer.weight.dtype if target_layer.weight is not None else torch.float32
            
            mu = info["mu"].to(device=device, dtype=target_dtype)
            Uf = info["U_forget"].to(device=device, dtype=target_dtype)
            Ur = info["U_residual"].to(device=device, dtype=target_dtype)
            Sr = info["S_residual"].to(device=device, dtype=target_dtype)
            z_min, z_max = info["z_min"].to(device=device, dtype=target_dtype), info["z_max"].to(device=device, dtype=target_dtype)
            inf_low = info["inf_low"].to(device=device, dtype=target_dtype)
            inf_high = info["inf_high"].to(device=device, dtype=target_dtype)

            w_name = self.param_to_name[target_layer.weight]
            b_name = self.param_to_name[target_layer.bias] if target_layer.bias is not None else None
            
            # Move snapshots to GPU only when needed for computation
            delta_W_raw = target_layer.weight - self.params_snapshot[w_name].to(device=device, dtype=target_dtype)
            delta_b = (target_layer.bias - self.params_snapshot[b_name].to(device=device, dtype=target_dtype)) if b_name else None
            
            if isinstance(target_layer, nn.Conv2d):
                mu_spatial = mu.view(1, -1, 1, 1)
                
                num_layers += 1
                layer_loss = torch.tensor(0.0, device=device, dtype=target_dtype)
                
                mean_response = torch.nn.functional.conv2d(
                    mu_spatial, delta_W_raw, 
                    bias=delta_b,
                    stride=target_layer.stride, 
                    padding=target_layer.padding,
                    dilation=target_layer.dilation,
                    groups=target_layer.groups
                )
                layer_loss = layer_loss + mean_response.pow(2).mean()

                if Ur.size(0) > 0:
                    Ur_spatial = Ur.view(Ur.size(0), -1, 1, 1)
                    
                    residual_responses = torch.nn.functional.conv2d(
                        Ur_spatial, delta_W_raw,
                        stride=target_layer.stride,
                        padding=target_layer.padding,
                        dilation=target_layer.dilation,
                        groups=target_layer.groups
                    )
                    
                    weighted_responses = residual_responses * Sr.view(-1, 1, 1, 1)
                    # Normalize by number of activations (output elements)
                    num_activations = weighted_responses.numel()
                    layer_loss = layer_loss + torch.norm(weighted_responses, p='fro').pow(2) / num_activations

                Uf_spatial = Uf.view(Uf.size(0), -1, 1, 1)
                
                forget_responses = torch.nn.functional.conv2d(
                    Uf_spatial, delta_W_raw,
                    stride=target_layer.stride,
                    padding=target_layer.padding,
                    dilation=target_layer.dilation,
                    groups=target_layer.groups
                )
                
                delta_f = forget_responses.view(Uf.size(0), -1).T
                
                dWp, dWn = torch.relu(delta_f), torch.relu(-delta_f)

                drift_low_1 = dWp @ inf_low - dWn @ z_min
                drift_low_2 = dWp @ z_min - dWn @ inf_low
                
                drift_high_1 = dWp @ z_max - dWn @ inf_high
                drift_high_2 = dWp @ inf_high - dWn @ z_max

                layer_loss = layer_loss + (drift_low_1.pow(2).mean() + drift_low_2.pow(2).mean())
                layer_loss = layer_loss + (drift_high_1.pow(2).mean() + drift_high_2.pow(2).mean())
                
            elif isinstance(target_layer, nn.Linear):
                delta_W = delta_W_raw
                num_layers += 1
                layer_loss = torch.tensor(0.0, device=device, dtype=target_dtype)

                db = delta_b if delta_b is not None else torch.tensor(0.0, device=device, dtype=target_dtype)
                
                global_shift = torch.matmul(delta_W, mu) + db
                layer_loss = layer_loss + global_shift.pow(2).mean()

                if Ur.size(0) > 0:
                    interference = delta_W @ Ur.T
                    weighted_interference = interference * Sr.unsqueeze(0)
                    # Normalize by number of activations (output elements)
                    num_activations = weighted_interference.numel()
                    layer_loss = layer_loss + torch.norm(weighted_interference, p='fro').pow(2) / num_activations

                delta_f = delta_W @ Uf.T
                dWp, dWn = torch.relu(delta_f), torch.relu(-delta_f)

                drift_low_1 = dWp @ inf_low - dWn @ z_min
                drift_low_2 = dWp @ z_min - dWn @ inf_low
                
                drift_high_1 = dWp @ z_max - dWn @ inf_high
                drift_high_2 = dWp @ inf_high - dWn @ z_max

                layer_loss = layer_loss + (drift_low_1.pow(2).mean() + drift_low_2.pow(2).mean())
                layer_loss = layer_loss + (drift_high_1.pow(2).mean() + drift_high_2.pow(2).mean())
            else:
                log.warning(f"Unknown layer type {type(target_layer)} for {layer_name}, skipping")
                continue
            
            total_loss = total_loss + layer_loss

        # Normalize by number of layers if requested
        if self.normalize_protection and num_layers > 0:
            total_loss = total_loss / num_layers

        return self.lambda_interval * total_loss

    def _collect_activations(self, model, layer_names: List[str], dataloader, device,
                            forward_fn: Callable = None,
                            data_transform_fn=None, betas=None, num_timesteps=1000,
                            process_fns: Optional[Dict[str, Callable[[torch.Tensor], None]]] = None):
        """
        Stream activations through hooks and invoke per-layer callbacks.

        This method intentionally does not store activations. Callers provide a
        per-layer callback that receives each reshaped activation batch on CPU.
        """
        if forward_fn is None:
            forward_fn = ddpm_forward_fn
            
        model.eval()
        layer_type_dict = {name: None for name in layer_names}
        hooks = []
        
        def make_hook(name):
            def hook(module, inp, out):
                if len(inp) > 0 and inp[0] is not None:
                    input_tensor = inp[0]

                    if layer_type_dict[name] is None:
                        layer_type_dict[name] = type(module).__name__

                    reshaped = self._reshape_hook_input(module, input_tensor)
                    reshaped_cpu = reshaped.detach().to(device="cpu", dtype=torch.float32)
                    try:
                        process_fn = process_fns.get(name) if process_fns is not None else None
                        if process_fn is not None:
                            process_fn(reshaped_cpu)
                    finally:
                        del reshaped_cpu, reshaped, input_tensor
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

        log.info(f"Streamed activations for layers: {list(layer_type_dict.keys())}")
        return layer_type_dict
    
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