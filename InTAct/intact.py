import logging
import torch
import torch.nn as nn
from typing import List, Dict, Optional

log = logging.getLogger(__name__)

class UnlearnIntervalProtection:
    def __init__(
        self,
        targets: List[str],
        lambda_interval: float = 10.0,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        reduced_dim: int = 32,
        infinity_scale: float = 20.0,
        use_actual_bounds: bool = False
    ):
        self.targets = targets
        self.lambda_interval = lambda_interval
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.reduced_dim = reduced_dim
        self.infinity_scale = infinity_scale
        self.use_actual_bounds = use_actual_bounds

        self.pca_info: List[Dict] = []
        self.params_snapshot = {}  # Only target layer parameters
        self.target_layers: Dict[str, nn.Module] = {}  # Maps target_name -> target_module
        self.param_to_name: Dict[nn.Parameter, str] = {}  # Maps parameter -> name

    def setup_protection(self, model: nn.Module, forget_dataloader, device, remain_dataloader=None,
                        data_transform_fn=None, betas=None, num_timesteps=1000):
        """
        Populates: pca_info, params_snapshot
        """

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
            data_transform_fn, betas, num_timesteps
        )
        
        # 1b. Optionally collect remain data for actual bounds calculation
        remain_acts_dict = None
        if self.use_actual_bounds and remain_dataloader is not None:
            log.info("Collecting remain data activations for actual bounds...")
            remain_acts_dict = self._collect_activations(
                model, target_names, remain_dataloader, device,
                data_transform_fn, betas, num_timesteps
            )
        
        # 2. Process each target layer
        for layer_name, acts_info in acts_dict.items():
            acts = acts_info['activations']
            original_shape = acts_info['original_shape']
            
            # Centered SVD
            mu = acts.mean(dim=0)
            Xc = acts - mu
            _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            
            k = min(self.reduced_dim, Vh.size(0))
            U_forget = Vh[:k] 
            U_residual = Vh[k:]
            S_residual = S[k:]

            # Define the Forget Box in centered PCA space
            Z = Xc @ U_forget.T
            z_min = torch.quantile(Z, self.lower_percentile, dim=0)
            z_max = torch.quantile(Z, self.upper_percentile, dim=0)
            
            # Calculate actual bounds from remain+forget if requested
            if self.use_actual_bounds and remain_acts_dict is not None and layer_name in remain_acts_dict:
                remain_acts = remain_acts_dict[layer_name]['activations']
                remain_Xc = remain_acts - mu
                remain_Z = remain_Xc @ U_forget.T
                
                # Combine forget and remain projections
                combined_Z = torch.cat([Z, remain_Z], dim=0)
                
                # Use actual min/max as infinity bounds
                inf_low = combined_Z.min(dim=0)[0]
                inf_high = combined_Z.max(dim=0)[0]
                
                log.info(f"Layer {layer_name}: Using actual bounds from remain+forget data")
            else:
                # Use scaled bounds (original behavior)
                inf_low = z_min - self.infinity_scale
                inf_high = z_max + self.infinity_scale

            self.pca_info.append({
                "layer_name": layer_name,
                "mu": mu.detach(),
                "U_forget": U_forget.detach(),
                "U_residual": U_residual.detach(),
                "S_residual": S_residual.detach(),
                "z_min": z_min.detach(),
                "z_max": z_max.detach(),
                "inf_low": inf_low.detach(),
                "inf_high": inf_high.detach(),
                "original_shape": original_shape  # Store for Conv2d layers
            })

        # 3. Build param_to_name mapping and snapshot only target layer parameters
        self.param_to_name = {p: n for n, p in model.named_parameters()}
        
        target_params = set()
        for target in self.target_layers.values():
            if hasattr(target, 'weight') and target.weight is not None:
                target_params.add(target.weight)
            if hasattr(target, 'bias') and target.bias is not None:
                target_params.add(target.bias)
        
        self.params_snapshot = {
            n: p.detach().clone() 
            for n, p in model.named_parameters() 
            if p in target_params
        }
        log.info(f"Snapshotted {len(self.params_snapshot)} target layer parameters")

    def compute_protection_loss(self, model: nn.Module, device) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=device)
        if not self.pca_info: return total_loss

        for info in self.pca_info:
            layer_name = info["layer_name"]
            target_layer = self.target_layers.get(layer_name)
            if target_layer is None: 
                log.warning(f"Target layer {layer_name} not found, skipping protection loss computation.")
                continue
            
            mu = info["mu"].to(device)
            Uf = info["U_forget"].to(device)
            Ur = info["U_residual"].to(device)
            Sr = info["S_residual"].to(device)
            z_min, z_max = info["z_min"].to(device), info["z_max"].to(device)
            
            # Use precomputed bounds (either actual or scaled)
            inf_low = info["inf_low"].to(device)
            inf_high = info["inf_high"].to(device)

            w_name = self.param_to_name[target_layer.weight]
            b_name = self.param_to_name[target_layer.bias] if target_layer.bias is not None else None
            
            delta_W = target_layer.weight - self.params_snapshot[w_name]
            delta_b = (target_layer.bias - self.params_snapshot[b_name]) if b_name else 0

            # === Dimension validation ===
            expected_input_dim = delta_W.shape[1]
            actual_input_dim = mu.shape[0]
            
            if expected_input_dim != actual_input_dim:
                log.warning(f"Dimension mismatch for target {layer_name}: "
                           f"expected {expected_input_dim}, got {actual_input_dim}. Skipping this layer.")
                continue

            # === Handle Linear vs Conv2d layers ===
            if isinstance(target_layer, nn.Linear):
                # --- 1. MEAN TRICK: Global Shift Protection ---
                global_shift = torch.matmul(delta_W, mu) + delta_b
                total_loss += global_shift.pow(2).mean()

                # --- 2. RESIDUAL PROTECTION (Energy-Scaled) ---
                interference = delta_W @ Ur.T
                weighted_interference = interference * Sr.to(device).unsqueeze(0)
                total_loss += torch.norm(weighted_interference, p='fro').pow(2)

                # --- 3. NEGATIVE SPACE IA PROTECTION ---
                delta_f = delta_W @ Uf.T
                dWp, dWn = torch.relu(delta_f), torch.relu(-delta_f)

                # Lower Negative Space IA: [-inf, z_min]
                drift_low_1 = dWp @ inf_low - dWn @ z_min
                drift_low_2 = dWp @ z_min - dWn @ inf_low
                
                # Upper Negative Space IA: [z_max, +inf]
                drift_high_1 = dWp @ z_max - dWn @ inf_high
                drift_high_2 = dWp @ inf_high - dWn @ z_max

                total_loss += (drift_low_1.pow(2).mean() + drift_low_2.pow(2).mean())
                total_loss += (drift_high_1.pow(2).mean() + drift_high_2.pow(2).mean())

            elif isinstance(target_layer, nn.Conv2d):
                # Reshape bounds for conv2d operations
                original_shape = info.get("original_shape")
                
                if original_shape is not None:
                    # Reshape to (1, C, H, W) for conv operations
                    try:
                        mu_view = mu.view(1, *original_shape)
                        z_min_view = z_min.view(1, *original_shape) if z_min.numel() == mu.numel() else z_min.view(1, -1, 1, 1)
                        z_max_view = z_max.view(1, *original_shape) if z_max.numel() == mu.numel() else z_max.view(1, -1, 1, 1)
                        inf_low_view = inf_low.view(1, *original_shape) if inf_low.numel() == mu.numel() else inf_low.view(1, -1, 1, 1)
                        inf_high_view = inf_high.view(1, *original_shape) if inf_high.numel() == mu.numel() else inf_high.view(1, -1, 1, 1)
                    except:
                        log.warning("Original shape mismatch, falling back to channel-only bounds")
                        # Fallback: assume channel-only bounds
                        mu_view = mu.view(1, -1, 1, 1)
                        z_min_view = z_min.view(1, -1, 1, 1)
                        z_max_view = z_max.view(1, -1, 1, 1)
                        inf_low_view = inf_low.view(1, -1, 1, 1)
                        inf_high_view = inf_high.view(1, -1, 1, 1)
                else:
                    # Default: assume channel-only bounds
                    mu_view = mu.view(1, -1, 1, 1)
                    z_min_view = z_min.view(1, -1, 1, 1)
                    z_max_view = z_max.view(1, -1, 1, 1)
                    inf_low_view = inf_low.view(1, -1, 1, 1)
                    inf_high_view = inf_high.view(1, -1, 1, 1)
                
                conv_kwargs = {
                    "stride": target_layer.stride,
                    "padding": target_layer.padding,
                    "dilation": target_layer.dilation,
                    "groups": target_layer.groups,
                }
                
                # Split weight changes into positive and negative
                dW_pos = torch.relu(delta_W)
                dW_neg = torch.relu(-delta_W)
                
                # --- 1. MEAN TRICK: Global Shift Protection ---
                mean_shift = nn.functional.conv2d(mu_view, delta_W, delta_b, **conv_kwargs)
                total_loss += mean_shift.pow(2).mean()
                
                # --- 2. RESIDUAL PROTECTION ---
                # For conv: project Ur back to spatial shape and convolve
                # This is approximate - we treat each output channel independently
                Ur_expanded = Ur.T  # (residual_dim, original_dim)
                interference_loss = torch.tensor(0.0, device=device)
                for i in range(Ur_expanded.size(0)):
                    ur_vec = Ur_expanded[i].view_as(mu_view)
                    interf = nn.functional.conv2d(ur_vec, delta_W, None, **conv_kwargs)
                    interference_loss += (interf.pow(2).mean() * Sr[i].pow(2))
                total_loss += interference_loss
                
                # --- 3. NEGATIVE SPACE IA PROTECTION ---
                # Compute interval bounds through convolution
                lower_bound_1 = nn.functional.conv2d(inf_low_view, dW_pos, None, **conv_kwargs) - \
                                nn.functional.conv2d(z_min_view, dW_neg, None, **conv_kwargs)
                lower_bound_2 = nn.functional.conv2d(z_min_view, dW_pos, None, **conv_kwargs) - \
                                nn.functional.conv2d(inf_low_view, dW_neg, None, **conv_kwargs)
                
                upper_bound_1 = nn.functional.conv2d(z_max_view, dW_pos, None, **conv_kwargs) - \
                                nn.functional.conv2d(inf_high_view, dW_neg, None, **conv_kwargs)
                upper_bound_2 = nn.functional.conv2d(inf_high_view, dW_pos, None, **conv_kwargs) - \
                                nn.functional.conv2d(z_max_view, dW_neg, None, **conv_kwargs)
                
                # Add bias contribution
                if isinstance(delta_b, torch.Tensor):
                    lower_bound_1 = lower_bound_1 + delta_b.view(1, -1, 1, 1)
                    lower_bound_2 = lower_bound_2 + delta_b.view(1, -1, 1, 1)
                    upper_bound_1 = upper_bound_1 + delta_b.view(1, -1, 1, 1)
                    upper_bound_2 = upper_bound_2 + delta_b.view(1, -1, 1, 1)
                
                total_loss += (lower_bound_1.pow(2).mean() + lower_bound_2.pow(2).mean())
                total_loss += (upper_bound_1.pow(2).mean() + upper_bound_2.pow(2).mean())

        return self.lambda_interval * total_loss

    def _collect_activations(self, model, layer_names: List[str], dataloader, device,
                            data_transform_fn=None, betas=None, num_timesteps=1000):
        model.eval()
        buf_dict = {name: [] for name in layer_names}
        shape_dict = {name: None for name in layer_names}
        hooks = []
        
        # Register hooks for all provided layers - collect INPUTS
        def make_hook(name):
            def hook(module, inp, out):
                # inp is a tuple of inputs, typically (input_tensor,) for Linear/Conv2d
                if len(inp) > 0 and inp[0] is not None:
                    input_tensor = inp[0]
                    # Store original shape before flattening (for Conv2d support)
                    if shape_dict[name] is None and len(input_tensor.shape) > 2:
                        shape_dict[name] = input_tensor.shape[1:]  # (C, H, W) or similar
                    buf_dict[name].append(input_tensor.detach().view(input_tensor.size(0), -1))
            return hook
        hooks = [layer_module.register_forward_hook(make_hook(layer_name)) for layer_name, layer_module in self.target_layers.items()]
        
        # Forward pass through all data
        with torch.no_grad():
            for batch in dataloader:
                x, c = batch
                n = x.size(0)
                x = x.to(device)
                c = c.to(device)
                
                # Apply data transform if provided (for diffusion models)
                if data_transform_fn is not None:
                    x = data_transform_fn(x)
                
                # Sample random timesteps for diffusion model
                t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
                t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
                
                # Add diffusion noise if betas provided
                if betas is not None:
                    e = torch.randn_like(x)
                    a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                    x_noisy = x * a.sqrt() + e * (1.0 - a).sqrt()
                else:
                    x_noisy = x
                
                # Call diffusion model with required arguments
                model(x_noisy, t.float(), c, mode="train")
        
        # Remove all hooks
        for h in hooks:
            h.remove()
        model.train()

        # Return both activations and shapes, skip layers with no activations
        result = {}
        log.info(f"Collected input activations for layers: {list(buf_dict.keys())}, with counts: {[len(buf_dict[name]) for name in buf_dict]}")
        for name in buf_dict:
            if len(buf_dict[name]) > 0:  # Only include layers that collected activations
                result[name] = {
                    'activations': torch.cat(buf_dict[name], dim=0),
                    'original_shape': shape_dict[name]
                }
            else:
                log.warning(f"Skipping layer {name} - no activations collected (layer not executed)")
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

# For Classification:

def intact_train_epoch(
    model, optimizer, criterion, forget_loader, device,
    interval_protection: Optional[UnlearnIntervalProtection] = None,
):
    model.train()
    total_loss = total_unlearn = total_protect = 0.0
    n = 0

    for batch in forget_loader:
        X, y = batch[:2]
        X, y = X.to(device), y.to(device)

        out = model(X)
        # Unlearning via Gradient Ascent (Negative Loss)
        unlearn_loss = -criterion(out, y) 

        protect_loss = (
            interval_protection.compute_protection_loss(model, device)
            if interval_protection else torch.tensor(0.0, device=device)
        )

        loss = unlearn_loss + protect_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_unlearn += unlearn_loss.item()
        total_protect += protect_loss.item()
        n += 1

    return total_loss / n, total_unlearn / n, total_protect / n

def intact_unlearn(
    model, forget_loader, criterion, optimizer, num_epochs, device,
    lambda_interval=10.0, lower_percentile=0.05, upper_percentile=0.95, reduced_dim=32,
    infinity_scale=20
):
    protection = UnlearnIntervalProtection(
        lambda_interval=lambda_interval,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        reduced_dim=reduced_dim,
        infinity_scale=infinity_scale
    )

    protection.setup_protection(model, forget_loader, device)

    for epoch in range(num_epochs):
        loss, unlearn, protect = intact_train_epoch(
            model, optimizer, criterion, forget_loader, device, protection
        )
        log.info(f"Epoch {epoch+1}/{num_epochs} | loss={loss:.4f} unlearn={unlearn:.4f} protect={protect:.4f}")

    return model