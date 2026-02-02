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
        use_actual_bounds: bool = False,
        normalize_protection: bool = True,  # Normalize protection loss by number of layers
    ):
        self.targets = targets
        self.lambda_interval = lambda_interval
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.reduced_dim = reduced_dim
        self.infinity_scale = infinity_scale
        self.use_actual_bounds = use_actual_bounds
        self.normalize_protection = normalize_protection

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
        
        # 2. Compute SVD on forget data and optionally collect projected remain data
        pca_components = {}  # Store mu and U_forget for each layer
        
        for layer_name, acts_info in acts_dict.items():
            acts = acts_info['activations']
            layer_type = acts_info.get('layer_type', 'Linear')
            
            acts_gpu = acts.to(device)
            
            # Centered SVD
            mu = acts_gpu.mean(dim=0)
            Xc = acts_gpu - mu
            _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
            
            k = min(self.reduced_dim, Vh.size(0))
            U_forget = Vh[:k] 
            U_residual = Vh[k:]
            S_residual = S[k:]

            # Define the Forget Box in centered PCA space
            Z_forget = Xc @ U_forget.T
            z_min = torch.quantile(Z_forget, self.lower_percentile, dim=0)
            z_max = torch.quantile(Z_forget, self.upper_percentile, dim=0)
            
            # Store PCA components for remain data projection
            pca_components[layer_name] = {
                'mu': mu,
                'U_forget': U_forget,
                'layer_type': layer_type
            }
            
            # Free GPU memory
            del acts_gpu, Xc
            
            # Calculate actual bounds from remain+forget if requested
            if self.use_actual_bounds and remain_dataloader is not None:
                # Start with forget data bounds
                combined_min = Z_forget.min(dim=0)[0]
                combined_max = Z_forget.max(dim=0)[0]
                del Z_forget
                
                # Will collect and project remain data on-the-fly
                inf_low = combined_min  # Temporary, will update after remain collection
                inf_high = combined_max
            else:
                # Use scaled bounds (original behavior)
                inf_low = z_min - self.infinity_scale
                inf_high = z_max + self.infinity_scale
                del Z_forget

            # Store PCA info (will update inf_low/inf_high after remain collection if needed)
            pca_entry = {
                "layer_name": layer_name,
                "mu": mu.detach().cpu(),
                "U_forget": U_forget.detach().cpu(),
                "U_residual": U_residual.detach().cpu(),
                "S_residual": S_residual.detach().cpu(),
                "z_min": z_min.detach().cpu(),
                "z_max": z_max.detach().cpu(),
                "inf_low": inf_low.detach().cpu(),
                "inf_high": inf_high.detach().cpu(),
                "layer_type": layer_type
            }
            self.pca_info.append(pca_entry)
        
        # 2b. Collect projected remain data and update bounds
        if self.use_actual_bounds and remain_dataloader is not None:
            log.info("Collecting and projecting remain data on-the-fly...")
            remain_projected = self._collect_projected_activations(
                model, pca_components, remain_dataloader, device,
                data_transform_fn, betas, num_timesteps
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
        Freeze all model parameters except those in target layers.
        Call this after setup_protection() to ensure only protected layers are trainable.
        """
        # Collect all parameters in target layers
        target_params = set()
        for target_layer in self.target_layers.values():
            if hasattr(target_layer, 'weight') and target_layer.weight is not None:
                target_params.add(target_layer.weight)
            if hasattr(target_layer, 'bias') and target_layer.bias is not None:
                target_params.add(target_layer.bias)
        
        # Freeze/unfreeze parameters
        frozen_count = 0
        trainable_count = 0
        for name, param in model.named_parameters():
            if param in target_params:
                param.requires_grad = True
                trainable_count += 1
            else:
                param.requires_grad = False
                frozen_count += 1
        
        log.info(f"Frozen {frozen_count} parameters, kept {trainable_count} target layer parameters trainable")

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
            
            mu = info["mu"].to(device)
            Uf = info["U_forget"].to(device)
            Ur = info["U_residual"].to(device)
            Sr = info["S_residual"].to(device)
            z_min, z_max = info["z_min"].to(device), info["z_max"].to(device)
            inf_low = info["inf_low"].to(device)
            inf_high = info["inf_high"].to(device)

            w_name = self.param_to_name[target_layer.weight]
            b_name = self.param_to_name[target_layer.bias] if target_layer.bias is not None else None
            
            # Move snapshots to GPU only when needed for computation
            delta_W_raw = target_layer.weight - self.params_snapshot[w_name].to(device)
            delta_b = (target_layer.bias - self.params_snapshot[b_name].to(device)) if b_name else torch.tensor(0.0, device=device)
            
            if isinstance(target_layer, nn.Conv2d):
                mu_spatial = mu.view(1, -1, 1, 1)
                
                num_layers += 1
                layer_loss = torch.tensor(0.0, device=device)
                
                mean_response = torch.nn.functional.conv2d(
                    mu_spatial, delta_W_raw, 
                    bias=delta_b if isinstance(delta_b, torch.Tensor) else None,
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
                    layer_loss = layer_loss + torch.norm(weighted_responses, p='fro').pow(2)

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
                layer_loss = torch.tensor(0.0, device=device)

                if isinstance(delta_b, torch.Tensor):
                    db = delta_b
                else:
                    db = torch.tensor(0.0, device=device)
                
                global_shift = torch.matmul(delta_W, mu) + db
                layer_loss = layer_loss + global_shift.pow(2).mean()

                if Ur.size(0) > 0:
                    interference = delta_W @ Ur.T
                    weighted_interference = interference * Sr.unsqueeze(0)
                    layer_loss = layer_loss + torch.norm(weighted_interference, p='fro').pow(2)

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
                            data_transform_fn=None, betas=None, num_timesteps=1000):
        model.eval()
        buf_dict = {name: [] for name in layer_names}
        layer_type_dict = {name: None for name in layer_names}
        hooks = []
        
        # Register hooks for all provided layers - collect INPUTS
        def make_hook(name):
            def hook(module, inp, out):
                # inp is a tuple of inputs, typically (input_tensor,) for Linear/Conv2d
                if len(inp) > 0 and inp[0] is not None:
                    input_tensor = inp[0]
                    
                    # Store layer type
                    if layer_type_dict[name] is None:
                        layer_type_dict[name] = type(module).__name__
                    
                    # Handle different layer types
                    if isinstance(module, nn.Conv2d):
                        # For ALL Conv2d layers: treat each spatial position as a sample
                        # Input: (B, C, H, W) -> (B*H*W, C)
                        # This works for any kernel size and gives channel-wise statistics
                        B, C, H, W = input_tensor.shape
                        # Reshape to (B*H*W, C) - each spatial position is a sample
                        reshaped = input_tensor.permute(0, 2, 3, 1).reshape(-1, C)
                        buf_dict[name].append(reshaped.detach().cpu())  # Move to CPU immediately
                    else:
                        buf_dict[name].append(input_tensor.detach().view(input_tensor.size(0), -1).cpu())  # Move to CPU immediately
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
            if len(buf_dict[name]) > 0:
                # Activations are already on CPU from the hook
                activations = torch.cat(buf_dict[name], dim=0)
                result[name] = {
                    'activations': activations,
                    'layer_type': layer_type_dict[name]
                }
                log.info(f"  Layer {name}: {result[name]['activations'].shape}, type={layer_type_dict[name]}")
            else:
                log.warning(f"Skipping layer {name} - no activations collected (layer not executed)")
        return result
    
    def _collect_projected_activations(self, model, pca_components: Dict, dataloader, device,
                                       data_transform_fn=None, betas=None, num_timesteps=1000):
        """
        Collect activations and project them on-the-fly using pre-computed PCA components.
        Returns dict of {layer_name: projected_activations} where projected_activations is [N, reduced_dim].
        
        This saves memory by storing only reduced-dimension projections instead of full activations.
        """
        model.eval()
        projected_dict = {name: [] for name in pca_components.keys()}
        hooks = []
        
        # Register hooks that project activations immediately
        def make_projection_hook(layer_name, mu, U_forget):
            def hook(module, inp, out):
                if len(inp) > 0 and inp[0] is not None:
                    input_tensor = inp[0]
                    
                    # Handle different layer types
                    if isinstance(module, nn.Conv2d):
                        B, C, H, W = input_tensor.shape
                        reshaped = input_tensor.permute(0, 2, 3, 1).reshape(-1, C)
                    else:
                        reshaped = input_tensor.view(input_tensor.size(0), -1)
                    
                    # Project on GPU, then move to CPU
                    centered = reshaped - mu
                    projected = centered @ U_forget.T
                    projected_dict[layer_name].append(projected.detach().cpu())
            return hook
        
        # Register hooks with PCA components already on GPU
        for layer_name, components in pca_components.items():
            layer_module = self.target_layers[layer_name]
            mu = components['mu']
            U_forget = components['U_forget']
            hook = layer_module.register_forward_hook(make_projection_hook(layer_name, mu, U_forget))
            hooks.append(hook)
        
        # Forward pass through all data
        with torch.no_grad():
            for batch in dataloader:
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
        
        # Remove hooks
        for h in hooks:
            h.remove()
        model.train()
        
        # Concatenate projected activations
        result = {}
        for name, projections in projected_dict.items():
            if len(projections) > 0:
                result[name] = torch.cat(projections, dim=0)
                log.info(f"  Projected layer {name}: {result[name].shape} (reduced from full activations)")
        
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
    
    # Freeze all parameters except target layers
    protection.freeze_non_target_params(model)

    for epoch in range(num_epochs):
        loss, unlearn, protect = intact_train_epoch(
            model, optimizer, criterion, forget_loader, device, protection
        )
        log.info(f"Epoch {epoch+1}/{num_epochs} | loss={loss:.4f} unlearn={unlearn:.4f} protect={protect:.4f}")

    return model