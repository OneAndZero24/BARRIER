import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class UnlearnIntervalProtection:
    """
    Unlearning with interval protection using negative space.
    
    Steps:
    1. Collect activations from forget classes
    2. Compute [forget_min, forget_max] intervals
    3. Protect everything outside: [-inf, forget_min] U [forget_max, +inf]
    
    Args:
        lambda_interval: Weight for the interval protection loss
        lower_percentile: Lower percentile for forget interval bounds
        upper_percentile: Upper percentile for forget interval bounds
        infinity_scale: Scale factor for "infinity" bounds
    """
    
    def __init__(
        self,
        lambda_interval: float = 1.0,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        infinity_scale: float = 10.0
    ):
        self.lambda_interval = lambda_interval
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.infinity_scale = infinity_scale
        
        self.params_snapshot = {}
        self.protected_intervals = []
        self.interval_layers = []
        
        log.info(f"UnlearnIntervalProtection initialized:")
        log.info(f"  lambda_interval={lambda_interval}")
        log.info(f"  percentiles=[{lower_percentile}, {upper_percentile}]")
        log.info(f"  infinity_scale={infinity_scale}")
    
    
    def setup_protection(self, model: nn.Module, forget_dataloader, device):
        """
        Set up interval protection from forget set.
        
        Args:
            model: The model to protect
            forget_dataloader: DataLoader with samples from classes to unlearn
            device: Device to run computations on
        """
        log.info("Setting up interval protection...")
        
        # Auto-detect feature layer before classifier
        self.interval_layers = []
        feature_layer = self._find_feature_layer(model)

        if feature_layer is not None:
            layer_name, layer_module = feature_layer
            self.interval_layers.append((layer_name, layer_module))
            log.info(f"Auto-detected feature layer: {layer_name}")
        
        if len(self.interval_layers) == 0:
            log.warning("No feature layers found! Protection will be disabled.")
            return
        
        # Collect activations from FORGET classes
        forget_activations = self._collect_activations(model, forget_dataloader, device)
        
        # Compute forget intervals and set up negative space protection
        self.protected_intervals = []
        for idx, (layer_name, layer) in enumerate(self.interval_layers):
            forget_acts = forget_activations[idx]
            
            if len(forget_acts) == 0:
                log.warning(f"No activations collected for layer {layer_name}")
                continue
            
            # Compute bounds for forget classes using percentiles
            sorted_acts, _ = torch.sort(forget_acts, dim=0)
            n_samples = sorted_acts.size(0)
            
            lower_idx = int(n_samples * self.lower_percentile)
            upper_idx = int(n_samples * self.upper_percentile)
            
            forget_min = sorted_acts[lower_idx]
            forget_max = sorted_acts[upper_idx]
            
            protected_interval = {
                'forget_min': forget_min,
                'forget_max': forget_max,
                'layer_name': layer_name
            }
            
            self.protected_intervals.append(protected_interval)
            
            log.info(f"Layer {layer_name}:")
            log.info(f"  Forget range: [{forget_min.mean().item():.4f}, {forget_max.mean().item():.4f}]")
            log.info(f"  Protection: NEGATIVE SPACE (everything outside forget range)")
        
        # Save parameter snapshot
        self.params_snapshot = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params_snapshot[name] = param.detach().clone()
        
        log.info(f"Protection setup complete. Tracking {len(self.params_snapshot)} parameters.")
    
    
    def _collect_activations(self, model, dataloader, device):
        """Collect activations from all feature layers."""
        model.eval()
        
        activation_buffers = {idx: [] for idx in range(len(self.interval_layers))}
        hook_handles = []
        
        # Register hooks to collect activations
        for idx, (layer_name, layer) in enumerate(self.interval_layers):
            def hook(module, input, output, idx=idx):
                # Flatten if needed (e.g., after pooling layers)
                if output.dim() > 2:
                    output = output.view(output.size(0), -1)
                activation_buffers[idx].append(output.detach())
            
            handle = layer.register_forward_hook(hook)
            hook_handles.append(handle)
        
        # Run forward passes
        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 3:
                    X, y, _ = batch
                else:
                    X, y = batch
                X = X.to(device)
                _ = model(X)
        
        # Remove hooks
        for handle in hook_handles:
            handle.remove()
        
        # Concatenate all activations
        result = []
        for idx in range(len(self.interval_layers)):
            if len(activation_buffers[idx]) > 0:
                acts = torch.cat(activation_buffers[idx], dim=0)
                result.append(acts)
            else:
                result.append(torch.tensor([]))
        
        model.train()
        return result
    
    
    def compute_protection_loss(self, model: nn.Module, device) -> torch.Tensor:
        """
        Compute the interval protection loss for negative space.
        
        Penalizes weight changes that affect activations in:
        - Interval 1: [-inf, forget_min]
        - Interval 2: [forget_max, +inf]
        
        Returns:
            Protection loss value
        """
        if len(self.protected_intervals) == 0 or len(self.params_snapshot) == 0:
            return torch.tensor(0.0, device=device)
        
        total_loss = torch.tensor(0.0, device=device)
        
        for idx, interval_info in enumerate(self.protected_intervals):
            layer_name = interval_info['layer_name']
            forget_min = interval_info['forget_min'].to(device)
            forget_max = interval_info['forget_max'].to(device)
            
            # Find the next linear layer after this feature layer
            next_linear = self._find_next_linear(model, layer_name)
            
            if next_linear is None:
                continue
            
            # Compute "infinity" bounds for negative space
            neg_inf = forget_min - self.infinity_scale * torch.abs(forget_min)
            pos_inf = forget_max + self.infinity_scale * torch.abs(forget_max)
            
            # Initialize bound regularizations
            lower_bound_reg_1 = None
            upper_bound_reg_1 = None
            lower_bound_reg_2 = None
            upper_bound_reg_2 = None
            
            for name, param in next_linear.named_parameters():
                # Find corresponding snapshot parameter
                param_full_name = None
                for mod_name, mod_param in model.named_parameters():
                    if mod_param is param:
                        param_full_name = mod_name
                        break
                
                if param_full_name is None or param_full_name not in self.params_snapshot:
                    continue
                
                prev_param = self.params_snapshot[param_full_name]
               
                if "weight" in name:
                    weight_diff = param - prev_param
                    weight_diff_pos = torch.relu(weight_diff)
                    weight_diff_neg = torch.relu(-weight_diff)
                    
                    # Interval 1: [-inf, forget_min] (negative space below forget)
                    lower_contrib_1 = weight_diff_pos @ neg_inf - weight_diff_neg @ forget_min
                    upper_contrib_1 = weight_diff_pos @ forget_min - weight_diff_neg @ neg_inf
                    
                    # Interval 2: [forget_max, +inf] (negative space above forget)
                    lower_contrib_2 = weight_diff_pos @ forget_max - weight_diff_neg @ pos_inf
                    upper_contrib_2 = weight_diff_pos @ pos_inf - weight_diff_neg @ forget_max
                    
                    if lower_bound_reg_1 is None:
                        lower_bound_reg_1 = lower_contrib_1
                        upper_bound_reg_1 = upper_contrib_1
                        lower_bound_reg_2 = lower_contrib_2
                        upper_bound_reg_2 = upper_contrib_2
                    else:
                        lower_bound_reg_1 = lower_bound_reg_1 + lower_contrib_1
                        upper_bound_reg_1 = upper_bound_reg_1 + upper_contrib_1
                        lower_bound_reg_2 = lower_bound_reg_2 + lower_contrib_2
                        upper_bound_reg_2 = upper_bound_reg_2 + upper_contrib_2
                
                elif "bias" in name:
                    bias_diff = param - prev_param
                    if lower_bound_reg_1 is not None:
                        lower_bound_reg_1 = lower_bound_reg_1 + bias_diff
                        upper_bound_reg_1 = upper_bound_reg_1 + bias_diff
                        lower_bound_reg_2 = lower_bound_reg_2 + bias_diff
                        upper_bound_reg_2 = upper_bound_reg_2 + bias_diff
            
            if lower_bound_reg_1 is not None:
                # Add loss from both negative space intervals
                total_loss += lower_bound_reg_1.pow(2).mean() + upper_bound_reg_1.pow(2).mean()
                total_loss += lower_bound_reg_2.pow(2).mean() + upper_bound_reg_2.pow(2).mean()
        
        return self.lambda_interval * total_loss
    
    
    def _find_next_linear(self, model: nn.Module, interval_layer_name: str) -> Optional[nn.Module]:
        """Find the next Linear layer after the specified feature layer."""
        module_list = list(model.named_modules())
        
        # Find index of interval layer
        interval_idx = None
        for i, (name, module) in enumerate(module_list):
            if name == interval_layer_name:
                interval_idx = i
                break
        
        if interval_idx is None:
            return None
        
        # Find next Linear layer
        for i in range(interval_idx + 1, len(module_list)):
            name, module = module_list[i]
            if isinstance(module, nn.Linear):
                return module
        
        return None
    
    
    def _find_feature_layer(self, model: nn.Module) -> Optional[Tuple[str, nn.Module]]:
        """
        Auto-detect the last feature layer before the classifier head.
        
        Returns:
            Tuple of (layer_name, layer_module) or None
        """
        module_list = list(model.named_modules())
        
        # Find the final classifier
        classifier_idx = None
        for i, (name, module) in enumerate(module_list):
            if isinstance(module, nn.Linear):
                # Look for top-level fc/classifier/head
                if any(keyword in name for keyword in ['fc', 'classifier', 'head']):
                    if name.count('.') <= 1:
                        classifier_idx = i
                        break
        
        if classifier_idx is None:
            log.warning("Could not find classifier layer")
            return None
        
        # Find the last meaningful layer before the classifier
        for i in range(classifier_idx - 1, -1, -1):
            name, module = module_list[i]
            
            # Skip container modules
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                continue
            
            # Look for pooling layers (ideal for feature extraction)
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.MaxPool2d)):
                log.info(f"Found pooling layer: {name}")
                return (name, module)
            
            # Look for last conv layers
            if isinstance(module, nn.Conv2d) and name.count('.') <= 1:
                log.info(f"Found conv layer: {name}")
                return (name, module)
            
            # For ResNet: look for layer4
            if 'layer4' in name and name.count('.') == 0:
                log.info(f"Found ResNet layer4: {name}")
                return (name, module)
            
            # For VGG: look for features module
            if name == 'features':
                log.info(f"Found VGG features: {name}")
                return (name, module)
        
        log.warning("Could not auto-detect feature layer")
        return None


def intact_train_epoch(
    model, 
    optimizer, 
    criterion, 
    forget_loader, 
    device,
    interval_protection: Optional[UnlearnIntervalProtection] = None
):
    """
    Train one epoch on forget set with interval protection.
    
    Returns:
        avg_loss: Average total loss
        avg_unlearn_loss: Average unlearning loss
        avg_protection_loss: Average protection loss
    """
    model.train()
    
    total_loss = 0.0
    total_unlearn_loss = 0.0
    total_protection_loss = 0.0
    n_batches = 0
    
    for batch in forget_loader:
        if len(batch) == 3:
            X, y, _ = batch
        else:
            X, y = batch
        
        X = X.to(device)
        y = y.to(device)
        
        # Forward pass
        output = model(X)
        
        # Unlearning loss (negative of standard loss to forget)
        unlearn_loss = -criterion(output, y)
        
        # Protection loss
        protection_loss = torch.tensor(0.0, device=device)
        if interval_protection is not None:
            protection_loss = interval_protection.compute_protection_loss(model, device)
        
        # Total loss
        loss = unlearn_loss + protection_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        total_unlearn_loss += unlearn_loss.item()
        total_protection_loss += protection_loss.item()
        n_batches += 1
    
    return (
        total_loss / n_batches,
        total_unlearn_loss / n_batches,
        total_protection_loss / n_batches
    )


def intact_unlearn(
    model,
    forget_loader,
    criterion,
    optimizer,
    num_epochs: int,
    device,
    lambda_interval: float = 1.0,
    lower_percentile: float = 0.05,
    upper_percentile: float = 0.95,
    infinity_scale: float = 10.0
):
    """
    Main InTAct unlearning procedure.
    
    Args:
        model: Model to unlearn from
        forget_loader: DataLoader for forget set
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of unlearning epochs
        device: Device to use
        lambda_interval: Weight for interval protection loss
        lower_percentile: Lower percentile for forget intervals
        upper_percentile: Upper percentile for forget intervals
        infinity_scale: Scale factor for infinity bounds
    
    Returns:
        model: Unlearned model
    """
    # Initialize interval protection
    interval_protection = UnlearnIntervalProtection(
        lambda_interval=lambda_interval,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        infinity_scale=infinity_scale
    )
    
    # Setup protection from forget set
    interval_protection.setup_protection(model, forget_loader, device)
    
    # Unlearning epochs
    log.info(f"\nStarting unlearning for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        avg_loss, avg_unlearn, avg_protection = intact_train_epoch(
            model, optimizer, criterion, forget_loader, device, interval_protection
        )
        
        log.info(f"Epoch {epoch+1}/{num_epochs}: "
                 f"loss={avg_loss:.4f}, unlearn={avg_unlearn:.4f}, protection={avg_protection:.4f}")
    
    return model
