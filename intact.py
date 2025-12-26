import sys
import time
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import utils

from .impl import iterative_unlearn

sys.path.append(".")
from imagenet import get_x_y_from_data_dict

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class UnlearnIntervalProtection:
    """
    Unlearning with interval protection for non-continual learning scenarios.
    
    This method protects the knowledge of classes we want to RETAIN by:
    1. Computing activation intervals on retain classes
    2. Penalizing weight changes that would affect activations in the protected intervals
    
    The goal is to unlearn specific classes while preserving the rest.
    
    Args:
        lambda_interval (float): Weight for the interval protection loss
        compute_intervals_from_data (bool): If True, computes intervals from data before unlearning
        infinity_scale (float): Scale factor for "infinity" bounds (default: 10.0)
    """
    
    def __init__(
        self,
        lambda_interval: float = 1.0,
        compute_intervals_from_data: bool = True,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        infinity_scale: float = 10.0
    ):
        self.lambda_interval = lambda_interval
        self.compute_intervals_from_data = compute_intervals_from_data
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.infinity_scale = infinity_scale
        
        self.params_snapshot = {}
        self.protected_intervals = []  # List of (min, max) tuples for each interval layer
        self.interval_layers = []
        
        log.info(f"UnlearnIntervalProtection initialized with lambda_interval={lambda_interval}, "
                 f"percentiles=[{lower_percentile}, {upper_percentile}], infinity_scale={infinity_scale}")
    
    
    def setup_protection(self, model: nn.Module, retain_dataloader, unlearn_dataloader, device):
        """
        Set up interval protection before unlearning starts.
        
        Steps:
        1. Collect activations from retain classes (what we want to keep)
        2. Compute protected intervals based on retain class activations
        3. Save parameter snapshot for computing weight changes
        
        Args:
            model: The model to protect
            retain_dataloader: DataLoader with samples from classes to retain
            unlearn_dataloader: DataLoader with samples from classes to unlearn (not used)
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
        else:
            # Fallback: look for IntervalActivation layers
            for name, module in model.named_modules():
                if type(module).__name__ == "IntervalActivation":
                    self.interval_layers.append((name, module))
                    log.info(f"Found IntervalActivation layer: {name}")
        
        if len(self.interval_layers) == 0:
            log.warning("No feature layers found! Protection will be disabled.")
            return
        
        # Collect activations from retain classes only
        retain_activations = self._collect_activations(model, retain_dataloader, device)
        
        # Compute protected intervals based on retain data
        self.protected_intervals = []
        for idx, (layer_name, layer) in enumerate(self.interval_layers):
            retain_acts = retain_activations[idx]  # Shape: (N_retain, features)
            
            if len(retain_acts) == 0:
                log.warning(f"No activations collected for layer {layer_name}")
                continue
            
            # Compute bounds for retain classes using percentiles
            sorted_acts, _ = torch.sort(retain_acts, dim=0)
            n_samples = sorted_acts.size(0)
            
            lower_idx = int(n_samples * self.lower_percentile)
            upper_idx = int(n_samples * self.upper_percentile)
            
            retain_min = sorted_acts[lower_idx]
            retain_max = sorted_acts[upper_idx]
            
            protected_interval = {
                'retain_min': retain_min,
                'retain_max': retain_max,
                'layer_name': layer_name
            }
            
            self.protected_intervals.append(protected_interval)
            
            log.info(f"Layer {layer_name}: Protected interval computed")
            log.info(f"  Retain range: [{retain_min.mean().item():.4f}, {retain_max.mean().item():.4f}]")
        
        # Save parameter snapshot
        self.params_snapshot = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.params_snapshot[name] = param.detach().clone()
        
        log.info(f"Interval protection setup complete. Tracking {len(self.params_snapshot)} parameters.")
    
    
    def _collect_activations(self, model, dataloader, device):
        """
        Collect activations from all feature layers.
        
        Returns:
            List of activation tensors, one per interval layer
        """
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
                # Handle both 2-value and 3-value unpacking
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
                acts = torch.cat(activation_buffers[idx], dim=0)  # (N, features)
                result.append(acts)
            else:
                result.append(torch.tensor([]))
        
        model.train()
        return result
    
    
    def compute_protection_loss(self, model: nn.Module, device) -> torch.Tensor:
        """
        Compute the interval protection loss.
        
        This loss penalizes weight changes that would cause the output to change
        within the protected intervals: [-inf, forget_min] and [forget_max, +inf].
        
        Returns:
            Tensor: Protection loss value
        """
        
        if len(self.protected_intervals) == 0 or len(self.params_snapshot) == 0:
            return torch.tensor(0.0, device=device)
        
        total_loss = torch.tensor(0.0, device=device)
        
        # For each interval layer, compute protection loss for two intervals
        for idx, interval_info in enumerate(self.protected_intervals):
            layer_name = interval_info['layer_name']
            forget_min = interval_info['retain_min'].to(device)
            forget_max = interval_info['retain_max'].to(device)
            
            # Find the next linear layer after this interval activation
            next_linear = self._find_next_linear(model, layer_name)
            
            if next_linear is None:
                continue
            
            # Compute "infinity" bounds
            neg_inf = forget_min - self.infinity_scale * torch.abs(forget_min)
            pos_inf = forget_max + self.infinity_scale * torch.abs(forget_max)
            
            # Apply loss to interval 1: [-inf, forget_min]
            lower_bound_reg_1 = None
            upper_bound_reg_1 = None
            
            # Apply loss to interval 2: [forget_max, +inf]
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
                    
                    # Interval 1: [-inf, forget_min]
                    lower_contrib_1 = weight_diff_pos @ neg_inf - weight_diff_neg @ forget_min
                    upper_contrib_1 = weight_diff_pos @ forget_min - weight_diff_neg @ neg_inf
                    
                    # Interval 2: [forget_max, +inf]
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
                    # Bias affects all outputs equally
                    if lower_bound_reg_1 is not None:
                        lower_bound_reg_1 = lower_bound_reg_1 + bias_diff
                        upper_bound_reg_1 = upper_bound_reg_1 + bias_diff
                        lower_bound_reg_2 = lower_bound_reg_2 + bias_diff
                        upper_bound_reg_2 = upper_bound_reg_2 + bias_diff
            
            if lower_bound_reg_1 is not None:
                # Add loss from both intervals
                total_loss += lower_bound_reg_1.sum().pow(2) + upper_bound_reg_1.sum().pow(2)
                total_loss += lower_bound_reg_2.sum().pow(2) + upper_bound_reg_2.sum().pow(2)
        
        return self.lambda_interval * total_loss
    
    
    def _find_next_linear(self, model: nn.Module, interval_layer_name: str) -> Optional[nn.Module]:
        """
        Find the next Linear layer after the specified IntervalActivation layer.
        """
        # Get ordered list of modules
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
            # Also check if it's an IncrementalClassifier with a Linear classifier
            if hasattr(module, 'classifier') and isinstance(module.classifier, nn.Linear):
                return module.classifier
        
        return None


    def _find_feature_layer(self, model: nn.Module) -> Optional[tuple]:
        """
        Auto-detect the last feature layer before the classifier head.
        
        For ResNet: avgpool layer (features are flattened after this)
        For VGG: last layer before classifier
        
        Returns:
            Tuple of (layer_name, layer_module) or None
        """
        # Strategy: find the last pooling layer or the layer right before fc/classifier
        
        # First, try to find common patterns
        module_list = list(model.named_modules())
        
        # Find the final classifier (fc, classifier, etc.)
        classifier_idx = None
        for i, (name, module) in enumerate(module_list):
            if isinstance(module, nn.Linear) and any(keyword in name for keyword in ['fc', 'classifier', 'head']):
                # Make sure it's a top-level or close to top-level module
                if name.count('.') <= 1:  # Top-level or one level deep
                    classifier_idx = i
                    break
        
        if classifier_idx is None:
            log.warning("Could not find classifier layer")
            return None
        
        # Now find the last meaningful layer before the classifier
        # Look backwards from classifier
        for i in range(classifier_idx - 1, -1, -1):
            name, module = module_list[i]
            
            # Skip certain layer types
            if isinstance(module, (nn.Sequential, nn.ModuleList)):
                continue
            
            # Look for pooling layers (ideal for feature extraction)
            if isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.AdaptiveMaxPool2d, nn.MaxPool2d)):
                log.info(f"Found pooling layer before classifier: {name}")
                return (name, module)
            
            # Look for last conv/block layers
            if isinstance(module, nn.Conv2d) and name.count('.') <= 1:
                log.info(f"Found conv layer before classifier: {name}")
                return (name, module)
            
            # For ResNet: look for layer4, layer3, etc.
            if 'layer4' in name and name.count('.') == 0:
                log.info(f"Found ResNet layer4: {name}")
                return (name, module)
            
            # For VGG: look for features module
            if name == 'features':
                log.info(f"Found VGG features: {name}")
                return (name, module)
        
        log.warning("Could not auto-detect feature layer")
        return None


def intact_iter(
    data_loaders, model, criterion, optimizer, epoch, args, mask=None, 
    interval_protection=None
):
    """
    InTAct unlearning iteration - trains on forget set with interval protection.
    
    The method:
    1. Trains on the FORGET set (to degrade forget class performance)
    2. Adds interval protection loss to preserve retain class knowledge
    """
    forget_loader = data_loaders["forget"]

    losses = utils.AverageMeter()
    protection_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    
    if args.imagenet_arch:
        for i, data in enumerate(forget_loader):
            image, target = get_x_y_from_data_dict(data, device)
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
                )

            # compute output on forget set
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            # Add interval protection loss
            protection_loss = torch.tensor(0.0, device=device)
            if interval_protection is not None:
                protection_loss = interval_protection.compute_protection_loss(model, device)
                loss = loss + protection_loss
            
            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            protection_losses.update(protection_loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Protection Loss {ploss.val:.4f} ({ploss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(forget_loader), end - start, 
                        loss=losses, ploss=protection_losses, top1=top1
                    )
                )
                start = time.time()
    else:
        for i, (image, target) in enumerate(forget_loader):
            if epoch < args.warmup:
                utils.warmup_lr(
                    epoch, i + 1, optimizer, one_epoch_step=len(forget_loader), args=args
                )

            image = image.to(device)
            target = target.to(device)
            
            # compute output on forget set
            output_clean = model(image)
            loss = criterion(output_clean, target)
            
            # Add interval protection loss
            protection_loss = torch.tensor(0.0, device=device)
            if interval_protection is not None:
                protection_loss = interval_protection.compute_protection_loss(model, device)
                loss = loss + protection_loss

            optimizer.zero_grad()
            loss.backward()

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name]

            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            protection_losses.update(protection_loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Protection Loss {ploss.val:.4f} ({ploss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(forget_loader), end - start, 
                        loss=losses, ploss=protection_losses, top1=top1
                    )
                )
                start = time.time()

    print("train_accuracy {top1.avg:.3f}".format(top1=top1))

    return top1.avg


def intact(data_loaders, model, criterion, args, mask=None):
    """
    InTAct: Interval-based Active unlearning with protection
    
    This method:
    1. Sets up interval protection based on retain set activations
    2. Trains on forget set while protecting retain class knowledge
    """
    device = f"cuda:{int(args.gpu)}" if torch.cuda.is_available() else "cpu"
    
    # Initialize interval protection
    lambda_interval = getattr(args, 'lambda_interval', 1.0)
    lower_percentile = getattr(args, 'lower_percentile', 0.05)
    upper_percentile = getattr(args, 'upper_percentile', 0.95)
    infinity_scale = getattr(args, 'infinity_scale', 10.0)
    
    interval_protection = UnlearnIntervalProtection(
        lambda_interval=lambda_interval,
        compute_intervals_from_data=True,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        infinity_scale=infinity_scale
    )
    
    # Setup protection (compute intervals from retain data)
    retain_loader = data_loaders["retain"]
    forget_loader = data_loaders["forget"]
    interval_protection.setup_protection(
        model, retain_loader, forget_loader, device
    )
    
    # Now run the iterative unlearning with protection
    @iterative_unlearn
    def intact_with_protection(data_loaders, model, criterion, optimizer, epoch, args, mask=None):
        return intact_iter(
            data_loaders, model, criterion, optimizer, epoch, args, mask,
            interval_protection=interval_protection
        )
    
    # Execute the unlearning
    intact_with_protection(data_loaders, model, criterion, args, mask)
    
    return model
