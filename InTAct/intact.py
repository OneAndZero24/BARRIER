import logging
import torch
import torch.nn as nn
from typing import Optional, List, Dict

log = logging.getLogger(__name__)

class UnlearnIntervalProtection:
    """
    Theoretical Max InTAct: Negative Space IA + Mean Reparametrization.
    
    Improvements:
    1. Mean Trick: Corrects the bias shift by coupling Delta W and Delta b.
    2. Scaled Residual: Uses singular values to weight the residual protection,
       avoiding the over-constraining nature of a raw Frobenius norm.
    """

    def __init__(
        self,
        lambda_interval: float = 10.0,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        reduced_dim: int = 32,
        infinity_scale: float = 20.0
    ):
        self.lambda_interval = lambda_interval
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.reduced_dim = reduced_dim
        self.infinity_scale = infinity_scale

        self.pca_info: List[Dict] = []
        self.params_snapshot = {}

    def setup_protection(self, model: nn.Module, forget_dataloader, device):
        log.info("Setting up InTAct with Mean Reparametrization...")
        
        feature_layer = self._find_feature_layer(model)
        if feature_layer is None: return
        layer_name, layer_module = feature_layer

        # 1. Collect Activations
        acts = self._collect_activations(model, layer_module, forget_dataloader, device)
        
        # 2. Centered SVD
        mu = acts.mean(dim=0)
        Xc = acts - mu
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        
        k = min(self.reduced_dim, Vh.size(0))
        U_forget = Vh[:k] 
        U_residual = Vh[k:]
        S_residual = S[k:]

        # 3. Define the Forget Box in centered PCA space
        Z = Xc @ U_forget.T
        z_min = torch.quantile(Z, self.lower_percentile, dim=0)
        z_max = torch.quantile(Z, self.upper_percentile, dim=0)

        self.pca_info = [{
            "layer_name": layer_name,
            "mu": mu.detach(),
            "U_forget": U_forget.detach(),
            "U_residual": U_residual.detach(),
            "S_residual": S_residual.detach(),
            "z_min": z_min.detach(),
            "z_max": z_max.detach()
        }]

        self.params_snapshot = {
            n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
        }

    def compute_protection_loss(self, model: nn.Module, device) -> torch.Tensor:
        if not self.pca_info: return torch.tensor(0.0, device=device)
        total_loss = torch.tensor(0.0, device=device)

        for info in self.pca_info:
            next_linear = self._find_next_linear(model, info["layer_name"])
            if not next_linear: continue

            mu = info["mu"].to(device)
            Uf = info["U_forget"].to(device)
            Ur = info["U_residual"].to(device)
            Sr = info["S_residual"].to(device)
            z_min, z_max = info["z_min"].to(device), info["z_max"].to(device)
            
            inf_low = z_min - self.infinity_scale
            inf_high = z_max + self.infinity_scale

            w_name = self._resolve_param_name(model, next_linear.weight)
            b_name = self._resolve_param_name(model, next_linear.bias)
            
            delta_W = next_linear.weight - self.params_snapshot[w_name]
            delta_b = next_linear.bias - self.params_snapshot[b_name] if next_linear.bias is not None else 0

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

        return self.lambda_interval * total_loss

    def _collect_activations(self, model, layer, dataloader, device):
        model.eval()
        buf = []
        def hook(_, __, out): buf.append(out.detach().view(out.size(0), -1))
        h = layer.register_forward_hook(hook)
        with torch.no_grad():
            for batch in dataloader: model(batch[0].to(device))
        h.remove()
        model.train()
        return torch.cat(buf, dim=0)

    def _resolve_param_name(self, model, param):
        for name, p in model.named_parameters():
            if p is param: return name
        return ""

    def _find_next_linear(self, model, layer_name):
        found = False
        for name, m in model.named_modules():
            if name == layer_name: found = True
            if found and isinstance(m, nn.Linear): return m
        return None

    def _find_feature_layer(self, model):
        for name, m in reversed(list(model.named_modules())):
            if isinstance(m, (nn.AdaptiveAvgPool2d, nn.AvgPool2d)): return name, m
        return None

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

        # InTAct Protection (Negative Space Shield)
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