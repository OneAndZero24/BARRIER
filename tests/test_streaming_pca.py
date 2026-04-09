import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from InTAct.intact import UnlearnIntervalProtection, classification_forward_fn


torch.manual_seed(0)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 3, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def collect_reference_activations(model, dataloader):
    acts = []

    def hook(module, inp, out):
        if len(inp) > 0 and inp[0] is not None:
            x = inp[0].reshape(-1, module.weight.shape[1]).detach().cpu().to(torch.float32)
            acts.append(x)

    handle = model.fc1.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            classification_forward_fn(model, batch, torch.device("cpu"))
    handle.remove()
    model.train()
    return torch.cat(acts, dim=0)


def reference_pca(activations, reduced_dim):
    mu = activations.mean(dim=0)
    centered = activations - mu
    _, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    k = min(reduced_dim, Vh.size(0))
    U_forget = Vh[:k]
    U_residual = Vh[k:]
    S_residual = S[k:]
    projected = centered @ U_forget.T
    z_min = torch.quantile(projected, 0.05, dim=0)
    z_max = torch.quantile(projected, 0.95, dim=0)
    return {
        "mu": mu,
        "U_forget": U_forget,
        "U_residual": U_residual,
        "S_residual": S_residual,
        "z_min": z_min,
        "z_max": z_max,
    }


def align_rows(reference, candidate):
    aligned = candidate.clone()
    signs = []
    for row_idx in range(reference.shape[0]):
        dot = torch.dot(reference[row_idx].flatten(), candidate[row_idx].flatten())
        sign = -1.0 if dot < 0 else 1.0
        signs.append(sign)
        if sign < 0:
            aligned[row_idx] = -aligned[row_idx]
    return aligned, torch.tensor(signs, dtype=reference.dtype)


def main():
    model = TinyClassifier()
    x = torch.randn(64, 4)
    y = torch.zeros(64, dtype=torch.long)
    dataloader = DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)

    reference_acts = collect_reference_activations(model, dataloader)
    reference = reference_pca(reference_acts, reduced_dim=3)

    protector = UnlearnIntervalProtection(targets=["fc1"], reduced_dim=3)
    protector.setup_protection(
        model,
        forget_dataloader=dataloader,
        device=torch.device("cpu"),
        forward_fn=classification_forward_fn,
    )

    assert len(protector.pca_info) == 1, "Expected one protected layer"
    actual = protector.pca_info[0]

    aligned_u_forget, forget_signs = align_rows(reference["U_forget"], actual["U_forget"])
    aligned_u_residual, _ = align_rows(reference["U_residual"], actual["U_residual"])
    aligned_projected = (reference_acts - actual["mu"]) @ aligned_u_forget.T
    aligned_z_min = torch.quantile(aligned_projected, 0.05, dim=0)
    aligned_z_max = torch.quantile(aligned_projected, 0.95, dim=0)

    actual_z_min_aligned = torch.where(forget_signs > 0, actual["z_min"], -actual["z_max"])
    actual_z_max_aligned = torch.where(forget_signs > 0, actual["z_max"], -actual["z_min"])

    assert torch.allclose(actual["mu"], reference["mu"], atol=1e-6, rtol=1e-5)
    assert torch.allclose(aligned_u_forget, reference["U_forget"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(aligned_u_residual, reference["U_residual"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(actual["S_residual"], reference["S_residual"], atol=1e-5, rtol=1e-4)
    assert torch.allclose(actual_z_min_aligned, aligned_z_min, atol=1e-5, rtol=1e-4)
    assert torch.allclose(actual_z_max_aligned, aligned_z_max, atol=1e-5, rtol=1e-4)

    print("streaming_pca_ok")


if __name__ == "__main__":
    main()