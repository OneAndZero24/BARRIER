"""
SD-specific helpers shared by the SD and Flux experiment code.

These were previously copy-pasted into every SD training/analysis script
(``train-scripts/intact_unlearn.py``, ``scapre/train.py``,
``scripts/activation_space_analysis.py``, ``scripts/svd_separation_scatter.py``).
They now live here so every experiment uses one implementation.

Contents
--------
- ``compact_target_tag``       short stable tag for a target-layer selection
- ``sd_forward_fn``            one SD (LDM) forward pass for InTAct activation
- ``setup_intact_protection``  build ``UnlearnIntervalProtection`` for an SD model
"""

import hashlib
import re

import torch

from barrier.intact import UnlearnIntervalProtection


def compact_target_tag(targets):
    """Build a short, stable tag for selected target layers."""
    if not targets:
        return "tgt_default"

    pattern = re.compile(r"^output_blocks\.(\d+)\.1\.transformer_blocks\.0\.(.+)$")
    parsed = [pattern.match(target) for target in targets]
    if all(match is not None for match in parsed):
        blocks = []
        layers = []
        for match in parsed:
            block_id = match.group(1)
            layer_name = match.group(2)
            if block_id not in blocks:
                blocks.append(block_id)
            if layer_name not in layers:
                layers.append(layer_name)

        aliases = []
        for layer in layers:
            if layer == "attn2.to_q":
                aliases.append("q")
            elif layer == "attn2.to_k":
                aliases.append("k")
            elif layer == "attn2.to_v":
                aliases.append("v")
            elif layer == "attn2.to_out.0":
                aliases.append("out0")
            else:
                aliases.append(layer.split(".")[-1].replace("to_", ""))

        tag = f"blk{'-'.join(blocks)}_{'-'.join(aliases)}"
        if len(tag) <= 48:
            return tag

    digest = hashlib.sha1("|".join(targets).encode("utf-8")).hexdigest()[:10]
    return f"tgth_{digest}_n{len(targets)}"


def sd_forward_fn(model, batch, device, prompts=None, data_transform_fn=None,
                  betas=None, num_timesteps=1000, mirror_t=True, **kwargs):
    """
    SD-specific forward function for InTAct activation collection.
    Takes raw image batches and handles the full encoding/forward pipeline.

    Args:
        model: Full SD model (LatentDiffusion) — needed for get_input()
        batch: Either tuple (images, labels) or just images (for NSFW datasets)
        device: CUDA device
        prompts: List of text prompts (indexed by labels if available)
        data_transform_fn: Optional transform applied to the latent ``x``
        betas: Noise schedule betas tensor (None = skip noise addition)
        num_timesteps: Number of diffusion timesteps
        mirror_t: When True, draw ``n//2+1`` steps and mirror them (canonical
            behaviour); when False, draw ``n`` steps directly on ``device``
            (behaviour of the analysis-script copies).
    """
    # Handle both (images, labels) and images-only batches
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], torch.Tensor):
        # batch is (images, labels) tuple from DataLoader
        images, labels = batch
    else:
        # batch is just images (NSFW datasets)
        images = batch
        labels = None

    images = torch.stack([item for item in images])
    images = images.to(device)
    n = images.size(0)

    # Get text prompts
    if prompts is not None and labels is not None:
        txt = [prompts[label] for label in labels]
    elif prompts is not None:
        # No labels (e.g. NSFW datasets) — repeat first prompt for all images
        txt = [prompts[0]] * n
    else:
        txt = [""] * n

    # Create batch dict for SD
    batch_dict = {
        "jpg": images.permute(0, 2, 3, 1),
        "txt": txt
    }

    # Encode to latent and get conditioning embeddings
    with torch.no_grad():
        x, c = model.get_input(batch_dict, model.first_stage_key)

    if data_transform_fn is not None:
        x = data_transform_fn(x)

    # Create timesteps
    if mirror_t:
        t = torch.randint(low=0, high=num_timesteps, size=(n // 2 + 1,)).to(device)
        t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
    else:
        t = torch.randint(0, num_timesteps, (n,), device=device).long()

    # Add noise if betas provided
    if betas is not None:
        e = torch.randn_like(x)
        a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x_noisy = x * a.sqrt() + e * (1.0 - a).sqrt()
    else:
        x_noisy = x

    # Forward through UNet (triggers hooks for activation collection)
    model.model.diffusion_model(x_noisy, t.float(), context=c)


def sd_forward_fn_model_schedule(model, batch, device, prompts=None, **kwargs):
    """
    Variant of :func:`sd_forward_fn` that derives the noise schedule from the
    model itself and draws timesteps directly on ``device`` (non-mirrored).

    Reproduces the behaviour of the analysis-script copies
    (``scripts/activation_space_analysis.py``, ``scripts/svd_separation_scatter.py``):
    activations are always collected with diffusion noise applied.  Schedule
    kwargs forwarded by ``setup_protection`` are ignored in favour of the
    model's own schedule.
    """
    kwargs.pop("betas", None)
    kwargs.pop("num_timesteps", None)
    betas = model.betas.to(device) if hasattr(model, "betas") else None
    return sd_forward_fn(
        model, batch, device, prompts=prompts,
        betas=betas,
        num_timesteps=getattr(model, "num_timesteps", 1000),
        mirror_t=False,
        **kwargs,
    )


def setup_intact_protection(
    model,
    forget_dl,
    remain_dl,
    descriptions,
    device,
    targets,
    lambda_interval=1.0,
    lower_percentile=0.05,
    upper_percentile=0.95,
    reduced_dim=32,
    infinity_scale=20.0,
    use_actual_bounds=False,
    normalize_protection=True,
    betas=None,
    num_timesteps=None,
):
    """
    Setup InTAct protection for an SD model (LatentDiffusion).

    Args:
        model: SD model (LatentDiffusion)
        forget_dl: Forget dataloader (raw, yields images/labels)
        remain_dl: Remain dataloader (optional)
        descriptions: List of class descriptions (prompts indexed by label)
        device: CUDA device
        targets: List of target layer patterns (e.g., ["to_q", "to_k", "to_v"])
        betas: Noise schedule pass-through. ``None`` skips noise during
            activation collection (scapre behaviour); pass
            ``model.betas.to(device)`` to add noise (intact_unlearn behaviour).
        num_timesteps: Diffusion timestep count pass-through.

    Returns:
        protection: UnlearnIntervalProtection instance
    """
    protection = UnlearnIntervalProtection(
        targets=targets,
        lambda_interval=lambda_interval,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        reduced_dim=reduced_dim,
        infinity_scale=infinity_scale,
        use_actual_bounds=use_actual_bounds,
        normalize_protection=normalize_protection,
    )

    # Create forward function with prompts pre-bound.
    # The full model is captured for encoding; forward_fn receives the
    # diffusion_model (already extracted by setup_protection).
    def forward_fn(diffusion_model, batch, dev, **kwargs):
        return sd_forward_fn(model, batch, dev, prompts=descriptions, **kwargs)

    setup_kwargs = {}
    if betas is not None:
        setup_kwargs["betas"] = betas
    if num_timesteps is not None:
        setup_kwargs["num_timesteps"] = num_timesteps

    protection.setup_protection(
        model.model.diffusion_model,
        forget_dl,
        device,
        remain_dataloader=remain_dl,
        forward_fn=forward_fn,
        **setup_kwargs,
    )

    return protection