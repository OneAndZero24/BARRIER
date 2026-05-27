from pathlib import Path
from typing import Optional

import torch


def load_barrier_unet(
    unet_checkpoint: str,
    *,
    base_model: str = "CompVis/stable-diffusion-v1-4",
    device: str = "cpu",
) -> "UNet2DConditionModel":
    """Load a BARRIER-exported UNet checkpoint into a standard diffusers UNet module."""
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    state_dict = torch.load(unet_checkpoint, map_location="cpu")
    unet.load_state_dict(state_dict)
    return unet.to(device)


def load_barrier_pipeline(
    unet_checkpoint: str,
    *,
    base_model: str = "CompVis/stable-diffusion-v1-4",
    device: str = "cuda",
    torch_dtype: Optional[torch.dtype] = None,
) -> "StableDiffusionPipeline":
    """
    Load a normal StableDiffusionPipeline and patch in the BARRIER unlearned UNet.

    This keeps downstream attack code unaware of BARRIER internals.
    """
    from diffusers import StableDiffusionPipeline

    dtype = torch_dtype
    if dtype is None:
        dtype = torch.float16 if device.startswith("cuda") else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        safety_checker=None,
        torch_dtype=dtype,
    )
    state_dict = torch.load(unet_checkpoint, map_location="cpu")
    pipe.unet.load_state_dict(state_dict)
    return pipe.to(device)


def load_barrier_stereo_diffuser(
    unet_checkpoint: str,
    *,
    device: str = "cuda",
    scheduler: str = "DDIM",
) -> "StableDiffuser":
    """
    Build the vendored STEREO StableDiffuser and inject BARRIER UNet weights.

    This adapter is thin by design and does not alter attack logic.
    """
    from SD.stereo.vendor.utils_vendor import StableDiffuser

    diffuser = StableDiffuser(scheduler=scheduler).to(device)
    state_dict = torch.load(Path(unet_checkpoint), map_location="cpu")
    diffuser.unet.load_state_dict(state_dict)
    diffuser.eval()
    return diffuser
