import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def _unwrap_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        return payload["state_dict"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Checkpoint payload is not a valid state_dict-like object")


def _looks_like_diffusers_unet(state_dict: Dict[str, torch.Tensor]) -> bool:
    keys = list(state_dict.keys())
    return "conv_in.weight" in state_dict or any(k.startswith("down_blocks.") for k in keys)


def _looks_like_compvis_unet(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(k.startswith("model.diffusion_model.") for k in state_dict.keys())


def export_barrier_checkpoint(
    source_checkpoint: str,
    output_unet_checkpoint: str,
    *,
    compvis_config_path: Optional[str] = None,
    diffusers_config_path: Optional[str] = None,
    device: str = "cpu",
) -> str:
    """
    Export a BARRIER checkpoint into a diffusers UNet state_dict file.

    Fast path: if source is already a diffusers UNet state_dict (.pt), copy/save it.
    Conversion path: if source is a CompVis-style full checkpoint, convert UNet only.
    """
    source_path = Path(source_checkpoint)
    output_path = Path(output_unet_checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(source_path, map_location="cpu")
    state_dict = _unwrap_state_dict(payload)

    if _looks_like_diffusers_unet(state_dict):
        if source_path.resolve() != output_path.resolve():
            shutil.copy2(source_path, output_path)
        return str(output_path)

    if _looks_like_compvis_unet(state_dict):
        if compvis_config_path is None or diffusers_config_path is None:
            raise ValueError(
                "compvis_config_path and diffusers_config_path are required for CompVis checkpoint conversion"
            )

        # Lazy import so users with pre-exported diffusers checkpoints do not need conversion deps.
        import sys

        train_scripts = Path(__file__).resolve().parent.parent / "SD" / "train-scripts"
        sys.path.insert(0, str(train_scripts))
        from convertModels import create_unet_diffusers_config, convert_ldm_unet_checkpoint  # type: ignore
        from omegaconf import OmegaConf

        original_config = OmegaConf.load(compvis_config_path)
        unet_config = create_unet_diffusers_config(original_config, image_size=512)
        converted = convert_ldm_unet_checkpoint(state_dict, unet_config, path=str(source_path), extract_ema=False)
        torch.save(converted, output_path)
        return str(output_path)

    raise ValueError(
        "Unsupported checkpoint format. Expected diffusers UNet state_dict or CompVis-style checkpoint."
    )
