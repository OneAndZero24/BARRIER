import argparse
from pathlib import Path
from types import SimpleNamespace
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SD.barrier_adapter import load_barrier_pipeline
from SD.checkpointing import export_barrier_checkpoint
from SD.stereo.attacks.stereo_vendor import attack_stereo
from SD.stereo.vendor.utils_vendor import StableDiffuser


def main(args: argparse.Namespace) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    exported = out / "smoke_unet.pt"
    export_barrier_checkpoint(
        source_checkpoint=args.checkpoint,
        output_unet_checkpoint=str(exported),
        compvis_config_path=args.compvis_config_path,
        diffusers_config_path=args.diffusers_config_path,
        device=args.device,
    )

    pipe = load_barrier_pipeline(
        unet_checkpoint=str(exported),
        base_model=args.base_model,
        device=args.device,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
    )

    # 1-step generation smoke check.
    img = pipe(prompt=["a person on a beach"], num_inference_steps=1, guidance_scale=7.5).images[0]
    img.save(out / "smoke_generation.png")
    del pipe

    attack_dir = out / "attack"
    attack_dir.mkdir(parents=True, exist_ok=True)
    attack_ckpt = attack_dir / "final_reo_unet.pt"
    attack_ckpt.write_bytes(exported.read_bytes())

    attack_args = SimpleNamespace(
        output_dir=str(attack_dir),
        unet_ckpt_to_attack="final_reo_unet.pt",
        attack_eval_images=args.attack_eval_images,
        initializer_token=args.initializer_token,
        ci_lr=args.ci_lr,
        ti_max_train_steps=1,
        learnable_property="object",
        device=args.device,
        generic_prompt="a photo of a",
        center_crop=False,
    )
    diffuser = StableDiffuser(scheduler="DDIM").to(args.device)
    attack_stereo(attack_args, diffuser)

    generated = list((attack_dir / "eval_ci_iteration").glob("*.png"))
    if not generated:
        raise RuntimeError("Smoke test failed: attack produced no images")

    print("Smoke test passed")
    print(f"Generated images: {len(generated)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight smoke test for BARRIER->STEREO attack pipeline")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--attack_eval_images", required=True)
    parser.add_argument("--output_dir", default="results/smoke")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--initializer_token", default="person")
    parser.add_argument("--ci_lr", type=float, default=5e-3)
    parser.add_argument("--compvis_config_path", default=None)
    parser.add_argument("--diffusers_config_path", default=None)
    main(parser.parse_args())
