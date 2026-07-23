"""
BARRIER/InTAct training on ImageNet-1K classes for ScaPre benchmark comparison.

Trains BARRIER on Diversi50 (50 concepts) or Confuse5 (10 concepts) target classes
using ImageNet-1K training images.  Produces a diffusers UNet checkpoint ready for
evaluation with `scapre/evaluate.py`.

Usage:
    # Diversi50
    python scapre/train.py --benchmark diversi50 \
        --imagenet_root /datasets/ImageNet \
        --base_method rl --targets to_q to_k to_v \
        --lambda_interval 4.0 --epochs 5 --lr 5e-6

    # Confuse5
    python scapre/train.py --benchmark confuse5 \
        --imagenet_root /datasets/ImageNet \
        --base_method rl --targets to_q to_k to_v \
        --lambda_interval 4.0 --epochs 5 --lr 5e-6
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
SD_DIR = SCRIPT_DIR.parent

sys.path.insert(0, str(SD_DIR / "train-scripts"))
sys.path.insert(0, str(SD_DIR.parent))  # For InTAct
sys.path.insert(0, str(SD_DIR))         # For LDM

from imagenet_data import (
    DIVERSI50_CONCEPTS,
    CONFUSE5_CONCEPTS,
    make_forget_remain_dataloaders,
)

from InTAct.intact import UnlearnIntervalProtection
from convertModels import savemodelDiffusers


def load_model_from_config(config_path, ckpt_path, device):
    from ldm.util import instantiate_from_config
    cfg = OmegaConf.load(config_path)
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(cfg.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", type=str, required=True,
                   choices=["diversi50", "confuse5"],
                   help="Which benchmark concepts to train on.")

    # Data
    p.add_argument("--imagenet_root", type=str, required=True,
                   help="Path to ImageNet-1K ILSVRC2012 directory (expects 'train/' subdir)")

    # Base method
    p.add_argument("--base_method", type=str, default="rl", choices=["ga", "rl"])
    p.add_argument("--alpha", type=float, default=0.0)

    # Training
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--image_size", type=int, default=512)

    # InTAct
    p.add_argument("--targets", type=str, nargs="+",
                   default=["to_q", "to_k", "to_v"])
    p.add_argument("--lambda_interval", type=float, default=4.0)
    p.add_argument("--lower_percentile", type=float, default=0.05)
    p.add_argument("--upper_percentile", type=float, default=0.95)
    p.add_argument("--reduced_dim", type=int, default=32)
    p.add_argument("--infinity_scale", type=float, default=18.0)
    p.add_argument("--use_actual_bounds", action="store_true")
    p.add_argument("--normalize_protection", action="store_true", default=True)
    p.add_argument("--bounds_fraction", type=float, default=1.0)

    # Model paths
    p.add_argument("--ckpt_path", type=str,
                   default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt")
    p.add_argument("--config_path", type=str,
                   default="configs/stable-diffusion/v1-intact.yaml")
    p.add_argument("--diffusers_config_path", type=str,
                   default="diffusers_unet_config.json")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--model_save_dir", type=str, default="models")
    p.add_argument("--model_name", type=str, default=None,
                   help="Override auto-generated model name (used for checkpoint save dir)")

    return p.parse_args()


def compact_target_tag(targets):
    import hashlib, re
    pattern = re.compile(r"^output_blocks\.(\d+)\.1\.transformer_blocks\.0\.(.+)$")
    parsed = [pattern.match(t) for t in targets]
    if all(m is not None for m in parsed):
        blocks = sorted(set(m.group(1) for m in parsed))
        aliases = []
        for layer in sorted(set(m.group(2) for m in parsed)):
            if layer == "attn2.to_q": aliases.append("q")
            elif layer == "attn2.to_k": aliases.append("k")
            elif layer == "attn2.to_v": aliases.append("v")
            elif layer == "attn2.to_out.0": aliases.append("out0")
            else: aliases.append(layer.split(".")[-1].replace("to_", ""))
        tag = f"blk{'-'.join(blocks)}_{'-'.join(aliases)}"
        if len(tag) <= 48:
            return tag
    digest = hashlib.sha1("|".join(targets).encode("utf-8")).hexdigest()[:10]
    return f"tgth_{digest}_n{len(targets)}"


def sd_forward_fn(model, batch, device, prompts=None, data_transform_fn=None,
                  betas=None, num_timesteps=1000):
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], torch.Tensor):
        images, labels = batch
    else:
        images = batch
        labels = None

    images = torch.stack([item for item in images])
    images = images.to(device)
    n = images.size(0)

    if prompts is not None and labels is not None:
        txt = [prompts[label] for label in labels]
    elif prompts is not None:
        txt = [prompts[0]] * n
    else:
        txt = [""] * n

    batch_dict = {"jpg": images.permute(0, 2, 3, 1), "txt": txt}
    with torch.no_grad():
        x, c = model.get_input(batch_dict, model.first_stage_key)

    if data_transform_fn is not None:
        x = data_transform_fn(x)

    t = torch.randint(0, num_timesteps, (n // 2 + 1,), device=device)
    t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]

    if betas is not None:
        e = torch.randn_like(x)
        a = (1 - betas).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x_noisy = x * a.sqrt() + e * (1.0 - a).sqrt()
    else:
        x_noisy = x

    model.model.diffusion_model(x_noisy, t.float(), context=c)


def setup_intact_protection(model, forget_dl, remain_dl, descriptions, device,
                            targets, lambda_interval, lower_percentile, upper_percentile,
                            reduced_dim, infinity_scale, use_actual_bounds, normalize_protection):
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
    protection.setup_protection(
        model.model.diffusion_model,
        forget_dl,
        device,
        remain_dataloader=remain_dl,
        forward_fn=lambda m, b, dev: sd_forward_fn(m, b, dev, prompts=descriptions,
                                                     betas=None, num_timesteps=1000),
        betas=None,
        num_timesteps=1000,
    )
    return protection


def compute_rl_loss(model, forget_images, forget_prompts, pseudo_prompts,
                    remain_batch, alpha, criteria, device):
    forget_batch = {"jpg": forget_images.permute(0, 2, 3, 1), "txt": forget_prompts}
    pseudo_batch = {"jpg": forget_images.permute(0, 2, 3, 1), "txt": pseudo_prompts}
    forget_input, forget_emb = model.get_input(forget_batch, model.first_stage_key)
    pseudo_input, pseudo_emb = model.get_input(pseudo_batch, model.first_stage_key)
    t = torch.randint(0, model.num_timesteps, (forget_input.shape[0],), device=device).long()
    noise = torch.randn_like(forget_input, device=device)
    forget_noisy = model.q_sample(x_start=forget_input, t=t, noise=noise)
    pseudo_noisy = model.q_sample(x_start=pseudo_input, t=t, noise=noise)
    forget_out = model.apply_model(forget_noisy, t, forget_emb)
    pseudo_out = model.apply_model(pseudo_noisy, t, pseudo_emb).detach()
    forget_loss = criteria(forget_out, pseudo_out)
    remain_loss = model.shared_step(remain_batch)[0]
    return forget_loss + alpha * remain_loss, forget_loss.item(), remain_loss.item()


def compute_ga_loss(model, forget_batch, remain_batch, alpha, device):
    forget_loss = -model.shared_step(forget_batch)[0]
    remain_loss = model.shared_step(remain_batch)[0]
    return forget_loss + alpha * remain_loss, forget_loss.item(), remain_loss.item()


def train_imagenet_intact(args):
    device = f"cuda:{args.device}"
    benchmark = args.benchmark
    concepts = DIVERSI50_CONCEPTS if benchmark == "diversi50" else CONFUSE5_CONCEPTS

    print(f"Training BARRIER/InTAct on {benchmark}: {len(concepts)} concepts")
    print(f"  Concepts: {', '.join(concepts[:5])}...")

    # --- load base model ---
    model = load_model_from_config(args.config_path, args.ckpt_path, device)
    model = model.to(device)
    if hasattr(model, 'logvar'):
        model.logvar = model.logvar.to(device)

    criteria = torch.nn.MSELoss()

    # --- data ---
    forget_dl, remain_dl, descriptions = make_forget_remain_dataloaders(
        args.imagenet_root, concepts, args.batch_size, args.image_size,
        bounds_fraction=args.bounds_fraction,
    )

    # --- InTAct protection ---
    protection = setup_intact_protection(
        model, forget_dl, remain_dl, descriptions, device,
        targets=args.targets,
        lambda_interval=args.lambda_interval,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
        reduced_dim=args.reduced_dim,
        infinity_scale=args.infinity_scale,
        use_actual_bounds=args.use_actual_bounds,
        normalize_protection=args.normalize_protection,
    )

    diffusion_model = model.model.diffusion_model
    protection.freeze_non_target_params(diffusion_model)
    trainable_params = protection.get_trainable_params(diffusion_model)
    print(f"Training {len(trainable_params)} parameters")

    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    model.train()

    tgt_tag = compact_target_tag(args.targets)
    name = args.model_name or f"compvis-intact-{args.base_method}_imagenet_{benchmark}-targets_{tgt_tag}-lambda_{args.lambda_interval}-epochs_{args.epochs}-lr_{args.lr}"

    # Setup pseudo-prompts for RL
    import random
    ALL_DESCRIPTIONS = [f"an image of a {name}" for name in
                        ["indoor scene", "outdoor scene", "nature photo", "city view",
                         "abstract texture", "blank background", "random object",
                         "everyday item", "food photograph", "landscape"]]

    from tqdm import tqdm
    for epoch in range(args.epochs):
        with tqdm(total=len(forget_dl), desc=f"Epoch {epoch}") as pbar:
            for i in range(len(forget_dl)):
                optimizer.zero_grad()

                try:
                    forget_images, forget_labels = next(iter(forget_dl))
                    remain_images, remain_labels = next(iter(remain_dl))
                except StopIteration:
                    forget_dl_iter = iter(forget_dl)
                    remain_dl_iter = iter(remain_dl)
                    forget_images, forget_labels = next(forget_dl_iter)
                    remain_images, remain_labels = next(remain_dl_iter)

                forget_images = forget_images.to(device)
                remain_images = remain_images.to(device)

                forget_prompts = [descriptions[lbl.item() if isinstance(lbl, torch.Tensor) else lbl] for lbl in forget_labels]
                remain_prompts = [descriptions[lbl.item() if isinstance(lbl, torch.Tensor) else lbl] for lbl in remain_labels]

                remain_batch = {
                    "jpg": remain_images.permute(0, 2, 3, 1),
                    "txt": remain_prompts,
                }

                if args.base_method == "ga":
                    forget_batch = {
                        "jpg": forget_images.permute(0, 2, 3, 1),
                        "txt": forget_prompts,
                    }
                    base_loss, fl, rl = compute_ga_loss(model, forget_batch, remain_batch, args.alpha, device)
                else:
                    pseudo_prompts = [random.choice(ALL_DESCRIPTIONS) for _ in forget_labels]
                    base_loss, fl, rl = compute_rl_loss(
                        model, forget_images, forget_prompts, pseudo_prompts,
                        remain_batch, args.alpha, criteria, device,
                    )

                intact_loss = protection.compute_protection_loss(diffusion_model, device)
                total_loss = base_loss + intact_loss
                total_loss.backward()
                optimizer.step()

                pbar.set_postfix({
                    "base": f"{base_loss.item():.4f}",
                    "intact": f"{intact_loss.item():.4f}",
                    "total": f"{total_loss.item():.4f}",
                })
                pbar.update(1)

    # --- save ---
    model_dir = Path(args.model_save_dir) / name
    model_dir.mkdir(parents=True, exist_ok=True)

    compvis_path = model_dir / f"{name}.pt"
    torch.save(model.state_dict(), compvis_path)
    print(f"Saved CompVis: {compvis_path}")

    savemodelDiffusers(name, args.config_path, args.diffusers_config_path,
                       device=device, save_dir=args.model_save_dir)
    diffusers_path = model_dir / f"diffusers-{name}.pt"
    print(f"Saved Diffusers: {diffusers_path}")

    print(f"\nModel: {name}")
    print(f"Evaluate with:")
    print(f"  python scapre/evaluate.py --benchmark {benchmark} "
          f"--ckpt_name {diffusers_path} --output_dir results/{benchmark}")


if __name__ == "__main__":
    args = parse_args()
    train_imagenet_intact(args)