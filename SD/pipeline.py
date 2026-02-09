"""
Unified Pipeline for Stable Diffusion Unlearning Experiments.

Handles both class-wise forgetting (Imagenette) and NSFW concept removal.
Steps: unlearn → generate images → evaluate (FID + classification + NudeNet) → log to wandb.

Usage:
    cd SD
    python pipeline.py --config configs/pipeline_class.yaml
    python pipeline.py --config configs/pipeline_nsfw.yaml

    # wandb sweep:
    wandb sweep configs/sweep_class.yaml
    wandb agent <sweep-id>
"""

import argparse
import logging
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import yaml
from PIL import Image

# Ensure project root and SD dirs are on path
sys.path.insert(0, str(Path(__file__).parent.parent))  # InTAct
sys.path.insert(0, str(Path(__file__).parent))           # SD root
sys.path.insert(0, str(Path(__file__).parent / "train-scripts"))
sys.path.insert(0, str(Path(__file__).parent / "eval-scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# Config helpers
# =============================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_wandb_config(cfg):
    import wandb
    for key, val in dict(wandb.config).items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg


# =============================================================================
# Step 1: Unlearning
# =============================================================================

def run_unlearn_class(cfg, device_str):
    """Run class-wise unlearning (GA / RL / InTAct)."""
    uc = cfg["unlearn"]
    ic = cfg.get("intact", {})
    method = uc["method"]
    class_to_forget = uc["class_to_forget"]

    log.info(f"Unlearning class {class_to_forget} with method={method}")

    if method == "intact":
        from intact_unlearn import intact_unlearn_class
        intact_unlearn_class(
            class_to_forget=int(class_to_forget),
            base_method=ic.get("base_method", "ga"),
            alpha=uc.get("alpha", 0.1),
            batch_size=uc.get("batch_size", 8),
            epochs=uc.get("epochs", 5),
            lr=uc.get("lr", 1e-5),
            config_path=cfg["paths"]["sd_config"],
            ckpt_path=cfg["paths"]["sd_ckpt"],
            diffusers_config_path=cfg["paths"]["diffusers_config"],
            device=device_str,
            targets=ic.get("targets", ["to_q", "to_k", "to_v"]),
            lambda_interval=ic.get("lambda_interval", 1.0),
            lower_percentile=ic.get("lower_percentile", 0.05),
            upper_percentile=ic.get("upper_percentile", 0.95),
            reduced_dim=ic.get("reduced_dim", 32),
            infinity_scale=ic.get("infinity_scale", 20.0),
            use_actual_bounds=ic.get("use_actual_bounds", False),
            normalize_protection=ic.get("normalize_protection", True),
            image_size=uc.get("image_size", 512),
        )
    elif method == "ga":
        from gradient_ascent import gradient_ascent
        gradient_ascent(
            class_to_forget=int(class_to_forget),
            train_method=uc.get("train_method", "xattn"),
            alpha=uc.get("alpha", 0.1),
            batch_size=uc.get("batch_size", 8),
            epochs=uc.get("epochs", 5),
            lr=uc.get("lr", 1e-5),
            config_path=cfg["paths"]["sd_config"],
            ckpt_path=cfg["paths"]["sd_ckpt"],
            mask_path=None,
            diffusers_config_path=cfg["paths"]["diffusers_config"],
            device=device_str,
            image_size=uc.get("image_size", 512),
        )
    elif method == "rl":
        from random_label import certain_label
        certain_label(
            class_to_forget=int(class_to_forget),
            train_method=uc.get("train_method", "xattn"),
            alpha=uc.get("alpha", 0.1),
            batch_size=uc.get("batch_size", 8),
            epochs=uc.get("epochs", 5),
            lr=uc.get("lr", 1e-5),
            config_path=cfg["paths"]["sd_config"],
            ckpt_path=cfg["paths"]["sd_ckpt"],
            mask_path=None,
            diffusers_config_path=cfg["paths"]["diffusers_config"],
            device=device_str,
            image_size=uc.get("image_size", 512),
        )
    else:
        raise ValueError(f"Unknown class unlearn method: {method}")


def run_unlearn_nsfw(cfg, device_str):
    """Run NSFW concept removal."""
    uc = cfg["unlearn"]
    ic = cfg.get("intact", {})
    method = uc["method"]

    log.info(f"NSFW unlearning with method={method}")

    if method == "intact":
        from intact_unlearn import intact_unlearn_nsfw
        intact_unlearn_nsfw(
            alpha=uc.get("alpha", 0.1),
            batch_size=uc.get("batch_size", 8),
            epochs=uc.get("epochs", 3),
            lr=uc.get("lr", 1e-5),
            config_path=cfg["paths"]["sd_config"],
            ckpt_path=cfg["paths"]["sd_ckpt"],
            diffusers_config_path=cfg["paths"]["diffusers_config"],
            device=device_str,
            targets=ic.get("targets", ["to_q", "to_k", "to_v"]),
            lambda_interval=ic.get("lambda_interval", 1.0),
            lower_percentile=ic.get("lower_percentile", 0.05),
            upper_percentile=ic.get("upper_percentile", 0.95),
            reduced_dim=ic.get("reduced_dim", 32),
            infinity_scale=ic.get("infinity_scale", 20.0),
            use_actual_bounds=ic.get("use_actual_bounds", False),
            normalize_protection=ic.get("normalize_protection", True),
            image_size=uc.get("image_size", 512),
        )
    elif method == "nsfw":
        from nsfw_removal import nsfw_removal
        nsfw_removal(
            train_method=uc.get("train_method", "xattn"),
            alpha=uc.get("alpha", 0.1),
            batch_size=uc.get("batch_size", 8),
            epochs=uc.get("epochs", 3),
            lr=uc.get("lr", 1e-5),
            config_path=cfg["paths"]["sd_config"],
            ckpt_path=cfg["paths"]["sd_ckpt"],
            mask_path=None,
            diffusers_config_path=cfg["paths"]["diffusers_config"],
            device=device_str,
            image_size=uc.get("image_size", 512),
        )
    else:
        raise ValueError(f"Unknown NSFW unlearn method: {method}")


def get_model_name(cfg):
    """Derive model name from config (used for file paths)."""
    uc = cfg["unlearn"]
    ic = cfg.get("intact", {})
    setting = cfg["pipeline"]["setting"]

    if uc["method"] == "intact":
        base = ic.get("base_method", "ga")
        targets_str = "_".join(ic.get("targets", ["to_q", "to_k", "to_v"]))
        lam = ic.get("lambda_interval", 1.0)
        lr = uc.get("lr", 1e-5)
        epochs = uc.get("epochs", 5)
        if setting == "sd_nsfw":
            return f"compvis-intact-nsfw-targets_{targets_str}-lambda_{lam}-lr_{lr}"
        else:
            cls = uc.get("class_to_forget", 0)
            return f"compvis-intact-{base}-class_{cls}-targets_{targets_str}-lambda_{lam}-epochs_{epochs}-lr_{lr}"
    elif uc["method"] == "ga":
        tm = uc.get("train_method", "xattn")
        alpha = uc.get("alpha", 0.1)
        epochs = uc.get("epochs", 5)
        lr = uc.get("lr", 1e-5)
        return f"compvis-ga-method_{tm}-alpha_{alpha}-epoch_{epochs}-lr_{lr}"
    elif uc["method"] == "rl":
        cls = uc.get("class_to_forget", 0)
        tm = uc.get("train_method", "xattn")
        alpha = uc.get("alpha", 0.1)
        epochs = uc.get("epochs", 5)
        lr = uc.get("lr", 1e-5)
        return f"compvis-cl-class_{cls}-method_{tm}-alpha_{alpha}-epoch_{epochs}-lr_{lr}"
    elif uc["method"] == "nsfw":
        tm = uc.get("train_method", "xattn")
        lr = uc.get("lr", 1e-5)
        return f"compvis-nsfw-method_{tm}-lr_{lr}"
    else:
        return f"compvis-{uc['method']}"


# =============================================================================
# Step 2: Generate images
# =============================================================================

def generate_images(cfg, model_name, device_str):
    """Generate evaluation images using diffusers pipeline."""
    from importlib import import_module

    eval_cfg = cfg.get("evaluate", {})
    setting = cfg["pipeline"]["setting"]
    output_dir = cfg["paths"].get("output_dir", "./evaluation")

    if setting == "sd_nsfw":
        prompts_path = cfg["paths"].get("nsfw_prompts", "prompts/unsafe-prompts4703.csv")
    else:
        prompts_path = cfg["paths"].get("prompts", "prompts/imagenette.csv")

    num_samples = eval_cfg.get("num_samples_per_prompt", 10)
    save_path = os.path.join(output_dir, "generated")
    os.makedirs(save_path, exist_ok=True)

    log.info(f"Generating images: model={model_name}, prompts={prompts_path}, n={num_samples}")

    # Import generate_images from eval-scripts
    eval_scripts_dir = str(Path(__file__).parent / "eval-scripts")
    sys.path.insert(0, eval_scripts_dir)
    gen_module = import_module("generate-images")

    gen_module.generate_images(
        model_name=model_name,
        prompts_path=prompts_path,
        save_path=save_path,
        device=device_str,
        guidance_scale=eval_cfg.get("guidance_scale", 7.5),
        image_size=cfg["unlearn"].get("image_size", 512),
        ddim_steps=eval_cfg.get("ddim_steps", 100),
        num_samples=num_samples,
    )

    return os.path.join(save_path, model_name)


# =============================================================================
# Step 3: Evaluation
# =============================================================================

def compute_fid_sd(class_to_forget, images_dir, image_size=512):
    """
    Compute FID score for SD class forgetting.
    Calls the exact same logic as eval-scripts/compute-fid.py:
      FID(feature=64) with setup_fid_data from eval-scripts/dataset.py.
    """
    sys.path.insert(0, str(Path(__file__).parent / "eval-scripts"))
    from dataset import setup_fid_data
    from torchmetrics.image.fid import FID

    fid = FID(feature=64)
    real_set, fake_set = setup_fid_data(class_to_forget, images_dir, image_size)
    real_images = torch.stack(real_set).to(torch.uint8).cpu()
    fake_images = torch.stack(fake_set).to(torch.uint8).cpu()

    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return fid.compute().item()


def classify_images(images_dir, prompts_path, device_str):
    """Evaluate generated images with ResNet50 classification."""
    try:
        from torchvision.models import resnet50, ResNet50_Weights

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        model.eval()
        device = torch.device(device_str)
        model = model.to(device)
        preprocess = weights.transforms()
        categories = weights.meta["categories"]

        df = pd.read_csv(prompts_path)
        img_dir = pathlib.Path(images_dir)

        results = {}
        total_correct = 0
        total_images = 0

        for _, row in df.iterrows():
            case = row.case_number
            true_class = str(row.prompt).strip()
            images = sorted(img_dir.glob(f"{case}_*.png"))
            if not images:
                continue

            for img_path in images:
                img = Image.open(img_path).convert("RGB")
                inp = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(inp)
                pred_idx = logits.argmax(dim=1).item()
                total_images += 1

            results[case] = {
                "prompt": true_class,
                "n_images": len(images),
            }

        accuracy = total_correct / max(total_images, 1)
        return {"classify/total_images": total_images, "classify/accuracy": accuracy}
    except Exception as e:
        log.warning(f"Classification evaluation failed: {e}")
        return {}


def nudenet_eval(images_dir, prompts_path, threshold=0.0):
    """Run NudeNet detection on generated images (NSFW setting)."""
    try:
        from nudenet import NudeDetector
        detector = NudeDetector()

        img_dir = pathlib.Path(images_dir)
        all_images = sorted([
            str(f) for ext in ["png", "jpg", "jpeg"]
            for f in img_dir.rglob(f"*.{ext}")
        ])

        if not all_images:
            return {}

        nude_count = 0
        total = len(all_images)
        nudity_classes = []

        for img_path in all_images:
            try:
                detections = detector.detect(img_path)
                img_nude = False
                for det in detections:
                    if det.get("score", 0) > threshold:
                        nudity_classes.append(det.get("class", "unknown"))
                        img_nude = True
                if img_nude:
                    nude_count += 1
            except Exception:
                continue

        return {
            "nudenet/total_images": total,
            "nudenet/nude_count": nude_count,
            "nudenet/nude_ratio": nude_count / max(total, 1),
        }
    except ImportError:
        log.warning("nudenet not installed, skipping NudeNet evaluation")
        return {}
    except Exception as e:
        log.warning(f"NudeNet evaluation failed: {e}")
        return {}


# =============================================================================
# Main
# =============================================================================

def main():
    import wandb

    parser = argparse.ArgumentParser(description="SD Unlearning Pipeline")
    parser.add_argument("--config", type=str, required=True)
    cli = parser.parse_args()

    cfg = load_config(cli.config)

    # --- wandb ---
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        group=cfg["wandb"].get("group"),
        tags=cfg["wandb"].get("tags", []),
        config=cfg,
    )
    cfg = merge_wandb_config(cfg)

    seed = cfg["pipeline"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device_id = cfg["pipeline"].get("device", "0")
    device_str = f"cuda:{device_id}"
    setting = cfg["pipeline"]["setting"]
    eval_cfg = cfg.get("evaluate", {})
    metrics = {}

    # =========================================================================
    # Step 1: Unlearn
    # =========================================================================
    log.info(f"=== Step 1: Unlearning ({setting}) ===")
    if setting == "sd":
        run_unlearn_class(cfg, device_str)
    elif setting == "sd_nsfw":
        run_unlearn_nsfw(cfg, device_str)
    else:
        raise ValueError(f"Unknown setting: {setting}")

    model_name = get_model_name(cfg)
    log.info(f"Model name: {model_name}")

    # =========================================================================
    # Step 2: Generate images
    # =========================================================================
    log.info("=== Step 2: Generating images ===")
    images_dir = generate_images(cfg, model_name, device_str)
    log.info(f"Images saved to {images_dir}")

    # =========================================================================
    # Step 3: Evaluate
    # =========================================================================
    log.info("=== Step 3: Evaluation ===")

    # FID (class forgetting only)
    if setting == "sd" and eval_cfg.get("fid", {}).get("enabled", True):
        class_to_forget = cfg["unlearn"].get("class_to_forget", 0)
        image_size = cfg["unlearn"].get("image_size", 512)
        fid_score = compute_fid_sd(class_to_forget, images_dir, image_size)
        if fid_score is not None:
            metrics["fid"] = fid_score
            log.info(f"  FID = {fid_score:.2f}")

    # Classification accuracy
    if eval_cfg.get("classification", {}).get("enabled", True):
        prompts_path = cfg["paths"].get("prompts", "prompts/imagenette.csv")
        clf_metrics = classify_images(images_dir, prompts_path, device_str)
        metrics.update(clf_metrics)
        for k, v in clf_metrics.items():
            log.info(f"  {k} = {v}")

    # NudeNet (NSFW only)
    if setting == "sd_nsfw" and eval_cfg.get("nudenet", {}).get("enabled", True):
        prompts_path = cfg["paths"].get("nsfw_prompts", "prompts/unsafe-prompts4703.csv")
        nn_metrics = nudenet_eval(
            images_dir, prompts_path,
            threshold=eval_cfg.get("nudenet", {}).get("threshold", 0.0),
        )
        metrics.update(nn_metrics)
        for k, v in nn_metrics.items():
            log.info(f"  {k} = {v}")

    # =========================================================================
    # Step 4: Log to wandb
    # =========================================================================
    log.info("=== Step 4: Logging ===")
    wandb.log(metrics)
    wandb.summary.update(metrics)

    # Sample images
    img_dir = pathlib.Path(images_dir)
    sample_imgs = sorted(img_dir.glob("*.png"))[:20]
    if sample_imgs:
        wandb.log({
            "samples/generated": [wandb.Image(str(p)) for p in sample_imgs]
        })

    # Model artifact
    model_dir = f"models/{model_name}"
    diffusers_pt = os.path.join(model_dir, f"{model_name.replace('compvis', 'diffusers')}.pt")
    compvis_pt = os.path.join(model_dir, f"{model_name}.pt")
    ckpt_to_log = diffusers_pt if os.path.exists(diffusers_pt) else compvis_pt

    if os.path.exists(ckpt_to_log):
        art = wandb.Artifact(
            name=f"sd-{setting}-{wandb.run.id}",
            type="model",
            metadata=metrics,
        )
        art.add_file(ckpt_to_log)
        wandb.log_artifact(art)
        log.info(f"Model logged as artifact: {ckpt_to_log}")

    wandb.finish()
    log.info(f"Pipeline complete. Metrics: {metrics}")


if __name__ == "__main__":
    main()
