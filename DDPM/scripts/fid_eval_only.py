"""
Minimal FID-only evaluation from a pre-existing DDPM checkpoint.
Skips unlearning — just generates FID samples + computes TF Inception-V3 FID.

Usage:
    cd DDPM
    python scripts/fid_eval_only.py \
        --ckpt_dir /shared/.../2026_08_11_015300 \
        --label_to_forget 0 \
        --run_label "IA NO SVD"
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import setup_cache  # noqa: E402  — must precede torch imports

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from functions import dict2namespace

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# =============================================================================
# Helpers (mirror pipeline.py so we can run standalone)
# =============================================================================

def build_runner_config_from_model(model_config_path):
    with open(model_config_path, "r") as f:
        return dict2namespace(yaml.safe_load(f))


def resolve_ckpt_path(ckpt_dir):
    if os.path.isdir(os.path.join(ckpt_dir, "ckpts")):
        return os.path.join(ckpt_dir, "ckpts", "ckpt.pth")
    if os.path.isfile(os.path.join(ckpt_dir, "ckpt.pth")):
        return os.path.join(ckpt_dir, "ckpt.pth")
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def compute_fid_reference(ref_dir, sample_dir):
    import tensorflow.compat.v1 as tf
    from evaluator import Evaluator, read_images_folder

    ref_arr = read_images_folder(ref_dir)
    sample_arr = read_images_folder(sample_dir)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    evaluator = Evaluator(sess)
    evaluator.warmup()

    ref_acts = evaluator.read_activations(ref_arr)
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_acts)

    sample_acts = evaluator.read_activations(sample_arr)
    sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_acts)

    inception_score = evaluator.compute_inception_score(sample_acts[0])
    fid = sample_stats.frechet_distance(ref_stats)
    sfid = sample_stats_spatial.frechet_distance(ref_stats_spatial)
    prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])

    sess.close()

    return {
        "FID": float(fid),
        "InceptionScore": float(inception_score),
        "sFID": float(sfid),
        "Precision": float(prec),
        "Recall": float(recall),
    }


def fid_sample_dir(ckpt_dir, cond_scale, label_to_forget):
    return os.path.join(
        ckpt_dir,
        f"fid_samples_guidance_{cond_scale}_excluded_class_{label_to_forget}",
    )


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="FID-only eval from checkpoint")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to pipeline output dir containing ckpts/")
    parser.add_argument("--model_config", type=str,
                        default="configs/cifar10_intact.yml")
    parser.add_argument("--label_to_forget", type=int, default=0)
    parser.add_argument("--n_samples_per_class", type=int, default=5000)
    parser.add_argument("--cond_scale", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ref_dataset_dir", type=str,
                        default="/shared/results/common/miksa/intact/DDPM/results/cifar10_without_label_0")
    parser.add_argument("--run_label", type=str, default="fid_eval",
                        help="Label for output and wandb")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip_generation", action="store_true",
                        help="Skip sample generation (use existing samples)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="intact-ddpm")
    parser.add_argument("--wandb_entity", type=str, default="oneandzero24")
    parser.add_argument("--wandb_group", type=str, default="fid-eval-only")
    args_cli = parser.parse_args()

    # ---- Resolve checkpoint ----
    ckpt_root = args_cli.ckpt_dir
    ckpt_path = resolve_ckpt_path(ckpt_root)
    log.info("Checkpoint: %s", ckpt_path)
    log.info("Label:      %s", args_cli.run_label)
    log.info("Forgotten:  class %d", args_cli.label_to_forget)

    # ---- Wandb ----
    if not args_cli.no_wandb:
        import wandb
        wandb.init(
            project=args_cli.wandb_project,
            entity=args_cli.wandb_entity,
            group=args_cli.wandb_group,
            name=args_cli.run_label,
            tags=["fid-only", args_cli.run_label],
            config=vars(args_cli),
            reinit=True,
        )
    else:
        wandb = None

    # ---- Seeds ----
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    # ---- Build runner config ----
    runner_config = build_runner_config_from_model(args_cli.model_config)
    runner_config.sampling.batch_size = args_cli.batch_size

    # ---- Build runner args ----
    runner_args = argparse.Namespace()
    runner_args.ckpt_folder = ckpt_root
    runner_args.mode = "sample_fid"
    runner_args.classes_to_generate = f"x{args_cli.label_to_forget}"
    runner_args.n_samples_per_class = args_cli.n_samples_per_class
    runner_args.cond_scale = args_cli.cond_scale
    runner_args.seed = args_cli.seed
    runner_args.sample_type = "generalized"
    runner_args.skip_type = "uniform"
    runner_args.timesteps = 1000
    runner_args.eta = 1.0
    runner_args.sequence = False
    runner_args.label_to_forget = args_cli.label_to_forget

    from runners.diffusion import Diffusion

    # ---- Generate FID samples (if not skipping) ----
    sample_dir = fid_sample_dir(
        ckpt_root, args_cli.cond_scale, args_cli.label_to_forget
    )

    if args_cli.skip_generation and os.path.isdir(sample_dir):
        log.info("Skipping generation — samples already at %s", sample_dir)
    else:
        log.info("Generating %d FID samples per class (excluding class %d) ...",
                 args_cli.n_samples_per_class, args_cli.label_to_forget)
        fid_runner = Diffusion(runner_args, runner_config)
        fid_runner.sample()
        log.info("Samples saved to %s", sample_dir)

    # ---- Compute FID ----
    ref_dir = args_cli.ref_dataset_dir
    if not os.path.isdir(ref_dir):
        log.warning("Reference dataset not found: %s", ref_dir)
        # Try generating it on the fly
        import torchvision
        ds = torchvision.datasets.CIFAR10(
            root=os.environ.get("DATAPATH", "../data"),
            train=True, download=True,
            transform=torchvision.transforms.ToTensor(),
        )
        targets = ds.targets
        indices = [i for i, t in enumerate(targets) if t != args_cli.label_to_forget]
        os.makedirs(ref_dir, exist_ok=True)
        img_id = 0
        for idx in indices[:500 * 9]:  # 500 per class × 9 classes
            torchvision.utils.save_image(
                ds[idx][0], os.path.join(ref_dir, f"{img_id}.png"), normalize=True
            )
            img_id += 1
        log.info("Generated reference dataset: %d images at %s", img_id, ref_dir)

    log.info("Computing FID ...")
    metrics = compute_fid_reference(ref_dir, sample_dir)

    # ---- Output ----
    print()
    print("=" * 60)
    print(f"  FID RESULTS — {args_cli.run_label}")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<18s} : {v:.4f}")
    print("=" * 60)
    print()

    if wandb:
        wandb.log(metrics)
        wandb.summary.update(metrics)
        wandb.finish()

    log.info("Done. FID = %.4f  (label: %s)", metrics["FID"], args_cli.run_label)


if __name__ == "__main__":
    main()
