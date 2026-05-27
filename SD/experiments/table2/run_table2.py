import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SD.barrier_adapter import load_barrier_pipeline
from SD.checkpointing import export_barrier_checkpoint


METHODS = {
    "esd": load_barrier_pipeline,
    "uce": load_barrier_pipeline,
    "concept-ablation": load_barrier_pipeline,
    "barrier": load_barrier_pipeline,
}


def _count_pngs(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.png"))


def run(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    attacks_dir = output_dir / "attacks" / f"{args.method}_{args.concept}"
    metrics_dir = output_dir / "metrics"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    attacks_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    exported_unet = checkpoints_dir / f"{args.method}_{args.concept}_unet.pt"
    if exported_unet.exists() and not args.force_export:
        print(f"[cache] Reusing exported checkpoint: {exported_unet}")
    else:
        export_barrier_checkpoint(
            source_checkpoint=args.checkpoint,
            output_unet_checkpoint=str(exported_unet),
            compvis_config_path=args.compvis_config_path,
            diffusers_config_path=args.diffusers_config_path,
            device=args.device,
        )

    # Keep attack implementation unchanged: it expects output_dir + relative filename.
    attack_ckpt_name = "final_reo_unet.pt"
    attack_ckpt_path = attacks_dir / attack_ckpt_name
    if not attack_ckpt_path.exists() or args.force_attack:
        shutil.copy2(exported_unet, attack_ckpt_path)

    attack_images_dir = attacks_dir / "eval_ci_iteration"
    attack_encoder = attacks_dir / "eval_ci_attack_on_stereo_text_encoder.pt"
    if attack_images_dir.exists() and attack_encoder.exists() and not args.force_attack:
        print(f"[cache] Reusing attack outputs under: {attacks_dir}")
    else:
        from SD.stereo.attacks.stereo_vendor import attack_stereo
        from SD.stereo.vendor.utils_vendor import StableDiffuser

        stereo_args = SimpleNamespace(
            output_dir=str(attacks_dir),
            unet_ckpt_to_attack=attack_ckpt_name,
            attack_eval_images=args.attack_eval_images,
            initializer_token=args.initializer_token,
            ci_lr=args.ci_lr,
            ti_max_train_steps=args.ti_max_train_steps,
            learnable_property=args.learnable_property,
            device=args.device,
            generic_prompt=args.generic_prompt,
            center_crop=args.center_crop,
        )
        diffuser = StableDiffuser(scheduler="DDIM").to(args.device)
        attack_stereo(stereo_args, diffuser)

    # Optional sanity check that checkpoint is pipeline-loadable.
    if args.verify_pipeline_load:
        pipe = METHODS[args.method](
            unet_checkpoint=str(exported_unet),
            base_model=args.base_model,
            device=args.device,
        )
        del pipe

    external_attack_logs = []
    if args.external_attacks:
        for attack_name in [a.strip() for a in args.external_attacks.split(",") if a.strip()]:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "SD" / "stereo" / "scripts" / "run_external_attacks.py"),
                "--attack",
                attack_name,
                "--attack_idx",
                str(args.attack_idx),
            ]
            if args.ud_config:
                cmd.extend(["--ud_config", args.ud_config])
            if args.rab_command:
                cmd.extend(["--rab_command", args.rab_command])
            if args.cce_variant:
                cmd.extend(["--cce_variant", args.cce_variant])
            if args.cce_command:
                cmd.extend(["--cce_command", args.cce_command])

            print(f"[external-attack] running {attack_name}")
            subprocess.run(cmd, check=True)
            external_attack_logs.append({"attack": attack_name, "status": "ok"})

    result = {
        "concept": args.concept,
        "method": args.method,
        "input_checkpoint": str(Path(args.checkpoint).resolve()),
        "exported_unet": str(exported_unet.resolve()),
        "attack_output_dir": str(attacks_dir.resolve()),
        "attack_images": _count_pngs(attack_images_dir),
        "attack_text_encoder_ckpt": str(attack_encoder.resolve()),
        "external_attacks": external_attack_logs,
    }

    metrics_json = metrics_dir / f"{args.method}_{args.concept}.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    table2_row_csv = metrics_dir / "table2_rows.csv"
    write_header = not table2_row_csv.exists()
    with open(table2_row_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)

    print(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BARRIER -> checkpoint export -> STEREO attack pipeline")
    parser.add_argument("--concept", default="nudity")
    parser.add_argument("--method", default="barrier", choices=list(METHODS.keys()))
    parser.add_argument("--checkpoint", required=True, help="Path to unlearned checkpoint (.pt)")
    parser.add_argument("--attack_eval_images", required=True, help="Gallery images used by STEREO attack")
    parser.add_argument("--output_dir", default="results/barrier_nudity")

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base_model", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--compvis_config_path", default=None)
    parser.add_argument("--diffusers_config_path", default=None)

    parser.add_argument("--initializer_token", default="person")
    parser.add_argument("--ci_lr", type=float, default=5e-3)
    parser.add_argument("--ti_max_train_steps", type=int, default=300)
    parser.add_argument("--learnable_property", default="object")
    parser.add_argument("--generic_prompt", default="a photo of a")
    parser.add_argument("--center_crop", action="store_true")

    parser.add_argument("--force_export", action="store_true")
    parser.add_argument("--force_attack", action="store_true")
    parser.add_argument("--verify_pipeline_load", action="store_true")

    parser.add_argument("--external_attacks", default="", help="Comma-separated: ud,rab,cce")
    parser.add_argument("--attack_idx", type=int, default=0)
    parser.add_argument("--ud_config", default=None)
    parser.add_argument("--rab_command", default=None)
    parser.add_argument("--cce_variant", default=None)
    parser.add_argument("--cce_command", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
