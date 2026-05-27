import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SD.experiments.table2.run_table2 import run


try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required for non-JSON sweep configs")
    return yaml.safe_load(text)


def main(args: argparse.Namespace) -> None:
    cfg = _load_config(Path(args.config))
    runs = cfg.get("runs", [])
    if not runs:
        raise ValueError("Sweep config must define a non-empty 'runs' list")

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    for i, run_cfg in enumerate(runs):
        run_name = run_cfg.get("name", f"run_{i}")
        run_output = out_root / run_name

        ns = argparse.Namespace(
            concept=cfg.get("concept", "nudity"),
            method=cfg.get("method", "barrier"),
            checkpoint=run_cfg["checkpoint"],
            attack_eval_images=cfg["attack_eval_images"],
            output_dir=str(run_output),
            device=cfg.get("device", "cuda"),
            base_model=cfg.get("base_model", "CompVis/stable-diffusion-v1-4"),
            compvis_config_path=cfg.get("compvis_config_path"),
            diffusers_config_path=cfg.get("diffusers_config_path"),
            initializer_token=cfg.get("initializer_token", "person"),
            ci_lr=run_cfg.get("ci_lr", cfg.get("ci_lr", 5e-3)),
            ti_max_train_steps=run_cfg.get("ti_max_train_steps", cfg.get("ti_max_train_steps", 300)),
            learnable_property=cfg.get("learnable_property", "object"),
            generic_prompt=cfg.get("generic_prompt", "a photo of a"),
            center_crop=cfg.get("center_crop", False),
            force_export=args.force_export,
            force_attack=args.force_attack,
            verify_pipeline_load=cfg.get("verify_pipeline_load", False),
            external_attacks=cfg.get("external_attacks", ""),
            attack_idx=run_cfg.get("attack_idx", cfg.get("attack_idx", 0)),
            ud_config=cfg.get("ud_config"),
            rab_command=cfg.get("rab_command"),
            cce_variant=cfg.get("cce_variant"),
            cce_command=cfg.get("cce_command"),
        )
        result = run(ns)
        result["run_name"] = run_name
        result.update({k: v for k, v in run_cfg.items() if k != "checkpoint"})
        results.append(result)

    sweep_csv = out_root / "sweep_results.csv"
    with open(sweep_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in results for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Placeholder paper-style summary: sort by attack image count desc.
    table_csv = out_root / "table2_barrier.csv"
    ranked = sorted(results, key=lambda x: x.get("attack_images", 0), reverse=True)
    with open(table_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["run_name", "concept", "method", "attack_images", "exported_unet", "attack_output_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"Wrote {sweep_csv}")
    print(f"Wrote {table_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep BARRIER checkpoint attack evaluations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--force_export", action="store_true")
    parser.add_argument("--force_attack", action="store_true")
    main(parser.parse_args())
