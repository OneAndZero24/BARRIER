#!/usr/bin/env python3
"""Run a Stable Diffusion class-forgetting sweep trial over all 10 classes.

Each W&B sweep trial represents one hyperparameter configuration. The trial
executes the full 10-class evaluation loop, aggregates the metrics, and logs
the averages back to W&B.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import wandb
import yaml


IMAGENETTE_CLASSES = [
    "tench",
    "english_springer",
    "cassette_player",
    "chain_saw",
    "church",
    "french_horn",
    "garbage_truck",
    "gas_pump",
    "golf_ball",
    "parachute",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SD class-forgetting sweep trial")
    parser.add_argument("--config", required=True, help="Base pipeline config to sweep from")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    return parser.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def merge_dot_config(cfg: Dict[str, Any], flat_config: Dict[str, Any]) -> Dict[str, Any]:
    for key, val in flat_config.items():
        parts = key.split(".")
        cur = cfg
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = val
    return cfg


def write_config(path: Path, cfg: Dict[str, Any]) -> None:
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def run_pipeline(script_dir: Path, cfg_path: Path, metrics_path: Path) -> Dict[str, Any]:
    command = [
        sys.executable,
        str(script_dir / "pipeline.py"),
        "--config",
        str(cfg_path),
        "--no-wandb",
        "--metrics-out",
        str(metrics_path),
    ]
    subprocess.run(command, check=True, cwd=str(script_dir))
    with metrics_path.open("r") as f:
        return json.load(f)


def numeric_metric(metrics: Dict[str, Any], key: str) -> Optional[float]:
    value = metrics.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent.parent
    base_cfg = load_yaml(script_dir / args.config)

    use_wandb = not args.no_wandb and bool(base_cfg.get("wandb", {}).get("project"))
    if use_wandb:
        wandb.init(
            project=base_cfg["wandb"]["project"],
            entity=base_cfg["wandb"].get("entity"),
            group=base_cfg["wandb"].get("group"),
            tags=base_cfg["wandb"].get("tags", []),
            config=base_cfg,
        )
        base_cfg = merge_dot_config(base_cfg, dict(wandb.config))

    run_id = wandb.run.id if use_wandb else "local"
    base_output_dir = Path(base_cfg["paths"]["output_dir"])
    base_model_dir = Path(base_cfg["paths"]["model_save_dir"])
    base_logs_dir = Path(base_cfg["paths"]["logs_dir"])
    base_cfg["paths"]["output_dir"] = str(base_output_dir / f"sweep_{run_id}")
    base_cfg["paths"]["model_save_dir"] = str(base_model_dir / f"sweep_{run_id}")
    base_cfg["paths"]["logs_dir"] = str(base_logs_dir / f"sweep_{run_id}")

    class_results: List[Dict[str, Any]] = []

    for class_id, class_name in enumerate(IMAGENETTE_CLASSES):
        class_cfg = copy.deepcopy(base_cfg)
        class_cfg["unlearn"]["class_to_forget"] = class_id
        class_cfg["paths"]["output_dir"] = str(Path(base_cfg["paths"]["output_dir"]) / f"class_{class_id}")

        with tempfile.TemporaryDirectory(prefix=f"sd_fulleval_{class_id}_") as tmpdir:
            tmp_path = Path(tmpdir)
            cfg_path = tmp_path / "config.yaml"
            metrics_path = tmp_path / "metrics.json"
            write_config(cfg_path, class_cfg)
            metrics = run_pipeline(script_dir, cfg_path, metrics_path)

        class_results.append({"class_id": class_id, "class_name": class_name, "metrics": metrics})

        if use_wandb:
            class_log = {f"class_{class_id}/{key}": value for key, value in metrics.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
            wandb.log(class_log)

    ua_values = [numeric_metric(item["metrics"], "UA") for item in class_results]
    remain_values = [numeric_metric(item["metrics"], "ACC_REST_AVG") for item in class_results]
    fid_values = [numeric_metric(item["metrics"], "FID") for item in class_results]

    avg_ua = mean([value for value in ua_values if value is not None]) if any(value is not None for value in ua_values) else None
    avg_remain = mean([value for value in remain_values if value is not None]) if any(value is not None for value in remain_values) else None
    avg_fid = mean([value for value in fid_values if value is not None]) if any(value is not None for value in fid_values) else None

    summary: Dict[str, Any] = {
        "avg/UA": avg_ua,
        "avg/UA_pct": None if avg_ua is None else avg_ua * 100.0,
        "avg/ACC_REST_AVG": avg_remain,
        "avg/ACC_REST_AVG_pct": None if avg_remain is None else avg_remain * 100.0,
        "avg/FID": avg_fid,
    }

    if avg_ua is not None and avg_remain is not None:
        summary["target_gap"] = abs(avg_ua - 0.98) + abs(avg_remain - 0.80)
        summary["target_score"] = -summary["target_gap"]
    else:
        summary["target_gap"] = None
        summary["target_score"] = None

    if use_wandb:
        wandb.summary.update({k: v for k, v in summary.items() if v is not None})
        wandb.log({k: v for k, v in summary.items() if v is not None})
        wandb.finish()

    print(json.dumps({"classes": class_results, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()