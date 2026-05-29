#!/usr/bin/env python3
"""Run a Stable Diffusion class-forgetting sweep trial over all 10 classes.

Each W&B sweep trial represents one hyperparameter configuration. The trial
executes the full 10-class evaluation loop, aggregates the metrics, and logs
the averages back to W&B.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
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


def trial_key_from_config(cfg: Dict[str, Any]) -> str:
    """Build a stable key for a hyperparameter setting.

    This lets a re-run of the same trial reuse the same output directory and
    continue from any class evaluations already completed on disk.
    """

    payload = {
        "pipeline": cfg.get("pipeline", {}),
        "unlearn": cfg.get("unlearn", {}),
        "intact": cfg.get("intact", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


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


def load_metrics_if_present(metrics_path: Path) -> Optional[Dict[str, Any]]:
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r") as f:
            metrics = json.load(f)
        if isinstance(metrics, dict):
            return metrics
    except Exception:
        return None
    return None


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        try:
            path.unlink()
        except FileNotFoundError:
            pass


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

    trial_key = trial_key_from_config(base_cfg)
    base_output_dir = Path(base_cfg["paths"]["output_dir"])
    trial_root = base_output_dir / f"sweep_{trial_key}"
    resume_root = trial_root / "resume"
    tmp_root = trial_root / "tmp"
    trial_root.mkdir(parents=True, exist_ok=True)
    resume_root.mkdir(parents=True, exist_ok=True)

    class_results: List[Dict[str, Any]] = []

    for class_id, class_name in enumerate(IMAGENETTE_CLASSES):
        class_cfg = copy.deepcopy(base_cfg)
        class_cfg["unlearn"]["class_to_forget"] = class_id
        class_cfg["unlearn"]["save_compvis"] = False
        class_cfg["unlearn"]["save_diffusers"] = False
        class_cfg["unlearn"]["save_history_logs"] = False
        class_cfg["paths"]["output_dir"] = str(trial_root / f"class_{class_id}")
        class_cfg["paths"]["model_save_dir"] = str(tmp_root / f"class_{class_id}" / "models")
        class_cfg["paths"]["logs_dir"] = str(tmp_root / f"class_{class_id}" / "logs")

        with tempfile.TemporaryDirectory(prefix=f"sd_fulleval_{class_id}_") as tmpdir:
            tmp_path = Path(tmpdir)
            cfg_path = tmp_path / "config.yaml"
            metrics_path = tmp_path / "metrics.json"
            existing_metrics_path = resume_root / f"class_{class_id}" / "metrics.json"
            metrics = load_metrics_if_present(existing_metrics_path)
            if metrics is None:
                existing_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                write_config(cfg_path, class_cfg)
                metrics = run_pipeline(script_dir, cfg_path, metrics_path)
                existing_metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
            remove_path(Path(class_cfg["paths"]["output_dir"]))
            remove_path(Path(class_cfg["paths"]["model_save_dir"]))
            remove_path(Path(class_cfg["paths"]["logs_dir"]))

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