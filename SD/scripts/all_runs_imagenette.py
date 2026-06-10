#!/usr/bin/env python3
"""Export all SD class-forgetting runs per class from two W&B sweeps to CSV.

The script reads per-class metrics logged by the full-eval sweep runner:
  - class_<id>/ACC_FORGET
  - class_<id>/ACC_REST_AVG

It extracts every run for each Imagenette class, includes the specified hyperparameters
(lambda, epochs, lr), and outputs the results into a CSV file for manual selection.

Example:
  python scripts/export_all_class_runs.py
  python scripts/export_all_class_runs.py lc1u39el kxi0f1xu --output my_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from typing import Any, Dict, List, Optional, Sequence

import wandb


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
    parser = argparse.ArgumentParser(
        description="Export all runs per forgotten class from two SD W&B sweeps to CSV"
    )
    parser.add_argument(
        "sweep_ids",
        nargs="*",
        default=["lc1u39el", "kxi0f1xu"],
        help="Two W&B sweep IDs to query (default: lc1u39el kxi0f1xu)",
    )
    parser.add_argument("--entity", default="oneandzero24", help="W&B entity")
    parser.add_argument("--project", default="intact-sd", help="W&B project")
    parser.add_argument(
        "--output", 
        default="all_runs_per_class.csv", 
        help="Output CSV filename (default: all_runs_per_class.csv)"
    )
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Consider runs in every state instead of filtering to finished runs.",
    )
    parser.add_argument(
        "--state",
        action="append",
        default=[],
        help="Optional run state filter(s). If omitted, the script keeps finished runs only.",
    )
    return parser.parse_args()


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def _find_in_config(config: Dict[str, Any], key: str, default: Any = "NA") -> Any:
    """Helper to search for a key in top-level and one-level-deep nested configs."""
    if key in config:
        return config[key]
    for v in config.values():
        if isinstance(v, dict) and key in v:
            return v[key]
    return default


def fmt_metric(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def collect_rows(
    api: "wandb.Api",
    entity: str,
    project: str,
    sweep_ids: Sequence[str],
    allowed_states: Sequence[str],
) -> List[Dict[str, Any]]:
    path = f"{entity}/{project}"
    rows: List[Dict[str, Any]] = []

    for sweep_id in sweep_ids:
        sweep = api.sweep(f"{path}/{sweep_id}")
        for run in sweep.runs:
            if allowed_states and run.state not in allowed_states:
                continue

            summary = run.summary or {}
            config = run.config or {}
            run_name = getattr(run, "name", None) or run.id
            run_url = getattr(run, "url", None) or ""
            
            # Extract requested hyperparameters
            lr = _find_in_config(config, "lr", _find_in_config(config, "learning_rate", "NA"))
            epochs = _find_in_config(config, "epochs", _find_in_config(config, "num_epochs", "NA"))
            lambda_val = _find_in_config(config, "lambda", "NA")

            for class_id, class_name in enumerate(IMAGENETTE_CLASSES):
                forget_key = f"class_{class_id}/ACC_FORGET"
                remain_key = f"class_{class_id}/ACC_REST_AVG"
                forget_acc = _to_float(summary.get(forget_key))
                remain_acc = _to_float(summary.get(remain_key))

                # Include the run if it has logged metrics for this class
                if forget_acc is not None and remain_acc is not None:
                    rows.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "sweep_id": sweep_id,
                            "run_id": run.id,
                            "run_name": run_name,
                            "lambda": lambda_val,
                            "epochs": epochs,
                            "lr": lr,
                            "forget_acc": forget_acc,
                            "remain_acc": remain_acc,
                            "run_url": run_url,
                            "state": run.state,
                        }
                    )

    return rows


def main() -> None:
    args = parse_args()
    if not args.sweep_ids:
        raise SystemExit("Provide at least one sweep ID, or omit to use the defaults.")

    api = wandb.Api()
    allowed_states = [] if args.all_states else (args.state or ["finished"])
    
    print(f"Fetching runs from sweeps: {', '.join(args.sweep_ids)}...")
    rows = collect_rows(api, args.entity, args.project, args.sweep_ids, allowed_states)
    
    if not rows:
        print("No candidate runs found with the requested filters.")
        return

    # Sort rows to make manual selection easier: Group by Class ID, then by lowest Forget Acc
    rows.sort(key=lambda x: (
        x["class_id"], 
        x["forget_acc"] if x["forget_acc"] is not None else float('inf')
    ))

    # Export to CSV
    headers = [
        "class_id",
        "class_name",
        "sweep_id",
        "run_name",
        "lambda",
        "epochs",
        "lr",
        "ACC_FORGET",
        "ACC_REST_AVG",
        "state",
        "run_url",
    ]

    with open(args.output, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                row["class_id"],
                row["class_name"],
                row["sweep_id"],
                row["run_name"],
                row["lambda"],
                row["epochs"],
                row["lr"],
                fmt_metric(row["forget_acc"]),
                fmt_metric(row["remain_acc"]),
                row["state"],
                row["run_url"],
            ])

    print(f"Successfully exported {len(rows)} data points to '{args.output}'.")


if __name__ == "__main__":
    main()