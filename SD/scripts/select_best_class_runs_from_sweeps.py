#!/usr/bin/env python3
"""Select the best SD class-forgetting run per class from two W&B sweeps.

The script reads per-class metrics logged by the full-eval sweep runner:
  - class_<id>/ACC_FORGET
  - class_<id>/ACC_REST_AVG

For each Imagenette class, it considers every run from both sweeps and picks
the run with the lowest forgotten-class accuracy. Ties are broken by the
highest average accuracy on the remaining classes.

Example:
  python scripts/select_best_class_runs_from_sweeps.py
  python scripts/select_best_class_runs_from_sweeps.py lc1u39el kxi0f1xu
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
        description="Select the best run per forgotten class from two SD W&B sweeps"
    )
    parser.add_argument(
        "sweep_ids",
        nargs="*",
        default=["lc1u39el", "kxi0f1xu"],
        help="Two W&B sweep IDs to compare (default: lc1u39el kxi0f1xu)",
    )
    parser.add_argument("--entity", default="oneandzero24", help="W&B entity")
    parser.add_argument("--project", default="intact-sd", help="W&B project")
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


def _nested_get(mapping: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def fmt_metric(value: Optional[float], digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return "(empty)"

    def esc(text: str) -> str:
        return text.replace("|", "\\|")

    lines = []
    lines.append("| " + " | ".join(esc(header) for header in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(esc(str(cell)) for cell in row) + " |")
    return "\n".join(lines)


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
            run_name = getattr(run, "name", None) or run.id
            run_url = getattr(run, "url", None) or ""
            class_to_forget = _nested_get(run.config or {}, "unlearn", "class_to_forget", default=None)

            for class_id, class_name in enumerate(IMAGENETTE_CLASSES):
                forget_key = f"class_{class_id}/ACC_FORGET"
                remain_key = f"class_{class_id}/ACC_REST_AVG"
                forget_acc = _to_float(summary.get(forget_key))
                remain_acc = _to_float(summary.get(remain_key))

                if forget_acc is None or remain_acc is None:
                    continue

                rows.append(
                    {
                        "sweep_id": sweep_id,
                        "run_id": run.id,
                        "run_name": run_name,
                        "run_url": run_url,
                        "state": run.state,
                        "class_id": class_id,
                        "class_name": class_name,
                        "class_to_forget": class_to_forget,
                        "forget_acc": forget_acc,
                        "remain_acc": remain_acc,
                    }
                )

    return rows


def pick_best_per_class(rows: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    best: Dict[int, Dict[str, Any]] = {}

    def sort_key(row: Dict[str, Any]) -> Tuple[float, float, str, str, str]:
        return (
            row["forget_acc"],
            -row["remain_acc"],
            str(row["run_name"]),
            str(row["sweep_id"]),
            str(row["run_id"]),
        )

    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["class_id"])].append(row)

    for class_id, candidates in grouped.items():
        best[class_id] = sorted(candidates, key=sort_key)[0]

    return best


def main() -> None:
    args = parse_args()
    if len(args.sweep_ids) != 2:
        raise SystemExit("Provide exactly two sweep IDs, or omit them to use the defaults.")

    api = wandb.Api()
    allowed_states = [] if args.all_states else (args.state or ["finished"])
    rows = collect_rows(api, args.entity, args.project, args.sweep_ids, allowed_states)
    if not rows:
        print("No candidate runs found with the requested filters.")
        return

    best = pick_best_per_class(rows)
    output_rows: List[List[str]] = []

    for class_id, class_name in enumerate(IMAGENETTE_CLASSES):
        winner = best.get(class_id)
        if winner is None:
            output_rows.append(
                [
                    str(class_id),
                    class_name,
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                    "NA",
                ]
            )
            continue

        output_rows.append(
            [
                str(class_id),
                class_name,
                winner["sweep_id"],
                winner["run_name"],
                fmt_metric(winner["forget_acc"]),
                fmt_metric(winner["remain_acc"]),
                winner["run_url"],
            ]
        )

    print(
        markdown_table(
            [
                "class_id",
                "class_name",
                "sweep_id",
                "run_name",
                "ACC_FORGET",
                "ACC_REST_AVG",
                "run_url",
            ],
            output_rows,
        )
    )


if __name__ == "__main__":
    main()