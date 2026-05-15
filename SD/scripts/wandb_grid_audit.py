#!/usr/bin/env python3
"""Audit SD full-eval grid runs in Weights & Biases.

Outputs:
1) Crash table with last known step/iteration for non-finished runs.
2) Matrix by class_to_forget (rows) and hyperparameter tag (columns),
   with each cell rendered as UA/FID.

Example:
  python scripts/wandb_grid_audit.py \
    --entity oneandzero24 \
    --project intact-sd \
    --group-prefix grid-
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import wandb

IMAGENETTE_CLASSES = {
    0: "tench",
    1: "english_springer",
    2: "cassette_player",
    3: "chain_saw",
    4: "church",
    5: "french_horn",
    6: "garbage_truck",
    7: "gas_pump",
    8: "golf_ball",
    9: "parachute",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit SD grid runs in W&B")
    p.add_argument("--entity", default="oneandzero24", help="W&B entity")
    p.add_argument("--project", default="intact-sd", help="W&B project")
    p.add_argument("--group-prefix", default="grid-", help="Only groups that start with this prefix")
    p.add_argument("--include-running", action="store_true", help="Keep running/queued runs in the matrix")
    p.add_argument(
        "--state",
        action="append",
        default=[],
        help="Optional state filter(s), e.g. --state finished --state failed",
    )
    return p.parse_args()


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    return None


def _nested_get(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def infer_last_iteration(run: "wandb.apis.public.Run") -> Optional[float]:
    """Best-effort inference of the last reached iteration/step for a run."""
    candidates: List[float] = []

    summary = run.summary or {}
    summary_keys = [
        "_step",
        "global_step",
        "train/global_step",
        "step",
        "train/step",
        "iter",
        "iteration",
        "epoch",
    ]
    for key in summary_keys:
        v = _to_float(summary.get(key))
        if v is not None:
            candidates.append(v)

    # Public API usually exposes this for quick access.
    lhs = getattr(run, "lastHistoryStep", None)
    lhsf = _to_float(lhs)
    if lhsf is not None:
        candidates.append(lhsf)

    if candidates:
        return max(candidates)

    # Fallback: scan a few likely history keys.
    history_keys = ["_step", "epoch", "global_step", "step", "iter", "iteration"]
    try:
        for row in run.scan_history(keys=history_keys, page_size=500):
            for key in history_keys:
                v = _to_float(row.get(key))
                if v is not None:
                    candidates.append(v)
    except Exception:
        return None

    return max(candidates) if candidates else None


def format_param_tag(alpha: Any, lam: Any, epochs: Any, lr: Any) -> str:
    return f"a{alpha}-lam{lam}-ep{epochs}-lr{lr}"


def fmt_metric(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "NA"
    return f"{v:.{digits}f}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "(empty)"

    def esc(s: str) -> str:
        return s.replace("|", "\\|")

    lines = []
    lines.append("| " + " | ".join(esc(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(esc(str(c)) for c in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    api = wandb.Api()

    path = f"{args.entity}/{args.project}"
    filters: Dict[str, Any] = {}
    if args.group_prefix:
        filters["group"] = {"$regex": f"^{args.group_prefix}"}
    if args.state:
        filters["state"] = {"$in": args.state}

    runs = list(api.runs(path=path, filters=filters))
    if not runs:
        print("No runs found for the requested filters.")
        return

    # Collect rows for detailed and crash views.
    detailed_rows: List[Dict[str, Any]] = []
    crash_rows: List[Dict[str, Any]] = []

    # Matrix accumulator for (class, param_tag).
    agg: Dict[Tuple[int, str], List[Tuple[Optional[float], Optional[float], str]]] = defaultdict(list)

    for run in runs:
        cfg = run.config or {}
        summary = run.summary or {}

        cls = _nested_get(cfg, "unlearn", "class_to_forget", default=None)
        alpha = _nested_get(cfg, "unlearn", "alpha", default=None)
        lam = _nested_get(cfg, "intact", "lambda_interval", default=None)
        epochs = _nested_get(cfg, "unlearn", "epochs", default=None)
        lr = _nested_get(cfg, "unlearn", "lr", default=None)
        group = getattr(run, "group", None)

        if cls is None:
            # Not a class-forget grid run; skip.
            continue

        try:
            cls_int = int(cls)
        except Exception:
            continue

        param_tag = format_param_tag(alpha, lam, epochs, lr)
        cls_name = IMAGENETTE_CLASSES.get(cls_int, f"class_{cls_int}")

        ua = _to_float(summary.get("UA"))
        fid = _to_float(summary.get("FID"))

        last_iter = infer_last_iteration(run)

        row = {
            "run": run.name,
            "state": run.state,
            "group": group,
            "class_id": cls_int,
            "class_name": cls_name,
            "alpha": alpha,
            "lambda": lam,
            "epochs": epochs,
            "lr": lr,
            "param_tag": param_tag,
            "UA": ua,
            "FID": fid,
            "last_iter": last_iter,
            "url": run.url,
        }
        detailed_rows.append(row)

        if run.state != "finished":
            crash_rows.append(row)

        if run.state == "finished" or args.include_running:
            agg[(cls_int, param_tag)].append((ua, fid, run.state))

    if not detailed_rows:
        print("Runs found, but none matched class grid schema (missing unlearn.class_to_forget).")
        return

    # Crash report.
    print("# Crash Report")
    if crash_rows:
        crash_rows_sorted = sorted(
            crash_rows,
            key=lambda r: (r["class_id"], r["param_tag"], r["run"]),
        )
        crash_md_rows = []
        for r in crash_rows_sorted:
            crash_md_rows.append([
                str(r["class_id"]),
                r["class_name"],
                r["param_tag"],
                str(r["state"]),
                "NA" if r["last_iter"] is None else str(int(r["last_iter"])),
                fmt_metric(r["UA"]),
                fmt_metric(r["FID"]),
                r["run"],
            ])
        print(
            markdown_table(
                [
                    "class_id",
                    "class_name",
                    "hyperparams",
                    "state",
                    "last_iter",
                    "UA",
                    "FID",
                    "run",
                ],
                crash_md_rows,
            )
        )
    else:
        print("No non-finished runs detected.")

    # Matrix by class x hyperparams with UA/FID cells.
    param_tags = sorted({r["param_tag"] for r in detailed_rows})
    class_ids = sorted({r["class_id"] for r in detailed_rows})

    print("\n# UA/FID Matrix (class x hyperparameters)")
    matrix_headers = ["class"] + param_tags
    matrix_rows: List[List[str]] = []

    for cls in class_ids:
        cls_label = f"{cls}:{IMAGENETTE_CLASSES.get(cls, f'class_{cls}') }"
        row_cells = [cls_label]
        for tag in param_tags:
            vals = agg.get((cls, tag), [])
            if not vals:
                row_cells.append("-")
                continue

            uas = [u for (u, _, _) in vals if u is not None]
            fids = [f for (_, f, _) in vals if f is not None]
            ua_mean = statistics.mean(uas) if uas else None
            fid_mean = statistics.mean(fids) if fids else None

            # If there are mixed run states in this cell, mark it.
            states = sorted({s for (_, _, s) in vals})
            state_suffix = "" if states == ["finished"] else f" ({','.join(states)})"
            row_cells.append(f"{fmt_metric(ua_mean)}/{fmt_metric(fid_mean)}{state_suffix}")

        matrix_rows.append(row_cells)

    print(markdown_table(matrix_headers, matrix_rows))

    # Flat detailed table for debugging / traceability.
    print("\n# Detailed Runs")
    detailed_rows_sorted = sorted(
        detailed_rows,
        key=lambda r: (r["class_id"], r["param_tag"], r["run"]),
    )
    detailed_md_rows = []
    for r in detailed_rows_sorted:
        detailed_md_rows.append([
            str(r["class_id"]),
            r["class_name"],
            r["param_tag"],
            str(r["state"]),
            "NA" if r["last_iter"] is None else str(int(r["last_iter"])),
            fmt_metric(r["UA"]),
            fmt_metric(r["FID"]),
            r["run"],
        ])
    print(
        markdown_table(
            ["class_id", "class_name", "hyperparams", "state", "last_iter", "UA", "FID", "run"],
            detailed_md_rows,
        )
    )


if __name__ == "__main__":
    main()
