#!/usr/bin/env python3
"""Collect DDPM class-wise forgetting results into a compact table.

The pipeline writes a metrics.json file into each run directory. This helper
searches a results tree, keeps the newest result per forgotten class, and prints
the values needed for the paper table: FA and FID.
"""

import argparse
import json
from pathlib import Path


CLASS_NAMES = {
    0: "Airplane",
    1: "Automobile",
    2: "Bird",
    3: "Cat",
    4: "Deer",
    5: "Dog",
    6: "Frog",
    7: "Horse",
    8: "Ship",
    9: "Truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True, help="Search root for metrics.json files")
    parser.add_argument(
        "--labels",
        type=str,
        default="1,3,5,7,9",
        help="Comma-separated forgotten labels to include in the table.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Optional CSV file to write alongside the printed table.",
    )
    return parser.parse_args()


def load_latest_metrics(root: Path, labels: set[int]) -> list[dict]:
    latest: dict[int, tuple[float, dict]] = {}
    for metrics_file in root.rglob("metrics.json"):
        try:
            payload = json.loads(metrics_file.read_text())
        except Exception:
            continue

        label = payload.get("label_to_forget")
        if label not in labels:
            continue

        mtime = metrics_file.stat().st_mtime
        current = latest.get(label)
        if current is None or mtime >= current[0]:
            entry = dict(payload)
            entry["metrics_path"] = str(metrics_file)
            latest[label] = (mtime, entry)

    rows = [latest[label][1] for label in sorted(latest)]
    return rows


def format_float(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def main() -> int:
    args = parse_args()
    labels = {int(x) for x in args.labels.split(",") if x.strip()}
    rows = load_latest_metrics(Path(args.root), labels)

    if not rows:
        print("No metrics.json files found for the requested labels.")
        return 1

    csv_rows = []
    print("| Class | FA | FID | Run |")
    print("| --- | ---: | ---: | --- |")
    for row in rows:
        label = int(row["label_to_forget"])
        class_name = CLASS_NAMES.get(label, str(label))
        fa = row.get("FA", row.get("classifier/acc_forgotten"))
        fid = row.get("FID")
        run_dir = row.get("run_dir", "-")
        print(f"| {class_name} | {format_float(fa)} | {format_float(fid)} | {run_dir} |")
        csv_rows.append(
            {
                "label": label,
                "class": class_name,
                "FA": fa,
                "FID": fid,
                "run_dir": run_dir,
            }
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as fp:
            fp.write("label,class,FA,FID,run_dir\n")
            for row in csv_rows:
                fp.write(
                    f'{row["label"]},{row["class"]},{format_float(row["FA"])},{format_float(row["FID"])},{row["run_dir"]}\n'
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())