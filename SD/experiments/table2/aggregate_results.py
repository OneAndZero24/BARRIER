import argparse
import csv
import json
from pathlib import Path


def main(args: argparse.Namespace) -> None:
    root = Path(args.results_root)
    metric_files = sorted(root.rglob("metrics/*.json"))
    if not metric_files:
        raise FileNotFoundError(f"No metrics JSON files found under {root}")

    rows = []
    for fp in metric_files:
        with open(fp, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    out_csv = root / "sweep_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for r in rows for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_table = root / "table2_barrier.csv"
    ranked = sorted(rows, key=lambda x: x.get("attack_images", 0), reverse=True)
    with open(out_table, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["concept", "method", "attack_images", "input_checkpoint", "attack_output_dir"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in ranked:
            writer.writerow({k: row.get(k) for k in fieldnames})

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_table}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate table2 run artifacts")
    parser.add_argument("--results_root", default="results")
    main(parser.parse_args())
