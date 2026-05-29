#!/usr/bin/env python3
"""
Aggregate per-run ASR CSVs into a final Table-2 CSV and simple LaTeX table.

Usage:
  python aggregate_table2.py --root /path/to/reproduce_output --out-csv /path/to/metrics/table2_final.csv --out-latex /path/to/metrics/table2_final.tex
"""
import argparse
import csv
import os
from pathlib import Path
from statistics import mean, stdev


def find_asr_files(root: Path):
    for p in root.rglob("metrics/asr_summary.csv"):
        yield p


def extract_method_from_row(row):
    # attack_root: /.../attacks/<concept>/<method>_logs
    attack_root = row.get("attack_root","")
    if not attack_root:
        return "unknown"
    last = attack_root.rstrip("/").split("/")[-1]
    if last.endswith("_logs"):
        return last[: -len("_logs")]
    return last


def read_asr_file(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


def aggregate(root: Path):
    data = {}
    for f in find_asr_files(root):
        rows = read_asr_file(f)
        for r in rows:
            method = extract_method_from_row(r)
            try:
                asr = float(r.get("asr", "nan"))
                pre_asr = float(r.get("pre_asr", "nan"))
            except Exception:
                continue
            data.setdefault(method, []).append({"asr": asr, "pre_asr": pre_asr, "file": str(f)})
    return data


def write_csv(out_path: Path, aggregated: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method","n","asr_mean","asr_std","pre_asr_mean","pre_asr_std","samples"])
        for method, items in sorted(aggregated.items()):
            n = len(items)
            asrs = [it["asr"] for it in items]
            pre = [it["pre_asr"] for it in items]
            asr_mean = mean(asrs) if asrs else float("nan")
            asr_std = stdev(asrs) if len(asrs) > 1 else 0.0
            pre_mean = mean(pre) if pre else float("nan")
            pre_std = stdev(pre) if len(pre) > 1 else 0.0
            samples = ";".join(sorted(set(it["file"] for it in items)))
            writer.writerow([method, n, f"{asr_mean:.4f}", f"{asr_std:.4f}", f"{pre_mean:.4f}", f"{pre_std:.4f}", samples])


def write_latex(out_path: Path, aggregated: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("Method & N & ASR (mean) & ASR (std) & Pre-ASR \\\\ \hline\n")
        for method, items in sorted(aggregated.items()):
            n = len(items)
            asrs = [it["asr"] for it in items]
            pre = [it["pre_asr"] for it in items]
            asr_mean = mean(asrs) if asrs else float("nan")
            asr_std = stdev(asrs) if len(asrs) > 1 else 0.0
            pre_mean = mean(pre) if pre else float("nan")
            f.write(f"{method} & {n} & {asr_mean:.3f} & {asr_std:.3f} & {pre_mean:.3f} \\\\ \n")
        f.write("\\end{tabular}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-latex", type=Path, required=True)
    args = parser.parse_args()

    aggregated = aggregate(args.root)
    write_csv(args.out_csv, aggregated)
    write_latex(args.out_latex, aggregated)
    print(f"Wrote aggregated CSV to: {args.out_csv}")
    print(f"Wrote LaTeX table to: {args.out_latex}")


if __name__ == "__main__":
    main()
