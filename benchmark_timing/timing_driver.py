#!/usr/bin/env python
"""
Timing driver for the BARRIER unlearning benchmark.

Executes one method run (argv) as a subprocess, sampling GPU VRAM via
nvidia-smi every poll interval, parsing walltime phase markers from stdout,
and appending a row to a JSONL results file.

Markers recognized (printed by the out-of-the-box harness scripts):
    <NAME>_SECONDS <float>            generic phase marker (per-epoch etc.)
    ESC unlearning time: <float> s    ESC one-shot erase (unlearn.py)
    ESC-T unlearning time: <float> s  ESC-T one-shot erase

Usage:
    python timing_driver.py --name salun --repeat 1 \
        --cwd /path/to/harness --out results.jsonl \
        --cmd python3 timing_runner.py --method salun --arch allcnn ...

After all runs:
    python timing_driver.py --summarize results.jsonl --csv results.csv

The single big benchmark script (run_timing_benchmark.sh) orchestrates env
setup, checkpoints, repeats and cleanup around this driver.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time

MARKER_RE = re.compile(r"^([A-Za-z_-]+_SECONDS)\s+([\d.]+)")
ESC_RE = re.compile(r"^(ESC|ESC-T) unlearning time:\s*([\d.]+)\s*s", re.MULTILINE)
RESULTS_DIR = os.environ.get("TIMING_RESULTS_DIR", ".")


def gpu_info():
    """(name, total_mb, used_mb) via nvidia-smi, or None on failure."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().split(",")
        return out[0].strip(), int(out[1].strip()), int(out[2].strip())
    except Exception:
        return None


def sample_vram(stop_event, samples):
    """Background sampler: list of used-MB readings while a method runs."""
    while not stop_event.is_set():
        info = gpu_info()
        if info is not None:
            samples.append(info[2])
        time.sleep(0.25)


def parse_stream(proc, name, repeat, out, meta):
    """Read proc stdout line by line; emit marker rows to `out` (jsonl)."""
    epoch_idx = 0
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode(errors="replace")
        sys.stdout.write(f"[{name} r{repeat}] {line}")
        sys.stdout.flush()

        m = MARKER_RE.search(line)
        if m:
            phase = m.group(1)
            if phase.endswith("EPOCH_SECONDS"):
                epoch_idx += 1
            out.write(json.dumps({
                "method": name, "repeat": repeat, "phase": phase,
                "epoch_index": epoch_idx, "walltime_s": float(m.group(2)),
                "gpu": meta["gpu_name"], "gpu_total_mb": meta["gpu_total_mb"],
            }) + "\n")
            out.flush()
            continue
        m2 = ESC_RE.search(line)
        if m2:
            phase = "esc_oneshot_unlearn"
            out.write(json.dumps({
                "method": name, "repeat": repeat, "phase": phase,
                "epoch_index": 1, "walltime_s": float(m2.group(2)),
                "gpu": meta["gpu_name"], "gpu_total_mb": meta["gpu_total_mb"],
            }) + "\n")
            out.flush()


def run_one(args, meta):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, args.out)
    log_path = os.path.join(RESULTS_DIR, "logs", f"{args.name}_r{args.repeat}.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    stop = threading.Event()
    samples = []
    sampler = threading.Thread(target=sample_vram, args=(stop, samples), daemon=True)
    sampler.start()

    t_start = time.time()
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            args.cmd, cwd=args.cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        with open(out_path, "a") as out:
            parse_stream(proc, args.name, args.repeat, out, meta)
        proc.wait()
    walltime = time.time() - t_start
    stop.set()
    sampler.join(timeout=5)

    peak = max(samples) if samples else -1

    # Fold the whole-process walltime + peak VRAM into the output as a row too
    with open(out_path, "a") as out:
        out.write(json.dumps({
            "method": args.name, "repeat": args.repeat,
            "phase": "process_total", "epoch_index": 0,
            "walltime_s": round(walltime, 4), "peak_vram_mb": peak,
            "gpu": meta["gpu_name"], "gpu_total_mb": meta["gpu_total_mb"],
        }) + "\n")

    ok = proc.returncode == 0
    print(f"[{args.name} r{args.repeat}] rc={proc.returncode} "
          f"walltime={walltime:.2f}s peak_vram={peak}MB")
    return ok


def summarize(args):
    rows = []
    with open(args.summarize) as f:
        for line in f:
            rows.append(json.loads(line))

    out_file = args.csv
    header = ["method", "repeat", "phase", "epoch_index", "walltime_s", "peak_vram_mb", "gpu", "gpu_total_mb"]
    with open(out_file, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in header) + "\n")

    # per-method per-phase summary (mean over repeats)
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        if r.get("phase") == "process_total":
            continue
        agg[(r["method"], r["phase"])].append(r["walltime_s"])
    mean_rows = []
    for (m, phase), vals in sorted(agg.items()):
        mean_rows.append((m, phase, sum(vals) / len(vals), len(vals)))
    with open(out_file.replace(".csv", "_summary.csv"), "w") as f:
        f.write("method,phase,mean_walltime_s,n\n")
        for m, phase, mean, n in mean_rows:
            f.write(f"{m},{phase},{mean:.4f},{n}\n")

    # per-method peak VRAM (max over repeats, from process_total rows)
    peaks = collections.defaultdict(list)
    for r in rows:
        if r.get("phase") == "process_total" and r.get("peak_vram_mb", -1) > 0:
            peaks[r["method"]].append(r["peak_vram_mb"])
    with open(out_file.replace(".csv", "_vram.csv"), "w") as f:
        f.write("method,peak_vram_mb_mean,peak_vram_mb_max,repeats\n")
        for m, vals in sorted(peaks.items()):
            f.write(f"{m},{sum(vals) / len(vals):.1f},{max(vals)},{len(vals)}\n")

    print(f"[summary] {len(rows)} rows -> {out_file}, *_summary.csv, *_vram.csv")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode")
    run = sub.add_parser("run")
    run.add_argument("--name", required=True)
    run.add_argument("--repeat", type=int, required=True)
    run.add_argument("--cwd", default=".")
    run.add_argument("--out", default="results.jsonl")
    run.add_argument("--cmd", nargs=argparse.REMAINDER, required=True)
    sum_ = sub.add_parser("summarize")
    sum_.add_argument("--summarize", required=True)
    sum_.add_argument("--csv", required=True)
    args = p.parse_args()

    if args.mode == "summarize":
        summarize(args)
        return

    info = gpu_info()
    meta = {
        "gpu_name": info[0] if info else "unknown",
        "gpu_total_mb": info[1] if info else -1,
    }
    ok = run_one(args, meta)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()