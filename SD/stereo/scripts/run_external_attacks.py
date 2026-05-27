import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _run(cmd, cwd: Path) -> None:
    print(f"[run] (cwd={cwd}) {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run_ud(args: argparse.Namespace) -> None:
    if not args.ud_config:
        raise ValueError("--ud_config is required for attack=ud")

    ud_root = REPO_ROOT / "SD" / "stereo" / "attacks" / "vendors" / "unlearndiffatk"
    cmd = [
        sys.executable,
        "src/execs/attack.py",
        "--config-file",
        args.ud_config,
        "--attacker.attack_idx",
        str(args.attack_idx),
        "--logger.name",
        f"attack_idx_{args.attack_idx}",
    ]
    _run(cmd, ud_root)


def run_rab(args: argparse.Namespace) -> None:
    rab_root = REPO_ROOT / "SD" / "stereo" / "attacks" / "vendors" / "ring-a-bell"
    if args.rab_command is None:
        raise ValueError(
            "Ring-A-Bell upstream is notebook-first; provide --rab_command as a shell command to execute in ring-a-bell/"
        )
    _run(["bash", "-lc", args.rab_command], rab_root)


def run_cce(args: argparse.Namespace) -> None:
    if not args.cce_variant:
        raise ValueError("--cce_variant is required for attack=cce")
    if not args.cce_command:
        raise ValueError("--cce_command is required for attack=cce")

    cce_root = REPO_ROOT / "SD" / "stereo" / "attacks" / "vendors" / "cce" / args.cce_variant
    _run(["bash", "-lc", args.cce_command], cce_root)


def main(args: argparse.Namespace) -> None:
    if args.attack == "ud":
        run_ud(args)
    elif args.attack == "rab":
        run_rab(args)
    elif args.attack == "cce":
        run_cce(args)

    output = {
        "attack": args.attack,
        "attack_idx": args.attack_idx,
        "ud_config": args.ud_config,
        "cce_variant": args.cce_variant,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vendored external attacks used in Table-2 style evaluation")
    parser.add_argument("--attack", choices=["ud", "rab", "cce"], required=True)
    parser.add_argument("--attack_idx", type=int, default=0)

    parser.add_argument("--ud_config", default=None)

    parser.add_argument("--rab_command", default=None)

    parser.add_argument("--cce_variant", default=None, help="One of: esd, fmn, sa, np, sld, uce, ac")
    parser.add_argument("--cce_command", default=None)

    parser.add_argument("--output_json", default=None)
    main(parser.parse_args())
