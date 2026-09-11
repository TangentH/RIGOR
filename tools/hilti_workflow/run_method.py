#!/usr/bin/env python3
"""Launch DA3 or PanoVGGT with one shared run/output convention."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_DA3_CONFIG = (
    REPO / "tools/hilti_workflow/configs/rigor_paper_free_scale.yaml"
)


def safe_component(value: str, label: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{label} must be one path component: {value!r}")
    return value


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=("da3", "panovggt"))
    parser.add_argument("--floor", required=True, help="For example floor_1 or floor_UG1")
    parser.add_argument("--date", required=True, help="Capture date, normally YYYY-MM-DD")
    parser.add_argument("--run", required=True, type=int, dest="run_number")
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")

    parser.add_argument("--rosbag", type=Path, help="DA3 input ROS2 bag")
    parser.add_argument("--workflow-config", type=Path, default=DEFAULT_DA3_CONFIG)

    parser.add_argument("--panovggt-repo", type=Path, help="Official PanoVGGT checkout")
    parser.add_argument("--checkpoint", type=Path, help="PanoVGGT model.pt")
    parser.add_argument("--image-dir", type=Path, help="Prepared levelled ERP frames")
    parser.add_argument("--erp-mask-npz", type=Path, help="Packed invalid-pixel masks")
    args, extra = parser.parse_known_args()
    return args, extra


def require_paths(args: argparse.Namespace, names: tuple[str, ...]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise SystemExit(f"{args.method} requires {flags}")


def build_command(
    args: argparse.Namespace,
    output_dir: Path,
    extra: list[str],
) -> list[str]:
    if args.method == "da3":
        require_paths(args, ("rosbag",))
        command = [
            sys.executable,
            str(REPO / "tools/hilti_workflow/run_hilti_rosbag_to_reconstruction.py"),
            str(args.rosbag),
            "--output-dir",
            str(output_dir),
            "--workflow-config",
            str(args.workflow_config),
        ]
    else:
        require_paths(
            args,
            ("panovggt_repo", "checkpoint", "image_dir", "erp_mask_npz"),
        )
        command = [
            sys.executable,
            str(REPO / "tools/hilti_workflow/panovggt/run_reconstruction.py"),
            "--repo",
            str(args.panovggt_repo),
            "--checkpoint",
            str(args.checkpoint),
            "--image-dir",
            str(args.image_dir),
            "--erp-mask-npz",
            str(args.erp_mask_npz),
            "--output-dir",
            str(output_dir),
        ]
    if args.force and args.method == "da3":
        command.append("--force")
    command.extend(extra)
    return command


def main() -> int:
    args, extra = parse_args()
    floor = safe_component(args.floor, "floor")
    date = safe_component(args.date, "date")
    if args.run_number < 1:
        raise ValueError("run must be at least 1")
    output_dir = (
        args.output_root.expanduser().resolve()
        / args.method
        / floor
        / date
        / f"run_{args.run_number}"
        / "reconstruction"
    )
    command = build_command(args, output_dir, extra)
    print(f"[method] {args.method}")
    print(f"[output] {output_dir}")
    print("[command] " + shlex.join(command))
    if args.dry_run:
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.run(command, cwd=REPO, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
