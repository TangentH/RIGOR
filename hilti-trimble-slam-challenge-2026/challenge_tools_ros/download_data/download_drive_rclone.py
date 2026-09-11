#!/usr/bin/env python3
"""
Google Drive Folder Downloader (rclone version)

Downloads all files and folders from a Google Drive folder using rclone.

Setup:
    1. Install rclone:
       macOS:   brew install rclone
       Linux:   sudo apt install rclone  (or: sudo dnf install rclone)
       Windows: choco install rclone  (or: winget install rclone)

    2. Configure Google Drive remote (one-time):
       rclone config
       - Choose "n" for new remote
       - Name it "gdrive"
       - Choose "drive" (Google Drive)
       - Leave client_id and client_secret blank
       - Choose "1" for full access
       - Follow browser auth flow

Usage:
    python download_drive_rclone.py
    python download_drive_rclone.py -f FOLDER_ID -o ./output
"""

import argparse
import csv
import re
import subprocess
import shutil
import sys
from pathlib import Path

# The folder ID from the URL
# https://drive.google.com/drive/u/3/folders/1BWFIfEL40Nvj-yeyre5O9dOiYCTWatv5
DEFAULT_FOLDER_ID = "1BWFIfEL40Nvj-yeyre5O9dOiYCTWatv5"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_MANIFEST = REPO_ROOT / "groundtruth" / "init_gt_poses.csv"
RUN_NAME_RE = re.compile(r"^(?P<floor>floor_.+?)_(?P<date>\d{4}-\d{2}-\d{2})_run_(?P<run>\d+)$")


def parse_run_name(run_name):
    """Return (floor, date, run_number) from floor_X_YYYY-MM-DD_run_Z."""
    match = RUN_NAME_RE.match(run_name)
    if not match:
        raise ValueError(f"Unexpected run name: {run_name}")
    return match.group("floor"), match.group("date"), int(match.group("run"))


def load_latest_runs_per_floor(manifest_path, include_floors=None, exclude_floors=None):
    """Select the latest date and highest run number for each floor."""
    include = set(include_floors or [])
    exclude = set(exclude_floors or [])
    latest = {}
    with Path(manifest_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_name = row.get("run_name") or row.get("#Sequence Name") or row.get("Sequence Name")
            if not run_name:
                raise KeyError(f"Could not find run name column in {manifest_path}")
            floor, date, run_number = parse_run_name(run_name)
            if include and floor not in include:
                continue
            if floor in exclude:
                continue
            key = (date, run_number)
            if floor not in latest or key > latest[floor]["key"]:
                latest[floor] = {
                    "key": key,
                    "run_name": run_name,
                    "floor": floor,
                    "date": date,
                    "run": run_number,
                    "relative_path": f"{floor}/{date}/run_{run_number}",
                }
    return [latest[floor] for floor in sorted(latest)]


def split_csv_arg(text):
    if not text:
        return []
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download files from a public Google Drive folder using rclone"
    )
    parser.add_argument(
        "--folder-id",
        "-f",
        default=DEFAULT_FOLDER_ID,
        help=f"Google Drive folder ID (default: {DEFAULT_FOLDER_ID})",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="./challenge_data",
        help="Output directory (default: ./challenge_data)",
    )
    parser.add_argument(
        "--latest-per-floor",
        action="store_true",
        help="Download only the latest available run for each floor, using groundtruth/init_gt_poses.csv as the run manifest",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=DEFAULT_RUN_MANIFEST,
        help=f"CSV manifest used by --latest-per-floor (default: {DEFAULT_RUN_MANIFEST})",
    )
    parser.add_argument(
        "--remote-prefix",
        default="",
        help="Optional prefix inside the Google Drive folder before floor_* directories, e.g. 'data'",
    )
    parser.add_argument(
        "--include-floors",
        default="",
        help="Comma-separated floor names to include with --latest-per-floor, e.g. floor_1,floor_2",
    )
    parser.add_argument(
        "--exclude-floors",
        default="",
        help="Comma-separated floor names to exclude with --latest-per-floor, e.g. floor_UG2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected rclone commands without downloading",
    )
    args = parser.parse_args()

    # Check if rclone is installed
    if not args.dry_run and not shutil.which("rclone"):
        print("ERROR: rclone is not installed!")
        print()
        print("Install with:")
        print("  macOS:   brew install rclone")
        print("  Linux:   sudo apt install rclone")
        print("  Windows: choco install rclone")
        print()
        sys.exit(1)

    # Check if gdrive remote is configured
    result = None
    if not args.dry_run:
        result = subprocess.run(
            ["rclone", "listremotes"],
            capture_output=True,
            text=True,
        )
    if not args.dry_run and "gdrive:" not in result.stdout:
        print("ERROR: rclone 'gdrive' remote not configured!")
        print()
        print("Run 'rclone config' and create a remote named 'gdrive':")
        print("  1. Choose 'n' for new remote")
        print("  2. Name it 'gdrive'")
        print("  3. Choose 'drive' (Google Drive)")
        print("  4. Leave client_id and client_secret blank")
        print("  5. Choose '1' for full access")
        print("  6. Follow the browser auth flow")
        print()
        sys.exit(1)

    folder_id = args.folder_id
    output_dir = Path(args.output)

    print("=" * 60)
    print("Google Drive Folder Downloader (rclone)")
    print("=" * 60)
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build rclone source path using configured gdrive remote
    source = f"gdrive,root_folder_id={folder_id}:"

    print(f"Folder ID: {folder_id}")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Starting download...")
    print("-" * 60)

    commands = []
    if args.latest_per_floor:
        runs = load_latest_runs_per_floor(
            args.run_manifest,
            include_floors=split_csv_arg(args.include_floors),
            exclude_floors=split_csv_arg(args.exclude_floors),
        )
        if not runs:
            print(f"ERROR: no runs selected from {args.run_manifest}")
            sys.exit(1)
        print("Selected latest runs per floor:")
        for run in runs:
            print(f"  {run['floor']}: {run['run_name']} -> {run['relative_path']}")

        prefix = args.remote_prefix.strip("/")
        for run in runs:
            rel = run["relative_path"]
            remote_rel = f"{prefix}/{rel}" if prefix else rel
            commands.append(
                [
                    "rclone",
                    "copy",
                    f"{source}{remote_rel}",
                    str(output_dir / rel),
                    "--progress",
                    "--transfers=4",
                    "--drive-acknowledge-abuse",
                ]
            )
    else:
        commands.append(
            [
                "rclone",
                "copy",
                source,
                str(output_dir),
                "--progress",
                "--transfers=4",
                "--drive-acknowledge-abuse",  # Download even if flagged
            ]
        )

    try:
        for cmd in commands:
            print()
            print("$ " + " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True)
        print("-" * 60)
        print("Download complete!" if not args.dry_run else "Dry run complete!")
        print(f"  Location: {output_dir.absolute()}")
    except subprocess.CalledProcessError as e:
        print("-" * 60)
        print(f"Download failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()
