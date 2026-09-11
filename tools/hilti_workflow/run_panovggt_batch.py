#!/usr/bin/env python3
"""Run a local, restartable PanoVGGT queue with one process per GPU.

The driver never downloads or deletes input data. Bind each worker externally
with CUDA_VISIBLE_DEVICES and use --worker-index/--num-workers for disjoint
manifest shards.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.hilti_workflow import run_hilti_batch as shared  # noqa: E402

SINGLE_RUNNER = (
    REPO / "tools/hilti_workflow/run_panovggt_rosbag_to_reconstruction.py"
)
DEFAULT_PANO_REPO = REPO / ".external/PanoVGGT"
DEFAULT_CHECKPOINT = DEFAULT_PANO_REPO / "checkpoints/model.pt"
DEFAULT_DEVICE_MASK = REPO / "device_mask_final.png"
DEFAULT_MASK_PROMPT = "person. human. worker."


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    paths = shared.load_paths_config(shared.DEFAULT_PATHS_CONFIG)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config", type=Path, default=shared.DEFAULT_PATHS_CONFIG
    )
    parser.add_argument(
        "--run-manifest", "--manifest", dest="run_manifest", type=Path,
        default=shared.DEFAULT_RUN_MANIFEST,
    )
    parser.add_argument("--include-runs", default="")
    parser.add_argument("--include-floors", default="")
    parser.add_argument("--exclude-floors", default="")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--worker-id", default="worker-0")
    parser.add_argument("--source-data-root", type=Path, default=paths["data_root"])
    parser.add_argument("--scratch-root", type=Path, default=paths["scratch_root"])
    parser.add_argument(
        "--output-root", type=Path,
        default=paths["output_root"].parent / "panovggt",
    )
    parser.add_argument("--panovggt-repo", type=Path, default=DEFAULT_PANO_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device-mask", type=Path, default=DEFAULT_DEVICE_MASK)
    parser.add_argument("--da3-env", default="da3")
    parser.add_argument("--gsam2-env", default="gsam2")
    parser.add_argument("--mask-prompt", default=DEFAULT_MASK_PROMPT)
    parser.add_argument(
        "--keep-work-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    args = parser.parse_args(argv)
    supplied = sys.argv[1:] if argv is None else argv
    explicit = {
        token.split("=", 1)[0] for token in supplied if token.startswith("--")
    }
    if "--paths-config" in explicit:
        configured = shared.load_paths_config(args.paths_config.expanduser().resolve())
        if "--source-data-root" not in explicit:
            args.source_data_root = configured["data_root"]
        if "--scratch-root" not in explicit:
            args.scratch_root = configured["scratch_root"]
        if "--output-root" not in explicit:
            args.output_root = configured["output_root"].parent / "panovggt"
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, --num-workers)")
    safe_worker(args.worker_id)
    return args


def safe_worker(value: str) -> str:
    if (
        not value
        or Path(value).name != value
        or not all(character.isalnum() or character in "._-" for character in value)
    ):
        raise ValueError(f"Invalid --worker-id: {value!r}")
    return value


def selected_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    runs = shared.load_runs(args.run_manifest)
    include_runs = {
        value.strip() for value in args.include_runs.split(",") if value.strip()
    }
    include_floors = {
        value.strip() for value in args.include_floors.split(",") if value.strip()
    }
    exclude_floors = {
        value.strip() for value in args.exclude_floors.split(",") if value.strip()
    }
    available = {str(run["run_name"]) for run in runs}
    missing = sorted(include_runs - available)
    if missing:
        raise ValueError(f"Unknown --include-runs entries: {missing}")
    runs = [
        run for run in runs
        if (not include_runs or run["run_name"] in include_runs)
        and (not include_floors or run["floor"] in include_floors)
        and run["floor"] not in exclude_floors
    ]
    runs.sort(key=lambda run: str(run["relative_path"]))
    runs = [
        run for index, run in enumerate(runs)
        if index % args.num_workers == args.worker_index
    ]
    return runs[: args.stop_after] if args.stop_after else runs


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_command(
    args: argparse.Namespace,
    run: dict[str, Any],
    bag: Path,
    output: Path,
    work: Path,
) -> list[str]:
    command = [
        "conda", "run", "--no-capture-output", "-n", args.da3_env,
        "python", str(SINGLE_RUNNER),
        "--rosbag", str(bag),
        "--output-dir", str(output),
        "--work-dir", str(work),
        "--relative-path", str(run["relative_path"]),
        "--panovggt-repo", str(args.panovggt_repo.expanduser().resolve()),
        "--checkpoint", str(args.checkpoint.expanduser().resolve()),
        "--device-mask", str(args.device_mask.expanduser().resolve()),
        "--da3-env", args.da3_env,
        "--gsam2-env", args.gsam2_env,
        "--mask-prompt", args.mask_prompt,
        "--no-delete-rosbag-on-success",
        "--keep-work-on-success"
        if args.keep_work_on_success
        else "--no-keep-work-on-success",
    ]
    return command


def static_input_provenance(args: argparse.Namespace) -> dict[str, Any]:
    cached = getattr(args, "_static_input_provenance", None)
    if isinstance(cached, dict):
        return cached
    value = {
        "panovggt_config": shared.file_identity(
            args.panovggt_repo.expanduser().resolve()
            / "training/config/default.yaml"
        ),
        "checkpoint": shared.file_identity(args.checkpoint.expanduser().resolve()),
        "device_mask": shared.file_identity(args.device_mask.expanduser().resolve()),
        "workflow": shared.file_identity(SINGLE_RUNNER),
    }
    args._static_input_provenance = value
    return value


def completed_pano_output(
    args: argparse.Namespace, output: Path, relative: str
) -> bool:
    if not shared.completed_output(output, expected_relative_path=relative):
        return False
    try:
        manifest = json.loads(
            (output / "workflow_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    recorded = manifest.get("input_provenance")
    expected = static_input_provenance(args)
    return isinstance(recorded, dict) and all(
        recorded.get(name) == identity for name, identity in expected.items()
    )


def run_logged(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + shlex.join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        return process.wait()


def main() -> int:
    args = parse_args()
    worker = safe_worker(args.worker_id)
    source_root = args.source_data_root.expanduser().resolve()
    scratch_root = args.scratch_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    runs = selected_runs(args)
    status_path = output_root / "_logs" / worker / "batch_status.jsonl"
    print(f"[worker] {args.worker_index + 1}/{args.num_workers}")
    print("[visible-gpus] " + os.environ.get("CUDA_VISIBLE_DEVICES", "<inherited>"))
    print(f"[runs] {len(runs)}")
    failed = 0
    for index, run in enumerate(runs, 1):
        relative = str(run["relative_path"])
        output = output_root / relative / "reconstruction"
        work = scratch_root / "panovggt" / relative / "_work"
        row: dict[str, Any] = {
            "time": shared.now(),
            "run_name": run["run_name"],
            "worker_index": args.worker_index,
            "output_dir": str(output),
        }
        if (
            completed_pano_output(args, output, relative)
            and not args.rerun_existing
        ):
            row["status"] = "skipped_verified_complete"
            append_jsonl(status_path, row)
            print(f"[{index}/{len(runs)}] skip {run['run_name']}")
            continue
        try:
            bag = shared.find_rosbag(relative, source_root, scratch_root)
            if bag is None:
                raise FileNotFoundError(
                    f"No local ROS bag for {relative}; downloads are not performed"
                )
            command = build_command(args, run, bag, output, work)
            print(f"[{index}/{len(runs)}] {run['run_name']}")
            print("[command] " + shlex.join(command))
            if args.list_runs or args.dry_run:
                row["status"] = "listed" if args.list_runs else "dry_run"
                row["rosbag"] = str(bag)
            else:
                if (output / "reconstruction.ply").exists():
                    raise RuntimeError(
                        "Incomplete output contains reconstruction.ply; refusing overwrite"
                    )
                output.mkdir(parents=True, exist_ok=True)
                returncode = run_logged(
                    command,
                    output_root / "_logs" / worker / f"{run['run_name']}.log",
                )
                if returncode:
                    raise RuntimeError(f"PanoVGGT worker exited with {returncode}")
                if not completed_pano_output(args, output, relative):
                    raise RuntimeError(
                        "worker returned without hash-verified complete outputs"
                    )
                row["status"] = "complete"
        except Exception as error:
            failed += 1
            row["status"] = "failed"
            row["error"] = f"{type(error).__name__}: {error}"
            print(f"[failed] {run['run_name']}: {error}", file=sys.stderr)
            failure_dir = output / "logs"
            failure_dir.mkdir(parents=True, exist_ok=True)
            (failure_dir / "batch_error_traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        append_jsonl(status_path, row)
    print(f"[batch-finished] runs={len(runs)} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
