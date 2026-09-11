#!/usr/bin/env python3
"""Run a local, restartable HILTI reconstruction queue.

The batch driver never downloads or deletes data. Launch one process per GPU
and bind it externally with CUDA_VISIBLE_DEVICES. Multiple workers divide the
manifest deterministically with --worker-index and --num-workers; one worker
is the default.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
DEFAULT_PATHS_CONFIG = Path(
    os.environ.get(
        "HILTI_PATHS_CONFIG",
        REPO
        / (
            "hilti_paths.local.yaml"
            if (REPO / "hilti_paths.local.yaml").is_file()
            else "hilti_paths.yaml"
        ),
    )
).expanduser()
DEFAULT_RUN_MANIFEST = (
    REPO / "tools" / "hilti_workflow" / "manifests" / "hilti_all_runs.json"
)
DEFAULT_WORKFLOW_CONFIG = (
    REPO
    / "tools"
    / "hilti_workflow"
    / "configs"
    / "rigor_paper_free_scale.yaml"
)
SINGLE_RUNNER = (
    REPO
    / "tools"
    / "hilti_workflow"
    / "run_hilti_rosbag_to_reconstruction.py"
)
REQUIRED_OUTPUTS = (
    "reconstruction.ply",
    "camera_poses.txt",
    "camera_poses.ply",
    "workflow_manifest.json",
)
RUN_PATTERN = re.compile(
    r"^(?P<floor>floor_.+?)_(?P<date>\d{4}-\d{2}-\d{2})_run_(?P<run>\d+)$"
)


def parse_simple_paths(path: Path) -> dict[str, Path]:
    values: dict[str, Path] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        text = value.strip().strip("'\"")
        if not text:
            continue
        text = text.format(repo=REPO, user=getpass.getuser())
        resolved = Path(text).expanduser()
        values[key.strip()] = (
            resolved if resolved.is_absolute() else (REPO / resolved).resolve()
        )
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = parse_simple_paths(DEFAULT_PATHS_CONFIG)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=DEFAULT_PATHS_CONFIG,
        help="Portable path-only YAML; command-line paths take precedence",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=DEFAULT_RUN_MANIFEST,
        help="Shipped JSON run manifest or a CSV with a sequence column",
    )
    parser.add_argument(
        "--source-data-root",
        type=Path,
        default=defaults.get("data_root", REPO / "data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=defaults.get("output_root", REPO / "outputs" / "da3"),
    )
    parser.add_argument(
        "--workflow-config",
        type=Path,
        default=DEFAULT_WORKFLOW_CONFIG,
    )
    parser.add_argument("--da3-env", default="da3")
    parser.add_argument("--gsam2-env", default="gsam2")
    parser.add_argument(
        "--include-runs",
        default="",
        help="Comma-separated exact sequence names",
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="Zero-based shard index for this independently GPU-bound worker",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of independently launched workers sharing this manifest",
    )
    parser.add_argument(
        "--worker-id",
        default="worker-0",
        help="Filesystem-safe label for persistent status logs",
    )
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--list-runs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument(
        "--keep-intermediate",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--keep-work",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    supplied = sys.argv[1:] if argv is None else argv
    explicit = {
        token.split("=", 1)[0]
        for token in supplied
        if token.startswith("--")
    }
    if "--paths-config" in explicit:
        configured = parse_simple_paths(args.paths_config.expanduser().resolve())
        if "--source-data-root" not in explicit and "data_root" in configured:
            args.source_data_root = configured["data_root"]
        if "--output-root" not in explicit and "output_root" in configured:
            args.output_root = configured["output_root"]
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must be in [0, --num-workers)")
    if (
        not args.worker_id
        or Path(args.worker_id).name != args.worker_id
        or not re.fullmatch(r"[A-Za-z0-9._-]+", args.worker_id)
    ):
        parser.error("--worker-id must be one safe path component")
    return args


def parse_sequence_name(sequence: str) -> dict[str, Any]:
    match = RUN_PATTERN.fullmatch(sequence)
    if match is None:
        raise ValueError(f"Unsupported sequence name: {sequence!r}")
    run_number = int(match.group("run"))
    relative_path = (
        f"{match.group('floor')}/{match.group('date')}/run_{run_number}"
    )
    return {
        "sequence": sequence,
        "run_name": sequence,
        "floor": match.group("floor"),
        "date": match.group("date"),
        "run": run_number,
        "run_number": run_number,
        "relative_path": relative_path,
    }


def _normalize_json_run(raw: dict[str, Any]) -> dict[str, Any]:
    relative = str(raw.get("relative_path", "")).strip("/")
    sequence = str(raw.get("run_name") or raw.get("sequence") or "").strip()
    if not sequence and relative:
        parts = Path(relative).parts
        if len(parts) != 3 or not parts[2].startswith("run_"):
            raise ValueError(f"Unsupported relative_path: {relative!r}")
        sequence = f"{parts[0]}_{parts[1]}_{parts[2]}"
    run = parse_sequence_name(sequence)
    if relative and relative != run["relative_path"]:
        raise ValueError(
            f"Manifest identity mismatch for {sequence}: {relative!r}"
        )
    if str(raw.get("rosbag", "")).strip():
        run["rosbag"] = str(raw["rosbag"]).strip()
    return run


def load_runs(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_runs = payload.get("runs") if isinstance(payload, dict) else None
        if not isinstance(raw_runs, list) or not raw_runs:
            raise RuntimeError(f"No runs found in manifest: {path}")
        runs = []
        for raw in raw_runs:
            if not isinstance(raw, dict):
                raise RuntimeError(f"Invalid run entry in manifest: {raw!r}")
            runs.append(_normalize_json_run(raw))
        return runs

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise RuntimeError(f"Empty run manifest: {path}")
        fields = {
            field.lstrip("#").strip().lower(): field
            for field in reader.fieldnames
        }
        sequence_field = next(
            (
                fields[name]
                for name in ("sequence name", "sequence", "run_name", "run")
                if name in fields
            ),
            None,
        )
        if sequence_field is None:
            raise RuntimeError(
                "Run manifest must contain Sequence Name or sequence"
            )
        rosbag_field = next(
            (
                fields[name]
                for name in ("rosbag", "rosbag_path", "bag")
                if name in fields
            ),
            None,
        )
        runs: list[dict[str, Any]] = []
        for row in reader:
            sequence = str(row.get(sequence_field, "")).strip()
            if not sequence or sequence.startswith("#"):
                continue
            run = parse_sequence_name(sequence)
            if rosbag_field and str(row.get(rosbag_field, "")).strip():
                run["rosbag"] = str(row[rosbag_field]).strip()
            runs.append(run)
    if not runs:
        raise RuntimeError(f"No runs found in manifest: {path}")
    return runs

def run_relative_path(run: dict[str, Any]) -> Path:
    if str(run.get("relative_path", "")).strip("/"):
        return Path(str(run["relative_path"]).strip("/"))
    return (
        Path(str(run["floor"]))
        / str(run["date"])
        / f"run_{int(run['run_number'])}"
    )


def resolve_rosbag(run: dict[str, Any], source_root: Path) -> Path:
    explicit = run.get("rosbag")
    if explicit:
        path = Path(str(explicit)).expanduser()
        if not path.is_absolute():
            path = source_root / path
        if path.is_file():
            return path.resolve()
    relative = run_relative_path(run)
    candidates = (
        source_root / relative / "rosbag" / "rosbag.db3",
        source_root / relative / "rosbag.db3",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "No local ROS bag for "
        f"{run['sequence']}; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def output_directory(run: dict[str, Any], output_root: Path) -> Path:
    return output_root / run_relative_path(run) / "reconstruction"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    return {
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _identity_matches(path: Path, value: Any) -> bool:
    return (
        isinstance(value, dict)
        and type(value.get("size_bytes")) is int
        and value["size_bytes"] > 0
        and isinstance(value.get("sha256"), str)
        and len(value["sha256"]) == 64
        and file_identity(path) == {
            "size_bytes": value["size_bytes"],
            "sha256": value["sha256"],
        }
    )


def completed_output(
    path: Path,
    *,
    expected_relative_path: str | None = None,
    workflow_config: Path | None = None,
) -> bool:
    if any(
        not (path / name).is_file() or (path / name).stat().st_size <= 0
        for name in REQUIRED_OUTPUTS
    ):
        return False
    try:
        manifest = json.loads(
            (path / "workflow_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict) or not manifest.get("final_outputs"):
        return False
    if (
        expected_relative_path
        and manifest.get("relative_path") != expected_relative_path.strip("/")
    ):
        return False
    provenance = manifest.get("final_output_provenance")
    if not isinstance(provenance, dict):
        return False
    for name in REQUIRED_OUTPUTS[:-1]:
        if not _identity_matches(path / name, provenance.get(name)):
            return False
    if workflow_config is not None:
        workflow_config = workflow_config.expanduser().resolve()
        if not workflow_config.is_file() or not _identity_matches(
            workflow_config, manifest.get("workflow_config_provenance")
        ):
            return False
    return True


def validate_completed_run(
    path: Path,
    expected_relative_path: str | None = None,
    workflow_config: Path | None = None,
) -> list[str]:
    """Compatibility API used by the PanoVGGT workflow."""
    return [] if completed_output(
        path,
        expected_relative_path=expected_relative_path,
        workflow_config=workflow_config,
    ) else ["output contract or content provenance is incomplete"]


def write_camera_centers_ply(poses_txt: Path, output_ply: Path) -> None:
    centers: list[tuple[float, float, float]] = []
    for line in poses_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(values) == 16:
            centers.append((values[3], values[7], values[11]))
    if not centers:
        raise RuntimeError(f"No valid camera centers in {poses_txt}")
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(centers)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]
    for index, (x, y, z) in enumerate(centers):
        fraction = index / max(1, len(centers) - 1)
        lines.append(
            f"{x:.8f} {y:.8f} {z:.8f} 30 "
            f"{int(80 + 175 * fraction)} {int(255 - 175 * fraction)}"
        )
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_ply.with_name(
        f".{output_ply.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write("\n".join(lines) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output_ply)
    finally:
        temporary.unlink(missing_ok=True)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_paths_config(path: Path) -> dict[str, Path]:
    configured = parse_simple_paths(path)
    return {
        "data_root": configured.get("data_root", REPO / "data"),
        "scratch_root": configured.get("scratch_root", REPO / ".runtime/scratch"),
        "output_root": configured.get("output_root", REPO / "outputs/da3"),
    }


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return load_runs(path)


def load_runs_from_csv(
    path: Path, include_floors: str = "", exclude_floors: str = ""
) -> list[dict[str, Any]]:
    include = {item.strip() for item in include_floors.split(",") if item.strip()}
    exclude = {item.strip() for item in exclude_floors.split(",") if item.strip()}
    return [
        run
        for run in load_runs(path)
        if (not include or str(run["floor"]) in include)
        and str(run["floor"]) not in exclude
    ]


def find_rosbag(relative: str, data_root: Path, scratch_root: Path) -> Path | None:
    for root in (data_root, scratch_root):
        for suffix in (Path("rosbag/rosbag.db3"), Path("rosbag.db3")):
            candidate = root / relative.strip("/") / suffix
            if candidate.is_file():
                return candidate.resolve()
    return None

def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_command(
    args: argparse.Namespace,
    run: dict[str, Any],
    bag: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(SINGLE_RUNNER),
        str(bag),
        "--output-dir",
        str(output),
        "--workflow-config",
        str(args.workflow_config.expanduser().resolve()),
        "--da3-env",
        args.da3_env,
        "--gsam2-env",
        args.gsam2_env,
        "--no-delete-rosbag-on-success",
        "--relative-path",
        str(run_relative_path(run)),
        "--keep-intermediate"
        if args.keep_intermediate
        else "--no-keep-intermediate",
        "--keep-work" if args.keep_work else "--no-keep-work",
    ]


def main() -> int:
    args = parse_args()
    source_root = args.source_data_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    runs = load_runs(args.run_manifest.expanduser().resolve())
    requested = {
        value.strip()
        for value in args.include_runs.split(",")
        if value.strip()
    }
    if requested:
        available = {str(run["sequence"]) for run in runs}
        missing = sorted(requested - available)
        if missing:
            raise KeyError(f"Runs not present in manifest: {missing}")
        runs = [run for run in runs if run["sequence"] in requested]
    runs = [
        run
        for index, run in enumerate(runs)
        if index % args.num_workers == args.worker_index
    ]
    if args.stop_after:
        runs = runs[: args.stop_after]

    status_path = (
        output_root / "_logs" / args.worker_id / "batch_status.jsonl"
    )
    print(f"[worker] {args.worker_index + 1}/{args.num_workers}")
    print(
        "[visible-gpus] "
        + os.environ.get("CUDA_VISIBLE_DEVICES", "<inherited>")
    )
    print(f"[runs] {len(runs)}")
    failed = 0
    for index, run in enumerate(runs, 1):
        output = output_directory(run, output_root)
        state: dict[str, Any] = {
            "time_unix": time.time(),
            "sequence": run["sequence"],
            "worker_index": args.worker_index,
            "output_dir": str(output),
        }
        if completed_output(
            output,
            expected_relative_path=str(run_relative_path(run)),
            workflow_config=args.workflow_config,
        ) and not args.rerun_existing:
            state["status"] = "skipped_verified_complete"
            append_jsonl(status_path, state)
            print(f"[{index}/{len(runs)}] skip {run['sequence']}")
            continue
        try:
            bag = resolve_rosbag(run, source_root)
            command = build_command(args, run, bag, output)
            print(f"[{index}/{len(runs)}] {run['sequence']}")
            print("[command] " + shlex.join(command))
            if args.list_runs or args.dry_run:
                state["status"] = (
                    "listed" if args.list_runs else "dry_run"
                )
                state["rosbag"] = str(bag)
                append_jsonl(status_path, state)
                continue
            output.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(command, cwd=REPO, check=False)
            if result.returncode != 0:
                raise RuntimeError(
                    f"single-run workflow exited with {result.returncode}"
                )
            if not completed_output(
                output,
                expected_relative_path=str(run_relative_path(run)),
                workflow_config=args.workflow_config,
            ):
                raise RuntimeError(
                    "workflow returned successfully without a verified output"
                )
            state["status"] = "complete"
        except Exception as error:
            failed += 1
            state["status"] = "failed"
            state["error"] = f"{type(error).__name__}: {error}"
            print(
                f"[failed] {run['sequence']}: {error}",
                file=sys.stderr,
            )
        append_jsonl(status_path, state)
    print(f"[batch-finished] runs={len(runs)} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
