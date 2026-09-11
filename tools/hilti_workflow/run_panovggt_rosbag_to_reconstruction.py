#!/usr/bin/env python3
"""Run the frozen HILTI PanoVGGT protocol from one ROS2 bag.

The method reconstruction never consumes GT. Every expensive stage is
restartable, the original bag is retained by default, and the published output
is the cleaned reconstruction with its reconstruction-space camera poses.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
HILTI_REPO = REPO / "hilti-trimble-slam-challenge-2026"
GSAM2_REPO = REPO / "Grounded-SAM-2"
KALIB = HILTI_REPO / "config/hilti_openvins/kalibr_imucam_chain.yaml"
DEFAULT_PANO_REPO = REPO / ".external/PanoVGGT"
DEFAULT_PANO_CHECKPOINT = DEFAULT_PANO_REPO / "checkpoints/model.pt"
DEFAULT_DEVICE_MASK = REPO / "device_mask_final.png"
HF_GROUNDING_REVISION = "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
# Keep the yaw-45 ring that gives GroundingDINO favourable horizontal views,
# and add the two gravity-defined polar faces. PanoVGGT consumes the entire
# levelled ERP, so its semantic exclusion mask must cover the entire ERP too;
# the old yaw-only ring permanently missed both polar caps (including the
# camera carrier below the horizon).
PANO_POSES = ";".join(
    [*(f"{yaw},0,0" for yaw in range(0, 360, 45)), "0,90,0", "0,-90,0"]
)
PANO_VIEW_COUNT = len(PANO_POSES.split(";"))
DEFAULT_MASK_PROMPT = "person. human. worker."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rosbag", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--relative-path", required=True, help="floor/date/run_N")
    parser.add_argument("--panovggt-repo", type=Path, default=DEFAULT_PANO_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_PANO_CHECKPOINT)
    parser.add_argument("--device-mask", type=Path, default=DEFAULT_DEVICE_MASK)
    parser.add_argument("--da3-env", default="da3")
    parser.add_argument("--gsam2-env", default="gsam2")
    parser.add_argument(
        "--mask-prompt",
        default=DEFAULT_MASK_PROMPT,
        help="Grounding prompt for semantic exclusion masks; changing it defines an ablation.",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--keep-work-on-success",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retain restartable work products after success (default: true).",
    )
    parser.add_argument(
        "--no-delete-rosbag-on-success",
        dest="delete_rosbag_on_success",
        action="store_false",
        default=False,
        help="Record the publication workflow's mandatory input-retention policy.",
    )
    return parser.parse_args()


def conda_python(environment: str, script: Path, *args: object) -> list[str]:
    return [
        "conda", "run", "--no-capture-output", "-n", environment,
        "python", str(script), *(str(value) for value in args),
    ]


def run(command: list[str], log: Path, cwd: Path = REPO) -> float:
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\n$ " + " ".join(command) + "\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            stream.write(line)
        returncode = process.wait()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
    return time.monotonic() - started


def image_count(path: Path) -> int:
    suffixes = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(item.is_file() and item.suffix.lower() in suffixes for item in path.glob("*"))


def write_checkpoint(
    path: Path,
    relative_path: str,
    status: str,
    stages: list[str],
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "updated_at_unix": time.time(),
        "relative_path": relative_path,
        "status": status,
        "completed_stages": stages,
    }
    if error:
        payload["error"] = error
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_run_receipt(work: Path, request: dict[str, Any]) -> Path:
    receipt = work / "run_request.json"
    if receipt.is_file():
        try:
            recorded = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid PanoVGGT run receipt: {receipt}") from error
        if recorded != request:
            raise RuntimeError(
                "PanoVGGT work directory belongs to different inputs or "
                "settings; use a new --work-dir"
            )
    elif any(work.iterdir()):
        raise RuntimeError(
            "PanoVGGT work directory has no verifiable run receipt; use a "
            "new --work-dir instead of reusing legacy caches"
        )
    else:
        atomic_write_json(receipt, request)
    return receipt


def require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {label}: {path}")


def sequential_inference_ready(directory: Path) -> bool:
    """Return true only for a complete, explicitly sequential PanoVGGT export."""
    reconstruction = directory / "reconstruction.ply"
    manifest = directory / "workflow_manifest.json"
    if not reconstruction.is_file() or not manifest.is_file():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("outputs", {}).get("selected_variant") == "sequential"


def copy_file(source: Path, destination: Path) -> None:
    require_file(source, source.name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    from tools.hilti_workflow.run_hilti_batch import (
        file_identity,
        validate_completed_run,
        write_camera_centers_ply,
    )

    relative_path = args.relative_path.strip("/")
    if len(Path(relative_path).parts) != 3:
        raise ValueError("--relative-path must be floor/date/run_N")
    output = args.output_dir.expanduser().resolve()
    work = args.work_dir.expanduser().resolve()
    logs = output / "logs"
    checkpoint = work / "run_checkpoint.json"
    final_checkpoint = logs / "run_checkpoint.json"
    existing = validate_completed_run(output, relative_path)
    if not existing:
        print(f"[skip-verified] {relative_path}: {output}", flush=True)
        return 0
    if (output / "reconstruction.ply").exists():
        raise RuntimeError(
            f"Refusing to overwrite an incomplete reconstruction: {output}; problems={existing}"
        )
    for path, label in (
        (args.rosbag, "ROS bag"),
        (args.panovggt_repo / "training/config/default.yaml", "PanoVGGT config"),
        (args.checkpoint, "PanoVGGT checkpoint"),
        (args.device_mask, "static device mask"),
        (KALIB, "camera/IMU calibration"),
    ):
        require_file(path, label)

    if work.exists() and not args.resume:
        raise RuntimeError(f"Work directory exists but --no-resume was requested: {work}")
    work.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    run_request = {
        "schema_version": 1,
        "inputs": {
            "rosbag": file_identity(args.rosbag),
            "kalibr_yaml": file_identity(KALIB),
            "panovggt_config": file_identity(
                args.panovggt_repo / "training/config/default.yaml"
            ),
            "checkpoint": file_identity(args.checkpoint),
            "device_mask": file_identity(args.device_mask),
            "workflow": file_identity(Path(__file__)),
        },
        "settings": {
            "extract_stride": 10,
            "erp_resolution": [1036, 518],
            "imu_method": "complementary",
            "imu_tau": 2.0,
            "accel_gate_sigma": 0.2,
            "semantic_poses": PANO_POSES,
            "semantic_view_resolution": [768, 768],
            "semantic_view_fov_deg": 110,
            "mask_prompt": args.mask_prompt,
            "mask_thresholds": {"box": 0.18, "text": 0.18},
            "mask_close_pixels": 5,
            "mask_dilate_pixels": 15,
            "chunk_size": 24,
            "stride": 12,
            "points_per_frame": 40000,
            "cleanup_voxel_size": 0.05,
            "cleanup_min_points_per_voxel": 2,
            "cleanup_min_occupied_neighbor_voxels": 3,
            "cleanup_neighbor_connectivity": 26,
        },
    }
    ensure_run_receipt(work, run_request)
    stages: list[str] = []
    if checkpoint.is_file():
        try:
            previous = json.loads(checkpoint.read_text(encoding="utf-8"))
            if previous.get("relative_path") == relative_path:
                stages = list(previous.get("completed_stages", []))
        except (OSError, json.JSONDecodeError, TypeError):
            stages = []
    write_checkpoint(checkpoint, relative_path, "running", stages)

    intermediate = work / "intermediate"
    raw_erp = intermediate / "raw_erp"
    levelled_erp = intermediate / "levelled_erp"
    views = intermediate / "semantic_views"
    semantic = intermediate / "semantic_masks"
    semantic_packed = intermediate / "semantic_erp_masks.npz"
    device = intermediate / "device_masks"
    union = intermediate / "union_erp_masks.npz"
    inference = work / "panovggt_inference"
    cleaned = work / "reconstruction_cleaned.ply"
    stage_times: dict[str, float] = {}

    def complete(name: str) -> None:
        if name not in stages:
            stages.append(name)
        write_checkpoint(checkpoint, relative_path, "running", stages)

    try:
        if image_count(raw_erp) == 0:
            stage_times["extract_erp"] = run(
                conda_python(
                    args.da3_env,
                    REPO / "tools/hilti_workflow/extraction/extract_hilti_frames.py",
                    "--bag", args.rosbag, "--yaml", KALIB, "--out_dir", raw_erp,
                    "--mask0", "", "--mask1", "", "--stride", 10,
                ),
                logs / "01_extract_erp.log",
            )
        complete("extract_erp")
        frame_count = image_count(raw_erp)
        if frame_count < 2:
            raise RuntimeError(f"ERP extraction produced only {frame_count} frames")

        if image_count(levelled_erp) != frame_count:
            stage_times["level_erp"] = run(
                conda_python(
                    args.da3_env,
                    REPO / "tools/hilti_workflow/view_generation/imu_level_equirect_to_equirect.py",
                    "--bag", args.rosbag, "--input-dir", raw_erp, "--yaml", KALIB,
                    "--output-dir", levelled_erp, "--width", 1036, "--height", 518,
                    "--imu-tau", 2.0, "--imu-method", "complementary",
                    "--accel-gate-sigma", 0.2, "--rotate180", "--quality", 95,
                ),
                logs / "02_level_erp.log",
            )
        complete("level_erp")

        if image_count(views) != frame_count * PANO_VIEW_COUNT:
            stage_times["render_semantic_views"] = run(
                conda_python(
                    args.da3_env,
                    REPO / "tools/hilti_workflow/view_generation/equirect_pose_sequence_dir.py",
                    "--input_dir", levelled_erp, "--output_dir", views,
                    "--width", 768, "--height", 768, "--fov_deg", 110,
                    "--poses", PANO_POSES,
                ),
                logs / "03_render_semantic_views.log",
            )
        complete("render_semantic_views")

        mask_count = len(list((semantic / "masks_npy").glob("*.npy")))
        if mask_count != frame_count * PANO_VIEW_COUNT:
            stage_times["gsam2_masks"] = run(
                conda_python(
                    args.gsam2_env,
                    GSAM2_REPO / "tools/hilti_person_mask_batch.py",
                    "--input-dir", views, "--output-dir", semantic,
                    "--prompt", args.mask_prompt, "--grounding-backend", "hf",
                    "--hf-grounding-model", "IDEA-Research/grounding-dino-base",
                    "--hf-grounding-revision", HF_GROUNDING_REVISION,
                    "--box-threshold", 0.18, "--text-threshold", 0.18,
                    "--mask-close-pixels", 5, "--mask-dilate-pixels", 15,
                    "--grounding-batch-size", 2, "--export-mode", "npy",
                    "--skip-existing",
                ),
                logs / "04_gsam2_masks.log",
                GSAM2_REPO,
            )
        complete("gsam2_masks")

        if not semantic_packed.is_file():
            stage_times["backproject_masks"] = run(
                conda_python(
                    args.da3_env,
                    REPO / "tools/hilti_workflow/panovggt/backproject_masks.py",
                    "--erp-dir", levelled_erp, "--mask-dir", semantic / "masks_npy",
                    "--output", semantic_packed, "--report", intermediate / "semantic_erp_masks.json",
                    "--fov", 110, "--view-width", 768, "--view-height", 768,
                    "--min-view-votes", 1, "--max-view-mask-fraction", 0.85,
                    "--mapping", "inverse-gather",
                ),
                logs / "05_backproject_masks.log",
            )
        complete("backproject_masks")

        device_manifest = device / "manifest.json"
        if len(list((device / "masks_npy").glob("*.npy"))) != frame_count:
            stage_times["device_mask"] = run(
                conda_python(
                    args.da3_env,
                    REPO / "tools/hilti_workflow/panovggt/level_device_mask.py",
                    "--bag", args.rosbag, "--raw-erp-dir", raw_erp,
                    "--levelled-erp-dir", levelled_erp, "--yaml", KALIB,
                    "--static-mask", args.device_mask, "--output-dir", device,
                    "--manifest", device_manifest, "--imu-tau", 2.0,
                    "--accel-gate-sigma", 0.2, "--dilate-pixels", 3,
                ),
                logs / "06_device_mask.log",
            )
        complete("device_mask")

        if not union.is_file():
            stage_times["union_masks"] = run(
                conda_python(
                    args.da3_env,
                    REPO / "tools/hilti_workflow/panovggt/union_packed_masks.py",
                    "--primary-packed", semantic_packed,
                    "--secondary-npy-dir", device / "masks_npy",
                    "--output", union, "--report", intermediate / "union_erp_masks.json",
                ),
                logs / "07_union_masks.log",
            )
        complete("union_masks")

        inference_core = inference / "reconstruction.ply"
        if not sequential_inference_ready(inference):
            command = conda_python(
                args.da3_env,
                REPO / "tools/hilti_workflow/panovggt/run_reconstruction.py",
                "--repo", args.panovggt_repo, "--checkpoint", args.checkpoint,
                "--image-dir", levelled_erp, "--erp-mask-npz", union,
                "--output-dir", inference, "--chunk-size", 24, "--stride", 12,
                "--points-per-frame", 40000,
                "--window-checkpoint-dir", work / "window_checkpoints",
            )
            stage_times["panovggt"] = run(command, logs / "08_panovggt.log")
        complete("panovggt")

        if not cleaned.is_file():
            stage_times["clean_pointcloud"] = run(
                conda_python(
                    args.da3_env,
                    REPO / "tools/hilti_workflow/postprocess/clean_pointcloud_outliers.py",
                    "--input", inference_core, "--output", cleaned,
                    "--voxel-size", 0.05, "--min-points-per-voxel", 2,
                    "--min-occupied-neighbor-voxels", 3,
                    "--neighbor-connectivity", 26,
                ),
                logs / "09_clean_pointcloud.log",
            )
        complete("clean_pointcloud")

        selected_cloud = cleaned
        selected_poses = inference / "camera_poses.txt"

        copy_file(selected_cloud, output / "reconstruction.ply")
        copy_file(selected_poses, output / "camera_poses.txt")
        write_camera_centers_ply(output / "camera_poses.txt", output / "camera_poses.ply")
        copy_file(inference / "workflow_manifest.json", logs / "panovggt_inference_manifest.json")
        audit_artifacts: dict[str, str] = {}
        for source in (
            inference / "sequential_alignment.csv",
            inference / "chunk_sim3_sequential.npz",
        ):
            if source.is_file():
                destination = logs / source.name
                copy_file(source, destination)
                audit_artifacts[source.stem] = str(destination.relative_to(output))
        for source in (
            intermediate / "semantic_erp_masks.json",
            intermediate / "union_erp_masks.json",
            device_manifest,
            cleaned.with_suffix(".json"),
        ):
            if source.is_file():
                copy_file(source, logs / source.name)

        manifest = {
            "method": "panovggt",
            "relative_path": relative_path,
            "created_at_unix": time.time(),
            "selected_variant": "cleaned_sequential",
            "final_outputs": {
                "pointcloud": str(output / "reconstruction.ply"),
                "camera_poses": str(output / "camera_poses.txt"),
                "camera_pose_ply": str(output / "camera_poses.ply"),
            },
            "final_output_provenance": {
                "reconstruction.ply": file_identity(
                    output / "reconstruction.ply"
                ),
                "camera_poses.txt": file_identity(output / "camera_poses.txt"),
                "camera_poses.ply": file_identity(output / "camera_poses.ply"),
            },
            "input_provenance": dict(run_request["inputs"]),
            "run_request": run_request,
            "stage_timings_s": stage_times,
            "audit_artifacts": audit_artifacts,
            "settings": {
                "extract_stride": 10,
                "erp_resolution": [1036, 518],
                "imu_method": "complementary",
                "imu_tau": 2.0,
                "accel_gate_sigma": 0.2,
                "yaw_views": list(range(0, 360, 45)),
                "view_resolution": [768, 768],
                "view_fov_deg": 110,
                "mask_prompt": args.mask_prompt,
                "mask_thresholds": {"box": 0.18, "text": 0.18},
                "mask_close_pixels": 5,
                "mask_dilate_pixels": 15,
                "chunk_size": 24,
                "stride": 12,
                "points_per_frame": 40000,
                "delete_rosbag_on_success": args.delete_rosbag_on_success,
                "keep_work_on_success": args.keep_work_on_success,
                "gt_used": False,
            },
        }
        atomic_write_json(output / "workflow_manifest.json", manifest)
        problems = validate_completed_run(output, relative_path)
        if problems:
            raise RuntimeError(f"Final output validation failed: {problems}")
        complete("publish_verified_output")
        write_checkpoint(checkpoint, relative_path, "complete", stages)
        copy_file(checkpoint, final_checkpoint)

        if not args.keep_work_on_success:
            shutil.rmtree(work)
        print(f"[done] {relative_path} -> {output}", flush=True)
        return 0
    except BaseException as exc:
        write_checkpoint(checkpoint, relative_path, "failed", stages, f"{type(exc).__name__}: {exc}")
        if checkpoint.is_file():
            copy_file(checkpoint, final_checkpoint)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
