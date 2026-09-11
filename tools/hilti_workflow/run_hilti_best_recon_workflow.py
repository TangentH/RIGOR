#!/usr/bin/env python3
"""Run the paper DA3 reconstruction from prepared perspective views.

This compatibility entry point implements only the public paper pipeline:
materialize the frozen DA3 configuration, run DA3, remove isolated 5 cm
occupancy support, and publish a content-addressed manifest. Evaluation
alignment and visualization are separate tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORKFLOW_CONFIG = (
    Path(__file__).resolve().parent
    / "configs"
    / "rigor_paper_free_scale.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir", required=True, type=Path, help="Prepared perspective views"
    )
    parser.add_argument(
        "--confidence-zero-mask-dir",
        required=True,
        type=Path,
        help="One confidence-suppression .npy mask per perspective view",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--workflow-config", type=Path, default=DEFAULT_WORKFLOW_CONFIG
    )
    parser.add_argument(
        "--skip-da3",
        action="store_true",
        help="Reuse an exact, content-addressed DA3 stage",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Regenerate configuration files. Existing reconstructions remain "
            "protected and require an exact provenance match."
        ),
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO / path


def run(
    command: list[str],
    *,
    cwd: Path | None = None,
    log: Path | None = None,
    env: dict[str, str] | None = None,
) -> float:
    started_at = time.perf_counter()
    printable = " ".join(str(part) for part in command)
    print(f"\n[run] {printable}", flush=True)
    if log is None:
        subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            check=True,
        )
        return time.perf_counter() - started_at

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\n\n$ {printable}\n")
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd) if cwd else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            stream.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    return time.perf_counter() - started_at


def deep_set(data: dict[str, Any], keys: list[str], value: Any) -> None:
    current = data
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def build_da3_config(
    workflow: dict[str, Any],
    image_dir: Path,
    mask_dir: Path,
    output_dir: Path,
) -> Path:
    """Materialize the frozen workflow as a DA3 streaming configuration."""
    del image_dir  # Images are supplied to DA3 on its command line.
    reconstruction = workflow["reconstruction"]
    config = load_yaml(repo_path(reconstruction["base_da3_config"]))

    model_values = {
        "chunk_size": int(reconstruction["chunk_size"]),
        "overlap": int(reconstruction["overlap"]),
        "loop_chunk_size": int(reconstruction["loop_chunk_size"]),
        "loop_enable": bool(reconstruction["loop_enable"]),
        "delete_temp_files": bool(
            reconstruction.get("delete_temp_files", False)
        ),
        "reuse_existing_unaligned_chunks": bool(
            reconstruction.get("reuse_existing_unaligned_chunks", False)
        )
        or os.environ.get("DA3_REUSE_EXISTING_UNALIGNED_CHUNKS") == "1",
        "skip_aligned_cache_write": bool(
            reconstruction.get("skip_aligned_cache_write", False)
        )
        or os.environ.get("DA3_SKIP_ALIGNED_CACHE_WRITE") == "1",
        "confidence_zero_mask_dir": str(mask_dir),
        "confidence_zero_mask_strict": True,
    }
    if reconstruction.get("process_res") is not None:
        model_values["process_res"] = int(reconstruction["process_res"])
    if reconstruction.get("process_res_method"):
        model_values["process_res_method"] = str(
            reconstruction["process_res_method"]
        )
    for key, value in model_values.items():
        deep_set(config, ["Model", key], value)

    deep_set(
        config,
        ["Model", "Pointcloud_Save", "sample_ratio"],
        float(reconstruction["sample_ratio"]),
    )
    deep_set(
        config,
        ["Model", "Pointcloud_Save", "conf_threshold_coef"],
        float(reconstruction["conf_threshold_coef"]),
    )
    deep_set(
        config,
        ["Model", "Pointcloud_Save", "canonical_overlap_ownership"],
        bool(reconstruction.get("canonical_overlap_ownership", False)),
    )

    single_view_threshold = reconstruction.get(
        "loop_single_view_similarity_threshold",
        reconstruction.get("loop_similarity_threshold"),
    )
    if single_view_threshold is None:
        raise KeyError(
            "reconstruction.loop_single_view_similarity_threshold is required"
        )
    deep_set(
        config,
        ["Loop", "SALAD", "min_frame_gap"],
        int(reconstruction["loop_min_frame_gap"]),
    )
    deep_set(
        config,
        ["Loop", "event_nms_chunk_radius"],
        int(reconstruction.get("loop_event_nms_chunk_radius", 0)),
    )
    deep_set(
        config,
        ["Loop", "SALAD", "single_view_similarity_threshold"],
        float(single_view_threshold),
    )
    deep_set(
        config,
        ["Loop", "SALAD", "top_k"],
        int(reconstruction["loop_top_k"]),
    )
    candidate_mode = str(reconstruction.get("loop_candidate_mode", "image"))
    deep_set(config, ["Loop", "SALAD", "candidate_mode"], candidate_mode)
    if candidate_mode == "capture_cyclic":
        cyclic = reconstruction.get("loop_capture_cyclic_consensus", {})
        deep_set(config, ["Loop", "SALAD", "min_gap_unit"], "capture")
        deep_set(config, ["Loop", "SALAD", "nms_unit"], "capture")
        for field, default, cast in (
            ("mean_similarity_threshold", 0.65, float),
            ("min_view_similarity", 0.45, float),
            ("support_similarity", 0.50, float),
            ("min_support_views", 4, int),
        ):
            deep_set(
                config,
                ["Loop", "SALAD", "capture_cyclic_consensus", field],
                cast(cyclic.get(field, default)),
            )

    deep_set(
        config,
        ["Loop", "SIM3_Optimizer", "fix_scale"],
        bool(reconstruction.get("loop_pose_graph_fix_scale", False)),
    )

    geometry = reconstruction.get("loop_geometry_verification", {})
    for field, default, cast in (
        ("enabled", False, bool),
        ("max_side_alignment_error", 0.25, float),
        ("min_scale", 0.8, float),
        ("max_scale", 1.25, float),
        ("min_rig_acceptance", 0.8, float),
    ):
        deep_set(
            config,
            ["Loop", "GeometryVerification", field],
            cast(geometry.get(field, default)),
        )
    cross_side = geometry.get("cross_side_dense", {})
    for field, default, cast in (
        ("enabled", False, bool),
        ("record_diagnostics", False, bool),
        ("confidence_quantile", 0.75, float),
        ("min_confidence", 0.0, float),
        ("voxel_size", 0.10, float),
        ("max_points_per_side", 30_000, int),
        ("preselection_multiplier", 8, int),
        ("trim_quantile", 0.90, float),
        ("overlap_distance", 0.25, float),
        ("min_points_per_side", 100, int),
        ("min_mutual_count", 100, int),
        ("max_mutual_trimmed_rmse", 0.25, float),
        ("max_mutual_median", 0.15, float),
        ("min_overlap_a_to_b", 0.35, float),
        ("min_overlap_b_to_a", 0.35, float),
    ):
        deep_set(
            config,
            ["Loop", "GeometryVerification", "cross_side_dense", field],
            cast(cross_side.get(field, default)),
        )

    pose_filter = reconstruction.get("yaw4_pose_filter", {})
    for field, default, cast in (
        ("enabled", True, bool),
        ("yaw_group_size", 4, int),
        ("max_frame_group_dist", 0.10, float),
        ("max_median_group_max", 0.10, float),
        ("rig_repair_before_alignment", False, bool),
        ("rig_repair_after_alignment", False, bool),
        ("rig_paired_baseline_export", False, bool),
        ("rig_repair_loop_predictions", False, bool),
        ("rig_rotation_threshold_deg", 5.0, float),
        ("rig_min_inliers", 3, int),
        ("rig_validate_overlap", True, bool),
        ("rig_overlap_pixel_stride", 4, int),
        ("rig_overlap_ray_tolerance_pixels", 1.5, float),
        ("rig_overlap_confidence_quantile", 0.25, float),
        ("rig_overlap_max_depth", 15.0, float),
        ("rig_overlap_min_matched_rays", 100, int),
        ("rig_overlap_max_median_ratio", 1.0, float),
        ("rig_overlap_max_p90_ratio", 1.0, float),
        ("rig_reject_unrecoverable_group", False, bool),
        ("rig_check_orientation", True, bool),
        ("rig_outlier_action", "repair", str),
    ):
        deep_set(
            config,
            ["Model", "Yaw4_Pose_Filter", field],
            cast(pose_filter.get(field, default)),
        )
    center_depth_ratio = pose_filter.get("rig_center_depth_ratio")
    deep_set(
        config,
        ["Model", "Yaw4_Pose_Filter", "rig_center_depth_ratio"],
        float(center_depth_ratio) if center_depth_ratio is not None else None,
    )

    generated = output_dir / "configs" / "da3_streaming_generated.yaml"
    save_yaml(generated, config)
    save_yaml(
        output_dir / "configs" / "workflow_config_used.yaml",
        workflow,
    )
    return generated


def start_gpu_logger(
    output_dir: Path, interval_s: int
) -> subprocess.Popen | None:
    if shutil.which("nvidia-smi") is None:
        return None
    gpu_log = output_dir / "logs" / "gpu_memory_log.csv"
    gpu_log.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv",
            "-l",
            str(interval_s),
            "-f",
            str(gpu_log),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_provenance(
    path: Path, *, point_count: int | None = None
) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    provenance: dict[str, Any] = {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }
    if point_count is not None:
        provenance["point_count"] = int(point_count)
    return provenance


def provenance_matches_file(provenance: Any, path: Path) -> bool:
    if not isinstance(provenance, dict) or not path.is_file():
        return False
    try:
        return (
            int(provenance["size_bytes"]) == path.stat().st_size
            and str(provenance["sha256"]) == sha256_file(path)
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def directory_content_signature(
    directory: Path,
    suffixes: set[str],
) -> dict[str, Any]:
    """Hash a flat input set by relative name, size, and content."""
    resolved = directory.expanduser().resolve()
    paths = sorted(
        path
        for path in resolved.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )
    if not paths:
        raise RuntimeError(f"No provenance inputs found in {resolved}")
    entries: list[dict[str, Any]] = []
    for path in paths:
        before = path.stat()
        digest = sha256_file(path)
        after = path.stat()
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise RuntimeError(f"Input changed while hashing: {path}")
        if after.st_size <= 0:
            raise RuntimeError(f"Provenance input is empty: {path}")
        entries.append(
            {
                "name": path.relative_to(resolved).as_posix(),
                "size_bytes": int(after.st_size),
                "sha256": digest,
            }
        )
    names = [entry["name"] for entry in entries]
    return {
        "schema_version": 2,
        "root": str(resolved),
        "count": len(entries),
        "total_size_bytes": sum(
            int(entry["size_bytes"]) for entry in entries
        ),
        "all_nonempty": True,
        "first_name": names[0],
        "last_name": names[-1],
        "filename_set_sha256": canonical_json_sha256(names),
        "content_tree_sha256": canonical_json_sha256(entries),
    }


def da3_stage_request(
    args: argparse.Namespace,
    da3_config: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "da3_streaming",
        "image_tree": directory_content_signature(
            args.image_dir, {".jpg", ".jpeg", ".png", ".webp"}
        ),
        "mask_tree": directory_content_signature(
            args.confidence_zero_mask_dir, {".npy"}
        ),
        "workflow_config": file_provenance(args.workflow_config),
        "generated_da3_config": file_provenance(da3_config),
        "implementation": {
            "workflow_driver": file_provenance(Path(__file__).resolve()),
            "da3_entrypoint": file_provenance(
                REPO / "da3_streaming" / "da3_streaming.py"
            ),
        },
    }


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_stage_record(
    path: Path,
    *,
    stage: str,
    status: str,
    request: dict[str, Any],
    outputs: dict[str, Any] | None = None,
) -> None:
    record: dict[str, Any] = {
        "schema_version": 2,
        "stage": stage,
        "status": status,
        "request": request,
        "request_sha256": canonical_json_sha256(request),
        "updated_at_unix": time.time(),
    }
    if outputs is not None:
        record["outputs"] = outputs
        record["completed_at_unix"] = time.time()
    atomic_write_json(path, record)


def completed_stage_is_reusable(
    path: Path,
    *,
    stage: str,
    request: dict[str, Any],
    output_paths: dict[str, Path],
) -> bool:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    recorded_request = (
        record.get("request") if isinstance(record, dict) else None
    )
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != 2
        or record.get("stage") != stage
        or record.get("status") != "complete"
        or not isinstance(recorded_request, dict)
        or recorded_request != request
        or record.get("request_sha256")
        != canonical_json_sha256(request)
    ):
        return False
    outputs = record.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(output_paths):
        return False
    return all(
        provenance_matches_file(outputs.get(name), output_path)
        for name, output_path in output_paths.items()
    )


def load_complete_stage_record(
    path: Path, stage: str
) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Missing or invalid stage provenance: {path}"
        ) from error
    request = record.get("request") if isinstance(record, dict) else None
    outputs = record.get("outputs") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != 2
        or record.get("stage") != stage
        or record.get("status") != "complete"
        or not isinstance(request, dict)
        or record.get("request_sha256")
        != canonical_json_sha256(request)
        or not isinstance(outputs, dict)
        or not outputs
    ):
        raise RuntimeError(
            f"Stage provenance is incomplete or inconsistent: {path}"
        )
    for name, provenance in outputs.items():
        if not isinstance(provenance, dict) or not provenance.get("path"):
            raise RuntimeError(
                f"Stage output provenance is invalid: {stage}/{name}"
            )
        output_path = Path(str(provenance["path"])).expanduser().resolve()
        if not provenance_matches_file(provenance, output_path):
            raise RuntimeError(
                f"Stage output no longer matches provenance: {stage}/{name}"
            )
    return record


def run_da3(
    args: argparse.Namespace,
    workflow: dict[str, Any],
    da3_config: Path,
) -> None:
    raw = args.output_dir / "pcd" / "combined_pcd.ply"
    poses = args.output_dir / "camera_poses.txt"
    provenance_path = (
        args.output_dir / "logs" / "stage_da3_streaming_provenance.json"
    )
    request = da3_stage_request(args, da3_config)
    if raw.exists():
        if completed_stage_is_reusable(
            provenance_path,
            stage="da3_streaming",
            request=request,
            output_paths={"raw_pointcloud": raw, "camera_poses": poses},
        ):
            print(
                f"[skip-da3] content-addressed stage is complete: {raw}",
                flush=True,
            )
            return
        raise RuntimeError(
            "Refusing to reuse or overwrite an existing DA3 point cloud "
            f"without matching content provenance: {raw}. "
            "Use a new output/work root."
        )
    if args.skip_da3:
        raise RuntimeError(
            "--skip-da3 requires a complete content-addressed DA3 stage; "
            f"no reusable output exists at {raw}"
        )
    write_stage_record(
        provenance_path,
        stage="da3_streaming",
        status="running",
        request=request,
    )
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    interval = int(
        workflow.get("runtime", {}).get("gpu_log_interval_s", 5)
    )
    monitor = start_gpu_logger(args.output_dir, interval)
    try:
        run(
            [
                sys.executable,
                "-u",
                "da3_streaming.py",
                "--image_dir",
                str(args.image_dir),
                "--config",
                str(da3_config),
                "--output_dir",
                str(args.output_dir),
            ],
            cwd=REPO / "da3_streaming",
            log=args.output_dir / "logs" / "da3_streaming.log",
            env=environment,
        )
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
    if not raw.is_file() or not poses.is_file():
        raise RuntimeError(
            "DA3 returned successfully without both raw point cloud and "
            "camera poses"
        )
    if da3_stage_request(args, da3_config) != request:
        raise RuntimeError("DA3 inputs changed while the stage was running")
    write_stage_record(
        provenance_path,
        stage="da3_streaming",
        status="complete",
        request=request,
        outputs={
            "raw_pointcloud": file_provenance(raw),
            "camera_poses": file_provenance(poses),
        },
    )


def run_clean_pointcloud(
    output_dir: Path,
    workflow: dict[str, Any],
) -> Path:
    settings = workflow["pointcloud_clean"]
    raw = output_dir / "pcd" / "combined_pcd.ply"
    cleaned = output_dir / "pcd" / "combined_pcd_clean_voxel005.ply"
    if not settings.get("enabled", True):
        return raw
    if not raw.is_file():
        raise FileNotFoundError(
            f"Missing raw point cloud for cleaning: {raw}"
        )
    implementation = (
        REPO
        / "tools"
        / "hilti_workflow"
        / "postprocess"
        / "clean_pointcloud_outliers.py"
    )
    provenance_path = (
        output_dir
        / "logs"
        / "stage_pointcloud_cleaning_provenance.json"
    )
    request = {
        "schema_version": 1,
        "stage": "pointcloud_cleaning",
        "input": file_provenance(raw),
        "settings": dict(settings),
        "implementation": file_provenance(implementation),
    }
    if cleaned.exists():
        if completed_stage_is_reusable(
            provenance_path,
            stage="pointcloud_cleaning",
            request=request,
            output_paths={"cleaned_pointcloud": cleaned},
        ):
            print(
                f"[skip-clean] content-addressed stage is complete: {cleaned}",
                flush=True,
            )
            return cleaned
        raise RuntimeError(
            "Refusing to reuse or overwrite a cleaned point cloud without "
            f"matching content provenance: {cleaned}. "
            "Use a new output/work root."
        )
    write_stage_record(
        provenance_path,
        stage="pointcloud_cleaning",
        status="running",
        request=request,
    )
    command = [
        sys.executable,
        str(implementation),
        "--input",
        str(raw),
        "--output",
        str(cleaned),
        "--voxel-size",
        str(settings["voxel_size"]),
        "--min-points-per-voxel",
        str(settings["min_points_per_voxel"]),
        "--min-occupied-neighbor-voxels",
        str(settings["min_occupied_neighbor_voxels"]),
        "--neighbor-connectivity",
        str(settings["neighbor_connectivity"]),
    ]
    if settings.get("save_removed", False):
        command.append("--save-removed")
    run(command, log=output_dir / "logs" / "postprocess.log")
    if not cleaned.is_file():
        raise RuntimeError(
            f"Point-cloud cleaning did not create {cleaned}"
        )
    current_request = {
        **request,
        "input": file_provenance(raw),
        "implementation": file_provenance(implementation),
    }
    if current_request != request:
        raise RuntimeError(
            "Point-cloud cleaning inputs changed during the stage"
        )
    write_stage_record(
        provenance_path,
        stage="pointcloud_cleaning",
        status="complete",
        request=request,
        outputs={"cleaned_pointcloud": file_provenance(cleaned)},
    )
    return cleaned


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    workflow: dict[str, Any],
    outputs: dict[str, Any],
    stage_timings_s: dict[str, float],
) -> None:
    da3_stage = load_complete_stage_record(
        output_dir
        / "logs"
        / "stage_da3_streaming_provenance.json",
        "da3_streaming",
    )
    generated = Path(str(outputs["generated_da3_config"])).resolve()
    if da3_stage.get("request") != da3_stage_request(args, generated):
        raise RuntimeError(
            "DA3 inputs changed before final manifest publication"
        )

    cleaning_enabled = bool(
        workflow.get("pointcloud_clean", {}).get("enabled", True)
    )
    cleaning_stage = None
    if cleaning_enabled:
        cleaning_stage = load_complete_stage_record(
            output_dir
            / "logs"
            / "stage_pointcloud_cleaning_provenance.json",
            "pointcloud_cleaning",
        )
        settings = workflow["pointcloud_clean"]
        implementation = (
            REPO
            / "tools"
            / "hilti_workflow"
            / "postprocess"
            / "clean_pointcloud_outliers.py"
        )
        current_request = {
            "schema_version": 1,
            "stage": "pointcloud_cleaning",
            "input": file_provenance(
                output_dir / "pcd" / "combined_pcd.ply"
            ),
            "settings": dict(settings),
            "implementation": file_provenance(implementation),
        }
        if cleaning_stage.get("request") != current_request:
            raise RuntimeError(
                "Point-cloud cleaning inputs changed before manifest "
                "publication"
            )

    cleaned = Path(str(outputs["cleaned_pointcloud"])).resolve()
    poses = output_dir / "camera_poses.txt"
    if not cleaned.is_file() or not poses.is_file():
        raise RuntimeError(
            "Cannot publish workflow manifest without cleaned geometry "
            "and poses"
        )
    manifest = {
        "schema_version": 3,
        "created_at_unix": time.time(),
        "image_dir": str(args.image_dir.expanduser().resolve()),
        "confidence_zero_mask_dir": str(
            args.confidence_zero_mask_dir.expanduser().resolve()
        ),
        "workflow_config": str(
            args.workflow_config.expanduser().resolve()
        ),
        "output_dir": str(output_dir.expanduser().resolve()),
        "input_provenance": {
            "image_tree": da3_stage["request"]["image_tree"],
            "mask_tree": da3_stage["request"]["mask_tree"],
            "workflow_config": file_provenance(args.workflow_config),
            "workflow_config_effective": file_provenance(
                output_dir / "configs" / "workflow_config_used.yaml"
            ),
            "generated_da3_config": da3_stage["request"][
                "generated_da3_config"
            ],
        },
        "stage_provenance": {
            "da3_streaming": da3_stage,
            "pointcloud_cleaning": cleaning_stage,
        },
        "selected_output": {
            "pointcloud": file_provenance(cleaned),
            "camera_poses": file_provenance(poses),
        },
        "outputs": outputs,
        "stage_timings_s": stage_timings_s,
    }
    atomic_write_json(output_dir / "workflow_manifest.json", manifest)


def main() -> int:
    args = parse_args()
    args.image_dir = args.image_dir.expanduser().resolve()
    args.confidence_zero_mask_dir = (
        args.confidence_zero_mask_dir.expanduser().resolve()
    )
    args.output_dir = args.output_dir.expanduser().resolve()
    args.workflow_config = args.workflow_config.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workflow = load_yaml(args.workflow_config)
    generated = build_da3_config(
        workflow,
        args.image_dir,
        args.confidence_zero_mask_dir,
        args.output_dir,
    )
    outputs: dict[str, Any] = {
        "generated_da3_config": str(generated),
        "raw_pointcloud": str(
            args.output_dir / "pcd" / "combined_pcd.ply"
        ),
        "camera_poses": str(args.output_dir / "camera_poses.txt"),
    }
    timings: dict[str, float] = {}
    total_started = time.perf_counter()

    started = time.perf_counter()
    run_da3(args, workflow, generated)
    timings["da3_streaming"] = time.perf_counter() - started

    started = time.perf_counter()
    cleaned = run_clean_pointcloud(args.output_dir, workflow)
    timings["pointcloud_cleaning"] = time.perf_counter() - started
    outputs["cleaned_pointcloud"] = str(cleaned)

    timings["total"] = time.perf_counter() - total_started
    write_manifest(
        args.output_dir,
        args,
        workflow,
        outputs,
        timings,
    )
    print(json.dumps(outputs, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
