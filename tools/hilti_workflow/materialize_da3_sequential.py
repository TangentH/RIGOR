#!/usr/bin/env python3
"""Materialize the paper's DA3-Sequential baseline from frozen RIGOR caches.

This utility never invokes the DA3 backbone and never edits or deletes the
input cache tree. Each admitted run writes reconstruction and provenance
files under

  OUTPUT_ROOT/da3_sequential/FLOOR/DATE/RUN/reconstruction/

The primary outputs are ``reconstruction.ply``, ``camera_poses.txt``, and
``manifest.json``; point-cloud cleanup statistics and a log accompany them.

Scientific definition
---------------------
DA3-Sequential is the common DA3 trunk before loop closure *and* before our
virtual-rig repair.  The frozen configuration applies rig repair only after
the sequential chunk transforms have been estimated.  Consequently this tool
recovers those already-computed transforms from the sealed pre-loop pose
artifact, but regenerates poses and points from the unmodified
``_tmp_results_unaligned/chunk_*.npy`` predictions.  It does not copy the
pre-loop pose file because that file was written after local rig correction.

For the three runs where no pre-loop pose file was emitted, the final pose file
is admitted only after machine-readable evidence proves that zero loop edges
passed verification and no optimizer outcome exists; in that case the graph
transform is exactly the sequential transform.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


SCHEMA_VERSION = 2
PROTOCOL_ID = "da3-sequential-no-loop-no-rig-repair-v2-cleaned"
METHOD = "da3_sequential"
DEFAULT_REPO = Path(__file__).resolve().parents[2]
RUN_RE = re.compile(r"^(floor_[^/]+)/(\d{4}-\d{2}-\d{2})/(run_\d+)$")


@dataclass(frozen=True)
class RunSpec:
    rel: str
    run_root: Path
    work: Path
    cache_dir: Path
    config: Path
    pose_source: Path
    pose_source_kind: str
    output_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache-only DA3-Sequential materialization queue"
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        required=True,
        help="RIGOR work root containing FLOOR/DATE/RUN/_work/_da3_work caches",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root under which da3_sequential/FLOOR/DATE/RUN is written",
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument(
        "--status-jsonl",
        type=Path,
        default=None,
        help="Queue status file (default: OUTPUT_ROOT/da3_sequential_status.jsonl)",
    )
    parser.add_argument(
        "--repo-revision",
        default=None,
        help="Revision recorded in manifests when the source tree has no .git directory",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Relative FLOOR/DATE/RUN; repeat to select multiple runs",
    )
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Write outputs. Without this flag the command is a read-only dry-run.",
    )
    parser.add_argument(
        "--hash-caches",
        action="store_true",
        help="Hash every NPY during dry-run too (materialization always hashes).",
    )
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True).encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def append_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(payload)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def numbered_chunks(cache_dir: Path) -> list[Path]:
    found: dict[int, Path] = {}
    for path in cache_dir.glob("chunk_*.npy"):
        try:
            index = int(path.stem.split("_")[-1])
        except ValueError as exc:
            raise ValueError(f"invalid chunk filename: {path}") from exc
        if index in found:
            raise ValueError(f"duplicate chunk index {index}: {cache_dir}")
        found[index] = path
    if not found:
        raise FileNotFoundError(f"no frozen unaligned chunks: {cache_dir}")
    expected = list(range(max(found) + 1))
    if sorted(found) != expected:
        raise ValueError(f"non-contiguous chunk cache: {cache_dir}")
    return [found[index] for index in expected]


def zero_accepted_loops(work: Path) -> bool:
    outcome = work / "sim3_optimizer_outcome.json"
    if outcome.exists():
        return False
    table = work / "loop_geometry_verification.csv"
    if not table.is_file():
        return False
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return False
    return all(str(row.get("accepted", "")).strip().lower() == "false" for row in rows)


def discover(args: argparse.Namespace) -> list[RunSpec]:
    selected = set(args.run)
    specs: list[RunSpec] = []
    for cache_dir in sorted(args.cache_root.glob("floor_*/*/run_*/_work/_da3_work/_tmp_results_unaligned")):
        run_root = cache_dir.parents[2]
        rel = "/".join(run_root.relative_to(args.cache_root).parts[:3])
        if not RUN_RE.fullmatch(rel):
            continue
        if selected and rel not in selected:
            continue
        work = cache_dir.parent
        config = work / "configs/da3_streaming_generated.yaml"
        pre = work / "camera_poses_pre_loop.txt"
        final = work / "camera_poses.txt"
        if pre.is_file():
            pose_source, kind = pre, "camera_poses_pre_loop"
        elif final.is_file() and zero_accepted_loops(work):
            pose_source, kind = final, "final_proven_zero_accepted_loops"
        else:
            raise RuntimeError(
                f"{rel}: no admissible sealed sequential pose source"
            )
        specs.append(
            RunSpec(
                rel=rel,
                run_root=run_root,
                work=work,
                cache_dir=cache_dir,
                config=config,
                pose_source=pose_source,
                pose_source_kind=kind,
                output_dir=args.output_root / METHOD / rel / "reconstruction",
            )
        )
    if selected:
        missing = selected - {spec.rel for spec in specs}
        if missing:
            raise FileNotFoundError(f"selected runs not discovered: {sorted(missing)}")
    if args.max_runs > 0:
        specs = specs[: args.max_runs]
    return specs


def check_frozen_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = yaml.safe_load(path.read_text())
    model = config["Model"]
    repair = model["Yaw4_Pose_Filter"]
    point = model["Pointcloud_Save"]
    expected = {
        "chunk_size": 32,
        "overlap": 16,
        "rig_repair_before_alignment": False,
        "rig_repair_after_alignment": True,
        "rig_outlier_action": "repair",
        "canonical_overlap_ownership": True,
        "sample_ratio": 0.05,
        "conf_threshold_coef": 0.75,
    }
    observed = {
        "chunk_size": int(model["chunk_size"]),
        "overlap": int(model["overlap"]),
        "rig_repair_before_alignment": bool(repair["rig_repair_before_alignment"]),
        "rig_repair_after_alignment": bool(repair["rig_repair_after_alignment"]),
        "rig_outlier_action": str(repair["rig_outlier_action"]),
        "canonical_overlap_ownership": bool(point["canonical_overlap_ownership"]),
        "sample_ratio": float(point["sample_ratio"]),
        "conf_threshold_coef": float(point["conf_threshold_coef"]),
    }
    if observed != expected:
        raise ValueError(f"config is not the frozen trunk: {observed!r}")
    return config


def load_prediction(path: Path) -> Any:
    prediction = np.load(path, allow_pickle=True).item()
    required = ("depth", "conf", "extrinsics", "intrinsics", "processed_images")
    for key in required:
        if not hasattr(prediction, key):
            raise ValueError(f"{path}: missing prediction field {key}")
    count = len(prediction.depth)
    if not all(len(getattr(prediction, key)) == count for key in required[1:]):
        raise ValueError(f"{path}: inconsistent prediction lengths")
    return prediction


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def local_c2w(extrinsics: np.ndarray) -> np.ndarray:
    result = np.repeat(np.eye(4)[None], len(extrinsics), axis=0)
    result[:, :3, :4] = np.asarray(extrinsics, dtype=np.float64)
    result = np.linalg.inv(result)
    result[:, :3, :3] = np.asarray(
        [project_rotation(rotation) for rotation in result[:, :3, :3]]
    )
    return result


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = np.clip((np.trace(a.T @ b) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def recover_chunk_transform(
    local: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    """Recover x_target = s R x_local + t, rejecting repaired pose rows."""
    rotations = np.asarray(
        [project_rotation(g[:3, :3] @ l[:3, :3].T) for l, g in zip(local, target)]
    )
    pairwise = np.asarray(
        [[rotation_error_deg(a, b) for b in rotations] for a in rotations]
    )
    medoid = int(np.argmin(np.median(pairwise, axis=1)))
    keep = pairwise[medoid] <= 0.01
    if int(keep.sum()) < 4:
        raise ValueError("fewer than four unchanged orientations recover chunk gauge")
    rotation = project_rotation(rotations[keep].sum(axis=0))

    x = (rotation @ local[keep, :3, 3].T).T
    y = target[keep, :3, 3]
    for _ in range(3):
        xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
        denominator = float(np.sum(xc * xc))
        if denominator <= 1e-10:
            raise ValueError("chunk motion is insufficient to recover scale")
        scale = float(np.sum(xc * yc) / denominator)
        translation = y.mean(axis=0) - scale * x.mean(axis=0)
        residual = np.linalg.norm(scale * x + translation - y, axis=1)
        median = float(np.median(residual))
        refined = residual <= max(1e-4, 5.0 * median + 1e-6)
        if refined.all():
            break
        x, y = x[refined], y[refined]
        if len(x) < 4:
            raise ValueError("center residual rejection left too few pose rows")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid recovered scale: {scale}")
    center_residual = np.linalg.norm(scale * x + translation - y, axis=1)
    if float(center_residual.max(initial=0.0)) > 5e-4:
        raise ValueError("recovered chunk transform fails center parity")
    return scale, rotation, translation, {
        "orientation_rows": int(keep.sum()),
        "fit_rows": int(len(x)),
        "center_residual_max_m": float(center_residual.max(initial=0.0)),
        "rotation_cluster_max_deg": float(pairwise[medoid, keep].max(initial=0.0)),
    }


def transform_pose(pose: np.ndarray, sim3: tuple[float, np.ndarray, np.ndarray]) -> np.ndarray:
    scale, rotation, translation = sim3
    out = np.eye(4)
    out[:3, :3] = project_rotation(rotation @ pose[:3, :3])
    out[:3, 3] = scale * (rotation @ pose[:3, 3]) + translation
    return out


def inspect_and_recover(
    spec: RunSpec, hash_caches: bool
) -> tuple[
    dict[str, Any],
    list[Path],
    list[tuple[float, np.ndarray, np.ndarray]],
    list[np.ndarray],
]:
    config = check_frozen_config(spec.config)
    model = config["Model"]
    chunk_size, overlap = int(model["chunk_size"]), int(model["overlap"])
    step = chunk_size - overlap
    chunks = numbered_chunks(spec.cache_dir)
    poses = np.loadtxt(spec.pose_source, dtype=np.float64).reshape(-1, 4, 4)
    transforms, owned_poses = [], []
    cache_records = []
    total_frames = 0
    recovery_rows = []
    for index, path in enumerate(chunks):
        prediction = load_prediction(path)
        length = len(prediction.depth)
        if length <= 0 or length > chunk_size:
            raise ValueError(f"{path}: invalid chunk length {length}")
        start = index * step
        end = start + length
        total_frames = max(total_frames, end)
        local_end = length if index == len(chunks) - 1 else length - overlap
        if local_end <= 0:
            raise ValueError(f"{path}: ownership is empty")
        global_indices = np.arange(start, start + local_end)
        if int(global_indices[-1]) >= len(poses):
            raise ValueError(f"{path}: pose source is shorter than cache coverage")
        local = local_c2w(prediction.extrinsics[:local_end])
        target = poses[global_indices]
        scale, rotation, translation, recovery = recover_chunk_transform(local, target)
        transforms.append((scale, rotation, translation))
        owned_poses.append(
            np.asarray([transform_pose(pose, transforms[-1]) for pose in local])
        )
        recovery_rows.append({"chunk": index, **recovery})
        record = {"name": path.name, "size": path.stat().st_size}
        if hash_caches:
            record["sha256"] = sha256(path)
        cache_records.append(record)
        del prediction
    if total_frames != len(poses):
        raise ValueError(
            f"{spec.rel}: cache covers {total_frames} frames, pose source has {len(poses)}"
        )
    flattened = np.concatenate(owned_poses, axis=0)
    if len(flattened) != total_frames:
        raise AssertionError("canonical ownership did not produce one pose per frame")
    audit = {
        "run": spec.rel,
        "status": "admissible",
        "chunk_count": len(chunks),
        "frame_count": total_frames,
        "pose_source": str(spec.pose_source),
        "pose_source_kind": spec.pose_source_kind,
        "config": str(spec.config),
        "cache_records": cache_records,
        "transform_recovery": recovery_rows,
        "rig_repair": "off",
        "loop_closure": "off",
    }
    return audit, chunks, transforms, owned_poses


def seed_for(base: int, run: str, chunk: int) -> int:
    digest = hashlib.sha256(f"{base}:{PROTOCOL_ID}:{run}:{chunk}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def selected_chunk_payload(
    prediction: Any,
    sim3: tuple[float, np.ndarray, np.ndarray],
    owned_count: int,
    threshold_coef: float,
    sample_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    conf = np.asarray(prediction.conf)
    threshold = float(np.mean(conf)) * threshold_coef
    owned_conf = conf[:owned_count]
    valid = (owned_conf >= threshold) & (owned_conf > 1e-5)
    eligible = np.flatnonzero(valid.reshape(-1))
    sample_count = int(len(eligible) * sample_ratio)
    if sample_count == 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8), threshold, len(eligible)
    chosen = rng.choice(eligible, size=sample_count, replace=False)
    height, width = owned_conf.shape[-2:]
    frame = chosen // (height * width)
    pixel = chosen % (height * width)
    vv, uu = pixel // width, pixel % width
    depth = np.asarray(prediction.depth)[frame, vv, uu].astype(np.float64)
    frame_intrinsics = np.asarray(prediction.intrinsics)[:owned_count].astype(np.float64)
    rays = np.stack((uu, vv, np.ones_like(uu)), axis=1).astype(np.float64)
    camera = np.empty((len(chosen), 3), dtype=np.float64)
    for index in np.unique(frame):
        mask = frame == index
        camera[mask] = (np.linalg.inv(frame_intrinsics[index]) @ rays[mask].T).T
    camera *= depth[:, None]
    frame_c2w = local_c2w(np.asarray(prediction.extrinsics)[:owned_count])
    local = frame_c2w[frame]
    points = np.einsum("nij,nj->ni", local[:, :3, :3], camera) + local[:, :3, 3]
    scale, rotation, translation = sim3
    points = scale * (rotation @ points.T).T + translation
    colors = np.asarray(prediction.processed_images)[frame, vv, uu, :3]
    return points.astype(np.float32), colors.astype(np.uint8), threshold, len(eligible)


def write_pose_file(path: Path, blocks: list[np.ndarray]) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for pose in np.concatenate(blocks, axis=0):
                handle.write(" ".join(f"{value:.17g}" for value in pose.reshape(-1)))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def write_cloud(
    path: Path,
    spec: RunSpec,
    chunks: list[Path],
    transforms: list[tuple[float, np.ndarray, np.ndarray]],
    owned_poses: list[np.ndarray],
    config: dict[str, Any],
    base_seed: int,
) -> tuple[int, list[dict[str, Any]]]:
    payload_fd, payload_name = tempfile.mkstemp(prefix=".sequential-payload.", dir=path.parent)
    total = 0
    records = []
    try:
        with os.fdopen(payload_fd, "wb") as payload:
            for index, (chunk, sim3, pose_block) in enumerate(
                zip(chunks, transforms, owned_poses)
            ):
                prediction = load_prediction(chunk)
                rng = np.random.default_rng(seed_for(base_seed, spec.rel, index))
                points, colors, threshold, eligible = selected_chunk_payload(
                    prediction,
                    sim3,
                    len(pose_block),
                    float(config["Model"]["Pointcloud_Save"]["conf_threshold_coef"]),
                    float(config["Model"]["Pointcloud_Save"]["sample_ratio"]),
                    rng,
                )
                packed = np.empty(
                    len(points),
                    dtype=np.dtype(
                        [("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                         ("red", "u1"), ("green", "u1"), ("blue", "u1")]
                    ),
                )
                if len(points):
                    packed["x"], packed["y"], packed["z"] = points.T
                    packed["red"], packed["green"], packed["blue"] = colors.T
                    payload.write(packed.tobytes())
                total += len(points)
                records.append(
                    {
                        "chunk": index,
                        "eligible_points": eligible,
                        "sampled_points": len(points),
                        "confidence_threshold": threshold,
                        "seed": seed_for(base_seed, spec.rel, index),
                    }
                )
                del prediction, points, colors
            payload.flush()
            os.fsync(payload.fileno())
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as output, open(payload_name, "rb") as payload:
                output.write(b"ply\nformat binary_little_endian 1.0\n")
                output.write(f"element vertex {total}\n".encode())
                output.write(
                    b"property float x\nproperty float y\nproperty float z\n"
                    b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                    b"end_header\n"
                )
                while block := payload.read(8 << 20):
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
            os.replace(name, path)
        except BaseException:
            try:
                os.unlink(name)
            except FileNotFoundError:
                pass
            raise
    finally:
        try:
            os.unlink(payload_name)
        except FileNotFoundError:
            pass
    return total, records


def git_commit(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def materialize_one(spec: RunSpec, args_dict: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    materialize = bool(args_dict["materialize"])
    audit, chunks, transforms, owned_poses = inspect_and_recover(
        spec, hash_caches=materialize or bool(args_dict["hash_caches"])
    )
    if not materialize:
        return {**audit, "mode": "dry_run", "elapsed_s": time.time() - started}

    output = spec.output_dir
    output.parent.mkdir(parents=True, exist_ok=True)
    cloud = output / "reconstruction.ply"
    poses = output / "camera_poses.txt"
    manifest_path = output / "manifest.json"
    if any(path.exists() for path in (cloud, poses, manifest_path)):
        if all(path.is_file() for path in (cloud, poses, manifest_path)):
            old = json.loads(manifest_path.read_text())
            old_sampling = old.get("sampling")
            expected_seeds = [
                seed_for(int(args_dict["seed"]), spec.rel, index)
                for index in range(len(chunks))
            ]
            sampling_matches = (
                isinstance(old_sampling, list)
                and len(old_sampling) == len(expected_seeds)
                and all(
                    isinstance(row, dict) and row.get("seed") == expected
                    for row, expected in zip(old_sampling, expected_seeds)
                )
            )
            if (
                old.get("protocol_id") == PROTOCOL_ID
                and old.get("run") == spec.rel
                and old.get("config_sha256") == sha256(spec.config)
                and old.get("pose_source_sha256") == sha256(spec.pose_source)
                and old.get("source", {}).get("cache_records") == audit["cache_records"]
                and old.get("sampling_base_seed") == int(args_dict["seed"])
                and sampling_matches
                and old.get("outputs", {}).get("reconstruction_sha256") == sha256(cloud)
                and old.get("outputs", {}).get("camera_poses_sha256") == sha256(poses)
            ):
                return {"run": spec.rel, "status": "verified_complete", "mode": "materialize"}
        raise FileExistsError(f"refusing to overwrite stale/partial output: {output}")

    config = check_frozen_config(spec.config)
    workflow_path = Path(args_dict["repo"]) / "tools/hilti_workflow/configs/rigor_paper_free_scale.yaml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    clean = dict(workflow["pointcloud_clean"])
    if clean != {
        "enabled": True,
        "voxel_size": 0.05,
        "min_points_per_voxel": 2,
        "min_occupied_neighbor_voxels": 3,
        "neighbor_connectivity": "26",
        "save_removed": False,
    }:
        raise ValueError(f"unexpected frozen pointcloud_clean settings: {clean!r}")

    attempt = output.with_name(f".{output.name}.attempt.{os.getpid()}")
    if attempt.exists():
        raise FileExistsError(f"refusing stale attempt directory: {attempt}")
    attempt.mkdir(parents=True)
    raw_cloud = attempt / "source_sampled_pre_clean.ply"
    clean_cloud = attempt / "reconstruction.ply"
    attempt_poses = attempt / "camera_poses.txt"
    clean_log = attempt / "pointcloud_clean.log"
    raw_vertex_count, sampling = write_cloud(
        raw_cloud, spec, chunks, transforms, owned_poses, config, int(args_dict["seed"])
    )
    write_pose_file(attempt_poses, owned_poses)
    cleaner = Path(args_dict["repo"]) / "tools/hilti_workflow/postprocess/clean_pointcloud_outliers.py"
    command = [
        sys.executable, str(cleaner), "--input", str(raw_cloud), "--output", str(clean_cloud),
        "--voxel-size", str(clean["voxel_size"]),
        "--min-points-per-voxel", str(clean["min_points_per_voxel"]),
        "--min-occupied-neighbor-voxels", str(clean["min_occupied_neighbor_voxels"]),
        "--neighbor-connectivity", str(clean["neighbor_connectivity"]),
    ]
    with clean_log.open("wb") as log_handle:
        completed = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT)
    if completed.returncode != 0 or not clean_cloud.is_file():
        raise RuntimeError(f"frozen point-cloud cleaning failed; retained attempt: {attempt}")
    clean_stats_path = clean_cloud.with_suffix(".json")
    clean_stats = json.loads(clean_stats_path.read_text(encoding="utf-8"))
    vertex_count = int(clean_stats["kept_points"])
    if int(clean_stats["input_points"]) != raw_vertex_count:
        raise RuntimeError("point-cloud cleaner input count mismatch")
    raw_cloud.unlink()
    clean_stats_path.replace(attempt / "pointcloud_clean_stats.json")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "method": METHOD,
        "run": spec.rel,
        "definition": {
            "loop_closure": False,
            "rig_repair": False,
            "backbone_rerun": False,
            "source": "sealed frozen _tmp_results_unaligned caches",
            "chunk_transforms": "recovered from sealed pre-loop trajectory",
            "canonical_overlap_ownership": True,
            "sampling": "deterministic uniform without replacement per chunk",
        },
        "frozen_repo_commit": args_dict["repo_commit"],
        "source": audit,
        "config_sha256": sha256(spec.config),
        "pose_source_sha256": sha256(spec.pose_source),
        "sampling_base_seed": int(args_dict["seed"]),
        "sampling": sampling,
        "pointcloud_clean": {
            "settings": clean,
            "workflow_config": str(workflow_path),
            "workflow_config_sha256": sha256(workflow_path),
            "implementation": str(cleaner),
            "implementation_sha256": sha256(cleaner),
            "raw_sampled_vertices": raw_vertex_count,
            "cleaned_vertices": vertex_count,
            "stats": clean_stats,
        },
        "outputs": {
            "reconstruction": "reconstruction.ply",
            "reconstruction_vertices": vertex_count,
            "reconstruction_sha256": sha256(clean_cloud),
            "camera_poses": "camera_poses.txt",
            "camera_pose_count": int(sum(len(block) for block in owned_poses)),
            "camera_poses_sha256": sha256(attempt_poses),
        },
        "elapsed_s": time.time() - started,
        "completed_unix_s": time.time(),
    }
    atomic_json(attempt / "manifest.json", manifest)
    os.replace(attempt, output)
    return {
        "run": spec.rel,
        "status": "complete",
        "mode": "materialize",
        "vertices": vertex_count,
        "elapsed_s": manifest["elapsed_s"],
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.status_jsonl is None:
        args.status_jsonl = args.output_root / "da3_sequential_status.jsonl"
    # Loading pickled Prediction objects requires the frozen repository paths,
    # but this script never imports or instantiates the DA3 model.
    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "da3_streaming"))
    specs = discover(args)
    if not specs:
        raise RuntimeError("no runs selected")
    repo_commit = args.repo_revision or git_commit(args.repo)
    args_dict = {
        "materialize": args.materialize,
        "hash_caches": args.hash_caches,
        "seed": args.seed,
        "repo_commit": repo_commit,
        "repo": str(args.repo.resolve()),
    }
    print(
        json.dumps(
            {
                "mode": "materialize" if args.materialize else "dry_run",
                "run_count": len(specs),
                "workers": args.workers,
                "protocol_id": PROTOCOL_ID,
                "repo_commit": repo_commit,
            },
            sort_keys=True,
        )
    )
    failures = 0
    if args.workers == 1:
        for spec in specs:
            try:
                result = materialize_one(spec, args_dict)
            except BaseException as exc:
                failures += 1
                result = {
                    "run": spec.rel,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            append_status(args.status_jsonl, result)
            print(json.dumps(result, sort_keys=True))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(materialize_one, spec, args_dict): spec for spec in specs}
            for future in concurrent.futures.as_completed(futures):
                spec = futures[future]
                try:
                    result = future.result()
                except BaseException as exc:
                    failures += 1
                    result = {
                        "run": spec.rel,
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                append_status(args.status_jsonl, result)
                print(json.dumps(result, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
