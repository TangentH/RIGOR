#!/usr/bin/env python3
"""Register a reconstruction with the geometry protocol used in the paper.

The coarse transform is a Sim(3) fitted between timestamp-associated estimated
and ground-truth camera centres. A deterministic, scale-adjusting ICP proposal
is then estimated against the ROI ground-truth cloud and accepted only when all
frozen quality bounds pass. If it fails, the trajectory Sim(3) is retained.
Ground truth is consumed only by this evaluation program.

The input cloud is never modified. DA3 exports one pose per rendered yaw view,
so exact rendered image filenames are required to recover capture timestamps and
collapse the co-located views into one physical camera centre.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

SCHEMA_VERSION = 1
TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{10,})(?!\d)")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
    "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        return self.scale * (points @ self.rotation.T) + self.translation

    def compose_rigid_after(self, rotation: np.ndarray, translation: np.ndarray) -> "Similarity":
        return Similarity(
            self.scale,
            rotation @ self.rotation,
            rotation @ self.translation + translation,
        )

    def compose_similarity_after(
        self, scale: float, rotation: np.ndarray, translation: np.ndarray
    ) -> "Similarity":
        return Similarity(
            scale * self.scale,
            rotation @ self.rotation,
            scale * (rotation @ self.translation) + translation,
        )

    def matrix(self) -> np.ndarray:
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = self.scale * self.rotation
        result[:3, 3] = self.translation
        return result

@dataclass(frozen=True)
class PlyLayout:
    header: bytes
    count: int
    dtype: np.dtype

def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def atomic_json(path: Path, payload: dict[str, Any]) -> bytes:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    atomic_bytes(path, data)
    return data

def atomic_symlink(target: Path | str, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    try:
        temporary.unlink()
    except FileNotFoundError:
        pass
    os.symlink(str(target), temporary)
    os.replace(temporary, link)

def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                return digest.hexdigest()
            digest.update(block)

def load_pose_centres(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64)
    raw = raw.reshape(1, -1) if raw.ndim == 1 else raw
    if raw.shape[1] != 16 or len(raw) < 3 or not np.isfinite(raw).all():
        raise ValueError(f"Invalid camera-to-world pose file: {path}")
    return raw.reshape(-1, 4, 4)[:, :3, 3]

def load_tum_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(value) for value in line.split()]
            if len(values) >= 4:
                rows.append(values[:4])
    if len(rows) < 3:
        raise ValueError(f"Too few GT trajectory rows: {path}")
    array = np.asarray(rows, dtype=np.float64)
    return array[:, 0], array[:, 1:4]

def timestamps_from_images(image_dir: Path) -> np.ndarray:
    values: list[float] = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        tokens = TIMESTAMP_RE.findall(path.stem)
        if not tokens:
            raise ValueError(f"No timestamp in retained frame name: {path.name}")
        values.append(int(max(tokens, key=len)) * 1e-9)
    if not values:
        raise ValueError(f"No timestamped images in {image_dir}")
    return np.asarray(values, dtype=np.float64)

def timestamps_from_archive(archive: Path, pose_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    groups: dict[str, list[tuple[str, int]]] = {}
    with tarfile.open(archive, "r") as stream:
        for member in stream:
            if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            tokens = TIMESTAMP_RE.findall(Path(member.name).stem)
            if not tokens:
                continue
            group = str(Path(member.name).parent)
            groups.setdefault(group, []).append((member.name, int(max(tokens, key=len))))
    candidates = []
    for group, entries in groups.items():
        values = np.asarray([value for _, value in sorted(entries)], dtype=np.float64) * 1e-9
        if len(values) == pose_count or len(np.unique(values)) == pose_count:
            preference = 0
            if len(values) == pose_count and "pinhole_yaw4_imu" in group:
                preference = 3
            elif len(np.unique(values)) == pose_count and "equirect_leveled_video" in group:
                preference = 2
            elif len(np.unique(values)) == pose_count and group.endswith("/equirect"):
                preference = 1
            digest = hashlib.sha256(
                "\n".join(f"{name}:{value}" for name, value in sorted(entries)).encode()
            ).hexdigest()
            candidates.append((preference, group, values, digest))
    if not candidates:
        observed = {group: (len(entries), len({v for _, v in entries})) for group, entries in groups.items()}
        raise RuntimeError(f"No timestamp group in archive matches {pose_count} poses: {observed}")
    preference, group, values, digest = max(candidates, key=lambda item: (item[0], item[1]))
    return values, {
        "kind": "sealed_tar_member_names",
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "member_group": group,
        "timestamped_member_count": int(len(values)),
        "timestamp_member_manifest_sha256": digest,
    }

def timestamps_from_source(source: Path, pose_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    if source.is_dir():
        values = timestamps_from_images(source)
        return values, {
            "kind": "retained_image_directory",
            "path": str(source),
            "timestamped_member_count": int(len(values)),
        }
    if source.is_file() and source.name.endswith(".tar"):
        return timestamps_from_archive(source, pose_count)
    raise FileNotFoundError(f"Missing timestamp source: {source}")

def physical_trajectory(pose_path: Path, image_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    centres = load_pose_centres(pose_path)
    raw_times, timestamp_provenance = timestamps_from_source(image_dir, len(centres))
    unique_times = np.unique(raw_times)
    if len(raw_times) == len(centres):
        grouped = []
        spreads = []
        for timestamp in unique_times:
            members = centres[raw_times == timestamp]
            centre = np.median(members, axis=0)
            grouped.append(centre)
            spreads.append(float(np.linalg.norm(members - centre, axis=1).max()))
        positions = np.asarray(grouped)
    elif len(unique_times) == len(centres):
        positions = centres
        spreads = [0.0] * len(centres)
    else:
        raise ValueError(
            f"Pose/image timestamp mismatch: poses={len(centres)}, images={len(raw_times)}, "
            f"unique_timestamps={len(unique_times)}"
        )
    return unique_times, positions, {
        "rendered_pose_count": int(len(centres)),
        "physical_pose_count": int(len(positions)),
        "timestamped_image_count": int(len(raw_times)),
        "max_same_capture_centre_spread_m": float(max(spreads)),
        "timestamp_provenance": timestamp_provenance,
    }

def associate(
    est_t: np.ndarray,
    est_p: np.ndarray,
    gt_t: np.ndarray,
    gt_p: np.ndarray,
    *,
    ignore_before: float,
    max_gap: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    keep = gt_t >= ignore_before
    eligible_t, eligible_p = gt_t[keep], gt_p[keep]
    if len(eligible_t) < 3:
        raise ValueError("Fewer than three eligible GT poses")
    right = np.searchsorted(est_t, eligible_t, side="left")
    inside = (eligible_t >= est_t[0]) & (eligible_t <= est_t[-1])
    right_safe = np.clip(right, 0, len(est_t) - 1)
    left_safe = np.clip(right - 1, 0, len(est_t) - 1)
    exact = np.abs(est_t[right_safe] - eligible_t) < 1e-9
    gap = est_t[right_safe] - est_t[left_safe]
    matched = inside & (exact | (gap <= max_gap))
    matched_t, matched_gt = eligible_t[matched], eligible_p[matched]
    if len(matched_t) < 3:
        raise RuntimeError("Fewer than three temporally matched GT poses")
    interpolated = np.column_stack(
        [np.interp(matched_t, est_t, est_p[:, axis]) for axis in range(3)]
    )
    coverage = len(matched_t) / len(eligible_t)
    return interpolated, matched_gt, {
        "eligible_gt_pose_count": int(len(eligible_t)),
        "matched_gt_pose_count": int(len(matched_t)),
        "fraction": float(coverage),
        "percent": float(100.0 * coverage),
        "first_eligible_timestamp": float(eligible_t[0]),
        "last_eligible_timestamp": float(eligible_t[-1]),
    }

def fit_sim3(source: np.ndarray, target: np.ndarray) -> Similarity:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Degenerate estimated trajectory")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid Sim3 scale: {scale}")
    translation = target_mean - scale * (source_mean @ rotation.T)
    return Similarity(scale, rotation, translation)

def parse_ply(path: Path) -> PlyLayout:
    with path.open("rb") as stream:
        header = bytearray()
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"Truncated PLY header: {path}")
            header.extend(line)
            if len(header) > (1 << 20):
                raise ValueError(f"PLY header too large: {path}")
            if line.strip() == b"end_header":
                break
    lines = header.decode("ascii").splitlines()
    if "format binary_little_endian 1.0" not in lines:
        raise ValueError("Only binary_little_endian PLY is supported")
    vertex_count: int | None = None
    current_element: str | None = None
    fields: list[tuple[str, str]] = []
    nonvertex_nonzero: list[str] = []
    for line in lines:
        parts = line.split()
        if parts[:1] == ["element"] and len(parts) == 3:
            current_element = parts[1]
            count = int(parts[2])
            if current_element == "vertex":
                vertex_count = count
            elif count:
                nonvertex_nonzero.append(current_element)
        elif parts[:1] == ["property"] and current_element == "vertex":
            if len(parts) != 3 or parts[1] == "list" or parts[1] not in PLY_TYPES:
                raise ValueError(f"Unsupported vertex property: {line}")
            fields.append((parts[2], PLY_TYPES[parts[1]]))
    if vertex_count is None or vertex_count <= 0:
        raise ValueError("PLY has no non-empty vertex element")
    if nonvertex_nonzero:
        raise ValueError(f"PLY contains non-vertex payload: {nonvertex_nonzero}")
    dtype = np.dtype(fields, align=False)
    if not {"x", "y", "z"}.issubset(dtype.names or ()):
        raise ValueError("PLY vertex schema lacks x/y/z")
    expected = len(header) + vertex_count * dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"Unexpected PLY byte size: expected={expected}, actual={path.stat().st_size}")
    return PlyLayout(bytes(header), vertex_count, dtype)

def transform_ply(source: Path, output: Path, transform: Similarity, chunk_points: int = 1_000_000) -> int:
    layout = parse_ply(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    vertices = np.memmap(
        source, mode="r", dtype=layout.dtype, offset=len(layout.header), shape=(layout.count,)
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(layout.header)
            for start in range(0, layout.count, chunk_points):
                block = np.array(vertices[start : start + chunk_points], copy=True)
                xyz = np.column_stack((block["x"], block["y"], block["z"])).astype(np.float64)
                moved = transform.apply(xyz)
                block["x"], block["y"], block["z"] = moved[:, 0], moved[:, 1], moved[:, 2]
                if {"nx", "ny", "nz"}.issubset(layout.dtype.names or ()):
                    normals = np.column_stack((block["nx"], block["ny"], block["nz"])).astype(np.float64)
                    normals = normals @ transform.rotation.T
                    length = np.linalg.norm(normals, axis=1, keepdims=True)
                    normals /= np.maximum(length, 1e-12)
                    block["nx"], block["ny"], block["nz"] = normals[:, 0], normals[:, 1], normals[:, 2]
                block.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
        parse_ply(temporary)
        os.replace(temporary, output)
    finally:
        del vertices
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return layout.count

def sample_ply_xyz(path: Path, limit: int) -> np.ndarray:
    layout = parse_ply(path)
    vertices = np.memmap(path, mode="r", dtype=layout.dtype, offset=len(layout.header), shape=(layout.count,))
    indices = np.linspace(0, layout.count - 1, min(limit, layout.count), dtype=np.int64)
    points = np.column_stack((vertices[indices]["x"], vertices[indices]["y"], vertices[indices]["z"]))
    del vertices
    return np.asarray(points, dtype=np.float64)

def fit_rigid(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    covariance = (target - target_mean).T @ (source - source_mean)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    rotation = u @ correction @ vt
    return rotation, target_mean - source_mean @ rotation.T

def symmetric_trimmed_rmse(source: np.ndarray, target: np.ndarray, trim: float = 0.8) -> float:
    a = cKDTree(target).query(source, k=1, workers=-1)[0]
    b = cKDTree(source).query(target, k=1, workers=-1)[0]
    def one(values: np.ndarray) -> float:
        cutoff = np.quantile(values, trim)
        kept = values[values <= cutoff]
        return float(np.sqrt(np.mean(kept**2)))
    return 0.5 * (one(a) + one(b))

def rigid_icp_candidate(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rotation_total, translation_total = np.eye(3), np.zeros(3)
    moved = source.copy()
    stages = []
    tree = cKDTree(target)
    for max_distance in (2.0, 1.0, 0.5, 0.25):
        for _ in range(12):
            distances, indices = tree.query(moved, k=1, workers=-1)
            keep = distances <= max_distance
            if int(keep.sum()) < 50:
                break
            threshold = np.quantile(distances[keep], 0.8)
            keep &= distances <= threshold
            rotation, translation = fit_rigid(moved[keep], target[indices[keep]])
            moved = moved @ rotation.T + translation
            rotation_total = rotation @ rotation_total
            translation_total = rotation @ translation_total + translation
            if np.linalg.norm(translation) < 1e-5:
                break
        stages.append({"max_correspondence_m": max_distance, "kept": int(keep.sum())})
    return rotation_total, translation_total, {"stages": stages}

def similarity_icp_candidate(
    source: np.ndarray, target: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray, dict[str, Any]]:
    scale_total, rotation_total, translation_total = 1.0, np.eye(3), np.zeros(3)
    moved = source.copy()
    stages = []
    tree = cKDTree(target)
    for max_distance in (2.0, 1.0, 0.5, 0.25):
        for _ in range(12):
            distances, indices = tree.query(moved, k=1, workers=-1)
            keep = distances <= max_distance
            if int(keep.sum()) < 50:
                break
            threshold = np.quantile(distances[keep], 0.8)
            keep &= distances <= threshold
            delta = fit_sim3(moved[keep], target[indices[keep]])
            moved = delta.apply(moved)
            scale_total = delta.scale * scale_total
            rotation_total = delta.rotation @ rotation_total
            translation_total = (
                delta.scale * (delta.rotation @ translation_total) + delta.translation
            )
            if np.linalg.norm(delta.translation) < 1e-5 and abs(delta.scale - 1.0) < 1e-6:
                break
        stages.append({"max_correspondence_m": max_distance, "kept": int(keep.sum())})
    return scale_total, rotation_total, translation_total, {"stages": stages}

def rotation_degrees(rotation: np.ndarray) -> float:
    cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cloud", required=True, type=Path)
    parser.add_argument("--camera-poses", required=True, type=Path)
    timestamps = parser.add_mutually_exclusive_group(required=True)
    timestamps.add_argument(
        "--image-dir", type=Path,
        help="Rendered images whose filenames contain exact ROS timestamps.",
    )
    timestamps.add_argument(
        "--timestamp-archive", type=Path,
        help="Retained tar archive containing those timestamped rendered images.",
    )
    parser.add_argument("--ground-truth-trajectory", required=True, type=Path)
    parser.add_argument("--ground-truth-cloud", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-name", default="run")
    parser.add_argument(
        "--gt-trajectory-to-geometry", type=Path,
        help=("Optional 4x4 transform mapping GT trajectory positions into the "
              "coordinate frame of --ground-truth-cloud. Omit when they already agree."),
    )
    parser.add_argument("--coverage-threshold", type=float, default=0.99)
    parser.add_argument("--ignore-before-time", type=float, default=10005.0)
    parser.add_argument("--max-interpolation-gap", type=float, default=0.75)
    parser.add_argument(
        "--icp-reference-cloud", type=Path,
        help=("Cloud used to estimate the ICP delta; defaults to --source-cloud. "
              "Use only for a declared shared-gauge controlled comparison."),
    )
    parser.add_argument("--icp-sample-points", type=int, default=250_000)
    parser.add_argument("--icp-min-improvement", type=float, default=0.005)
    parser.add_argument("--icp-min-source-fitness", type=float, default=0.05)
    parser.add_argument("--icp-max-translation", type=float, default=2.0)
    parser.add_argument("--icp-max-rotation-deg", type=float, default=12.0)
    parser.add_argument("--icp-min-relative-scale", type=float, default=0.5)
    parser.add_argument("--icp-max-relative-scale", type=float, default=2.0)
    parser.add_argument(
        "--disable-icp-scale", action="store_true",
        help="Estimate only a rigid ICP delta (the paper protocol adjusts scale).",
    )
    parser.add_argument("--disable-icp", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def similarity_dict(transform: Similarity) -> dict[str, Any]:
    return {
        "scale": transform.scale,
        "rotation_3x3": transform.rotation.tolist(),
        "translation_3": transform.translation.tolist(),
        "matrix_4x4": transform.matrix().tolist(),
    }


def load_frame_transform(path: Path | None) -> tuple[np.ndarray, dict[str, Any]]:
    if path is None:
        return np.eye(4, dtype=np.float64), {"operation": "identity_geometry_gt"}
    matrix = np.asarray(np.loadtxt(path), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"Expected one finite 4x4 matrix in {path}")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"Malformed homogeneous transform in {path}")
    return matrix, {
        "operation": "provided_gt_trajectory_to_geometry",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "matrix_4x4": matrix.tolist(),
    }


def transform_positions(positions: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return positions @ matrix[:3, :3].T + matrix[:3, 3]

def trajectory_residuals(
    transform: Similarity, source: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    residual = np.linalg.norm(transform.apply(source) - target, axis=1)
    return {
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "median_m": float(np.median(residual)),
        "p95_m": float(np.percentile(residual, 95)),
        "max_m": float(np.max(residual)),
    }


def estimate_trajectory_transform(args: argparse.Namespace) -> tuple[Similarity, dict[str, Any]]:
    timestamp_source = args.image_dir or args.timestamp_archive
    est_t, est_p, trajectory_info = physical_trajectory(args.camera_poses, timestamp_source)
    gt_t, gt_p = load_tum_positions(args.ground_truth_trajectory)
    frame_transform, frame_report = load_frame_transform(args.gt_trajectory_to_geometry)
    gt_p = transform_positions(gt_p, frame_transform)
    source, target, coverage = associate(
        est_t,
        est_p,
        gt_t,
        gt_p,
        ignore_before=args.ignore_before_time,
        max_gap=args.max_interpolation_gap,
    )
    if coverage["fraction"] + 1e-12 < args.coverage_threshold:
        raise RuntimeError(
            f"strict_coverage_gate_failed:{coverage['fraction']:.9f}"
            f"<{args.coverage_threshold:.9f}"
        )
    transform = fit_sim3(source, target)
    coverage.update(
        required_fraction=args.coverage_threshold,
        passed=True,
    )
    report = {
        "coverage": coverage,
        "trajectory": trajectory_info,
        "ground_truth_frame_bridge": frame_report,
        "matched_trajectory_residual": trajectory_residuals(transform, source, target),
    }
    return transform, report

def prepare_output(output_dir: Path, force: bool) -> None:
    products = (
        "aligned.ply", "aligned_trajectory_sim3.ply",
        "aligned_trajectory_sim3_scale_icp_selected.ply",
        "trajectory_sim3_transform.json", "registration_report.json",
    )
    existing = [
        output_dir / name for name in products
        if (output_dir / name).exists() or (output_dir / name).is_symlink()
    ]
    if existing and not force:
        raise FileExistsError(f"Refusing to overwrite registration products: {existing}")
    if force:
        for path in existing:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_args(args: argparse.Namespace) -> None:
    if not 0.99 <= args.coverage_threshold <= 1.0:
        raise ValueError("--coverage-threshold must be in [0.99, 1.0]")
    if args.icp_sample_points < 50:
        raise ValueError("--icp-sample-points must be at least 50")
    if not 0.0 <= args.icp_min_improvement < 1.0:
        raise ValueError("--icp-min-improvement must be in [0, 1)")
    if not 0.0 <= args.icp_min_source_fitness <= 1.0:
        raise ValueError("--icp-min-source-fitness must be in [0, 1]")
    if not 0.0 < args.icp_min_relative_scale <= args.icp_max_relative_scale:
        raise ValueError("Invalid relative-scale bounds")


def run_registration(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    source = args.source_cloud.resolve()
    target = args.ground_truth_cloud.resolve()
    icp_reference = (args.icp_reference_cloud or source).resolve()
    required = (
        source, target, icp_reference, args.camera_poses.resolve(),
        (args.image_dir or args.timestamp_archive).resolve(),
        args.ground_truth_trajectory.resolve(),
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    prepare_output(args.output_dir, args.force)

    trajectory_transform, trajectory_report = estimate_trajectory_transform(args)
    transform_report = {
        "schema_version": SCHEMA_VERSION,
        "operation": "evaluation_only_trajectory_sim3",
        "run": args.run_name,
        "inputs": {
            "estimated_poses": str(args.camera_poses.resolve()),
            "estimated_poses_sha256": sha256_file(args.camera_poses),
            "timestamp_source": str((args.image_dir or args.timestamp_archive).resolve()),
            "ground_truth_trajectory": str(args.ground_truth_trajectory.resolve()),
            "ground_truth_trajectory_sha256": sha256_file(args.ground_truth_trajectory),
            "ground_truth_frame_bridge": trajectory_report["ground_truth_frame_bridge"],
        },
        "coverage": trajectory_report["coverage"],
        "trajectory": trajectory_report["trajectory"],
        "transform": similarity_dict(trajectory_transform),
        "matched_trajectory_residual": trajectory_report["matched_trajectory_residual"],
    }
    atomic_json(args.output_dir / "trajectory_sim3_transform.json", transform_report)

    sim3_cloud = args.output_dir / "aligned_trajectory_sim3.ply"
    point_count = transform_ply(source, sim3_cloud, trajectory_transform)
    selected_cloud = args.output_dir / "aligned_trajectory_sim3_scale_icp_selected.ply"
    selected_transform = trajectory_transform
    selection = "trajectory_sim3"
    icp_report: dict[str, Any] = {
        "enabled": not args.disable_icp,
        "attempted": False,
        "accepted": False,
        "adjust_scale": not args.disable_icp_scale,
    }
    if not args.disable_icp:
        icp_report["attempted"] = True
        source_sample = trajectory_transform.apply(
            sample_ply_xyz(icp_reference, args.icp_sample_points)
        )
        target_sample = sample_ply_xyz(target, args.icp_sample_points)
        before = symmetric_trimmed_rmse(source_sample, target_sample)
        before_distances = cKDTree(target_sample).query(source_sample, k=1, workers=-1)[0]
        source_fitness = float(np.mean(before_distances <= 0.5))
        if args.disable_icp_scale:
            rotation, translation, diagnostics = rigid_icp_candidate(source_sample, target_sample)
            relative_scale = 1.0
        else:
            relative_scale, rotation, translation, diagnostics = similarity_icp_candidate(
                source_sample, target_sample
            )
        moved = relative_scale * (source_sample @ rotation.T) + translation
        after = symmetric_trimmed_rmse(moved, target_sample)
        improvement = (before - after) / max(before, 1e-12)
        translation_m = float(np.linalg.norm(translation))
        rotation_deg = rotation_degrees(rotation)
        accepted = (
            improvement >= args.icp_min_improvement
            and source_fitness >= args.icp_min_source_fitness
            and translation_m <= args.icp_max_translation
            and rotation_deg <= args.icp_max_rotation_deg
            and args.icp_min_relative_scale <= relative_scale <= args.icp_max_relative_scale
        )
        icp_report.update({
            **diagnostics,
            "accepted": accepted,
            "before_symmetric_trimmed_rmse_m": before,
            "after_symmetric_trimmed_rmse_m": after,
            "relative_improvement": improvement,
            "translation_m": translation_m,
            "rotation_deg": rotation_deg,
            "source_fitness_at_05m": source_fitness,
            "required_source_fitness_at_05m": args.icp_min_source_fitness,
            "relative_scale": relative_scale,
            "required_relative_scale_range": [
                args.icp_min_relative_scale, args.icp_max_relative_scale
            ],
            "rotation_3x3": rotation.tolist(),
            "translation_3": translation.tolist(),
        })
        if accepted:
            selected_transform = trajectory_transform.compose_similarity_after(
                relative_scale, rotation, translation
            )
            transform_ply(source, selected_cloud, selected_transform)
            selection = "trajectory_sim3_plus_scale_icp"

    if not selected_cloud.exists():
        atomic_symlink(sim3_cloud.name, selected_cloud)
    atomic_symlink(selected_cloud.name, args.output_dir / "aligned.ply")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "operation": "evaluation_only_gt_registration",
        "gt_usage": "registration_and_evaluation_only",
        "run": args.run_name,
        "selection": selection,
        "source": {
            "path": str(source), "sha256": sha256_file(source), "point_count": point_count,
        },
        "ground_truth_cloud": {"path": str(target), "sha256": sha256_file(target)},
        "icp_reference_cloud": {
            "path": str(icp_reference), "sha256": sha256_file(icp_reference),
        },
        "coverage": trajectory_report["coverage"],
        "trajectory_sim3": similarity_dict(trajectory_transform),
        "scale_icp": icp_report,
        "selected_transform": similarity_dict(selected_transform),
        "outputs": {
            "trajectory_sim3": str(sim3_cloud),
            "selected": str(selected_cloud),
            "aligned_alias": str(args.output_dir / "aligned.ply"),
        },
    }
    atomic_json(args.output_dir / "registration_report.json", report)
    return report


def main() -> int:
    report = run_registration(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
