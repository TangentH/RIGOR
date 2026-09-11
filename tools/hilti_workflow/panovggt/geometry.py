"""Geometry and export helpers for the PanoVGGT HILTI adapter.

This module contains no run-specific events, paths, or ground-truth data.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_official(repo: Path):
    """Load the official PanoVGGT inference module from a checkout."""
    repo = repo.expanduser().resolve()
    inference = repo / "inference.py"
    if not inference.exists():
        raise FileNotFoundError(f"PanoVGGT inference.py not found: {inference}")
    sys.path.insert(0, str(repo))
    spec = importlib.util.spec_from_file_location("panovggt_official", inference)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import official inference.py: {inference}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def project_so3(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def apply_sim3(points: np.ndarray, transform):
    scale, rotation, translation = transform
    return scale * (points @ rotation.T) + translation


def apply_pose(pose: np.ndarray, transform):
    scale, rotation, translation = transform
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation @ pose[:3, :3]
    result[:3, 3] = scale * (rotation @ pose[:3, 3]) + translation
    return result


def compose(outer, inner):
    sa, ra, ta = outer
    sb, rb, tb = inner
    return sa * sb, ra @ rb, sa * (ra @ tb) + ta


def accumulate(relative):
    output = [(1.0, np.eye(3), np.zeros(3))]
    for transform in relative:
        output.append(compose(output[-1], transform))
    return output


def estimate_pose_sim3(source_poses: np.ndarray, target_poses: np.ndarray):
    rotations = np.stack(
        [
            target_poses[index, :3, :3] @ source_poses[index, :3, :3].T
            for index in range(len(source_poses))
        ]
    )
    rotation = project_so3(rotations.mean(axis=0))
    source = source_poses[:, :3, 3] @ rotation.T
    target = target_poses[:, :3, 3]
    scales = []
    for index in range(len(source)):
        for other in range(index + 1, len(source)):
            denominator = np.linalg.norm(source[index] - source[other])
            numerator = np.linalg.norm(target[index] - target[other])
            if denominator > 1e-6 and numerator > 1e-6:
                scales.append(numerator / denominator)
    if not scales:
        raise ValueError("Repeated poses do not constrain scale")
    scale = float(np.median(scales))
    translation = np.median(target - scale * source, axis=0)
    residual = np.linalg.norm(scale * source + translation - target, axis=1)
    return (scale, rotation, translation), residual


def window_starts(total: int, size: int, stride: int):
    if total <= size:
        return [0]
    starts = list(range(0, total - size + 1, stride))
    last = total - size
    if starts[-1] != last:
        starts.append(last)
    return starts


def select_pixels(
    points: np.ndarray,
    count: int,
    seed: int,
    mask: np.ndarray | None = None,
):
    flat = points.reshape(-1, 3)
    ranges = np.linalg.norm(flat, axis=1)
    valid = np.isfinite(flat).all(axis=1) & np.isfinite(ranges) & (ranges > 1e-5)
    if mask is not None:
        if mask.shape != points.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (points.shape[1], points.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        valid &= ~mask.reshape(-1)
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return indices
    values = ranges[indices]
    low, high = np.quantile(values, (0.005, 0.995))
    indices = indices[(values >= low) & (values <= high)]
    if len(indices) > count:
        indices = np.sort(
            np.random.default_rng(seed).choice(indices, count, replace=False)
        )
    return indices


def load_packed_masks(path: Path, image_paths: list[Path]):
    bundle = np.load(path, allow_pickle=False)
    shape = tuple(int(value) for value in bundle["shape"])
    masks = np.unpackbits(
        bundle["masks"], axis=1, count=int(np.prod(shape[1:]))
    ).reshape(shape).astype(bool)
    stems = [str(value) for value in bundle["stems"]]
    expected = [item.stem for item in image_paths]
    if stems != expected:
        raise ValueError("Packed ERP mask stems do not exactly match input order")
    return masks


def frame_colour(image_path: Path, height: int, width: int):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode {image_path}")
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def save_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    data = np.empty(len(xyz), dtype=dtype)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(data)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(data.tobytes())


def export_variant(
    output_dir: Path,
    name: str,
    samples,
    colours,
    nodes,
    local_poses,
    frame_pose_nodes,
    transforms,
):
    xyz_parts = [
        apply_sim3(points, transforms[node]).astype(np.float32)
        for points, node in zip(samples, nodes)
    ]
    xyz = np.concatenate(xyz_parts)
    rgb = np.concatenate(colours)
    ply_path = output_dir / f"reconstruction_{name}.ply"
    pose_path = output_dir / f"camera_poses_{name}_c2w.npy"
    save_ply(ply_path, xyz, rgb)
    poses = np.stack(
        [
            apply_pose(local_poses[index], transforms[frame_pose_nodes[index]])
            for index in range(len(local_poses))
        ]
    )
    np.save(pose_path, poses)
    return {"ply": str(ply_path), "poses": str(pose_path), "points": int(len(xyz))}
