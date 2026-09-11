#!/usr/bin/env python3
"""Clean obvious isolated outliers from a large colored point cloud.

The default filter is voxel-neighborhood based: points are not downsampled, but
points in sparse/isolated voxels are removed. This is much cheaper than running
a full nearest-neighbor statistical filter on tens of millions of points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--voxel-size", type=float, default=0.05)
    p.add_argument("--min-points-per-voxel", type=int, default=2)
    p.add_argument(
        "--min-occupied-neighbor-voxels",
        type=int,
        default=3,
        help="Minimum occupied voxels in the 3x3x3 neighborhood, including itself.",
    )
    p.add_argument(
        "--neighbor-connectivity",
        choices=["6", "18", "26"],
        default="26",
        help="Neighborhood used to count occupied neighbor voxels.",
    )
    p.add_argument("--save-removed", action="store_true")
    return p.parse_args()


def make_neighbor_offsets(connectivity: str) -> np.ndarray:
    offsets = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == "6" and manhattan > 1:
                    continue
                if connectivity == "18" and manhattan > 2:
                    continue
                offsets.append((dx, dy, dz))
    return np.asarray(offsets, dtype=np.int64)


def voxel_hash(voxels: np.ndarray, dims: np.ndarray) -> np.ndarray:
    return voxels[:, 0] + dims[0] * (voxels[:, 1] + dims[1] * voxels[:, 2])


def compute_keep_mask(
    points: np.ndarray,
    voxel_size: float,
    min_points_per_voxel: int,
    min_occupied_neighbor_voxels: int,
    connectivity: str,
) -> tuple[np.ndarray, dict[str, object]]:
    voxel = np.floor(points / voxel_size).astype(np.int64)
    min_voxel = voxel.min(axis=0)
    shifted = voxel - min_voxel
    dims = shifted.max(axis=0) + 1

    hashes = voxel_hash(shifted, dims)
    unique_hashes, inverse, counts = np.unique(
        hashes, return_inverse=True, return_counts=True
    )
    unique_shifted = shifted[np.unique(inverse, return_index=True)[1]]

    order = np.argsort(unique_hashes)
    sorted_hashes = unique_hashes[order]

    neighbor_counts = np.zeros(len(unique_hashes), dtype=np.uint8)
    offsets = make_neighbor_offsets(connectivity)
    for offset in offsets:
        candidate = unique_shifted + offset
        valid = np.all((candidate >= 0) & (candidate < dims), axis=1)
        candidate_hash = np.full(len(unique_hashes), -1, dtype=np.int64)
        candidate_hash[valid] = voxel_hash(candidate[valid], dims)
        found = np.zeros(len(unique_hashes), dtype=bool)
        pos = np.searchsorted(sorted_hashes, candidate_hash[valid])
        ok = pos < len(sorted_hashes)
        valid_indices = np.flatnonzero(valid)
        found[valid_indices[ok]] = sorted_hashes[pos[ok]] == candidate_hash[valid][ok]
        neighbor_counts += found.astype(np.uint8)

    keep_unique = (
        (counts >= min_points_per_voxel)
        & (neighbor_counts >= min_occupied_neighbor_voxels)
    )
    keep = keep_unique[inverse]

    stats = {
        "input_points": int(len(points)),
        "kept_points": int(np.count_nonzero(keep)),
        "removed_points": int(len(points) - np.count_nonzero(keep)),
        "removed_ratio": float(1.0 - np.count_nonzero(keep) / max(len(points), 1)),
        "voxel_size": voxel_size,
        "min_points_per_voxel": min_points_per_voxel,
        "min_occupied_neighbor_voxels": min_occupied_neighbor_voxels,
        "neighbor_connectivity": connectivity,
        "num_occupied_voxels": int(len(unique_hashes)),
        "num_kept_voxels": int(np.count_nonzero(keep_unique)),
        "bounds_min": points.min(axis=0).tolist(),
        "bounds_max": points.max(axis=0).tolist(),
        "voxel_dims": dims.tolist(),
    }
    return keep, stats


def write_binary_ply_float32(path: Path, points: np.ndarray, colors: np.ndarray | None) -> None:
    """Write a compact binary PLY with float32 xyz and uchar RGB.

    Open3D writes point coordinates as double by default, which nearly doubles
    large PLY files. DA3's original PLY uses float xyz + uchar RGB, so keep the
    postprocessed output in the same compact layout.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float32)
    if colors is not None and len(colors) == len(pts):
        col = np.asarray(colors)
        if np.issubdtype(col.dtype, np.floating):
            col = np.rint(np.clip(col, 0.0, 1.0) * 255.0)
        col = np.asarray(col, dtype=np.uint8)
    else:
        col = np.full((len(pts), 3), 255, dtype=np.uint8)

    vertex = np.empty(
        len(pts),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = pts[:, 0]
    vertex["y"] = pts[:, 1]
    vertex["z"] = pts[:, 2]
    vertex["red"] = col[:, 0]
    vertex["green"] = col[:, 1]
    vertex["blue"] = col[:, 2]

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertex)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        vertex.tofile(f)


def main() -> None:
    args = parse_args()
    print(f"Reading {args.input}")
    pcd = o3d.io.read_point_cloud(str(args.input))
    points = np.asarray(pcd.points)
    if len(points) == 0:
        raise ValueError("Input point cloud is empty")
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"input_points={len(points)}")

    keep, stats = compute_keep_mask(
        points,
        voxel_size=args.voxel_size,
        min_points_per_voxel=args.min_points_per_voxel,
        min_occupied_neighbor_voxels=args.min_occupied_neighbor_voxels,
        connectivity=args.neighbor_connectivity,
    )
    print(json.dumps(stats, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_binary_ply_float32(
        args.output,
        points[keep],
        colors[keep] if colors is not None and len(colors) == len(points) else None,
    )
    print(f"output={args.output}")

    stats_path = args.output.with_suffix(".json")
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"stats={stats_path}")

    if args.save_removed:
        removed_path = args.output.with_name(args.output.stem + "_removed.ply")
        write_binary_ply_float32(
            removed_path,
            points[~keep],
            colors[~keep] if colors is not None and len(colors) == len(points) else None,
        )
        print(f"removed={removed_path}")


if __name__ == "__main__":
    main()
