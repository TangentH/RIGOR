#!/usr/bin/env python3
"""Diagnose and project DA3 yaw4 poses onto the known virtual-camera rig."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


YAW_RE = re.compile(r"(?:^|_)yaw(?P<yaw>\d{3})(?:_|\.|$)")


@dataclass(frozen=True)
class RigFit:
    accepted: bool
    inliers: tuple[int, ...]
    center: np.ndarray
    rotation: np.ndarray
    center_residuals: np.ndarray
    rotation_residuals_deg: np.ndarray
    corrected_c2w: np.ndarray
    consensus_c2w: np.ndarray


def rotation_y(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def view_rotation(yaw_deg: float, rotate180: bool) -> np.ndarray:
    rotation = rotation_y(yaw_deg)
    if rotate180:
        rotation = rotation @ np.diag([-1.0, -1.0, 1.0])
    return rotation


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return u @ correction @ vt


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    return project_rotation(np.sum(rotations, axis=0))


def rotation_error_deg(reference: np.ndarray, estimate: np.ndarray) -> float:
    delta = reference.T @ estimate
    cosine = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def extrinsics_to_c2w(extrinsics: np.ndarray) -> np.ndarray:
    c2w = []
    for ext in np.asarray(extrinsics, dtype=np.float64):
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :4] = ext
        pose = np.linalg.inv(w2c)
        pose[:3, :3] = project_rotation(pose[:3, :3])
        c2w.append(pose)
    return np.asarray(c2w)


def c2w_to_extrinsics(c2w: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.inv(pose)[:3, :4] for pose in c2w], dtype=np.float64)


def _sample_overlap_view(
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsic: np.ndarray,
    c2w: np.ndarray,
    yaw_deg: float,
    *,
    pixel_stride: int,
    confidence_quantile: float,
    max_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth.shape
    vv, uu = np.meshgrid(
        np.arange(pixel_stride // 2, height, pixel_stride),
        np.arange(pixel_stride // 2, width, pixel_stride),
        indexing="ij",
    )
    pixels = np.stack((uu.ravel(), vv.ravel(), np.ones(uu.size)), axis=1)
    values = depth[vv, uu].ravel().astype(np.float64)
    conf = confidence[vv, uu].ravel().astype(np.float64)
    finite_conf = conf[np.isfinite(conf)]
    if not finite_conf.size:
        return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
    threshold = np.quantile(finite_conf, confidence_quantile)
    valid = (
        np.isfinite(values)
        & (values > 0.0)
        & (values < max_depth)
        & np.isfinite(conf)
        & (conf > 0.0)
        & (conf >= threshold)
    )
    pixels, values = pixels[valid], values[valid]
    rays = (np.linalg.inv(intrinsic) @ pixels.T).T
    camera_points = rays * values[:, None]
    world_points = (c2w[:3, :3] @ camera_points.T).T + c2w[:3, 3]
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    rig_directions = (view_rotation(yaw_deg, True) @ rays.T).T
    rig_directions /= np.maximum(
        np.linalg.norm(rig_directions, axis=1, keepdims=True), 1e-12
    )
    return rig_directions, world_points, values


def _mutual_direction_matches(
    directions_a: np.ndarray,
    directions_b: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(directions_a) or not len(directions_b):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    distance, index_b = cKDTree(directions_b).query(directions_a, k=1)
    index_a = np.flatnonzero(distance <= tolerance)
    index_b = index_b[index_a]
    if not len(index_a):
        return index_a, index_b
    _, reverse_a = cKDTree(directions_a).query(directions_b[index_b], k=1)
    mutual = reverse_a == index_a
    return index_a[mutual], index_b[mutual]


def evaluate_overlap_repair(
    depth: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    raw_c2w: np.ndarray,
    corrected_c2w: np.ndarray,
    yaws_deg: np.ndarray,
    *,
    pixel_stride: int = 4,
    ray_tolerance_pixels: float = 1.5,
    confidence_quantile: float = 0.25,
    max_depth: float = 15.0,
) -> dict[str, object]:
    """Compare raw and proposed poses using only adjacent virtual-view overlap."""
    order = np.argsort(yaws_deg)
    depth = np.asarray(depth)[order]
    confidence = np.asarray(confidence)[order]
    intrinsics = np.asarray(intrinsics)[order]
    raw_c2w = np.asarray(raw_c2w)[order]
    corrected_c2w = np.asarray(corrected_c2w)[order]
    yaws_deg = np.asarray(yaws_deg)[order]

    width = depth.shape[-1]
    median_fx = float(np.median(intrinsics[:, 0, 0]))
    horizontal_fov = 2.0 * math.atan(width / (2.0 * median_fx))
    angle_per_pixel = horizontal_fov / width
    tolerance = 2.0 * math.sin(
        angle_per_pixel * pixel_stride * ray_tolerance_pixels / 2.0
    )
    sample_kwargs = {
        "pixel_stride": pixel_stride,
        "confidence_quantile": confidence_quantile,
        "max_depth": max_depth,
    }
    raw = [
        _sample_overlap_view(
            depth[i], confidence[i], intrinsics[i], raw_c2w[i], yaws_deg[i],
            **sample_kwargs,
        )
        for i in range(4)
    ]
    repaired = [
        _sample_overlap_view(
            depth[i], confidence[i], intrinsics[i], corrected_c2w[i], yaws_deg[i],
            **sample_kwargs,
        )
        for i in range(4)
    ]

    raw_distances, repaired_distances, normalizers = [], [], []
    pair_counts = []
    for first in range(4):
        second = (first + 1) % 4
        selected_a, selected_b = _mutual_direction_matches(
            raw[first][0], raw[second][0], tolerance
        )
        pair_counts.append(int(len(selected_a)))
        if not len(selected_a):
            continue
        raw_distances.append(
            np.linalg.norm(
                raw[first][1][selected_a] - raw[second][1][selected_b], axis=1
            )
        )
        repaired_distances.append(
            np.linalg.norm(
                repaired[first][1][selected_a] - repaired[second][1][selected_b],
                axis=1,
            )
        )
        normalizers.append(
            np.maximum(
                (raw[first][2][selected_a] + raw[second][2][selected_b]) / 2.0,
                1e-6,
            )
        )
    if not raw_distances:
        return {"matched_rays": 0, "pair_counts": pair_counts}

    raw_normalized = np.concatenate(raw_distances) / np.concatenate(normalizers)
    repaired_normalized = np.concatenate(repaired_distances) / np.concatenate(normalizers)
    return {
        "matched_rays": int(len(raw_normalized)),
        "pair_counts": pair_counts,
        "raw_normalized_median": float(np.median(raw_normalized)),
        "repaired_normalized_median": float(np.median(repaired_normalized)),
        "raw_normalized_p90": float(np.quantile(raw_normalized, 0.9)),
        "repaired_normalized_p90": float(np.quantile(repaired_normalized, 0.9)),
    }


def fit_yaw4_group(
    c2w: np.ndarray,
    yaws_deg: np.ndarray,
    *,
    center_threshold: float,
    rotation_threshold_deg: float,
    min_inliers: int = 3,
    rotate180: bool = True,
    check_orientation: bool = True,
) -> RigFit:
    c2w = np.asarray(c2w, dtype=np.float64)
    yaws_deg = np.asarray(yaws_deg, dtype=np.float64)
    count = len(c2w)
    if count != 4 or len(yaws_deg) != count:
        return _rejected_fit(c2w)

    view_rotations = np.asarray([view_rotation(yaw, rotate180) for yaw in yaws_deg])
    centers = c2w[:, :3, 3]
    base_candidates = np.asarray(
        [project_rotation(c2w[index, :3, :3] @ view_rotations[index].T) for index in range(count)]
    )

    best: tuple[float, tuple[int, ...], np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    for subset_size in range(count, min_inliers - 1, -1):
        valid_at_size = []
        for subset in itertools.combinations(range(count), subset_size):
            selected = np.asarray(subset, dtype=np.int64)
            center = np.median(centers[selected], axis=0)
            rotation = mean_rotation(base_candidates[selected])
            center_residuals = np.linalg.norm(centers - center, axis=1)
            rotation_residuals = np.asarray(
                [rotation_error_deg(rotation, candidate) for candidate in base_candidates]
            )
            if np.any(center_residuals[selected] > center_threshold):
                continue
            if check_orientation and np.any(
                rotation_residuals[selected] > rotation_threshold_deg
            ):
                continue
            score = float(
                np.sum((center_residuals[selected] / max(center_threshold, 1e-12)) ** 2)
            )
            if check_orientation:
                score += float(
                    np.sum(
                        (
                            rotation_residuals[selected]
                            / max(rotation_threshold_deg, 1e-12)
                        )
                        ** 2
                    )
                )
            valid_at_size.append(
                (score, subset, center, rotation, center_residuals, rotation_residuals)
            )
        if valid_at_size:
            best = min(valid_at_size, key=lambda item: item[0])
            break

    if best is None:
        return _rejected_fit(c2w)

    _, inliers, center, rotation, center_residuals, rotation_residuals = best
    consensus = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    for index, relative in enumerate(view_rotations):
        consensus[index, :3, :3] = rotation @ relative
        consensus[index, :3, 3] = center
    corrected = c2w.copy()
    outliers = sorted(set(range(count)) - set(inliers))
    corrected[outliers] = consensus[outliers]
    return RigFit(
        accepted=True,
        inliers=tuple(inliers),
        center=center,
        rotation=rotation,
        center_residuals=center_residuals,
        rotation_residuals_deg=rotation_residuals,
        corrected_c2w=corrected,
        consensus_c2w=consensus,
    )


def _rejected_fit(c2w: np.ndarray) -> RigFit:
    count = len(c2w)
    return RigFit(
        accepted=False,
        inliers=(),
        center=np.full(3, np.nan),
        rotation=np.full((3, 3), np.nan),
        center_residuals=np.full(count, np.nan),
        rotation_residuals_deg=np.full(count, np.nan),
        corrected_c2w=np.asarray(c2w, dtype=np.float64).copy(),
        consensus_c2w=np.asarray(c2w, dtype=np.float64).copy(),
    )


def parse_yaw(name: str) -> float:
    match = YAW_RE.search(name)
    if match is None:
        raise ValueError(f"Could not parse yaw from image name: {name}")
    return float(match.group("yaw"))


def parse_capture_key(name: str) -> str:
    """Return the source-frame key shared by the four synthetic yaw views."""
    match = YAW_RE.search(name)
    if match is None:
        raise ValueError(f"Could not parse yaw from image name: {name}")
    prefix = name[: match.start()]
    frame_start = prefix.rfind("frame_")
    return prefix[frame_start:] if frame_start >= 0 else prefix


def analyze_chunk(
    extrinsics: np.ndarray,
    names: np.ndarray,
    *,
    center_threshold: float,
    depth: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
    center_depth_ratio: float | None = None,
    rotation_threshold_deg: float,
    min_inliers: int,
    rotate180: bool,
    check_orientation: bool = True,
    validate_repair_overlap: bool = False,
    overlap_pixel_stride: int = 4,
    overlap_ray_tolerance_pixels: float = 1.5,
    overlap_confidence_quantile: float = 0.25,
    overlap_max_depth: float = 15.0,
    overlap_min_matched_rays: int = 100,
    overlap_max_median_ratio: float = 1.0,
    overlap_max_p90_ratio: float = 1.0,
) -> tuple[list[dict[str, object]], list[dict[str, object]], np.ndarray, np.ndarray, np.ndarray]:
    if center_depth_ratio is not None and center_depth_ratio <= 0.0:
        raise ValueError("center_depth_ratio must be positive")
    if depth is not None and len(depth) != len(extrinsics):
        raise ValueError("depth and extrinsics must have the same first dimension")
    if validate_repair_overlap and any(
        value is None for value in (depth, confidence, intrinsics)
    ):
        raise ValueError(
            "depth, confidence, and intrinsics are required for overlap validation"
        )

    c2w = extrinsics_to_c2w(extrinsics)
    corrected_c2w = c2w.copy()
    consensus_c2w = c2w.copy()
    inlier_mask = np.zeros(len(c2w), dtype=bool)
    frame_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []

    grouped_indices: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        grouped_indices.setdefault(parse_capture_key(str(name)), []).append(index)

    expected_yaws = {0.0, 90.0, 180.0, 270.0}
    for group_id, (capture_key, index_list) in enumerate(grouped_indices.items()):
        selected = np.asarray(index_list, dtype=np.int64)
        start = int(selected.min())
        end = int(selected.max()) + 1
        group_names = np.asarray(names[selected])
        yaws = np.asarray([parse_yaw(str(name)) for name in group_names])
        eligible = len(selected) == 4 and set(yaws.tolist()) == expected_yaws
        effective_center_threshold = center_threshold
        median_predicted_depth = None
        if eligible and center_depth_ratio is not None:
            if depth is None:
                raise ValueError("depth is required when center_depth_ratio is set")
            group_depth = np.asarray(depth)[selected]
            positive_depth = group_depth[np.isfinite(group_depth) & (group_depth > 0.0)]
            if positive_depth.size:
                median_predicted_depth = float(np.median(positive_depth))
                effective_center_threshold = (
                    median_predicted_depth * center_depth_ratio
                )
        fit = (
            fit_yaw4_group(
                c2w[selected],
                yaws,
                center_threshold=effective_center_threshold,
                rotation_threshold_deg=rotation_threshold_deg,
                min_inliers=min_inliers,
                rotate180=rotate180,
                check_orientation=check_orientation,
            )
            if eligible
            else _rejected_fit(c2w[selected])
        )
        repair_candidate = fit.accepted and len(fit.inliers) < 4
        overlap_metrics: dict[str, object] = {}
        repair_validated = not validate_repair_overlap
        if repair_candidate and validate_repair_overlap:
            overlap_metrics = evaluate_overlap_repair(
                np.asarray(depth)[selected],
                np.asarray(confidence)[selected],
                np.asarray(intrinsics)[selected],
                c2w[selected],
                fit.corrected_c2w,
                yaws,
                pixel_stride=overlap_pixel_stride,
                ray_tolerance_pixels=overlap_ray_tolerance_pixels,
                confidence_quantile=overlap_confidence_quantile,
                max_depth=overlap_max_depth,
            )
            matched = int(overlap_metrics.get("matched_rays", 0))
            raw_median = float(overlap_metrics.get("raw_normalized_median", math.inf))
            repaired_median = float(
                overlap_metrics.get("repaired_normalized_median", math.inf)
            )
            raw_p90 = float(overlap_metrics.get("raw_normalized_p90", math.inf))
            repaired_p90 = float(overlap_metrics.get("repaired_normalized_p90", math.inf))
            repair_validated = (
                matched >= overlap_min_matched_rays
                and repaired_median < raw_median * overlap_max_median_ratio
                and repaired_p90 <= raw_p90 * overlap_max_p90_ratio
            )
        repair_applied = repair_candidate and repair_validated
        if fit.accepted:
            if not repair_candidate or repair_validated:
                corrected_c2w[selected] = fit.corrected_c2w
            consensus_c2w[selected] = fit.consensus_c2w
            for local_index in fit.inliers:
                inlier_mask[selected[local_index]] = True

        group_rows.append(
            {
                "group_id": group_id,
                "capture_key": capture_key,
                "start_local": start,
                "end_local_exclusive": end,
                "local_indices": ",".join(str(index) for index in selected),
                "eligible": eligible,
                "accepted": fit.accepted,
                "num_inliers": len(fit.inliers),
                "inliers": ",".join(str(index) for index in fit.inliers),
                "repair_candidate": repair_candidate,
                "repair_validated": repair_validated if repair_candidate else None,
                "repair_applied": repair_applied,
                "overlap_matched_rays": overlap_metrics.get("matched_rays"),
                "overlap_raw_normalized_median": overlap_metrics.get(
                    "raw_normalized_median"
                ),
                "overlap_repaired_normalized_median": overlap_metrics.get(
                    "repaired_normalized_median"
                ),
                "overlap_raw_normalized_p90": overlap_metrics.get(
                    "raw_normalized_p90"
                ),
                "overlap_repaired_normalized_p90": overlap_metrics.get(
                    "repaired_normalized_p90"
                ),
                "center_threshold": effective_center_threshold,
                "median_predicted_depth": median_predicted_depth,
                "center_depth_ratio": center_depth_ratio,
                "max_center_residual": _finite_max(fit.center_residuals),
                "max_rotation_residual_deg": _finite_max(fit.rotation_residuals_deg),
            }
        )
        inlier_set = set(fit.inliers)
        for offset, (local_index, name) in enumerate(zip(selected, group_names)):
            frame_rows.append(
                {
                    "local_idx": int(local_index),
                    "group_id": group_id,
                    "capture_key": capture_key,
                    "eligible": eligible,
                    "yaw_deg": yaws[offset],
                    "center_threshold": effective_center_threshold,
                    "median_predicted_depth": median_predicted_depth,
                    "center_depth_ratio": center_depth_ratio,
                    "center_residual": _finite_value(fit.center_residuals, offset),
                    "rotation_residual_deg": _finite_value(fit.rotation_residuals_deg, offset),
                    "inlier": offset in inlier_set,
                    "name": str(name),
                }
            )

    return (
        frame_rows,
        group_rows,
        inlier_mask,
        c2w_to_extrinsics(corrected_c2w),
        c2w_to_extrinsics(consensus_c2w),
    )


def _finite_value(values: np.ndarray, index: int) -> float | None:
    value = float(values[index])
    return value if math.isfinite(value) else None


def _finite_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if len(finite) else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path, help="raw_predictions_summary.npz")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--center-threshold", type=float, default=0.10)
    parser.add_argument("--rotation-threshold-deg", type=float, default=5.0)
    parser.add_argument("--min-inliers", type=int, default=3)
    parser.add_argument("--rotate180", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = np.load(args.predictions, allow_pickle=True)
    frame_rows, group_rows, inlier_mask, corrected_extrinsics, consensus_extrinsics = analyze_chunk(
        data["extrinsics"],
        data["names"],
        center_threshold=args.center_threshold,
        rotation_threshold_deg=args.rotation_threshold_deg,
        min_inliers=args.min_inliers,
        rotate180=args.rotate180,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "yaw4_rig_frames.csv", frame_rows)
    write_csv(args.output_dir / "yaw4_rig_groups.csv", group_rows)
    np.save(args.output_dir / "yaw4_rig_inlier_mask.npy", inlier_mask)
    np.save(args.output_dir / "yaw4_rig_corrected_extrinsics.npy", corrected_extrinsics)
    np.save(args.output_dir / "yaw4_rig_consensus_extrinsics.npy", consensus_extrinsics)

    accepted = sum(bool(row["accepted"]) for row in group_rows)
    full = sum(int(row["num_inliers"]) == 4 for row in group_rows)
    summary = {
        "predictions": str(args.predictions),
        "center_threshold": args.center_threshold,
        "rotation_threshold_deg": args.rotation_threshold_deg,
        "min_inliers": args.min_inliers,
        "rotate180": args.rotate180,
        "num_frames": len(frame_rows),
        "num_groups": len(group_rows),
        "accepted_groups": accepted,
        "full_groups": full,
        "three_inlier_groups": sum(int(row["num_inliers"]) == 3 for row in group_rows),
        "rejected_groups": len(group_rows) - accepted,
        "kept_frames": int(np.count_nonzero(inlier_mask)),
    }
    (args.output_dir / "yaw4_rig_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
