#!/usr/bin/env python3
"""Evaluate a DA3-Streaming Hilti reconstruction against released trajectory GT.

DA3-Streaming writes one camera-to-world 4x4 matrix for every rendered image.
The Hilti workflow often renders several yaw views from one physical panorama,
so this tool first collapses poses that share the ROS image timestamp into one
physical camera centre. Translation evaluation is then performed on the ground
truth timestamps, after interpolating the sampled estimate in time.

The exported TUM files contain the selected virtual-view orientation. They are
useful for inspecting this reconstruction, but are not a drop-in submission
trajectory for the physical Insta360 camera without an orientation conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


SCORE_A = 100.0
SCORE_C = 0.46051701859880917
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
TIMESTAMP_RE = re.compile(r"(?<!\d)(\d{10,})(?!\d)")
VIEW_RE = re.compile(r"(?:^|_)v(?P<view>\d+)(?:_|$)")


@dataclass(frozen=True)
class Trajectory:
    timestamps: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray

    @property
    def count(self) -> int:
        return int(self.timestamps.shape[0])

    def subset(self, mask: np.ndarray) -> "Trajectory":
        return Trajectory(self.timestamps[mask], self.positions[mask], self.rotations[mask])


@dataclass(frozen=True)
class PhysicalTrajectory:
    trajectory: Trajectory
    rendered_pose_count: int
    group_sizes: np.ndarray
    centre_spread_max_m: np.ndarray
    timestamp_source: str


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply_positions(self, positions: np.ndarray) -> np.ndarray:
        return self.scale * (positions @ self.rotation.T) + self.translation

    def apply_rotations(self, rotations: np.ndarray) -> np.ndarray:
        return self.rotation[None, :, :] @ rotations


@dataclass(frozen=True)
class AssociatedTrajectory:
    estimated: Trajectory
    ground_truth: Trajectory
    eligible_gt_count: int
    matched_gt_count: int
    coverage_fraction: float
    first_eval_timestamp: float
    last_eval_timestamp: float


@dataclass(frozen=True)
class ModeResult:
    mode: str
    transform: Similarity
    estimated: Trajectory
    ground_truth: Trajectory
    error_3d_m: np.ndarray
    error_xy_m: np.ndarray
    rpe_translation_m: np.ndarray
    metrics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-poses", required=True, type=Path, help="DA3 camera_poses.txt")
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Rendered DA3 input images. Their filenames provide exact ROS timestamps.",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="Approximate first physical-frame time; use only when rendered images were not retained.",
    )
    parser.add_argument(
        "--sample-period",
        type=float,
        help="Approximate seconds between physical frames, used with --start-time.",
    )
    parser.add_argument(
        "--views-per-frame",
        type=int,
        default=1,
        help="Rendered views per physical frame for approximate timestamps (typically 4).",
    )
    parser.add_argument(
        "--timestamp-scale",
        type=float,
        default=1e-9,
        help="Scale applied to integer timestamps parsed from image filenames.",
    )
    parser.add_argument(
        "--primary-view-index",
        type=int,
        default=0,
        help="Virtual view whose orientation is retained in diagnostic TUM exports.",
    )
    parser.add_argument("--ground-truth", required=True, type=Path, help="Released Hilti TUM GT file")
    parser.add_argument("--run-name", default=None, help="Label used in the report")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--modes",
        default="rigid,sim3",
        help="Comma-separated evaluation transforms: rigid, sim3, none.",
    )
    parser.add_argument(
        "--poster-mode",
        default=None,
        help="Mode to use in poster_panel.png; defaults to rigid when available.",
    )
    parser.add_argument(
        "--ignore-before-time",
        type=float,
        default=10005.0,
        help="Absolute GT time at which evaluation starts; official runs ignore the first 5 s.",
    )
    parser.add_argument(
        "--max-interpolation-gap",
        type=float,
        default=0.75,
        help="Do not match GT samples bracketed by estimate poses farther apart than this.",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.99,
        help="Required temporal coverage for the coverage-gated proxy score.",
    )
    parser.add_argument("--rpe-horizon", type=float, default=10.0)
    parser.add_argument("--rpe-tolerance", type=float, default=0.05)
    parser.add_argument("--floorplan", type=Path, help="Optional floorplan PNG for trajectory overlay")
    parser.add_argument("--floorplan-resolution", type=float, default=0.01, help="Metres per floorplan pixel")
    parser.add_argument("--pointcloud", type=Path, help="Optional PLY to export through one alignment")
    parser.add_argument(
        "--aligned-pointcloud-mode",
        choices=("rigid", "sim3", "none"),
        default=None,
        help="Write an aligned PLY for this enabled evaluation mode.",
    )
    return parser.parse_args()


def _proper_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    result = u @ vt
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1
        result = u @ vt
    return result


def load_c2w_matrices(path: Path) -> np.ndarray:
    raw = np.loadtxt(path, dtype=np.float64)
    raw = raw.reshape(1, -1) if raw.ndim == 1 else raw
    if raw.shape[1] != 16:
        raise ValueError(f"Expected 16 values per DA3 pose row in {path}, found {raw.shape[1]}")
    matrices = raw.reshape(-1, 4, 4)
    if not np.isfinite(matrices).all():
        raise ValueError(f"Non-finite camera poses found in {path}")
    return matrices


def load_tum_trajectory(path: Path) -> Trajectory:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [float(value) for value in line.split()]
            if len(parts) >= 8:
                rows.append(parts[:8])
    if len(rows) < 2:
        raise ValueError(f"Need at least two TUM poses in {path}")
    array = np.asarray(rows, dtype=np.float64)
    rotations = Rotation.from_quat(array[:, 4:8]).as_matrix()
    return Trajectory(array[:, 0], array[:, 1:4], rotations)


def _parse_timestamp(path: Path, timestamp_scale: float) -> float:
    candidates = TIMESTAMP_RE.findall(path.stem)
    if not candidates:
        raise ValueError(f"Cannot parse a ROS timestamp from image filename: {path.name}")
    token = max(candidates, key=len)
    return int(token) * timestamp_scale


def _parse_view_index(path: Path) -> int | None:
    match = VIEW_RE.search(path.stem)
    return int(match.group("view")) if match else None


def _timestamp_records_from_images(
    image_dir: Path, pose_count: int, timestamp_scale: float
) -> list[tuple[float, int | None]]:
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(image_paths) != pose_count:
        raise ValueError(
            f"Rendered image count ({len(image_paths)}) does not match DA3 pose count ({pose_count}): {image_dir}"
        )
    return [(_parse_timestamp(path, timestamp_scale), _parse_view_index(path)) for path in image_paths]


def _timestamp_records_approximate(
    pose_count: int, start_time: float, sample_period: float, views_per_frame: int
) -> list[tuple[float, int]]:
    if views_per_frame <= 0:
        raise ValueError("--views-per-frame must be positive")
    if sample_period <= 0:
        raise ValueError("--sample-period must be positive")
    return [
        (start_time + (index // views_per_frame) * sample_period, index % views_per_frame)
        for index in range(pose_count)
    ]


def load_physical_trajectory(
    pose_path: Path,
    *,
    image_dir: Path | None,
    start_time: float | None,
    sample_period: float | None,
    views_per_frame: int,
    timestamp_scale: float,
    primary_view_index: int,
) -> PhysicalTrajectory:
    matrices = load_c2w_matrices(pose_path)
    if image_dir is not None:
        records = _timestamp_records_from_images(image_dir, len(matrices), timestamp_scale)
        timestamp_source = f"image_filenames:{image_dir}"
    elif start_time is not None and sample_period is not None:
        records = _timestamp_records_approximate(
            len(matrices), start_time, sample_period, views_per_frame
        )
        timestamp_source = (
            f"approximate:start={start_time:.9f},period={sample_period:.9f},views={views_per_frame}"
        )
    else:
        raise ValueError(
            "Provide --image-dir for exact ROS timestamps, or both --start-time and "
            "--sample-period for an explicitly approximate evaluation."
        )

    grouped: dict[float, list[tuple[int, int | None]]] = {}
    for index, (timestamp, view_index) in enumerate(records):
        grouped.setdefault(timestamp, []).append((index, view_index))

    timestamps: list[float] = []
    positions: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    sizes: list[int] = []
    spreads: list[float] = []
    for timestamp in sorted(grouped):
        members = grouped[timestamp]
        centres = np.asarray([matrices[index, :3, 3] for index, _ in members])
        centre = np.median(centres, axis=0)
        selected_index = next(
            (index for index, view in members if view == primary_view_index), members[0][0]
        )
        timestamps.append(timestamp)
        positions.append(centre)
        rotations.append(_proper_rotation(matrices[selected_index, :3, :3]))
        sizes.append(len(members))
        spreads.append(float(np.max(np.linalg.norm(centres - centre, axis=1))))

    trajectory = Trajectory(
        np.asarray(timestamps, dtype=np.float64),
        np.asarray(positions, dtype=np.float64),
        np.asarray(rotations, dtype=np.float64),
    )
    if trajectory.count < 2:
        raise ValueError("Need at least two distinct physical-frame timestamps for evaluation")
    return PhysicalTrajectory(
        trajectory=trajectory,
        rendered_pose_count=len(matrices),
        group_sizes=np.asarray(sizes, dtype=np.int32),
        centre_spread_max_m=np.asarray(spreads, dtype=np.float64),
        timestamp_source=timestamp_source,
    )


def interpolate_trajectory(trajectory: Trajectory, timestamps: np.ndarray) -> Trajectory:
    positions = np.column_stack(
        [np.interp(timestamps, trajectory.timestamps, trajectory.positions[:, axis]) for axis in range(3)]
    )
    rotations = Slerp(trajectory.timestamps, Rotation.from_matrix(trajectory.rotations))(timestamps).as_matrix()
    return Trajectory(np.asarray(timestamps), positions, rotations)


def associate_ground_truth(
    estimated: Trajectory,
    ground_truth: Trajectory,
    *,
    ignore_before_time: float,
    max_interpolation_gap: float,
) -> AssociatedTrajectory:
    eligible = ground_truth.subset(ground_truth.timestamps >= ignore_before_time)
    if eligible.count == 0:
        raise ValueError("No ground-truth poses remain after --ignore-before-time")
    inside = (eligible.timestamps >= estimated.timestamps[0]) & (
        eligible.timestamps <= estimated.timestamps[-1]
    )
    right = np.searchsorted(estimated.timestamps, eligible.timestamps, side="left")
    exact = (right < estimated.count) & (
        np.abs(estimated.timestamps[np.minimum(right, estimated.count - 1)] - eligible.timestamps)
        < 1e-9
    )
    left_safe = np.clip(right - 1, 0, estimated.count - 1)
    right_safe = np.clip(right, 0, estimated.count - 1)
    bracket_gaps = estimated.timestamps[right_safe] - estimated.timestamps[left_safe]
    matched = inside & (exact | (bracket_gaps <= max_interpolation_gap))
    matched_gt = eligible.subset(matched)
    if matched_gt.count < 3:
        raise RuntimeError("Fewer than three GT poses can be temporally matched to this estimate")
    interpolated = interpolate_trajectory(estimated, matched_gt.timestamps)
    return AssociatedTrajectory(
        estimated=interpolated,
        ground_truth=matched_gt,
        eligible_gt_count=eligible.count,
        matched_gt_count=matched_gt.count,
        coverage_fraction=matched_gt.count / eligible.count,
        first_eval_timestamp=float(eligible.timestamps[0]),
        last_eval_timestamp=float(eligible.timestamps[-1]),
    )


def fit_rigid(source: np.ndarray, target: np.ndarray) -> Similarity:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (target - target_mean).T @ (source - source_mean) / len(source)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    translation = target_mean - source_mean @ rotation.T
    return Similarity(1.0, rotation, translation)


def fit_sim3(source: np.ndarray, target: np.ndarray) -> Similarity:
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if variance <= 1e-12:
        raise ValueError("Cannot fit Sim(3) to a degenerate trajectory")
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = target_mean - scale * (source_mean @ rotation.T)
    return Similarity(scale, rotation, translation)


def path_length(positions: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()) if len(positions) > 1 else 0.0


def stats(prefix: str, values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {f"{prefix}_{key}": None for key in ("mean", "median", "rmse", "p95", "max")}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(values**2))),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_max": float(np.max(values)),
    }


def compute_rpe_translation(
    estimated_positions: np.ndarray,
    ground_truth_positions: np.ndarray,
    timestamps: np.ndarray,
    *,
    horizon: float,
    tolerance: float,
) -> np.ndarray:
    values: list[float] = []
    for index, timestamp in enumerate(timestamps[:-1]):
        target = timestamp + horizon
        future = int(np.argmin(np.abs(timestamps - target)))
        if future <= index or abs(timestamps[future] - target) > tolerance:
            continue
        est_delta = estimated_positions[future] - estimated_positions[index]
        gt_delta = ground_truth_positions[future] - ground_truth_positions[index]
        values.append(float(np.linalg.norm(est_delta - gt_delta)))
    return np.asarray(values, dtype=np.float64)


def proxy_score(errors: np.ndarray) -> float:
    return float(np.mean(SCORE_A * np.exp(-SCORE_C * errors)))


def evaluate_mode(
    mode: str,
    associated: AssociatedTrajectory,
    *,
    coverage_threshold: float,
    rpe_horizon: float,
    rpe_tolerance: float,
) -> ModeResult:
    if mode == "rigid":
        transform = fit_rigid(associated.estimated.positions, associated.ground_truth.positions)
        interpretation = "no-scale SE(3) fit; primary SLAM comparison"
    elif mode == "sim3":
        transform = fit_sim3(associated.estimated.positions, associated.ground_truth.positions)
        interpretation = "scale-adjusted diagnostic; not an official no-scale result"
    elif mode == "none":
        transform = Similarity(1.0, np.eye(3), np.zeros(3))
        interpretation = "no alignment; use for floorplan-registered localization output"
    else:
        raise ValueError(f"Unsupported evaluation mode: {mode}")

    positions = transform.apply_positions(associated.estimated.positions)
    aligned = Trajectory(
        associated.estimated.timestamps,
        positions,
        transform.apply_rotations(associated.estimated.rotations),
    )
    error_3d = np.linalg.norm(positions - associated.ground_truth.positions, axis=1)
    error_xy = np.linalg.norm(positions[:, :2] - associated.ground_truth.positions[:, :2], axis=1)
    rpe = compute_rpe_translation(
        positions,
        associated.ground_truth.positions,
        associated.ground_truth.timestamps,
        horizon=rpe_horizon,
        tolerance=rpe_tolerance,
    )
    covered = associated.coverage_fraction >= coverage_threshold
    metrics: dict[str, Any] = {
        "interpretation": interpretation,
        "alignment_scale": transform.scale,
        "proxy_score_3d": proxy_score(error_3d),
        "proxy_score_xy": proxy_score(error_xy),
        "coverage_gated_proxy_score_3d": proxy_score(error_3d) if covered else 0.0,
        "coverage_gated_proxy_score_xy": proxy_score(error_xy) if covered else 0.0,
        **stats("ate_3d_m", error_3d),
        **stats("ate_xy_m", error_xy),
        **stats("rpe_translation_m", rpe),
    }
    return ModeResult(mode, transform, aligned, associated.ground_truth, error_3d, error_xy, rpe, metrics)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _flatten(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            output.update(_flatten(value, full_key))
        elif not isinstance(value, (list, tuple, np.ndarray)):
            output[full_key] = _json_safe(value)
    return output


def write_metrics_csv(path: Path, payload: dict[str, Any]) -> None:
    flattened = _flatten(payload)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened))
        writer.writeheader()
        writer.writerow(flattened)


def write_tum(path: Path, trajectory: Trajectory) -> None:
    quaternions = Rotation.from_matrix(np.asarray([_proper_rotation(r) for r in trajectory.rotations])).as_quat()
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# timestamp tx ty tz qx qy qz qw\n")
        for timestamp, position, quat in zip(trajectory.timestamps, trajectory.positions, quaternions):
            values = [timestamp, *position.tolist(), *quat.tolist()]
            handle.write(" ".join(f"{value:.9f}" for value in values) + "\n")


def write_matched_csv(path: Path, result: ModeResult) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "estimated_x",
                "estimated_y",
                "estimated_z",
                "gt_x",
                "gt_y",
                "gt_z",
                "error_xy_m",
                "error_3d_m",
            ]
        )
        for timestamp, estimate, gt, error_xy, error_3d in zip(
            result.estimated.timestamps,
            result.estimated.positions,
            result.ground_truth.positions,
            result.error_xy_m,
            result.error_3d_m,
        ):
            writer.writerow([f"{timestamp:.9f}", *estimate, *gt, error_xy, error_3d])


def _setup_floorplan_axis(ax: Any, floorplan: Path | None, resolution: float) -> None:
    if floorplan is None:
        return
    image = np.asarray(Image.open(floorplan).convert("L"))
    height, width = image.shape
    ax.imshow(
        image,
        cmap="gray",
        origin="upper",
        extent=(0.0, width * resolution, 0.0, height * resolution),
        alpha=0.72,
    )


def plot_mode(
    result: ModeResult,
    plots_dir: Path,
    *,
    floorplan: Path | None,
    floorplan_resolution: float,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    figure, ax = plt.subplots(figsize=(8, 7))
    _setup_floorplan_axis(ax, floorplan, floorplan_resolution)
    ax.plot(
        result.ground_truth.positions[:, 0],
        result.ground_truth.positions[:, 1],
        color="tab:green",
        linewidth=2.0,
        label="Ground truth",
    )
    ax.plot(
        result.estimated.positions[:, 0],
        result.estimated.positions[:, 1],
        color="tab:purple",
        linewidth=1.7,
        label=f"DA3 ({result.mode})",
    )
    ax.set_title(f"DA3 vs Ground Truth ({result.mode})")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / f"trajectory_overlay_{result.mode}.png", dpi=220)
    plt.close(figure)

    elapsed = result.estimated.timestamps - result.estimated.timestamps[0]
    figure, ax = plt.subplots(figsize=(9, 4))
    ax.plot(elapsed, result.error_3d_m, label="3D", color="tab:blue")
    ax.plot(elapsed, result.error_xy_m, label="XY", color="tab:orange", alpha=0.9)
    ax.set_xlabel("Time since evaluated segment start [s]")
    ax.set_ylabel("Translation error [m]")
    ax.set_title(f"Translation Error ({result.mode})")
    ax.grid(alpha=0.3)
    ax.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / f"translation_error_{result.mode}.png", dpi=220)
    plt.close(figure)


def plot_poster_panel(
    result: ModeResult,
    path: Path,
    *,
    coverage_fraction: float,
    floorplan: Path | None,
    floorplan_resolution: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    _setup_floorplan_axis(axes[0], floorplan, floorplan_resolution)
    axes[0].plot(result.ground_truth.positions[:, 0], result.ground_truth.positions[:, 1], color="#19984c", lw=2, label="Ground truth")
    axes[0].plot(result.estimated.positions[:, 0], result.estimated.positions[:, 1], color="#6b38a8", lw=2, label="DA3")
    axes[0].axis("equal")
    axes[0].set_xlabel("X [m]")
    axes[0].set_ylabel("Y [m]")
    axes[0].set_title(f"Trajectory Overlay ({result.mode})")
    axes[0].legend(frameon=False)

    axes[1].hist(result.error_3d_m, bins=28, color="#2171b5", alpha=0.9)
    axes[1].set_xlabel("3D translation error [m]")
    axes[1].set_ylabel("GT samples")
    axes[1].set_title("Error Distribution")
    summary = (
        f"ATE RMSE: {result.metrics['ate_3d_m_rmse']:.2f} m\n"
        f"ATE p95: {result.metrics['ate_3d_m_p95']:.2f} m\n"
        f"Coverage: {coverage_fraction * 100:.2f}%\n"
        f"Score proxy: {result.metrics['coverage_gated_proxy_score_3d']:.1f}"
    )
    axes[1].text(
        0.98,
        0.97,
        summary,
        ha="right",
        va="top",
        transform=axes[1].transAxes,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#aaaaaa"},
    )
    figure.tight_layout()
    figure.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(figure)


def export_aligned_pointcloud(path: Path, output: Path, transform: Similarity) -> int:
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("open3d is required for --aligned-pointcloud-mode") from exc
    cloud = o3d.io.read_point_cloud(str(path))
    points = np.asarray(cloud.points)
    if points.size == 0:
        raise RuntimeError(f"Point cloud is empty: {path}")
    cloud.points = o3d.utility.Vector3dVector(transform.apply_positions(points))
    if cloud.has_normals():
        cloud.normals = o3d.utility.Vector3dVector(cloud.normals)
        cloud.normals = o3d.utility.Vector3dVector(
            np.asarray(cloud.normals) @ transform.rotation.T
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_point_cloud(str(output), cloud):
        raise RuntimeError(f"Failed to write aligned point cloud: {output}")
    return int(points.shape[0])


def write_report(
    path: Path,
    *,
    run_name: str,
    physical: PhysicalTrajectory,
    associated: AssociatedTrajectory,
    modes: dict[str, ModeResult],
    poster_mode: str,
    approximate_timestamps: bool,
    aligned_cloud: Path | None,
) -> None:
    lines = [
        f"# DA3 Evaluation: {run_name}",
        "",
        "## Inputs and Coverage",
        "",
        f"- Rendered DA3 poses: `{physical.rendered_pose_count}`",
        f"- Physical timestamp groups: `{physical.trajectory.count}`",
        f"- Timestamp source: `{physical.timestamp_source}`",
        f"- Evaluated GT coverage: `{associated.coverage_fraction * 100:.3f}%` "
        f"({associated.matched_gt_count}/{associated.eligible_gt_count})",
        f"- Maximum yaw-view centre spread: `{physical.centre_spread_max_m.max():.4f} m`",
        "",
        "## Metrics",
        "",
        "| Alignment | Scale | ATE 3D RMSE [m] | ATE 3D p95 [m] | Proxy score 3D |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for mode, result in modes.items():
        lines.append(
            f"| {mode} | {result.metrics['alignment_scale']:.4f} | "
            f"{result.metrics['ate_3d_m_rmse']:.4f} | {result.metrics['ate_3d_m_p95']:.4f} | "
            f"{result.metrics['coverage_gated_proxy_score_3d']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Poster Artifacts",
            "",
            f"- Main panel: [poster_panel.png](plots/poster_panel.png) (`{poster_mode}` mode)",
            f"- Trajectory overlay: [trajectory_overlay_{poster_mode}.png](plots/trajectory_overlay_{poster_mode}.png)",
            f"- Error plot: [translation_error_{poster_mode}.png](plots/translation_error_{poster_mode}.png)",
        ]
    )
    for mode in modes:
        lines.append(f"- {mode} panel: [poster_panel_{mode}.png](plots/poster_panel_{mode}.png)")
    if aligned_cloud is not None:
        lines.append(f"- Aligned point cloud: [{aligned_cloud.name}](aligned_pointcloud/{aligned_cloud.name})")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `rigid` is the no-scale, reference-frame-aligned SLAM comparison.",
            "- `sim3` measures reconstruction quality after correcting scale; report its scale separately and do not present it as a no-scale challenge score.",
            "- `none` is only meaningful for an output already registered into floorplan coordinates.",
            "- Proxy scores reproduce the published exponential translation-error formula locally; the official evaluation server remains authoritative.",
            "- DA3 yaw-view orientations are virtual-camera orientations. The TUM exports here support diagnostics and translation plots, not direct physical-camera submission.",
        ]
    )
    if approximate_timestamps:
        lines.append(
            "- Timestamps were reconstructed from a fixed period rather than retained ROS-stamped rendered images; rerun with `--image-dir` before using the figures as final results."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    modes = tuple(part.strip() for part in args.modes.split(",") if part.strip())
    if not modes:
        raise ValueError("--modes must contain at least one mode")
    for mode in modes:
        if mode not in {"rigid", "sim3", "none"}:
            raise ValueError(f"Unsupported evaluation mode in --modes: {mode}")
    poster_mode = args.poster_mode or ("rigid" if "rigid" in modes else modes[0])
    if poster_mode not in modes:
        raise ValueError("--poster-mode must also be included in --modes")
    if args.aligned_pointcloud_mode and args.aligned_pointcloud_mode not in modes:
        raise ValueError("--aligned-pointcloud-mode must also be included in --modes")
    if args.aligned_pointcloud_mode and args.pointcloud is None:
        raise ValueError("--aligned-pointcloud-mode requires --pointcloud")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    physical = load_physical_trajectory(
        args.camera_poses,
        image_dir=args.image_dir,
        start_time=args.start_time,
        sample_period=args.sample_period,
        views_per_frame=args.views_per_frame,
        timestamp_scale=args.timestamp_scale,
        primary_view_index=args.primary_view_index,
    )
    ground_truth = load_tum_trajectory(args.ground_truth)
    associated = associate_ground_truth(
        physical.trajectory,
        ground_truth,
        ignore_before_time=args.ignore_before_time,
        max_interpolation_gap=args.max_interpolation_gap,
    )
    results = {
        mode: evaluate_mode(
            mode,
            associated,
            coverage_threshold=args.coverage_threshold,
            rpe_horizon=args.rpe_horizon,
            rpe_tolerance=args.rpe_tolerance,
        )
        for mode in modes
    }

    for mode, result in results.items():
        plot_mode(
            result,
            plots_dir,
            floorplan=args.floorplan,
            floorplan_resolution=args.floorplan_resolution,
        )
        write_tum(args.output_dir / f"trajectory_{mode}_at_gt_times.tum", result.estimated)
        write_matched_csv(args.output_dir / f"matched_trajectory_{mode}.csv", result)
        plot_poster_panel(
            result,
            plots_dir / f"poster_panel_{mode}.png",
            coverage_fraction=associated.coverage_fraction,
            floorplan=args.floorplan,
            floorplan_resolution=args.floorplan_resolution,
        )
    plot_poster_panel(
        results[poster_mode],
        plots_dir / "poster_panel.png",
        coverage_fraction=associated.coverage_fraction,
        floorplan=args.floorplan,
        floorplan_resolution=args.floorplan_resolution,
    )

    pointcloud_summary: dict[str, Any] | None = None
    aligned_cloud_path: Path | None = None
    if args.aligned_pointcloud_mode:
        aligned_cloud_path = (
            args.output_dir / "aligned_pointcloud" / f"pointcloud_{args.aligned_pointcloud_mode}.ply"
        )
        point_count = export_aligned_pointcloud(
            args.pointcloud, aligned_cloud_path, results[args.aligned_pointcloud_mode].transform
        )
        pointcloud_summary = {
            "source": str(args.pointcloud),
            "mode": args.aligned_pointcloud_mode,
            "output": str(aligned_cloud_path),
            "point_count": point_count,
        }

    raw_path_length = path_length(associated.estimated.positions)
    gt_path_length = path_length(associated.ground_truth.positions)
    payload: dict[str, Any] = {
        "run_name": args.run_name or args.camera_poses.parent.name,
        "inputs": {
            "camera_poses": str(args.camera_poses),
            "image_dir": None if args.image_dir is None else str(args.image_dir),
            "ground_truth": str(args.ground_truth),
            "floorplan": None if args.floorplan is None else str(args.floorplan),
            "pointcloud": None if args.pointcloud is None else str(args.pointcloud),
        },
        "timestamping": {
            "source": physical.timestamp_source,
            "approximate": args.image_dir is None,
            "first_estimate_timestamp": float(physical.trajectory.timestamps[0]),
            "last_estimate_timestamp": float(physical.trajectory.timestamps[-1]),
        },
        "trajectory_quality": {
            "rendered_pose_count": physical.rendered_pose_count,
            "physical_pose_count": physical.trajectory.count,
            "views_per_physical_frame_median": float(np.median(physical.group_sizes)),
            "views_per_physical_frame_min": int(physical.group_sizes.min()),
            "views_per_physical_frame_max": int(physical.group_sizes.max()),
            "yaw_view_centre_spread_median_m": float(np.median(physical.centre_spread_max_m)),
            "yaw_view_centre_spread_p95_m": float(np.percentile(physical.centre_spread_max_m, 95)),
            "yaw_view_centre_spread_max_m": float(physical.centre_spread_max_m.max()),
            "raw_estimated_path_length_m": raw_path_length,
            "gt_path_length_m": gt_path_length,
            "raw_to_gt_path_length_ratio": raw_path_length / gt_path_length if gt_path_length else None,
        },
        "coverage": {
            "ignore_before_time": args.ignore_before_time,
            "max_interpolation_gap_s": args.max_interpolation_gap,
            "required_fraction": args.coverage_threshold,
            "eligible_gt_pose_count": associated.eligible_gt_count,
            "matched_gt_pose_count": associated.matched_gt_count,
            "fraction": associated.coverage_fraction,
            "percent": associated.coverage_fraction * 100.0,
            "passes_required_fraction": associated.coverage_fraction >= args.coverage_threshold,
        },
        "modes": {mode: result.metrics for mode, result in results.items()},
        "pointcloud_export": pointcloud_summary,
        "artifacts": {
            "poster_mode": poster_mode,
            "poster_panel": str(plots_dir / "poster_panel.png"),
            "poster_panels": {
                mode: str(plots_dir / f"poster_panel_{mode}.png") for mode in results
            },
            "report": str(args.output_dir / "report.md"),
        },
    }
    write_json(args.output_dir / "metrics.json", payload)
    write_metrics_csv(args.output_dir / "metrics.csv", payload)
    write_report(
        args.output_dir / "report.md",
        run_name=payload["run_name"],
        physical=physical,
        associated=associated,
        modes=results,
        poster_mode=poster_mode,
        approximate_timestamps=args.image_dir is None,
        aligned_cloud=aligned_cloud_path,
    )
    return payload


def main() -> int:
    args = parse_args()
    payload = run_evaluation(args)
    coverage = payload["coverage"]
    print(f"run={payload['run_name']}")
    print(
        "coverage="
        f"{coverage['percent']:.3f}% "
        f"({coverage['matched_gt_pose_count']}/{coverage['eligible_gt_pose_count']})"
    )
    for mode, metrics in payload["modes"].items():
        print(
            f"{mode}: scale={metrics['alignment_scale']:.5f} "
            f"ate_3d_rmse_m={metrics['ate_3d_m_rmse']:.4f} "
            f"proxy_score_3d={metrics['coverage_gated_proxy_score_3d']:.2f}"
        )
    print(f"metrics={args.output_dir / 'metrics.json'}")
    print(f"poster_panel={args.output_dir / 'plots' / 'poster_panel.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
