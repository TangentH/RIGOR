from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from tools.hilti_workflow.evaluation.register_trajectory_sim3_icp import (
    fit_sim3,
    run_registration,
    sample_ply_xyz,
    similarity_icp_candidate,
    symmetric_trimmed_rmse,
)


def write_binary_ply(path: Path, points: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = 10, 20, 30
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode()
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def write_yaw4_poses_and_images(
    pose_path: Path, image_dir: Path, centres: np.ndarray, times: np.ndarray
) -> None:
    matrices = []
    image_dir.mkdir()
    for capture, (centre, timestamp) in enumerate(zip(centres, times)):
        for view in range(4):
            matrix = np.eye(4)
            matrix[:3, 3] = centre
            matrices.append(matrix.reshape(-1))
            timestamp_ns = int(round(timestamp * 1e9))
            (image_dir / f"{capture:05d}_v{view:02d}_{timestamp_ns}_yaw{view * 90:03d}.jpg").write_bytes(b"x")
    np.savetxt(pose_path, np.asarray(matrices))


def test_sim3_fit_recovers_known_transform() -> None:
    source = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.1, 1.3, 0.4], [1.7, 0.8, 1.1]]
    )
    angle = np.deg2rad(31.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    scale = 1.4
    translation = np.array([2.0, -3.0, 0.7])
    target = scale * (source @ rotation.T) + translation
    fitted = fit_sim3(source, target)
    assert abs(fitted.scale - scale) < 1e-12
    np.testing.assert_allclose(fitted.rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(fitted.translation, translation, atol=1e-12)




def test_scale_icp_candidate_improves_nearby_cloud() -> None:
    rng = np.random.default_rng(7)
    source = rng.normal(size=(4000, 3)) * np.array([3.0, 1.4, 0.7])
    angle = np.deg2rad(0.7)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    target = 1.015 * (source @ rotation.T) + np.array([0.025, -0.018, 0.012])
    before = symmetric_trimmed_rmse(source, target)
    scale, fitted_rotation, translation, _ = similarity_icp_candidate(source, target)
    moved = scale * (source @ fitted_rotation.T) + translation
    after = symmetric_trimmed_rmse(moved, target)
    assert after < 0.05 * before
    assert abs(scale - 1.015) < 5e-4

def test_end_to_end_trajectory_initializer_preserves_source(tmp_path: Path) -> None:
    count = 24
    parameter = np.linspace(0.0, 2.0, count)
    centres = np.column_stack((2.0 * parameter, np.sin(parameter), 0.3 * parameter**2))
    times = 10005.1 + np.arange(count) * 0.1
    angle = np.deg2rad(-22.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0],
         [np.sin(angle), np.cos(angle), 0.0],
         [0.0, 0.0, 1.0]]
    )
    scale = 1.6
    translation = np.array([-4.0, 1.5, 0.4])
    gt_positions = scale * (centres @ rotation.T) + translation

    pose_path = tmp_path / "camera_poses.txt"
    image_dir = tmp_path / "images"
    write_yaw4_poses_and_images(pose_path, image_dir, centres, times)
    gt_trajectory = tmp_path / "groundtruth.txt"
    with gt_trajectory.open("w", encoding="utf-8") as stream:
        for timestamp, point in zip(times, gt_positions):
            stream.write(
                f"{timestamp:.9f} {point[0]:.12f} {point[1]:.12f} {point[2]:.12f} 0 0 0 1\n"
            )

    source_points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    source_cloud = tmp_path / "source.ply"
    gt_cloud = tmp_path / "gt.ply"
    write_binary_ply(source_cloud, source_points)
    write_binary_ply(gt_cloud, scale * (source_points @ rotation.T) + translation)
    source_digest = hashlib.sha256(source_cloud.read_bytes()).hexdigest()
    output = tmp_path / "registered"
    args = argparse.Namespace(
        source_cloud=source_cloud,
        camera_poses=pose_path,
        image_dir=image_dir,
        timestamp_archive=None,
        ground_truth_trajectory=gt_trajectory,
        ground_truth_cloud=gt_cloud,
        output_dir=output,
        run_name="synthetic",
        gt_trajectory_to_geometry=None,
        coverage_threshold=0.99,
        ignore_before_time=10005.0,
        max_interpolation_gap=0.75,
        icp_reference_cloud=None,
        icp_sample_points=250_000,
        icp_min_improvement=0.005,
        icp_min_source_fitness=0.05,
        icp_max_translation=2.0,
        icp_max_rotation_deg=12.0,
        icp_min_relative_scale=0.5,
        icp_max_relative_scale=2.0,
        disable_icp_scale=False,
        disable_icp=True,
        force=False,
    )
    report = run_registration(args)

    assert report["selection"] == "trajectory_sim3"
    assert report["coverage"]["fraction"] == 1.0
    assert report["scale_icp"]["attempted"] is False
    assert abs(report["trajectory_sim3"]["scale"] - scale) < 1e-9
    assert hashlib.sha256(source_cloud.read_bytes()).hexdigest() == source_digest
    expected = scale * (source_points @ rotation.T) + translation
    np.testing.assert_allclose(sample_ply_xyz((output / "aligned.ply").resolve(), 10), expected, atol=2e-6)
