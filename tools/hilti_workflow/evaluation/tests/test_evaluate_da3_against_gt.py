from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "evaluate_da3_against_gt.py"
SPEC = importlib.util.spec_from_file_location("evaluate_da3_against_gt", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluation
SPEC.loader.exec_module(evaluation)


def write_pose_file(path: Path, physical_positions: np.ndarray, views: int = 2) -> None:
    rows = []
    for position in physical_positions:
        for view in range(views):
            matrix = np.eye(4)
            matrix[:3, 3] = position + np.array([0.0, 0.002 * view, 0.0])
            rows.append(" ".join(str(value) for value in matrix.reshape(-1)))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_image_names(path: Path, timestamps: np.ndarray, views: int = 2) -> None:
    path.mkdir()
    for frame, timestamp in enumerate(timestamps):
        timestamp_ns = int(round(timestamp * 1e9))
        for view in range(views):
            filename = f"{frame:05d}_v{view:02d}_frame_{frame:05d}_{timestamp_ns}_yaw{view * 90:03d}.jpg"
            (path / filename).touch()


def write_tum(path: Path, timestamps: np.ndarray, positions: np.ndarray) -> None:
    rows = [
        f"{timestamp:.9f} {position[0]} {position[1]} {position[2]} 0 0 0 1"
        for timestamp, position in zip(timestamps, positions)
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


class EvaluateDa3AgainstGtTest(unittest.TestCase):
    def test_cli_defaults_to_paper_rpe_horizon(self) -> None:
        old_argv = sys.argv
        try:
            sys.argv = [
                "evaluate_da3_against_gt.py",
                "--camera-poses",
                "poses.txt",
                "--ground-truth",
                "groundtruth.txt",
                "--output-dir",
                "evaluation",
            ]
            args = evaluation.parse_args()
        finally:
            sys.argv = old_argv

        self.assertEqual(args.rpe_horizon, 10.0)

    def test_collapses_yaw_views_by_timestamp_and_reports_spread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            poses = root / "camera_poses.txt"
            images = root / "images"
            times = np.array([10005.0, 10005.5, 10006.0])
            positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
            write_pose_file(poses, positions, views=4)
            write_image_names(images, times, views=4)

            physical = evaluation.load_physical_trajectory(
                poses,
                image_dir=images,
                start_time=None,
                sample_period=None,
                views_per_frame=1,
                timestamp_scale=1e-9,
                primary_view_index=0,
            )

            self.assertEqual(physical.rendered_pose_count, 12)
            self.assertEqual(physical.trajectory.count, 3)
            np.testing.assert_allclose(physical.trajectory.timestamps, times)
            np.testing.assert_array_equal(physical.group_sizes, [4, 4, 4])
            self.assertGreater(float(physical.centre_spread_max_m.max()), 0.0)

    def test_run_evaluation_recovers_rigid_transform_and_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            poses = root / "camera_poses.txt"
            images = root / "images"
            gt_path = root / "groundtruth.txt"
            output = root / "evaluation"
            times = np.array([10005.0, 10005.5, 10006.0, 10006.5, 10007.0])
            gt_positions = np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [2.0, 2.0, 0.0]]
            )
            estimate_positions = gt_positions + np.array([5.0, -3.0, 0.4])
            write_pose_file(poses, estimate_positions)
            write_image_names(images, times)
            write_tum(gt_path, times, gt_positions)

            payload = evaluation.run_evaluation(
                argparse.Namespace(
                    camera_poses=poses,
                    image_dir=images,
                    start_time=None,
                    sample_period=None,
                    views_per_frame=1,
                    timestamp_scale=1e-9,
                    primary_view_index=0,
                    ground_truth=gt_path,
                    run_name="synthetic",
                    output_dir=output,
                    modes="rigid,sim3",
                    poster_mode="rigid",
                    ignore_before_time=10005.0,
                    max_interpolation_gap=0.75,
                    coverage_threshold=0.99,
                    rpe_horizon=1.0,
                    rpe_tolerance=0.05,
                    floorplan=None,
                    floorplan_resolution=0.01,
                    pointcloud=None,
                    aligned_pointcloud_mode=None,
                )
            )

            self.assertAlmostEqual(payload["coverage"]["fraction"], 1.0)
            self.assertLess(payload["modes"]["rigid"]["ate_3d_m_rmse"], 0.002)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "plots" / "poster_panel.png").is_file())
            self.assertTrue((output / "plots" / "poster_panel_rigid.png").is_file())
            self.assertTrue((output / "plots" / "poster_panel_sim3.png").is_file())
            self.assertTrue((output / "trajectory_rigid_at_gt_times.tum").is_file())

    def test_incomplete_gt_temporal_coverage_gates_proxy_score(self) -> None:
        rotations = np.repeat(np.eye(3)[None, :, :], 3, axis=0)
        estimated = evaluation.Trajectory(
            np.array([10005.0, 10006.0, 10007.0]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            rotations,
        )
        gt = evaluation.Trajectory(
            np.array([10005.0, 10006.0, 10007.0, 10008.0]),
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            np.repeat(np.eye(3)[None, :, :], 4, axis=0),
        )
        associated = evaluation.associate_ground_truth(
            estimated, gt, ignore_before_time=10005.0, max_interpolation_gap=1.1
        )
        result = evaluation.evaluate_mode(
            "rigid",
            associated,
            coverage_threshold=0.99,
            rpe_horizon=1.0,
            rpe_tolerance=0.05,
        )

        self.assertAlmostEqual(associated.coverage_fraction, 0.75)
        self.assertEqual(result.metrics["coverage_gated_proxy_score_3d"], 0.0)


if __name__ == "__main__":
    unittest.main()
