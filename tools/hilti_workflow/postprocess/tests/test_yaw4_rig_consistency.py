from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "yaw4_rig_consistency.py"
SPEC = importlib.util.spec_from_file_location("yaw4_rig_consistency", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def rotation_x(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def make_group(base_rotation: np.ndarray, center: np.ndarray) -> np.ndarray:
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    for index, yaw in enumerate((0.0, 90.0, 180.0, 270.0)):
        poses[index, :3, :3] = base_rotation @ MODULE.view_rotation(yaw, True)
        poses[index, :3, 3] = center
    return poses


class Yaw4RigConsistencyTest(unittest.TestCase):
    def fit(self, poses: np.ndarray, *, check_orientation: bool = True):
        return MODULE.fit_yaw4_group(
            poses,
            np.array([0.0, 90.0, 180.0, 270.0]),
            center_threshold=0.10,
            rotation_threshold_deg=5.0,
            min_inliers=3,
            rotate180=True,
            check_orientation=check_orientation,
        )

    def test_exact_rig_with_common_tilt_keeps_all_views(self):
        base = rotation_x(24.0) @ MODULE.rotation_y(37.0)
        poses = make_group(base, np.array([1.0, -2.0, 0.5]))
        fit = self.fit(poses)
        self.assertTrue(fit.accepted)
        self.assertEqual(fit.inliers, (0, 1, 2, 3))
        np.testing.assert_allclose(fit.corrected_c2w, poses, atol=1e-8)

    def test_one_pose_outlier_recovers_three_view_consensus(self):
        poses = make_group(rotation_x(11.0), np.array([0.2, 0.3, 0.4]))
        poses[3, :3, 3] += np.array([0.5, 0.0, 0.0])
        poses[3, :3, :3] = rotation_x(35.0) @ poses[3, :3, :3]
        fit = self.fit(poses)
        self.assertTrue(fit.accepted)
        self.assertEqual(fit.inliers, (0, 1, 2))
        np.testing.assert_allclose(fit.center, np.array([0.2, 0.3, 0.4]), atol=1e-8)
        np.testing.assert_allclose(fit.corrected_c2w[:3], poses[:3], atol=1e-8)
        np.testing.assert_allclose(fit.corrected_c2w[3], fit.consensus_c2w[3], atol=1e-8)

    def test_two_by_two_split_rejects_group(self):
        first = make_group(np.eye(3), np.zeros(3))
        second = make_group(rotation_x(30.0), np.array([0.5, 0.0, 0.0]))
        poses = first.copy()
        poses[2:] = second[2:]
        fit = self.fit(poses)
        self.assertFalse(fit.accepted)
        self.assertEqual(fit.inliers, ())

    def test_orientation_check_detects_rotation_only_outlier(self):
        poses = make_group(rotation_x(8.0), np.array([0.2, 0.3, 0.4]))
        poses[3, :3, :3] = rotation_x(25.0) @ poses[3, :3, :3]

        orientation_fit = self.fit(poses, check_orientation=True)
        center_only_fit = self.fit(poses, check_orientation=False)

        self.assertEqual(orientation_fit.inliers, (0, 1, 2))
        self.assertEqual(center_only_fit.inliers, (0, 1, 2, 3))

    def test_named_groups_skip_partial_loop_window_boundaries(self):
        poses = np.concatenate(
            [
                make_group(np.eye(3), np.zeros(3))[2:],
                make_group(rotation_x(5.0), np.ones(3)),
                make_group(rotation_x(10.0), np.full(3, 2.0))[:2],
            ]
        )
        names = np.asarray(
            [
                "frame_00000_100_yaw180.jpg",
                "frame_00000_100_yaw270.jpg",
                "frame_00001_200_yaw000.jpg",
                "frame_00001_200_yaw090.jpg",
                "frame_00001_200_yaw180.jpg",
                "frame_00001_200_yaw270.jpg",
                "frame_00002_300_yaw000.jpg",
                "frame_00002_300_yaw090.jpg",
            ]
        )
        extrinsics = MODULE.c2w_to_extrinsics(poses)
        _, groups, _, corrected, _ = MODULE.analyze_chunk(
            extrinsics,
            names,
            center_threshold=0.10,
            rotation_threshold_deg=5.0,
            min_inliers=3,
            rotate180=True,
        )
        self.assertEqual([row["eligible"] for row in groups], [False, True, False])
        self.assertTrue(groups[1]["accepted"])
        np.testing.assert_allclose(corrected, extrinsics, atol=1e-8)

    def test_capture_key_ignores_projected_view_prefix(self):
        names = [
            "00042_v00_frame_00042_10014014007000_yaw000.jpg",
            "00042_v01_frame_00042_10014014007000_yaw090.jpg",
            "00042_v02_frame_00042_10014014007000_yaw180.jpg",
            "00042_v03_frame_00042_10014014007000_yaw270.jpg",
        ]
        keys = {MODULE.parse_capture_key(name) for name in names}
        self.assertEqual(keys, {"frame_00042_10014014007000"})

    def test_depth_ratio_sets_a_per_group_center_threshold(self):
        poses = make_group(np.eye(3), np.zeros(3))
        poses[3, :3, 3] = np.array([0.15, 0.0, 0.0])
        names = np.asarray(
            [f"frame_00000_100_yaw{yaw:03d}.jpg" for yaw in (0, 90, 180, 270)]
        )
        depth = np.full((4, 2, 2), 4.0)

        _, groups, _, _, _ = MODULE.analyze_chunk(
            MODULE.c2w_to_extrinsics(poses),
            names,
            center_threshold=0.10,
            depth=depth,
            center_depth_ratio=0.05,
            rotation_threshold_deg=5.0,
            min_inliers=4,
            rotate180=True,
        )

        self.assertTrue(groups[0]["accepted"])
        self.assertAlmostEqual(groups[0]["center_threshold"], 0.20)
        self.assertAlmostEqual(groups[0]["median_predicted_depth"], 4.0)

    def test_depth_ratio_requires_depth(self):
        poses = make_group(np.eye(3), np.zeros(3))
        names = np.asarray(
            [f"frame_00000_100_yaw{yaw:03d}.jpg" for yaw in (0, 90, 180, 270)]
        )
        with self.assertRaisesRegex(ValueError, "depth is required"):
            MODULE.analyze_chunk(
                MODULE.c2w_to_extrinsics(poses),
                names,
                center_threshold=0.10,
                center_depth_ratio=0.05,
                rotation_threshold_deg=5.0,
                min_inliers=3,
                rotate180=True,
            )

    def test_overlap_gate_rejects_a_worsening_repair(self):
        poses = make_group(np.eye(3), np.zeros(3))
        poses[3, :3, 3] = np.array([0.5, 0.0, 0.0])
        names = np.asarray(
            [f"frame_00000_100_yaw{yaw:03d}.jpg" for yaw in (0, 90, 180, 270)]
        )
        maps = np.ones((4, 8, 8), dtype=np.float64)
        intrinsics = np.repeat(np.eye(3)[None], 4, axis=0)
        metrics = {
            "matched_rays": 200,
            "raw_normalized_median": 0.1,
            "repaired_normalized_median": 0.2,
            "raw_normalized_p90": 0.2,
            "repaired_normalized_p90": 0.3,
        }

        with mock.patch.object(MODULE, "evaluate_overlap_repair", return_value=metrics):
            _, groups, _, corrected, _ = MODULE.analyze_chunk(
                MODULE.c2w_to_extrinsics(poses),
                names,
                center_threshold=0.10,
                depth=maps,
                confidence=maps,
                intrinsics=intrinsics,
                rotation_threshold_deg=5.0,
                min_inliers=3,
                rotate180=True,
                validate_repair_overlap=True,
            )

        self.assertTrue(groups[0]["repair_candidate"])
        self.assertFalse(groups[0]["repair_validated"])
        self.assertFalse(groups[0]["repair_applied"])
        np.testing.assert_allclose(
            corrected, MODULE.c2w_to_extrinsics(poses), atol=1e-12
        )

    def test_overlap_gate_applies_an_improving_repair(self):
        poses = make_group(np.eye(3), np.zeros(3))
        poses[3, :3, 3] = np.array([0.5, 0.0, 0.0])
        names = np.asarray(
            [f"frame_00000_100_yaw{yaw:03d}.jpg" for yaw in (0, 90, 180, 270)]
        )
        maps = np.ones((4, 8, 8), dtype=np.float64)
        intrinsics = np.repeat(np.eye(3)[None], 4, axis=0)
        metrics = {
            "matched_rays": 200,
            "raw_normalized_median": 0.2,
            "repaired_normalized_median": 0.1,
            "raw_normalized_p90": 0.3,
            "repaired_normalized_p90": 0.2,
        }

        with mock.patch.object(MODULE, "evaluate_overlap_repair", return_value=metrics):
            _, groups, _, corrected, consensus = MODULE.analyze_chunk(
                MODULE.c2w_to_extrinsics(poses),
                names,
                center_threshold=0.10,
                depth=maps,
                confidence=maps,
                intrinsics=intrinsics,
                rotation_threshold_deg=5.0,
                min_inliers=3,
                rotate180=True,
                validate_repair_overlap=True,
            )

        self.assertTrue(groups[0]["repair_validated"])
        self.assertTrue(groups[0]["repair_applied"])
        np.testing.assert_allclose(corrected[3], consensus[3])


if __name__ == "__main__":
    unittest.main()
