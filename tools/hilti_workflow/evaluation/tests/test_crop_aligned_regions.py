from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.hilti_workflow.evaluation.crop_aligned_regions import roi_membership, safe_label


class CropAlignedRegionsTest(unittest.TestCase):
    def test_roi_membership_partitions_xy_and_z(self) -> None:
        vertices = np.zeros(4, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1")])
        vertices["x"] = [1, 3, 1, 20]
        vertices["y"] = [1, 3, 1, 1]
        vertices["z"] = [1, 1, 5, 1]
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :5] = True
        entry = {"matrix_3x3": np.eye(3).tolist(), "z_min": 0.0, "z_max": 2.0}
        np.testing.assert_array_equal(roi_membership(vertices, entry, mask), [True, True, False, False])

    def test_label_is_path_safe(self) -> None:
        self.assertEqual(safe_label("crop0"), "crop0")
        with self.assertRaises(ValueError):
            safe_label("../crop")


if __name__ == "__main__":
    unittest.main()
