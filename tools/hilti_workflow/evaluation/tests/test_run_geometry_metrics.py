from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.hilti_workflow.evaluation.run_geometry_metrics import (
    compute_metrics,
    discover,
    stable_seed,
)


class RunGeometryMetricsTest(unittest.TestCase):
    def test_identical_cloud_metrics(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        metrics = compute_metrics(points, points.copy(), [0.05])
        self.assertAlmostEqual(metrics["chamfer_distance"], 0.0)
        self.assertAlmostEqual(metrics["threshold_metrics"]["tau=0.05"]["fscore"], 1.0)

    def test_stable_seed_depends_on_role(self) -> None:
        self.assertEqual(stable_seed("floor/date/run", "gt"), stable_seed("floor/date/run", "gt"))
        self.assertNotEqual(stable_seed("floor/date/run", "gt"), stable_seed("floor/date/run", "prediction"))

    def test_discovery_preserves_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recon = root / "floor_1/2025-05-05/run_1/reconstruction"
            recon.mkdir(parents=True)
            (recon / "aligned_roi.ply").write_bytes(b"ply")
            self.assertEqual(discover(root, "aligned_roi.ply")[0][0], "floor_1/2025-05-05/run_1")


if __name__ == "__main__":
    unittest.main()
