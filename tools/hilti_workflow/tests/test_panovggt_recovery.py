from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np

from tools.hilti_workflow.panovggt.run_reconstruction import (
    load_window_checkpoint,
    write_window_checkpoint,
)
from tools.hilti_workflow.run_panovggt_batch import (
    build_command as build_batch_command,
    parse_args as parse_batch_args,
    safe_worker,
)
from tools.hilti_workflow.run_panovggt_rosbag_to_reconstruction import (
    DEFAULT_MASK_PROMPT,
    PANO_POSES,
    PANO_VIEW_COUNT,
    ensure_run_receipt,
    parse_args,
    sequential_inference_ready,
)


class PanoWindowCheckpointTests(unittest.TestCase):
    def test_frozen_full_erp_semantic_view_protocol(self) -> None:
        self.assertEqual(
            PANO_POSES,
            "0,0,0;45,0,0;90,0,0;135,0,0;180,0,0;"
            "225,0,0;270,0,0;315,0,0;0,90,0;0,-90,0",
        )
        self.assertEqual(PANO_VIEW_COUNT, 10)

    def test_worker_labels_are_path_safe(self) -> None:
        self.assertEqual(safe_worker("gpu0.recovery"), "gpu0.recovery")
        for value in ("", "../gpu0", "gpu 0", "gpu0/status"):
            with self.assertRaises(ValueError):
                safe_worker(value)

    def test_single_run_cli_has_mandatory_input_retention(self) -> None:
        argv = [
            "runner", "--rosbag", "bag.db3", "--output-dir", "output",
            "--work-dir", "work", "--relative-path", "floor_UG2/date/run_1",
            "--no-delete-rosbag-on-success",
        ]
        with unittest.mock.patch.object(sys, "argv", argv):
            args = parse_args()
            self.assertFalse(args.delete_rosbag_on_success)
            self.assertTrue(args.keep_work_on_success)
            self.assertFalse(hasattr(args, "skip_floorplan"))

    def test_batch_command_is_local_and_gpu_agnostic(self) -> None:
        argv = ["runner", "--worker-id", "gpu0"]
        with unittest.mock.patch.object(sys, "argv", argv):
            args = parse_batch_args()
        run = {
            "run_name": "floor_1_2025-05-05_run_1",
            "relative_path": "floor_1/2025-05-05/run_1",
        }
        command = build_batch_command(
            args, run, Path("bag.db3"), Path("output"), Path("work")
        )
        joined = " ".join(command).lower()
        self.assertNotIn("download", joined)
        self.assertNotIn("floorplan", joined)
        self.assertNotIn("cuda_visible_devices", joined)
        self.assertIn("--no-delete-rosbag-on-success", command)

    def test_run_receipt_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            work.mkdir()
            request = {"schema_version": 1, "inputs": {"bag": "digest-a"}}
            receipt = ensure_run_receipt(work, request)
            self.assertTrue(receipt.is_file())
            self.assertEqual(ensure_run_receipt(work, request), receipt)
            with self.assertRaisesRegex(RuntimeError, "different inputs"):
                ensure_run_receipt(
                    work, {"schema_version": 1, "inputs": {"bag": "digest-b"}}
                )

        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            work.mkdir()
            (work / "legacy-cache.npz").write_bytes(b"cache")
            with self.assertRaisesRegex(RuntimeError, "no verifiable run receipt"):
                ensure_run_receipt(work, request)

    def test_round_trip_and_input_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "window.npz"
            poses = np.repeat(np.eye(4, dtype=np.float64)[None], 2, axis=0)
            world = np.arange(2 * 3 * 4 * 3, dtype=np.float32).reshape(2, 3, 4, 3)
            stems = ["frame_000.jpg", "frame_001.jpg"]
            write_window_checkpoint(path, 3, 5, stems, poses, world)

            loaded = load_window_checkpoint(path, 3, 5, stems)
            self.assertIsNotNone(loaded)
            loaded_poses, loaded_world = loaded
            np.testing.assert_array_equal(loaded_poses, poses)
            np.testing.assert_array_equal(loaded_world, world)
            self.assertIsNone(load_window_checkpoint(path, 4, 6, stems))
            self.assertIsNone(
                load_window_checkpoint(path, 3, 5, ["different.jpg", "frame_001.jpg"])
            )

    def test_corrupt_checkpoint_is_not_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "window.npz"
            path.write_bytes(b"partial")
            self.assertIsNone(load_window_checkpoint(path, 0, 1, ["frame.jpg"]))

    def test_only_sequential_inference_exports_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(sequential_inference_ready(root))

            (root / "reconstruction.ply").write_bytes(b"ply")
            (root / "workflow_manifest.json").write_text(
                "{\"outputs\": {\"selected_variant\": \"loop\"}}\n",
                encoding="utf-8",
            )
            self.assertFalse(sequential_inference_ready(root))

            (root / "workflow_manifest.json").write_text(
                "{\"outputs\": {\"selected_variant\": \"sequential\"}}\n",
                encoding="utf-8",
            )
            self.assertTrue(sequential_inference_ready(root))

            (root / "workflow_manifest.json").write_text("{", encoding="utf-8")
            self.assertFalse(sequential_inference_ready(root))



if __name__ == "__main__":
    unittest.main()
