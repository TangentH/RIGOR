from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


WORKFLOW = Path(__file__).resolve().parents[1]

MERGE_PATH = WORKFLOW / "postprocess/merge_confidence_zero_masks.py"
MERGE_SPEC = importlib.util.spec_from_file_location(
    "merge_confidence_zero_masks_content_test", MERGE_PATH
)
assert MERGE_SPEC is not None and MERGE_SPEC.loader is not None
MERGE = importlib.util.module_from_spec(MERGE_SPEC)
MERGE_SPEC.loader.exec_module(MERGE)

VIEW_DIR = WORKFLOW / "view_generation"
sys.path.insert(0, str(VIEW_DIR))
try:
    STATIC_PATH = VIEW_DIR / "equirect_static_mask_to_yaw4_masks.py"
    STATIC_SPEC = importlib.util.spec_from_file_location(
        "equirect_static_mask_content_test", STATIC_PATH
    )
    assert STATIC_SPEC is not None and STATIC_SPEC.loader is not None
    STATIC = importlib.util.module_from_spec(STATIC_SPEC)
    STATIC_SPEC.loader.exec_module(STATIC)
finally:
    sys.path.remove(str(VIEW_DIR))


def save_bool_mask(path: Path, values: list[list[bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(values, dtype=bool))


class MaskContentProvenanceTests(unittest.TestCase):
    def test_same_name_same_size_replacement_changes_content_tree_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "frame.npy"
            save_bool_mask(path, [[False, False], [False, False]])
            files = MERGE.mask_files(root)
            filename_before = MERGE.filename_set_signature(files)
            tree_before = MERGE.content_tree_signature(root, files)
            size_before = path.stat().st_size

            save_bool_mask(path, [[True, True], [True, True]])
            self.assertEqual(path.stat().st_size, size_before)
            files_after = MERGE.mask_files(root)
            filename_after = MERGE.filename_set_signature(files_after)
            tree_after = MERGE.content_tree_signature(root, files_after)

            self.assertEqual(
                filename_before["filename_set_sha256"],
                filename_after["filename_set_sha256"],
            )
            self.assertEqual(
                filename_before["total_size_bytes"],
                filename_after["total_size_bytes"],
            )
            self.assertNotEqual(
                tree_before["tree_sha256"], tree_after["tree_sha256"]
            )
            self.assertNotEqual(
                tree_before["files"][0]["sha256"],
                tree_after["files"][0]["sha256"],
            )

    def test_static_file_identity_rejects_wrong_same_size_bag_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            bag = Path(temporary) / "rosbag.db3"
            bag.write_bytes(b"AAAA")
            expected = STATIC.stable_file_identity(bag)
            bag.write_bytes(b"BBBB")
            self.assertEqual(bag.stat().st_size, expected["size_bytes"])
            with self.assertRaisesRegex(RuntimeError, "rosbag changed"):
                STATIC.assert_file_identity_unchanged(
                    "rosbag", bag, expected
                )

    def _merge_args(
        self, primary: Path, secondary: Path, output: Path
    ) -> list[str]:
        return [
            "merge",
            "--primary-dir",
            str(primary),
            "--secondary-dir",
            str(secondary),
            "--output-dir",
            str(output),
        ]

    def test_merge_schema_v3_binds_request_and_all_content_trees(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "primary"
            secondary = root / "secondary"
            output = root / "merged/masks_npy"
            save_bool_mask(primary / "frame.npy", [[True, False]])
            save_bool_mask(secondary / "frame.npy", [[False, True]])

            with mock.patch.object(
                sys, "argv", self._merge_args(primary, secondary, output)
            ):
                self.assertEqual(MERGE.main(), 0)

            manifest = json.loads(
                (output.parent / "merge_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["schema_version"], 3)
            self.assertEqual(
                manifest["request_sha256"],
                MERGE.canonical_json_sha256(manifest["request"]),
            )
            self.assertEqual(
                set(manifest["content_trees"]),
                {"primary", "secondary", "merged"},
            )
            self.assertTrue(
                manifest["validation"]["inputs_unchanged_during_run"]
            )
            self.assertEqual(
                np.load(output / "frame.npy", allow_pickle=False).tolist(),
                [[True, True]],
            )

    def test_merge_rejects_input_changed_during_run_and_unpublishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "primary"
            secondary = root / "secondary"
            output = root / "merged/masks_npy"
            save_bool_mask(primary / "frame.npy", [[False, False]])
            save_bool_mask(secondary / "frame.npy", [[True, False]])
            manifest = output.parent / "merge_summary.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"stale": true}\n', encoding="utf-8")
            original_save = MERGE.atomic_save_npy
            changed = False

            def save_then_mutate(path: Path, array: np.ndarray) -> None:
                nonlocal changed
                original_save(path, array)
                if not changed:
                    save_bool_mask(
                        primary / "frame.npy", [[True, True]]
                    )
                    changed = True

            with (
                mock.patch.object(
                    sys,
                    "argv",
                    self._merge_args(primary, secondary, output),
                ),
                mock.patch.object(
                    MERGE, "atomic_save_npy", side_effect=save_then_mutate
                ),
                self.assertRaisesRegex(
                    RuntimeError, "primary input content tree changed"
                ),
            ):
                MERGE.main()
            self.assertFalse(manifest.exists())

    def test_merge_rejects_output_pollution_and_unpublishes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "primary"
            secondary = root / "secondary"
            output = root / "merged/masks_npy"
            save_bool_mask(primary / "frame.npy", [[False]])
            save_bool_mask(secondary / "frame.npy", [[True]])
            save_bool_mask(output / "pollution.npy", [[True]])
            manifest = output.parent / "merge_summary.json"
            manifest.write_text('{"stale": true}\n', encoding="utf-8")

            with (
                mock.patch.object(
                    sys,
                    "argv",
                    self._merge_args(primary, secondary, output),
                ),
                self.assertRaisesRegex(
                    RuntimeError, "Merged-mask filename parity failed"
                ),
            ):
                MERGE.main()
            self.assertFalse(manifest.exists())

    def _static_fixture(self, root: Path) -> dict[str, Path]:
        equirect = root / "equirect"
        pinhole = root / "pinhole"
        output = root / "device"
        equirect.mkdir()
        pinhole.mkdir()
        bag = root / "rosbag.db3"
        yaml_path = root / "kalibr.yaml"
        static_mask = root / "static_mask.png"
        bag.write_bytes(b"AAAA")
        yaml_path.write_text("kalibr\n", encoding="utf-8")
        Image.fromarray(
            np.asarray([[0, 255], [255, 0]], dtype=np.uint8)
        ).save(static_mask)
        Image.fromarray(
            np.zeros((2, 4, 3), dtype=np.uint8)
        ).save(equirect / "frame_123.png")
        Image.fromarray(
            np.zeros((2, 2, 3), dtype=np.uint8)
        ).save(pinhole / "00000_v00_frame_123_yaw000.jpg")
        return {
            "equirect": equirect,
            "pinhole": pinhole,
            "output": output,
            "bag": bag,
            "yaml": yaml_path,
            "mask": static_mask,
        }

    def _static_argv(self, paths: dict[str, Path]) -> list[str]:
        return [
            "static",
            "--bag",
            str(paths["bag"]),
            "--equirect-dir",
            str(paths["equirect"]),
            "--pinhole-dir",
            str(paths["pinhole"]),
            "--yaml",
            str(paths["yaml"]),
            "--mask",
            str(paths["mask"]),
            "--output-dir",
            str(paths["output"]),
            "--width",
            "2",
            "--height",
            "2",
            "--yaws",
            "0",
            "--progress-interval-s",
            "0",
        ]

    def _run_static(
        self,
        paths: dict[str, Path],
        load_imu_side_effect: object | None = None,
        atomic_write_side_effect: object | None = None,
    ) -> None:
        map_x, map_y = np.meshgrid(
            np.arange(2, dtype=np.float32),
            np.arange(2, dtype=np.float32),
        )
        load_imu = (
            load_imu_side_effect
            if load_imu_side_effect is not None
            else mock.DEFAULT
        )
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(sys, "argv", self._static_argv(paths))
            )
            stack.enter_context(
                mock.patch.object(
                    STATIC,
                    "load_cam0_from_yaml",
                    return_value=(np.eye(3), 0.0),
                )
            )
            imu_patch = stack.enter_context(
                mock.patch.object(STATIC, "load_imu_series")
            )
            if load_imu is mock.DEFAULT:
                imu_patch.return_value = (
                    np.asarray([123], dtype=np.int64),
                    np.asarray([[0.0, 0.0, 1.0]]),
                )
            else:
                imu_patch.side_effect = load_imu
            stack.enter_context(
                mock.patch.object(
                    STATIC, "extract_frame_timestamp", return_value=123
                )
            )
            stack.enter_context(
                mock.patch.object(
                    STATIC,
                    "level_rotation_from_gravity",
                    return_value=np.eye(3, dtype=np.float32),
                )
            )
            stack.enter_context(
                mock.patch.object(
                    STATIC,
                    "build_remap_from_rotation",
                    return_value=(map_x, map_y),
                )
            )
            if atomic_write_side_effect is not None:
                stack.enter_context(
                    mock.patch.object(
                        STATIC,
                        "atomic_write_json",
                        side_effect=atomic_write_side_effect,
                    )
                )
            STATIC.main()

    def test_static_schema_v2_binds_bag_trees_request_and_publishes_last(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._static_fixture(Path(temporary))
            writes: list[str] = []
            original_write = STATIC.atomic_write_json

            def record_write(path: Path, payload: dict[str, object]) -> None:
                writes.append(path.name)
                original_write(path, payload)

            self._run_static(
                paths, atomic_write_side_effect=record_write
            )
            manifest = json.loads(
                (paths["output"] / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(writes[-1], "manifest.json")
            self.assertEqual(
                manifest["inputs"]["bag_size_bytes"],
                paths["bag"].stat().st_size,
            )
            self.assertEqual(
                manifest["inputs"]["bag_sha256"],
                hashlib.sha256(paths["bag"].read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest["request_sha256"],
                STATIC.canonical_json_sha256(manifest["request"]),
            )
            self.assertEqual(
                manifest["inputs"]["equirect_image_tree"]["count"], 1
            )
            self.assertEqual(
                manifest["inputs"]["pinhole_image_tree"]["count"], 1
            )
            self.assertEqual(
                manifest["outputs"]["mask_content_tree"]["count"], 1
            )
            self.assertTrue(
                manifest["validation"]["inputs_unchanged_during_run"]
            )

    def test_static_rejects_bag_changed_while_running_and_no_manifest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._static_fixture(Path(temporary))
            manifest = paths["output"] / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{"stale": true}\n', encoding="utf-8")

            def mutate_bag(
                _bag: str, **_kwargs: object
            ) -> tuple[np.ndarray, np.ndarray]:
                paths["bag"].write_bytes(b"BBBB")
                return (
                    np.asarray([123], dtype=np.int64),
                    np.asarray([[0.0, 0.0, 1.0]]),
                )

            with self.assertRaisesRegex(RuntimeError, "rosbag changed"):
                self._run_static(
                    paths, load_imu_side_effect=mutate_bag
                )
            self.assertFalse(manifest.exists())

    def test_static_rejects_pinhole_tree_membership_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._static_fixture(Path(temporary))

            def add_pinhole(
                _bag: str, **_kwargs: object
            ) -> tuple[np.ndarray, np.ndarray]:
                Image.fromarray(
                    np.zeros((2, 2, 3), dtype=np.uint8)
                ).save(paths["pinhole"] / "late_input.jpg")
                return (
                    np.asarray([123], dtype=np.int64),
                    np.asarray([[0.0, 0.0, 1.0]]),
                )

            with self.assertRaisesRegex(
                RuntimeError, "pinhole image content tree changed"
            ):
                self._run_static(
                    paths, load_imu_side_effect=add_pinhole
                )
            self.assertFalse(
                (paths["output"] / "manifest.json").exists()
            )

    def test_static_rejects_output_pollution_and_no_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._static_fixture(Path(temporary))
            save_bool_mask(
                paths["output"] / "masks_npy/pollution.npy", [[True]]
            )
            manifest = paths["output"] / "manifest.json"
            manifest.write_text('{"stale": true}\n', encoding="utf-8")

            with self.assertRaisesRegex(
                RuntimeError, "output filename parity failed"
            ):
                self._run_static(paths)
            self.assertFalse(manifest.exists())


if __name__ == "__main__":
    unittest.main()
