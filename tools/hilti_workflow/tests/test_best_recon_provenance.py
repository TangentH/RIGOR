from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "run_hilti_best_recon_workflow.py"
)
SPEC = importlib.util.spec_from_file_location("run_hilti_best_recon_workflow", MODULE_PATH)
assert SPEC and SPEC.loader
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)


class BestReconProvenanceTests(unittest.TestCase):
    def make_inputs(self, root: Path) -> tuple[argparse.Namespace, Path]:
        images = root / "images"
        masks = root / "masks"
        output = root / "work"
        images.mkdir()
        masks.mkdir()
        output.mkdir()
        (images / "frame.jpg").write_bytes(b"image-a")
        (masks / "frame.npy").write_bytes(b"mask-aa")
        workflow_config = root / "workflow.yaml"
        workflow_config.write_text("test: true\n", encoding="utf-8")
        generated_config = root / "generated.yaml"
        generated_config.write_text("Model: {}\n", encoding="utf-8")
        args = argparse.Namespace(
            image_dir=images,
            confidence_zero_mask_dir=masks,
            workflow_config=workflow_config,
            output_dir=output,
            skip_da3=False,
            force=False,
        )
        return args, generated_config

    def test_content_tree_detects_same_name_same_size_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "frame.npy"
            path.write_bytes(b"aaaa")
            before = WORKFLOW.directory_content_signature(directory, {".npy"})
            path.write_bytes(b"bbbb")
            after = WORKFLOW.directory_content_signature(directory, {".npy"})
            self.assertEqual(before["filename_set_sha256"], after["filename_set_sha256"])
            self.assertEqual(before["total_size_bytes"], after["total_size_bytes"])
            self.assertNotEqual(before["content_tree_sha256"], after["content_tree_sha256"])

    def test_completed_stage_rejects_same_size_output_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output.ply"
            output.write_bytes(b"aaaa")
            request = {"stage": "unit", "value": 1}
            record = root / "stage.json"
            WORKFLOW.write_stage_record(
                record,
                stage="unit",
                status="complete",
                request=request,
                outputs={"output": WORKFLOW.file_provenance(output)},
            )
            self.assertTrue(
                WORKFLOW.completed_stage_is_reusable(
                    record,
                    stage="unit",
                    request=request,
                    output_paths={"output": output},
                )
            )
            output.write_bytes(b"bbbb")
            self.assertFalse(
                WORKFLOW.completed_stage_is_reusable(
                    record,
                    stage="unit",
                    request=request,
                    output_paths={"output": output},
                )
            )

    def test_run_da3_refuses_existing_unattested_raw_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, generated_config = self.make_inputs(root)
            raw = args.output_dir / "pcd/combined_pcd.ply"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"existing")
            (args.output_dir / "camera_poses.txt").write_text(
                "0 0 0 0 0 0 0 1\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "Refusing to reuse or overwrite"):
                WORKFLOW.run_da3(args, {"runtime": {"gpu_log_interval_s": 10}}, generated_config)
            self.assertEqual(raw.read_bytes(), b"existing")

    def test_manifest_binds_selected_clean_output_and_rechecks_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args, generated_config = self.make_inputs(root)
            pcd = args.output_dir / "pcd"
            logs = args.output_dir / "logs"
            pcd.mkdir(parents=True)
            logs.mkdir()
            raw = pcd / "combined_pcd.ply"
            clean = pcd / "combined_pcd_clean_voxel005.ply"
            poses = args.output_dir / "camera_poses.txt"
            raw.write_bytes(b"raw-cloud")
            clean.write_bytes(b"clean-cloud")
            poses.write_text("0 0 0 0 0 0 0 1\n", encoding="utf-8")

            da3_request = WORKFLOW.da3_stage_request(args, generated_config)
            WORKFLOW.write_stage_record(
                logs / "stage_da3_streaming_provenance.json",
                stage="da3_streaming",
                status="complete",
                request=da3_request,
                outputs={
                    "raw_pointcloud": WORKFLOW.file_provenance(raw),
                    "camera_poses": WORKFLOW.file_provenance(poses),
                },
            )
            clean_cfg = {
                "enabled": True,
                "voxel_size": 0.05,
                "min_points_per_voxel": 1,
                "min_occupied_neighbor_voxels": 1,
                "neighbor_connectivity": 6,
                "save_removed": False,
            }
            clean_request = {
                "schema_version": 1,
                "stage": "pointcloud_cleaning",
                "input": WORKFLOW.file_provenance(raw),
                "settings": dict(clean_cfg),
                "implementation": WORKFLOW.file_provenance(
                    WORKFLOW.REPO
                    / "tools/hilti_workflow/postprocess/clean_pointcloud_outliers.py"
                ),
            }
            WORKFLOW.write_stage_record(
                logs / "stage_pointcloud_cleaning_provenance.json",
                stage="pointcloud_cleaning",
                status="complete",
                request=clean_request,
                outputs={"cleaned_pointcloud": WORKFLOW.file_provenance(clean)},
            )
            outputs = {
                "generated_da3_config": str(generated_config),
                "cleaned_pointcloud": str(clean),
            }
            workflow = {"pointcloud_clean": clean_cfg}
            effective_config = (
                args.output_dir / "configs/workflow_config_used.yaml"
            )
            effective_config.parent.mkdir(parents=True, exist_ok=True)
            effective_config.write_text(
                "pointcloud_clean:\n  enabled: true\n", encoding="utf-8"
            )
            WORKFLOW.write_manifest(args.output_dir, args, workflow, outputs, {})
            manifest = json.loads(
                (args.output_dir / "workflow_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["schema_version"], 3)
            self.assertNotIn("use_floorplan_alignment", manifest["selected_output"])
            self.assertEqual(
                manifest["selected_output"]["pointcloud"]["sha256"],
                WORKFLOW.sha256_file(clean),
            )

            clean.write_bytes(b"other-cloud")
            with self.assertRaisesRegex(RuntimeError, "no longer matches provenance"):
                WORKFLOW.write_manifest(args.output_dir, args, workflow, outputs, {})


if __name__ == "__main__":
    unittest.main()
