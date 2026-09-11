import argparse
import json
import sys
from pathlib import Path

import pytest

from tools.hilti_workflow import (
    run_hilti_rosbag_to_reconstruction as workflow,
)
from tools.hilti_workflow.run_hilti_rosbag_to_reconstruction import (
    finalize_outputs,
    parse_args,
)


def make_args(root: Path) -> argparse.Namespace:
    workflow_config = root / "workflow.yaml"
    workflow_config.write_text("paper: true\n", encoding="utf-8")
    return argparse.Namespace(
        da3_env="da3", gsam2_env="gsam2", extract_stride=10,
        max_equirect_frames=0, extract_progress_interval_s=5.0,
        pinhole_progress_interval_s=5.0, pinhole_width=768,
        pinhole_height=512, pinhole_fov_deg=95.0,
        imu_method="complementary", imu_tau=2.0, accel_gate_sigma=0.2,
        yaws="0,90,180,270", rotate180=True,
        mask_prompt="person.helmet.", box_threshold=0.3, text_threshold=0.3,
        grounding_batch_size=2, hf_grounding_model="model",
        hf_grounding_revision="revision", workflow_config=workflow_config,
        relative_path="floor_1/2025-05-05/run_1", keep_intermediate=True,
        keep_work=True, delete_rosbag_on_success=False,
    )


def test_standalone_retains_intermediate_and_work_by_default(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["runner", "bag.db3"])
    args = parse_args()
    assert args.keep_intermediate is True
    assert args.keep_work is True

    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "bag.db3", "--no-keep-intermediate", "--no-keep-work"],
    )
    args = parse_args()
    assert args.keep_intermediate is False
    assert args.keep_work is False


def test_standalone_accepts_explicit_no_delete_rosbag_policy(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["runner", "bag.db3", "--no-delete-rosbag-on-success"],
    )

    args = parse_args()

    assert args.delete_rosbag_on_success is False


def test_finalize_outputs_publishes_reconstruction_space_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    work_dir = output_dir / "_da3_work"
    masks_dir = output_dir / "masks"
    pcd = work_dir / "pcd/combined_pcd.ply"
    poses = work_dir / "camera_poses.txt"
    pcd.parent.mkdir(parents=True)
    pcd.write_text("ply\nformat ascii 1.0\nelement vertex 0\nend_header\n", encoding="utf-8")
    poses.write_text(" ".join(str(value) for value in range(16)) + "\n", encoding="utf-8")

    args = make_args(tmp_path)

    finalize_outputs(output_dir, work_dir, masks_dir, 0.0, args, tmp_path / "bag.db3", {})

    manifest = json.loads((output_dir / "workflow_manifest.json").read_text())
    assert (output_dir / "reconstruction.ply").exists()
    assert (output_dir / "camera_poses.txt").exists()
    assert set(manifest["final_outputs"]) == {
        "pointcloud", "camera_poses", "camera_pose_ply"
    }
    assert set(manifest["final_output_provenance"]) == {
        "reconstruction.ply", "camera_poses.txt", "camera_poses.ply"
    }
    assert manifest["settings"]["keep_intermediate"] is True
    assert manifest["settings"]["keep_work"] is True
    assert "delete_rosbag_on_success" not in manifest["settings"]
    assert manifest["settings"]["rosbag_retention_policy"] == "retain"


def _status_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_skipped_pipeline_writes_append_only_status_and_recovery_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bag = tmp_path / "rosbag.db3"
    bag.write_bytes(b"synthetic")
    output = tmp_path / "output"
    argv = [
        "runner",
        str(bag),
        "--output-dir",
        str(output),
        "--skip-equirect",
        "--skip-pinhole",
        "--skip-masks",
        "--skip-da3",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert workflow.main() == 0

    logs = output / "logs"
    checkpoint_path = logs / workflow.CHECKPOINT_NAME
    status_path = logs / workflow.STATUS_JSONL_NAME
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    first_status = status_path.read_bytes()
    expected_stages = [
        "prepare_workdirs",
        "extract_equirect",
        "generate_pinhole_yaw4_imu",
        "gsam2_masks",
        "best_reconstruction_workflow",
        "finalize_outputs",
        "retention_cleanup",
    ]
    assert checkpoint["status"] == "complete"
    assert checkpoint["attempt"] == 1
    assert checkpoint["current_stage"] is None
    assert checkpoint["completed_stages"] == expected_stages
    assert checkpoint["stage_outcomes"]["prepare_workdirs"] == "ready"
    assert all(
        checkpoint["stage_outcomes"][stage] == "cli_skipped"
        for stage in expected_stages[1:-1]
    )
    rows = _status_rows(status_path)
    assert rows[0]["event"] == "started"
    assert rows[-1]["event"] == "complete"
    assert [
        row["stage"] for row in rows if row["event"] == "stage_complete"
    ] == expected_stages
    assert not list(logs.glob(".run_checkpoint.json.tmp-*"))

    assert workflow.main() == 0

    assert status_path.read_bytes().startswith(first_status)
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["attempt"] == 2
    assert checkpoint["completed_stages"] == expected_stages
    rows = _status_rows(status_path)
    second_started = [
        row for row in rows if row["event"] == "started" and row["attempt"] == 2
    ]
    assert second_started[0]["recovered_completed_stages"] == expected_stages


def test_stage_failure_is_persisted_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bag = tmp_path / "rosbag.db3"
    bag.write_bytes(b"synthetic")
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            str(bag),
            "--output-dir",
            str(output),
            "--skip-pinhole",
            "--skip-masks",
            "--skip-da3",
        ],
    )

    def fail_run(*args, **kwargs):
        raise RuntimeError("injected stage failure")

    monkeypatch.setattr(workflow, "run", fail_run)
    with pytest.raises(RuntimeError, match="injected stage failure"):
        workflow.main()

    logs = output / "logs"
    checkpoint = json.loads(
        (logs / workflow.CHECKPOINT_NAME).read_text(encoding="utf-8")
    )
    assert checkpoint["status"] == "failed"
    assert checkpoint["current_stage"] == "extract_equirect"
    assert checkpoint["completed_stages"] == ["prepare_workdirs"]
    assert checkpoint["error"] == {
        "message": "injected stage failure",
        "type": "RuntimeError",
    }
    events = _status_rows(logs / workflow.STATUS_JSONL_NAME)
    assert events[-1]["event"] == "failed"
    assert events[-1]["stage"] == "extract_equirect"
    assert not any(event["event"] == "complete" for event in events)


def test_atomic_checkpoint_failure_preserves_previous_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "logs" / workflow.CHECKPOINT_NAME
    workflow.atomic_write_json(checkpoint, {"status": "previous"})
    previous = checkpoint.read_bytes()

    def fail_fsync(_descriptor):
        raise OSError("injected sync failure")

    monkeypatch.setattr(workflow.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected sync failure"):
        workflow.atomic_write_json(checkpoint, {"status": "replacement"})

    assert checkpoint.read_bytes() == previous
    assert not list(checkpoint.parent.glob(".run_checkpoint.json.tmp-*"))


def test_atomic_copy_failure_preserves_previous_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "destination.bin"
    source.write_bytes(b"replacement")
    destination.write_bytes(b"previous")

    def fail_fsync(_descriptor):
        raise OSError("injected copy sync failure")

    monkeypatch.setattr(workflow.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected copy sync failure"):
        workflow.copy_file(source, destination)

    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".destination.bin.tmp-*"))


def test_finalize_outputs_refuses_different_existing_reconstruction(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "output"
    work_dir = output_dir / "_da3_work"
    masks_dir = output_dir / "masks"
    source = work_dir / "pcd/combined_pcd_clean_voxel005.ply"
    poses = work_dir / "camera_poses.txt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"new reconstruction")
    poses.write_text(" ".join(str(value) for value in range(16)) + "\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    reconstruction = output_dir / "reconstruction.ply"
    reconstruction.write_bytes(b"protected reconstruction")

    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        finalize_outputs(
            output_dir,
            work_dir,
            masks_dir,
            0.0,
            make_args(tmp_path),
            tmp_path / "bag.db3",
            {},
        )

    assert reconstruction.read_bytes() == b"protected reconstruction"
    assert not list(output_dir.glob(".reconstruction.ply.tmp-*"))


def test_stage_provenance_reuses_only_same_request_and_exact_filename_set(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    outputs = tmp_path / "views"
    outputs.mkdir()
    (outputs / "frame.jpg").write_bytes(b"rgb")
    request = {"bag": {"sha256": "a" * 64}, "fov_deg": 90.0}

    reusable, _ = workflow.prepare_stage_provenance(
        logs_dir=logs,
        stage="synthetic_views",
        request=request,
        output_dir=outputs,
        suffixes={".jpg"},
        expected_names={"frame.jpg"},
        force=True,
    )
    assert reusable is False
    complete = workflow.complete_stage_provenance(
        logs_dir=logs,
        stage="synthetic_views",
        request=request,
        output_dir=outputs,
        suffixes={".jpg"},
        expected_names={"frame.jpg"},
    )
    assert len(complete["outputs"]["content_tree_sha256"]) == 64
    reusable, recovered = workflow.prepare_stage_provenance(
        logs_dir=logs,
        stage="synthetic_views",
        request=request,
        output_dir=outputs,
        suffixes={".jpg"},
        expected_names={"frame.jpg"},
        force=False,
    )
    assert reusable is True
    assert recovered == complete

    # A same-name, same-size content change must not be hidden by filename/size
    # parity. This is especially important for fixed-shape .npy mask files.
    (outputs / "frame.jpg").write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="Refusing to relabel stale"):
        workflow.prepare_stage_provenance(
            logs_dir=logs,
            stage="synthetic_views",
            request=request,
            output_dir=outputs,
            suffixes={".jpg"},
            expected_names={"frame.jpg"},
            force=False,
        )

    (outputs / "frame.jpg").write_bytes(b"rgb")

    with pytest.raises(RuntimeError, match="Refusing to relabel stale"):
        workflow.prepare_stage_provenance(
            logs_dir=logs,
            stage="synthetic_views",
            request={**request, "fov_deg": 95.0},
            output_dir=outputs,
            suffixes={".jpg"},
            expected_names={"frame.jpg"},
            force=False,
        )

    (outputs / "extra.jpg").write_bytes(b"rgb")
    with pytest.raises(RuntimeError, match="Refusing to relabel stale"):
        workflow.prepare_stage_provenance(
            logs_dir=logs,
            stage="synthetic_views",
            request=request,
            output_dir=outputs,
            suffixes={".jpg"},
            expected_names={"frame.jpg"},
            force=False,
        )


@pytest.mark.parametrize("corruption", ["missing_request", "forged_hash"])
def test_stage_provenance_rejects_unverifiable_request(
    tmp_path: Path,
    corruption: str,
) -> None:
    logs = tmp_path / "logs"
    outputs = tmp_path / "views"
    outputs.mkdir()
    (outputs / "frame.jpg").write_bytes(b"rgb")
    request = {"bag": {"sha256": "a" * 64}, "fov_deg": 90.0}
    record = workflow.complete_stage_provenance(
        logs_dir=logs,
        stage="synthetic_views",
        request=request,
        output_dir=outputs,
        suffixes={".jpg"},
        expected_names={"frame.jpg"},
    )
    if corruption == "missing_request":
        record.pop("request")
    else:
        record["request_sha256"] = "b" * 64
    workflow.atomic_write_json(
        logs / "stage_synthetic_views_provenance.json", record
    )

    with pytest.raises(RuntimeError, match="Refusing to relabel stale"):
        workflow.prepare_stage_provenance(
            logs_dir=logs,
            stage="synthetic_views",
            request=request,
            output_dir=outputs,
            suffixes={".jpg"},
            expected_names={"frame.jpg"},
            force=False,
        )


def test_frontend_stage_chain_rejects_disconnected_upstream_content_digest(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    erp = tmp_path / "erp"
    views = tmp_path / "views"
    masks = tmp_path / "masks"
    for directory, name, data in (
        (erp, "frame.jpg", b"erp"),
        (views, "view.jpg", b"rgb"),
        (masks, "view.npy", b"mask"),
    ):
        directory.mkdir()
        (directory / name).write_bytes(data)
    extraction_request = {"bag": {"sha256": "a" * 64}}
    extraction = workflow.complete_stage_provenance(
        logs_dir=logs,
        stage="extract_equirect",
        request=extraction_request,
        output_dir=erp,
        suffixes={".jpg"},
    )
    pinhole_request = {"equirect_outputs": extraction["outputs"]}
    pinhole = workflow.complete_stage_provenance(
        logs_dir=logs,
        stage="generate_pinhole_yaw4_imu",
        request=pinhole_request,
        output_dir=views,
        suffixes={".jpg"},
    )
    masks_request = {"pinhole_outputs": pinhole["outputs"]}
    mask_record = workflow.complete_stage_provenance(
        logs_dir=logs,
        stage="gsam2_masks",
        request=masks_request,
        output_dir=masks,
        suffixes={".npy"},
    )
    stages = {
        "extract_equirect": extraction,
        "generate_pinhole_yaw4_imu": pinhole,
        "gsam2_masks": mask_record,
    }
    assert workflow.validate_frontend_stage_chain(stages) == []

    pinhole["request"] = {
        "equirect_outputs": {
            **extraction["outputs"],
            "content_tree_sha256": "c" * 64,
        }
    }
    pinhole["request_sha256"] = workflow.canonical_json_sha256(
        pinhole["request"]
    )
    problems = workflow.validate_frontend_stage_chain(stages)
    assert any(
        "extraction output digest is not chained" in value
        for value in problems
    )


def test_in_progress_same_protocol_stage_can_resume_but_changed_bag_cannot(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    outputs = tmp_path / "masks"
    outputs.mkdir()
    bag = tmp_path / "bag.db3"
    bag.write_bytes(b"first")
    request = {"bag": workflow.required_file_identity(bag), "prompt": "person."}
    reusable, _ = workflow.prepare_stage_provenance(
        logs_dir=logs,
        stage="synthetic_masks",
        request=request,
        output_dir=outputs,
        suffixes={".npy"},
        expected_names={"one.npy", "two.npy"},
        force=False,
    )
    assert reusable is False
    (outputs / "one.npy").write_bytes(b"partial")
    reusable, _ = workflow.prepare_stage_provenance(
        logs_dir=logs,
        stage="synthetic_masks",
        request=request,
        output_dir=outputs,
        suffixes={".npy"},
        expected_names={"one.npy", "two.npy"},
        force=False,
    )
    assert reusable is False

    bag.write_bytes(b"other")
    changed = {"bag": workflow.required_file_identity(bag), "prompt": "person."}
    with pytest.raises(RuntimeError, match="Refusing to relabel stale"):
        workflow.prepare_stage_provenance(
            logs_dir=logs,
            stage="synthetic_masks",
            request=changed,
            output_dir=outputs,
            suffixes={".npy"},
            expected_names={"one.npy", "two.npy"},
            force=False,
        )
