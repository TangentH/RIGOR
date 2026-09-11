from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.hilti_workflow import run_hilti_batch as batch


def test_shipped_json_manifest_has_fixed_30_run_universe() -> None:
    runs = batch.load_runs(batch.DEFAULT_RUN_MANIFEST)
    assert len(runs) == 30
    assert runs[0]["run_name"] == "floor_1_2025-05-05_run_1"
    assert runs[-1]["relative_path"] == "floor_UG2/2025-12-02/run_1"


def test_official_hash_prefixed_csv_header_is_supported(tmp_path: Path) -> None:
    manifest = tmp_path / "runs.csv"
    manifest.write_text(
        "#Sequence Name,Other\n"
        "floor_1_2025-05-05_run_1,value\n",
        encoding="utf-8",
    )
    runs = batch.load_runs(manifest)
    assert [run["relative_path"] for run in runs] == [
        "floor_1/2025-05-05/run_1"
    ]


def test_child_command_is_gpu_agnostic_and_local_only(tmp_path: Path) -> None:
    config = tmp_path / "paper.yaml"
    config.write_text("paper: true\n", encoding="utf-8")
    args = argparse.Namespace(
        workflow_config=config, da3_env="da3", gsam2_env="gsam2",
        keep_intermediate=True, keep_work=True,
    )
    run = batch.parse_sequence_name("floor_1_2025-05-05_run_1")
    command = batch.build_command(
        args, run, tmp_path / "bag.db3", tmp_path / "output"
    )
    joined = " ".join(command).lower()
    assert "cuda_visible_devices" not in joined
    assert "gdown" not in joined and "download" not in joined
    assert "floorplan" not in joined and "video" not in joined
    assert command[command.index("--relative-path") + 1] == run["relative_path"]


def test_completion_requires_matching_hashes_run_and_config(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    for name in batch.REQUIRED_OUTPUTS[:-1]:
        (output / name).write_bytes((name + " payload").encode())
    config = tmp_path / "paper.yaml"
    config.write_text("paper: true\n", encoding="utf-8")
    manifest = {
        "relative_path": "floor_1/2025-05-05/run_1",
        "final_outputs": {"pointcloud": str(output / "reconstruction.ply")},
        "final_output_provenance": {
            name: batch.file_identity(output / name)
            for name in batch.REQUIRED_OUTPUTS[:-1]
        },
        "workflow_config_provenance": batch.file_identity(config),
    }
    (output / "workflow_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert batch.completed_output(
        output, expected_relative_path=manifest["relative_path"],
        workflow_config=config,
    )
    (output / "camera_poses.txt").write_bytes(b"changed")
    assert not batch.completed_output(
        output, expected_relative_path=manifest["relative_path"],
        workflow_config=config,
    )
