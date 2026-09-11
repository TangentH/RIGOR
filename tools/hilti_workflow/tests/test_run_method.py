from __future__ import annotations

import argparse
from pathlib import Path

from tools.hilti_workflow import run_method


def test_da3_launcher_uses_paper_config_without_presentation_features(
    tmp_path: Path,
) -> None:
    bag = tmp_path / "rosbag.db3"
    bag.write_bytes(b"bag")
    args = argparse.Namespace(
        method="da3", rosbag=bag,
        workflow_config=run_method.DEFAULT_DA3_CONFIG, force=False,
    )
    command = run_method.build_command(args, tmp_path / "output", [])
    assert str(run_method.DEFAULT_DA3_CONFIG) in command
    assert "floorplan" not in " ".join(command).lower()
    assert "video" not in " ".join(command).lower()


def test_panovggt_launcher_preserves_prepared_input_interface(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(
        method="panovggt",
        panovggt_repo=tmp_path / "PanoVGGT",
        checkpoint=tmp_path / "model.pt",
        image_dir=tmp_path / "erp",
        erp_mask_npz=tmp_path / "masks.npz",
        force=False,
    )
    command = run_method.build_command(args, tmp_path / "output", [])
    assert command[-10::2] == [
        "--repo", "--checkpoint", "--image-dir", "--erp-mask-npz", "--output-dir"
    ]
