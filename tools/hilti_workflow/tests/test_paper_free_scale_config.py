from pathlib import Path

import yaml

from tools.hilti_workflow.run_hilti_best_recon_workflow import build_da3_config

REPO = Path(__file__).resolve().parents[3]


def test_paper_config_uses_free_scale_raw_optimizer_output(tmp_path):
    workflow_path = REPO / "tools/hilti_workflow/configs/rigor_paper_free_scale.yaml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    generated_path = build_da3_config(
        workflow,
        tmp_path / "images",
        tmp_path / "masks",
        tmp_path / "output",
    )
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
    assert "outcome_audit" not in generated["Loop"]["SIM3_Optimizer"]
    assert "loop_optimizer_outcome_audit" not in workflow["reconstruction"]
    assert generated["Loop"]["SIM3_Optimizer"]["fix_scale"] is False

    pinhole = workflow["pinhole"]
    reconstruction = workflow["reconstruction"]
    assert pinhole["fov_deg"] == 95.0
    assert pinhole["yaws"] == "0,90,180,270"
    assert reconstruction["chunk_size"] == 32
    assert reconstruction["overlap"] == 16
    assert reconstruction["canonical_overlap_ownership"] is True
    assert generated["Loop"]["SALAD"]["candidate_mode"] == "capture_cyclic"
    assert generated["Loop"]["SALAD"]["min_frame_gap"] == 80
    assert generated["Loop"]["GeometryVerification"]["cross_side_dense"]["enabled"] is True
    rig = generated["Model"]["Yaw4_Pose_Filter"]
    assert rig["rig_repair_after_alignment"] is True
    assert rig["rig_repair_loop_predictions"] is True
    assert rig["rig_outlier_action"] == "repair"


def test_no_repair_ablation_records_rig_evidence_without_writes(tmp_path):
    workflow_path = REPO / "tools/hilti_workflow/configs/rigor_ablation_no_repair.yaml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    repair = workflow["reconstruction"]["yaw4_pose_filter"]

    assert repair["rig_repair_after_alignment"] is True
    assert repair["rig_repair_loop_predictions"] is True
    assert repair["rig_outlier_action"] == "detect"
    assert workflow["reconstruction"]["loop_pose_graph_fix_scale"] is False
    assert "loop_optimizer_outcome_audit" not in workflow["reconstruction"]
