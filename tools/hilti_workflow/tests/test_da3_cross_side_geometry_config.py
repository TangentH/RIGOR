from pathlib import Path

import yaml

from tools.hilti_workflow.run_hilti_best_recon_workflow import build_da3_config

REPO = Path(__file__).resolve().parents[3]


def test_workflow_passes_all_cross_side_geometry_fields(tmp_path):
    workflow_path = REPO / "tools/hilti_workflow/configs/rigor_paper_free_scale.yaml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    configured = workflow["reconstruction"]["loop_geometry_verification"][
        "cross_side_dense"
    ]

    generated_path = build_da3_config(
        workflow,
        tmp_path / "images",
        tmp_path / "masks",
        tmp_path / "output",
    )
    generated = yaml.safe_load(generated_path.read_text(encoding="utf-8"))
    written = generated["Loop"]["GeometryVerification"]["cross_side_dense"]

    assert written == configured
    assert written["enabled"] is True
    assert written["record_diagnostics"] is True
