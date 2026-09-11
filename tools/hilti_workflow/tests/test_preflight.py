from tools.reproducibility.preflight import (
    missing_result_outputs,
    select_results_root,
)


def test_default_empty_results_root_does_not_enable_result_validation(tmp_path):
    default_root = tmp_path / "outputs"
    default_root.mkdir()

    assert select_results_root(
        None,
        [],
        {"results_root": default_root},
    ) is None


def test_explicit_empty_results_root_reports_missing_outputs(tmp_path):
    results_root = tmp_path / "outputs"
    results_root.mkdir()
    runs = [
        {
            "run_name": "floor_1_2025-05-05_run_1",
            "relative_path": "floor_1/2025-05-05/run_1",
        }
    ]

    selected = select_results_root(results_root, [], {})
    assert selected == results_root
    assert len(missing_result_outputs(selected, ["da3"], runs)) == 4
