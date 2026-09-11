from da3_streaming.loop_utils.geometry_verification import evaluate_loop_geometry


def test_accepts_floor6_rig_consistent_candidate():
    accepted, reasons = evaluate_loop_geometry(
        alignment_error_a=0.0298,
        alignment_error_b=0.0489,
        scale=1.0447,
        rig_acceptance_a=1.0,
        rig_acceptance_b=1.0,
        max_side_alignment_error=0.25,
        min_scale=0.8,
        max_scale=1.25,
        min_rig_acceptance=0.8,
    )
    assert accepted
    assert reasons == []


def test_rejects_floor6_single_view_false_candidate():
    accepted, reasons = evaluate_loop_geometry(
        alignment_error_a=0.0870,
        alignment_error_b=2.0671,
        scale=1.2907,
        rig_acceptance_a=1.0,
        rig_acceptance_b=0.0,
        max_side_alignment_error=0.25,
        min_scale=0.8,
        max_scale=1.25,
        min_rig_acceptance=0.8,
    )
    assert not accepted
    assert reasons == ["alignment_error_b", "scale", "rig_acceptance_b"]
