import math

import numpy as np

from da3_streaming.loop_utils.geometry_verification import (
    CROSS_SIDE_DIAGNOSTIC_FIELDS,
    empty_cross_side_diagnostics,
    evaluate_cross_side_dense_geometry,
    evaluate_loop_geometry,
)


def corridor_grid(*, offset=(0.0, 0.0, 0.0)):
    x, y = np.meshgrid(
        np.linspace(0.0, 3.0, 31),
        np.linspace(-1.0, 1.0, 21),
        indexing="xy",
    )
    # Non-planar relief prevents nearest-neighbour ties from dominating the
    # mutual correspondence diagnostic.
    z = 0.1 * np.sin(1.7 * x) + 0.05 * np.cos(2.1 * y)
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    points += np.asarray(offset, dtype=np.float64)
    confidence = np.linspace(1.0, 2.0, len(points), dtype=np.float64)
    return points, confidence


def diagnostic(a, ca, b, cb):
    return evaluate_cross_side_dense_geometry(
        a,
        ca,
        b,
        cb,
        confidence_quantile=0.0,
        voxel_size=0.04,
        max_points_per_side=2_000,
        preselection_multiplier=2,
        trim_quantile=0.90,
        overlap_distance=0.05,
        min_points_per_side=100,
    )


def gate(metrics, **overrides):
    values = {
        "alignment_error_a": 0.03,
        "alignment_error_b": 0.04,
        "scale": 1.0,
        "rig_acceptance_a": 1.0,
        "rig_acceptance_b": 1.0,
        "max_side_alignment_error": 0.25,
        "min_scale": 0.8,
        "max_scale": 1.25,
        "min_rig_acceptance": 0.8,
        "cross_side_enabled": True,
        "cross_side_status": metrics["cross_side_status"],
        "cross_side_mutual_count": metrics["cross_side_mutual_count"],
        "cross_side_mutual_trimmed_rmse": metrics["cross_side_mutual_trimmed_rmse"],
        "cross_side_mutual_median": metrics["cross_side_mutual_median"],
        "cross_side_overlap_a_to_b": metrics["cross_side_overlap_a_to_b"],
        "cross_side_overlap_b_to_a": metrics["cross_side_overlap_b_to_a"],
        "cross_side_min_mutual_count": 100,
        "cross_side_max_mutual_trimmed_rmse": 0.05,
        "cross_side_max_mutual_median": 0.05,
        "cross_side_min_overlap_a_to_b": 0.95,
        "cross_side_min_overlap_b_to_a": 0.95,
    }
    values.update(overrides)
    return evaluate_loop_geometry(**values)


def test_true_joint_overlap_passes_cross_side_gate():
    points, confidence = corridor_grid()
    shifted = points + np.array([0.005, -0.004, 0.003])

    metrics = diagnostic(points, confidence, shifted, confidence)
    accepted, reasons = gate(metrics)

    assert metrics["cross_side_status"] == "ok"
    assert metrics["cross_side_mutual_count"] >= 600
    assert metrics["cross_side_mutual_trimmed_rmse"] < 0.01
    assert metrics["cross_side_overlap_a_to_b"] == 1.0
    assert metrics["cross_side_overlap_b_to_a"] == 1.0
    assert accepted
    assert reasons == []


def test_spatially_disjoint_repeated_corridor_is_rejected():
    points, confidence = corridor_grid()
    repeated, repeated_confidence = corridor_grid(offset=(8.0, 0.0, 0.0))

    metrics = diagnostic(points, confidence, repeated, repeated_confidence)
    accepted, reasons = gate(metrics, cross_side_min_mutual_count=1)

    assert metrics["cross_side_status"] == "ok"
    assert metrics["cross_side_mutual_trimmed_rmse"] > 4.0
    assert metrics["cross_side_overlap_a_to_b"] == 0.0
    assert metrics["cross_side_overlap_b_to_a"] == 0.0
    assert not accepted
    assert "cross_side_mutual_trimmed_rmse" in reasons
    assert "cross_side_mutual_median" in reasons
    assert "cross_side_overlap_a_to_b" in reasons
    assert "cross_side_overlap_b_to_a" in reasons


def test_degenerate_or_nan_input_fails_closed_when_enabled():
    points = np.full((256, 3), np.nan)
    confidence = np.ones(256)
    valid, valid_confidence = corridor_grid()

    metrics = diagnostic(points, confidence, valid, valid_confidence)
    accepted, reasons = gate(metrics)

    assert metrics["cross_side_status"] == "insufficient_points"
    assert math.isnan(metrics["cross_side_mutual_trimmed_rmse"])
    assert not accepted
    assert reasons == ["cross_side_data"]


def test_diagnostics_are_deterministic_and_bounded():
    points, confidence = corridor_grid()
    dense = np.repeat(points, 20, axis=0)
    dense_confidence = np.repeat(confidence, 20)

    first = evaluate_cross_side_dense_geometry(
        dense,
        dense_confidence,
        dense,
        dense_confidence,
        confidence_quantile=0.0,
        voxel_size=0.04,
        max_points_per_side=128,
        preselection_multiplier=2,
        min_points_per_side=20,
    )
    second = evaluate_cross_side_dense_geometry(
        dense,
        dense_confidence,
        dense,
        dense_confidence,
        confidence_quantile=0.0,
        voxel_size=0.04,
        max_points_per_side=128,
        preselection_multiplier=2,
        min_points_per_side=20,
    )

    assert first == second
    assert first["cross_side_candidate_points_a"] == 256
    assert first["cross_side_points_a"] <= 128
    assert first["cross_side_points_b"] <= 128


def test_cross_side_gate_is_disabled_by_default():
    accepted, reasons = evaluate_loop_geometry(
        alignment_error_a=0.03,
        alignment_error_b=0.04,
        scale=1.0,
        rig_acceptance_a=1.0,
        rig_acceptance_b=1.0,
        max_side_alignment_error=0.25,
        min_scale=0.8,
        max_scale=1.25,
        min_rig_acceptance=0.8,
    )
    assert accepted
    assert reasons == []

    empty = empty_cross_side_diagnostics()
    assert tuple(empty) == CROSS_SIDE_DIAGNOSTIC_FIELDS
    assert empty["cross_side_status"] == "not_run"
