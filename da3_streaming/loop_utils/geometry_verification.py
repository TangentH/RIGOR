"""Acceptance checks and bounded diagnostics for loop-closure candidates."""

from __future__ import annotations

import math

import numpy as np

CROSS_SIDE_DIAGNOSTIC_FIELDS = (
    "cross_side_status",
    "cross_side_input_points_a",
    "cross_side_input_points_b",
    "cross_side_candidate_points_a",
    "cross_side_candidate_points_b",
    "cross_side_high_conf_points_a",
    "cross_side_high_conf_points_b",
    "cross_side_points_a",
    "cross_side_points_b",
    "cross_side_confidence_threshold_a",
    "cross_side_confidence_threshold_b",
    "cross_side_a_to_b_trimmed_rmse",
    "cross_side_b_to_a_trimmed_rmse",
    "cross_side_symmetric_trimmed_rmse",
    "cross_side_symmetric_median",
    "cross_side_mutual_count",
    "cross_side_mutual_fraction_a",
    "cross_side_mutual_fraction_b",
    "cross_side_mutual_trimmed_rmse",
    "cross_side_mutual_median",
    "cross_side_overlap_a_to_b",
    "cross_side_overlap_b_to_a",
)


def empty_cross_side_diagnostics(status: str = "not_run") -> dict[str, object]:
    """Return a stable CSV schema even when diagnostics are not requested."""
    row: dict[str, object] = {
        field: float("nan") for field in CROSS_SIDE_DIAGNOSTIC_FIELDS
    }
    row["cross_side_status"] = status
    for field in (
        "cross_side_input_points_a",
        "cross_side_input_points_b",
        "cross_side_candidate_points_a",
        "cross_side_candidate_points_b",
        "cross_side_high_conf_points_a",
        "cross_side_high_conf_points_b",
        "cross_side_points_a",
        "cross_side_points_b",
        "cross_side_mutual_count",
    ):
        row[field] = 0
    return row


def _bounded_indices(size: int, limit: int) -> np.ndarray:
    """Choose deterministic, evenly spaced indices without allocating size."""
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    # Integer arithmetic avoids the endpoint duplication possible with rounded
    # linspace values. The selected set is stable across processes and hosts.
    return (np.arange(limit, dtype=np.int64) * size) // limit


def _bounded_high_confidence_voxels(
    point_map: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_quantile: float,
    min_confidence: float,
    voxel_size: float,
    max_points: int,
    preselection_multiplier: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Create a deterministic high-confidence, voxelized point proxy.

    The point map can contain millions of pixels. We first take a bounded
    uniform index sample, then apply finite/confidence filtering and voxelize.
    Peak temporary memory is O(max_points times preselection_multiplier),
    rather than O(number of image pixels).
    """
    points = np.asarray(point_map)
    conf = np.asarray(confidence)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("point_map must end in XYZ coordinates")
    if points.shape[:-1] != conf.shape:
        raise ValueError("point_map and confidence shapes do not match")
    if not 0.0 <= confidence_quantile <= 1.0:
        raise ValueError("confidence_quantile must be in [0, 1]")
    if not math.isfinite(min_confidence):
        raise ValueError("min_confidence must be finite")
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    if max_points <= 0 or preselection_multiplier <= 0:
        raise ValueError("point limits must be positive")

    flat_points = points.reshape(-1, 3)
    flat_conf = conf.reshape(-1)
    input_count = len(flat_points)
    candidate_limit = max_points * preselection_multiplier
    indices = _bounded_indices(input_count, candidate_limit)
    candidates = np.asarray(flat_points[indices], dtype=np.float64)
    candidate_conf = np.asarray(flat_conf[indices], dtype=np.float64)

    finite = np.isfinite(candidates).all(axis=1) & np.isfinite(candidate_conf)
    finite &= candidate_conf > min_confidence
    candidates = candidates[finite]
    candidate_conf = candidate_conf[finite]
    if len(candidate_conf) == 0:
        return np.empty((0, 3), dtype=np.float64), {
            "input_points": input_count,
            "candidate_points": len(indices),
            "high_conf_points": 0,
            "points": 0,
            "confidence_threshold": float("nan"),
        }

    confidence_threshold = float(np.quantile(candidate_conf, confidence_quantile))
    keep = candidate_conf >= confidence_threshold
    candidates = candidates[keep]
    candidate_conf = candidate_conf[keep]
    high_conf_count = len(candidates)

    scaled = np.floor(candidates / voxel_size)
    if not np.isfinite(scaled).all():
        raise ValueError("voxel coordinates are non-finite")
    int64 = np.iinfo(np.int64)
    if np.any(scaled < int64.min) or np.any(scaled > int64.max):
        raise ValueError("voxel coordinates exceed int64 range")
    voxel_keys = scaled.astype(np.int64)

    # Highest confidence wins within each voxel. Stable sorting makes ties
    # deterministic, and np.unique orders voxels lexicographically.
    confidence_order = np.argsort(-candidate_conf, kind="stable")
    _, first = np.unique(voxel_keys[confidence_order], axis=0, return_index=True)
    voxel_points = candidates[confidence_order[first]]
    if len(voxel_points) > max_points:
        voxel_points = voxel_points[_bounded_indices(len(voxel_points), max_points)]

    return voxel_points, {
        "input_points": input_count,
        "candidate_points": len(indices),
        "high_conf_points": high_conf_count,
        "points": len(voxel_points),
        "confidence_threshold": confidence_threshold,
    }


def _trimmed_rmse(distances: np.ndarray, quantile: float) -> float:
    if len(distances) == 0:
        return float("nan")
    cutoff = float(np.quantile(distances, quantile))
    trimmed = distances[distances <= cutoff]
    if len(trimmed) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(trimmed))))


def evaluate_cross_side_dense_geometry(
    point_map_a: np.ndarray,
    confidence_a: np.ndarray,
    point_map_b: np.ndarray,
    confidence_b: np.ndarray,
    *,
    confidence_quantile: float = 0.75,
    min_confidence: float = 0.0,
    voxel_size: float = 0.10,
    max_points_per_side: int = 30_000,
    preselection_multiplier: int = 8,
    trim_quantile: float = 0.90,
    overlap_distance: float = 0.25,
    min_points_per_side: int = 100,
) -> dict[str, object]:
    """Measure whether joint-prediction sides A and B occupy the same geometry.

    A and B must be point maps from the same joint loop inference, whose
    extrinsics place them in a common coordinate frame. This function does no
    registration: doing so would let repeated but spatially disjoint corridors
    manufacture overlap. It reports directed overlap, symmetric nearest-
    neighbour diagnostics, and true mutual-nearest-neighbour statistics.

    Invalid or insufficient input produces a non-ok status and NaN metrics;
    callers can consequently reject it when the gate is enabled (fail closed).
    """
    row = empty_cross_side_diagnostics("invalid")
    try:
        if not 0.0 < trim_quantile <= 1.0:
            raise ValueError("trim_quantile must be in (0, 1]")
        if not math.isfinite(overlap_distance) or overlap_distance <= 0.0:
            raise ValueError("overlap_distance must be finite and positive")
        if min_points_per_side <= 0:
            raise ValueError("min_points_per_side must be positive")
        points_a, stats_a = _bounded_high_confidence_voxels(
            point_map_a,
            confidence_a,
            confidence_quantile=confidence_quantile,
            min_confidence=min_confidence,
            voxel_size=voxel_size,
            max_points=max_points_per_side,
            preselection_multiplier=preselection_multiplier,
        )
        points_b, stats_b = _bounded_high_confidence_voxels(
            point_map_b,
            confidence_b,
            confidence_quantile=confidence_quantile,
            min_confidence=min_confidence,
            voxel_size=voxel_size,
            max_points=max_points_per_side,
            preselection_multiplier=preselection_multiplier,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        row["cross_side_status"] = f"invalid:{type(exc).__name__}"
        return row

    for suffix, stats in (("a", stats_a), ("b", stats_b)):
        for key, value in stats.items():
            row[f"cross_side_{key}_{suffix}"] = value
    if len(points_a) < min_points_per_side or len(points_b) < min_points_per_side:
        row["cross_side_status"] = "insufficient_points"
        return row

    try:
        # Imported lazily so historical runs with diagnostics disabled do not
        # acquire a new import-time dependency or ABI surface.
        from scipy.spatial import cKDTree

        tree_a = cKDTree(points_a)
        tree_b = cKDTree(points_b)
        distance_ab, index_ab = tree_b.query(points_a, k=1, workers=1)
        distance_ba, index_ba = tree_a.query(points_b, k=1, workers=1)
    except (ImportError, TypeError, ValueError, RuntimeError) as exc:
        row["cross_side_status"] = f"nn_error:{type(exc).__name__}"
        return row

    distance_ab = np.asarray(distance_ab, dtype=np.float64)
    distance_ba = np.asarray(distance_ba, dtype=np.float64)
    if not np.isfinite(distance_ab).all() or not np.isfinite(distance_ba).all():
        row["cross_side_status"] = "non_finite_distances"
        return row

    row["cross_side_a_to_b_trimmed_rmse"] = _trimmed_rmse(distance_ab, trim_quantile)
    row["cross_side_b_to_a_trimmed_rmse"] = _trimmed_rmse(distance_ba, trim_quantile)
    symmetric = np.concatenate((distance_ab, distance_ba))
    row["cross_side_symmetric_trimmed_rmse"] = _trimmed_rmse(symmetric, trim_quantile)
    row["cross_side_symmetric_median"] = float(np.median(symmetric))
    row["cross_side_overlap_a_to_b"] = float(np.mean(distance_ab <= overlap_distance))
    row["cross_side_overlap_b_to_a"] = float(np.mean(distance_ba <= overlap_distance))

    source_a = np.arange(len(points_a), dtype=np.int64)
    mutual_mask = (
        np.asarray(index_ba, dtype=np.int64)[np.asarray(index_ab, dtype=np.int64)]
        == source_a
    )
    mutual_distances = distance_ab[mutual_mask]
    row["cross_side_mutual_count"] = len(mutual_distances)
    row["cross_side_mutual_fraction_a"] = float(len(mutual_distances) / len(points_a))
    row["cross_side_mutual_fraction_b"] = float(len(mutual_distances) / len(points_b))
    row["cross_side_mutual_trimmed_rmse"] = _trimmed_rmse(
        mutual_distances, trim_quantile
    )
    row["cross_side_mutual_median"] = (
        float(np.median(mutual_distances)) if len(mutual_distances) else float("nan")
    )
    row["cross_side_status"] = "ok"
    return row


def evaluate_loop_geometry(
    *,
    alignment_error_a: float,
    alignment_error_b: float,
    scale: float,
    rig_acceptance_a: float,
    rig_acceptance_b: float,
    max_side_alignment_error: float,
    min_scale: float,
    max_scale: float,
    min_rig_acceptance: float,
    cross_side_enabled: bool = False,
    cross_side_status: str = "not_run",
    cross_side_mutual_count: int = 0,
    cross_side_mutual_trimmed_rmse: float = float("nan"),
    cross_side_mutual_median: float = float("nan"),
    cross_side_overlap_a_to_b: float = float("nan"),
    cross_side_overlap_b_to_a: float = float("nan"),
    cross_side_min_mutual_count: int = 100,
    cross_side_max_mutual_trimmed_rmse: float = 0.25,
    cross_side_max_mutual_median: float = 0.15,
    cross_side_min_overlap_a_to_b: float = 0.35,
    cross_side_min_overlap_b_to_a: float = 0.35,
) -> tuple[bool, list[str]]:
    """Return whether a loop candidate passes all independent sanity checks."""
    reasons: list[str] = []
    for side, value in (("a", alignment_error_a), ("b", alignment_error_b)):
        if not math.isfinite(value) or value > max_side_alignment_error:
            reasons.append(f"alignment_error_{side}")
    if not math.isfinite(scale) or not min_scale <= scale <= max_scale:
        reasons.append("scale")
    for side, value in (("a", rig_acceptance_a), ("b", rig_acceptance_b)):
        if not math.isfinite(value) or value < min_rig_acceptance:
            reasons.append(f"rig_acceptance_{side}")
    if cross_side_enabled:
        if cross_side_status != "ok":
            reasons.append("cross_side_data")
        else:
            if cross_side_mutual_count < cross_side_min_mutual_count:
                reasons.append("cross_side_mutual_count")
            if (
                not math.isfinite(cross_side_mutual_trimmed_rmse)
                or cross_side_mutual_trimmed_rmse > cross_side_max_mutual_trimmed_rmse
            ):
                reasons.append("cross_side_mutual_trimmed_rmse")
            if (
                not math.isfinite(cross_side_mutual_median)
                or cross_side_mutual_median > cross_side_max_mutual_median
            ):
                reasons.append("cross_side_mutual_median")
            if (
                not math.isfinite(cross_side_overlap_a_to_b)
                or cross_side_overlap_a_to_b < cross_side_min_overlap_a_to_b
            ):
                reasons.append("cross_side_overlap_a_to_b")
            if (
                not math.isfinite(cross_side_overlap_b_to_a)
                or cross_side_overlap_b_to_a < cross_side_min_overlap_b_to_a
            ):
                reasons.append("cross_side_overlap_b_to_a")
    return not reasons, reasons
