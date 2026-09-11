import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
STREAMING_DIR = REPO_ROOT / "da3_streaming"
ORIGINAL_SYS_PATH = list(sys.path)
try:
    sys.path.insert(0, str(STREAMING_DIR))
    SPEC = importlib.util.spec_from_file_location(
        "da3_streaming_export_under_test",
        STREAMING_DIR / "da3_streaming.py",
    )
    STREAMING = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(STREAMING)
finally:
    sys.path[:] = ORIGINAL_SYS_PATH


def selected_global_frames(chunk_indices, overlap_s, overlap_e, *, enabled=True):
    selected = []
    for chunk_idx, (chunk_start, _) in enumerate(chunk_indices):
        local_slice = STREAMING.owned_local_frame_slice(
            chunk_idx,
            chunk_indices,
            overlap_s,
            overlap_e,
            enabled=enabled,
        )
        selected.extend(
            chunk_start + local_idx
            for local_idx in range(local_slice.start, local_slice.stop)
        )
    return selected


def make_stream(*, enabled):
    stream = object.__new__(STREAMING.DA3_Streaming)
    stream.chunk_indices = [(0, 8), (5, 13), (10, 18), (15, 20)]
    stream.overlap_s = 1
    stream.overlap_e = 2
    stream.pointcloud_canonical_overlap_ownership = enabled
    stream.config = {
        "Model": {
            "Pointcloud_Save": {
                "conf_threshold_coef": 0.5,
                "sample_ratio": 0.125,
            }
        }
    }
    return stream


def frame_arrays(frame_count):
    frame_ids = np.arange(frame_count, dtype=np.float32)
    points = np.zeros((frame_count, 1, 1, 3), dtype=np.float32)
    points[:, 0, 0, 0] = frame_ids
    colors = np.repeat(frame_ids[:, None, None, None], 3, axis=3).astype(np.uint8)
    confs = (frame_ids + 1.0)[:, None, None]
    return points, colors, confs


def test_canonical_ownership_covers_every_global_frame_exactly_once():
    chunks = [(0, 8), (5, 13), (10, 18), (15, 20)]

    selected = selected_global_frames(chunks, overlap_s=1, overlap_e=2)

    assert selected == list(range(20))
    assert len(selected) == len(set(selected))


def test_final_and_single_chunks_keep_their_tail():
    chunks = [(0, 8), (5, 11)]

    final_slice = STREAMING.owned_local_frame_slice(1, chunks, 1, 2)
    single_slice = STREAMING.owned_local_frame_slice(0, [(0, 5)], 1, 2)

    assert list(range(6))[final_slice] == [1, 2, 3, 4, 5]
    assert list(range(5))[single_slice] == [0, 1, 2, 3, 4]


def test_disabled_mode_preserves_all_legacy_local_frames_and_overlap_duplicates():
    chunks = [(0, 8), (5, 13)]

    selected = selected_global_frames(
        chunks,
        overlap_s=1,
        overlap_e=2,
        enabled=False,
    )

    assert selected == [*range(8), *range(5, 13)]
    assert selected.count(5) == 2
    assert selected.count(6) == 2
    assert selected.count(7) == 2
    assert not STREAMING.canonical_overlap_ownership_enabled(
        {"Model": {"Pointcloud_Save": {}}}
    )


def test_disabled_payload_is_bitwise_legacy_input_with_full_chunk_threshold():
    stream = make_stream(enabled=False)
    points, colors, confs = frame_arrays(8)

    exported_points, exported_colors, exported_confs, threshold = (
        stream._pointcloud_export_payload(
            1,
            points,
            colors,
            confs,
            flatten=False,
        )
    )

    np.testing.assert_array_equal(exported_points, points)
    np.testing.assert_array_equal(exported_colors, colors)
    np.testing.assert_array_equal(exported_confs, confs)
    assert threshold == float(np.mean(confs)) * 0.5


def test_primary_and_paired_baseline_use_identical_owned_frames(monkeypatch):
    stream = make_stream(enabled=True)
    points, colors, confs = frame_arrays(8)
    baseline_points = points.copy()
    baseline_points[..., 0] += 100.0
    calls = []
    monkeypatch.setattr(
        STREAMING,
        "save_confident_pointcloud_batch",
        lambda **kwargs: calls.append(kwargs),
    )

    stream._save_pointcloud_chunk(
        0,
        points,
        colors,
        confs,
        "primary.ply",
        flatten=False,
    )
    stream._save_pointcloud_chunk(
        0,
        baseline_points,
        colors,
        confs,
        "baseline.ply",
        flatten=False,
    )

    primary, baseline = calls
    np.testing.assert_array_equal(primary["points"][..., 0].reshape(-1), range(6))
    np.testing.assert_array_equal(
        baseline["points"][..., 0].reshape(-1), np.arange(6) + 100
    )
    np.testing.assert_array_equal(primary["colors"], baseline["colors"])
    np.testing.assert_array_equal(primary["confs"], baseline["confs"])
    assert primary["conf_threshold"] == pytest.approx(np.mean(confs) * 0.5)
    assert baseline["conf_threshold"] == primary["conf_threshold"]
    assert primary["sample_ratio"] == 0.125
    assert baseline["sample_ratio"] == 0.125


def test_enabled_mode_keeps_legacy_full_chunk_threshold_and_eligible_pixels():
    points, colors, confs = frame_arrays(8)
    confs[[0, 6, 7]] = 1000.0

    enabled_stream = make_stream(enabled=True)
    enabled_payload = enabled_stream._pointcloud_export_payload(
        1,
        points,
        colors,
        confs,
        flatten=True,
    )
    disabled_stream = make_stream(enabled=False)
    disabled_payload = disabled_stream._pointcloud_export_payload(
        1,
        points,
        colors,
        confs,
        flatten=True,
    )

    enabled_points, _, enabled_confs, enabled_threshold = enabled_payload
    disabled_points, _, disabled_confs, disabled_threshold = disabled_payload
    np.testing.assert_array_equal(enabled_points[:, 0], np.arange(1, 6))
    np.testing.assert_array_equal(disabled_points[:, 0], np.arange(8))
    expected_threshold = float(np.mean(confs)) * 0.5
    assert enabled_threshold == expected_threshold
    assert disabled_threshold == expected_threshold
    np.testing.assert_array_equal(
        enabled_confs,
        disabled_confs.reshape(8, -1)[1:6].reshape(-1),
    )
    legacy_eligible = (disabled_confs >= disabled_threshold) & (
        disabled_confs > 1e-5
    )
    owned_legacy_eligible = legacy_eligible.reshape(8, -1)[1:6].reshape(-1)
    enabled_eligible = (enabled_confs >= enabled_threshold) & (
        enabled_confs > 1e-5
    )
    np.testing.assert_array_equal(enabled_eligible, owned_legacy_eligible)
    assert len(enabled_confs) == 5
    assert len(disabled_confs) == 8
