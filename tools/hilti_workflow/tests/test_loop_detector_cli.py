from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tools.hilti_workflow.run_hilti_best_recon_workflow import build_da3_config

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "da3_streaming"))

from loop_utils import loop_detector


def detector_config(*, diagnostics=None, min_gap_unit="image", nms_unit="image"):
    return {
        "Weights": {"SALAD": "unused"},
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": 1,
                "single_view_similarity_threshold": 0.85,
                "top_k": 2,
                "min_frame_gap": 10,
                "min_gap_unit": min_gap_unit,
                "use_nms": True,
                "nms_threshold": 1,
                "nms_unit": nms_unit,
                "views_per_capture": 4,
                "diagnostics": diagnostics or {},
            }
        },
    }


def capture_consensus_config(**overrides):
    config = detector_config(min_gap_unit="capture", nms_unit="capture")
    salad = config["Loop"]["SALAD"]
    salad.update(
        candidate_mode="capture_cyclic",
        min_frame_gap=2,
        use_nms=False,
        capture_cyclic_consensus={
            "mean_similarity_threshold": 0.65,
            "min_view_similarity": 0.45,
            "support_similarity": 0.50,
            "min_support_views": 4,
        },
    )
    salad["capture_cyclic_consensus"].update(overrides)
    return config


def test_frozen_candidate_loop_input_protocol():
    config = yaml.safe_load(
        (REPO / "tools/hilti_workflow/configs/rigor_paper_free_scale.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["pinhole"]["fov_deg"] == 95.0
    assert config["pinhole"]["imu_method"] == "complementary"
    assert config["pinhole"]["imu_tau"] == 2.0
    assert config["reconstruction"]["delete_temp_files"] is False
    assert config["reconstruction"]["canonical_overlap_ownership"] is True
    assert config["reconstruction"]["loop_candidate_mode"] == "capture_cyclic"
    assert config["reconstruction"]["loop_min_frame_gap"] == 80
    assert config["reconstruction"]["loop_single_view_similarity_threshold"] == 0.85
    assert "loop_similarity_threshold" not in config["reconstruction"]
    assert config["reconstruction"]["loop_capture_cyclic_consensus"] == {
        "mean_similarity_threshold": 0.65,
        "min_view_similarity": 0.45,
        "support_similarity": 0.50,
        "min_support_views": 4,
    }


def test_frozen_workflow_writes_da3_temp_retention(tmp_path):
    workflow = yaml.safe_load(
        (REPO / "tools/hilti_workflow/configs/rigor_paper_free_scale.yaml").read_text(
            encoding="utf-8"
        )
    )
    generated = build_da3_config(
        workflow,
        tmp_path / "images",
        tmp_path / "masks",
        tmp_path / "output",
    )
    generated_config = yaml.safe_load(generated.read_text(encoding="utf-8"))
    assert generated_config["Model"]["delete_temp_files"] is False
    assert generated_config["Model"]["Pointcloud_Save"][
        "canonical_overlap_ownership"
    ] is True


def test_workflow_maps_opt_in_capture_cyclic_consensus(tmp_path):
    workflow = yaml.safe_load(
        (REPO / "tools/hilti_workflow/configs/rigor_paper_free_scale.yaml").read_text(
            encoding="utf-8"
        )
    )
    reconstruction = workflow["reconstruction"]
    reconstruction.update(
        loop_candidate_mode="capture_cyclic",
        loop_min_frame_gap=80,
        loop_capture_cyclic_consensus={
            "mean_similarity_threshold": 0.66,
            "min_view_similarity": 0.46,
            "support_similarity": 0.51,
            "min_support_views": 3,
        },
    )

    generated = build_da3_config(
        workflow,
        tmp_path / "images",
        tmp_path / "masks",
        tmp_path / "output",
    )
    salad = yaml.safe_load(generated.read_text(encoding="utf-8"))["Loop"]["SALAD"]

    assert salad["candidate_mode"] == "capture_cyclic"
    assert salad["min_frame_gap"] == 80
    assert salad["min_gap_unit"] == "capture"
    assert salad["nms_unit"] == "capture"
    assert salad["capture_cyclic_consensus"] == {
        "mean_similarity_threshold": 0.66,
        "min_view_similarity": 0.46,
        "support_similarity": 0.51,
        "min_support_views": 3,
    }


def test_standalone_cli_builds_canonical_config(monkeypatch, tmp_path):
    captured = {}

    class FakeDetector:
        def __init__(self, *, image_dir, output, config):
            captured.update(image_dir=image_dir, output=output, config=config)

        def run(self):
            captured["ran"] = True

    monkeypatch.setattr(loop_detector, "LoopDetector", FakeDetector)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "loop_detector.py",
            "--image_dir",
            str(tmp_path),
            "--ckpt_path",
            "weights.ckpt",
            "--single_view_similarity_threshold",
            "0.85",
            "--top_k",
            "20",
            "--min_frame_gap",
            "160",
            "--nms_threshold",
            "25",
            "--no_nms",
        ],
    )

    loop_detector.main()

    salad = captured["config"]["Loop"]["SALAD"]
    assert captured["ran"] is True
    assert captured["config"]["Weights"]["SALAD"] == "weights.ckpt"
    assert salad["single_view_similarity_threshold"] == 0.85
    assert "similarity_threshold" not in salad
    assert salad["top_k"] == 20
    assert salad["min_frame_gap"] == 160
    assert salad["nms_threshold"] == 25
    assert salad["use_nms"] is False
    assert salad["min_gap_unit"] == "image"
    assert salad["nms_unit"] == "image"
    assert salad["diagnostics"]["descriptors_npz"] is None


def test_standalone_cli_default_checkpoint_is_cwd_independent(monkeypatch, tmp_path):
    captured = {}

    class FakeDetector:
        def __init__(self, *, image_dir, output, config):
            captured.update(image_dir=image_dir, output=output, config=config)

        def run(self):
            pass

    monkeypatch.setattr(loop_detector, "LoopDetector", FakeDetector)
    monkeypatch.setattr(sys, "argv", ["loop_detector.py", "--image_dir", str(tmp_path)])
    loop_detector.main()

    assert captured["config"]["Weights"]["SALAD"] == loop_detector.DEFAULT_SALAD_CKPT
    assert Path(loop_detector.DEFAULT_SALAD_CKPT).is_absolute()


def test_loop_results_create_output_parent(tmp_path):
    config = {
        "Weights": {"SALAD": "unused"},
        "Loop": {
            "SALAD": {
                "image_size": [336, 336],
                "batch_size": 1,
                "similarity_threshold": 0.85,
                "top_k": 1,
                "min_frame_gap": 10,
                "use_nms": True,
                "nms_threshold": 25,
            }
        },
    }
    output = tmp_path / "nested" / "diagnostics" / "loops.txt"
    detector = loop_detector.LoopDetector(tmp_path, output=str(output), config=config)
    detector.image_paths = [tmp_path / "frame.jpg"]
    detector.loop_closures = []
    detector.save_results()

    assert output.is_file()
    assert "# Loop pairs:" in output.read_text(encoding="utf-8")


def test_capture_metadata_and_capture_gap_are_optional(tmp_path):
    detector = loop_detector.LoopDetector(
        tmp_path,
        config=detector_config(min_gap_unit="capture"),
    )
    detector.min_frame_gap = 1
    detector.image_paths = [
        tmp_path / f"{capture:05d}_v{view:02d}_frame_{capture:05d}_yaw{view * 90:03d}.jpg"
        for capture in range(3)
        for view in range(4)
    ]

    metadata = detector.get_image_metadata()
    assert metadata[6] == {
        "image_index": 6,
        "capture_index": 1,
        "view_index": 2,
        "image_name": "00001_v02_frame_00001_yaw180.jpg",
    }
    assert detector._passes_min_gap(0, 7) is False
    assert detector._passes_min_gap(0, 8) is True

    image_unit = loop_detector.LoopDetector(tmp_path, config=detector_config())
    image_unit.min_frame_gap = 1
    image_unit.image_paths = detector.image_paths
    assert image_unit._passes_min_gap(0, 7) is True


def test_capture_nms_suppresses_nearby_capture_pairs(tmp_path):
    detector = loop_detector.LoopDetector(
        tmp_path,
        config=detector_config(nms_unit="capture"),
    )
    capture_ids = [0, 1, 10, 11, 20, 21]
    detector.image_paths = [
        tmp_path / f"{capture:05d}_v00_frame_{capture:05d}_yaw000.jpg"
        for capture in capture_ids
    ]
    loops = [(2, 0, 0.95), (3, 1, 0.90), (5, 4, 0.80)]

    assert detector._apply_nms_filter(loops, 1) == [
        (2, 0, 0.95),
        (5, 4, 0.80),
    ]


def test_capture_cyclic_consensus_handles_opposite_heading(tmp_path):
    detector = loop_detector.LoopDetector(
        tmp_path, config=capture_consensus_config()
    )
    detector.image_paths = [
        tmp_path
        / f"{capture:05d}_v{view:02d}_frame_{capture:05d}_yaw{view * 90:03d}.jpg"
        for capture in (0, 10)
        for view in range(4)
    ]
    descriptors = np.zeros((8, 8), dtype=np.float32)
    for view in range(4):
        descriptors[view, view] = 1.0
        descriptors[4 + view, (view + 2) % 4] = [0.90, 0.80, 0.70, 0.68][view]
    detector.descriptors = torch.from_numpy(descriptors)

    loops = detector.find_loop_closures()

    assert len(loops) == 1
    image_a, image_b, score = loops[0]
    assert (image_a, image_b) == (4, 2)
    assert score == pytest.approx((0.90 + 0.80 + 0.70 + 0.68) / 4)


def test_capture_cyclic_consensus_rejects_single_view_alias(tmp_path):
    detector = loop_detector.LoopDetector(
        tmp_path, config=capture_consensus_config(min_support_views=3)
    )
    detector.image_paths = [
        tmp_path
        / f"{capture:05d}_v{view:02d}_frame_{capture:05d}_yaw{view * 90:03d}.jpg"
        for capture in (0, 10)
        for view in range(4)
    ]
    descriptors = np.zeros((8, 8), dtype=np.float32)
    descriptors[:4, :4] = np.eye(4, dtype=np.float32)
    descriptors[4, 0] = 0.99
    detector.descriptors = torch.from_numpy(descriptors)

    assert detector.find_loop_closures() == []


def test_capture_cyclic_consensus_fails_closed_on_incomplete_yaw4(tmp_path):
    detector = loop_detector.LoopDetector(
        tmp_path, config=capture_consensus_config()
    )
    detector.image_paths = [
        tmp_path / f"00000_v{view:02d}_frame_00000_yaw{view * 90:03d}.jpg"
        for view in range(4)
    ] + [
        tmp_path / f"00010_v{view:02d}_frame_00010_yaw{view * 90:03d}.jpg"
        for view in range(3)
    ]
    detector.descriptors = torch.eye(7, dtype=torch.float32)

    assert detector.find_loop_closures() == []


def test_optional_diagnostics_are_atomic_complete_and_path_safe(tmp_path):
    descriptor_path = tmp_path / "diagnostics" / "descriptors.npz"
    retrieval_path = tmp_path / "diagnostics" / "retrieval.csv"
    matrix_path = tmp_path / "diagnostics" / "capture_matrix.npz"
    pair_path = tmp_path / "diagnostics" / "capture_pairs.csv"
    config = detector_config(
        diagnostics={
            "descriptors_npz": str(descriptor_path),
            "retrieval_csv": str(retrieval_path),
            "retrieval_top_k": 2,
            "capture_matrix_npz": str(matrix_path),
            "capture_pairs_csv": str(pair_path),
        }
    )
    detector = loop_detector.LoopDetector(tmp_path, config=config)
    detector.image_paths = [
        tmp_path
        / "sensitive-parent"
        / f"{capture:05d}_v{view:02d}_frame_{capture:05d}_yaw{view * 90:03d}.jpg"
        for capture in range(2)
        for view in range(4)
    ]
    descriptors = np.arange(24, dtype=np.float32).reshape(8, 3) / 24.0
    detector.descriptors = torch.from_numpy(descriptors)
    detector.retrieval_indices = np.asarray(
        [[query, (query + 1) % 8, (query + 2) % 8] for query in range(8)],
        dtype=np.int64,
    )
    detector.retrieval_similarities = np.asarray(
        [[1.0, 0.9, 0.8] for _ in range(8)], dtype=np.float32
    )

    detector.save_diagnostics()

    with np.load(descriptor_path, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["descriptors"], descriptors)
        assert saved["capture_indices"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        assert saved["view_indices"].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
        assert all("sensitive-parent" not in name for name in saved["image_names"])

    with retrieval_path.open(newline="", encoding="utf-8") as stream:
        retrieval_rows = list(csv.DictReader(stream))
    assert len(retrieval_rows) == 16
    assert {row["rank"] for row in retrieval_rows} == {"1", "2"}
    assert all("/" not in row["query_image_name"] for row in retrieval_rows)
    assert all("/" not in row["neighbor_image_name"] for row in retrieval_rows)

    with np.load(matrix_path, allow_pickle=False) as saved:
        scores = saved["scores"]
        assert scores.shape == (2, 2, 4, 4)
        np.testing.assert_allclose(scores[0, 1], descriptors[:4] @ descriptors[4:].T)
        np.testing.assert_array_equal(saved["image_indices"], np.arange(8).reshape(2, 4))
        assert all(
            "sensitive-parent" not in name for name in saved["image_names"].flat
        )

    with pair_path.open(newline="", encoding="utf-8") as stream:
        pair_rows = list(csv.DictReader(stream))
    assert len(pair_rows) == 1
    assert pair_rows[0]["capture_index_a"] == "0"
    assert pair_rows[0]["capture_index_b"] == "1"
    assert "score_v3_v3" in pair_rows[0]
    assert all(
        "/" not in pair_rows[0][f"image_name_{side}_v{view}"]
        for side in ("a", "b")
        for view in range(4)
    )
    assert not list((tmp_path / "diagnostics").glob(".*.tmp-*"))


def test_retrieval_top_k_zero_requests_full_ranking(tmp_path):
    retrieval_path = tmp_path / "retrieval_full.csv"
    detector = loop_detector.LoopDetector(
        tmp_path,
        config=detector_config(
            diagnostics={
                "retrieval_csv": str(retrieval_path),
                "retrieval_top_k": 0,
            }
        ),
    )
    detector.image_paths = [tmp_path / f"{index:05d}_v00.jpg" for index in range(4)]
    detector.retrieval_indices = np.asarray(
        [
            [0, 1, 2, 3],
            [1, 0, 2, 3],
            [2, 0, 1, 3],
            [3, 0, 1, 2],
        ],
        dtype=np.int64,
    )
    detector.retrieval_similarities = np.asarray(
        [[1.0, 0.9, 0.8, 0.7]] * 4, dtype=np.float32
    )

    detector.save_retrieval_diagnostics()

    with retrieval_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert detector._diagnostic_neighbor_limit(4) == 3


def test_full_retrieval_request_expands_faiss_search_only_when_enabled(tmp_path):
    detector = loop_detector.LoopDetector(
        tmp_path,
        config=detector_config(
            diagnostics={
                "retrieval_csv": str(tmp_path / "retrieval.csv"),
                "retrieval_top_k": 0,
            }
        ),
    )
    detector.image_paths = [tmp_path / f"{index:05d}_v00.jpg" for index in range(4)]
    detector.descriptors = torch.eye(4, dtype=torch.float32)

    assert detector.find_loop_closures() == []
    assert detector.retrieval_indices.shape == (4, 4)
    assert detector.retrieval_similarities.shape == (4, 4)


def test_atomic_npz_failure_does_not_replace_existing_file(monkeypatch, tmp_path):
    target = tmp_path / "descriptors.npz"
    target.write_bytes(b"previous-complete-result")

    def fail_save(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(loop_detector.np, "savez", fail_save)
    with pytest.raises(RuntimeError, match="injected failure"):
        loop_detector._atomic_write_npz(target, descriptors=np.ones((1, 2)))

    assert target.read_bytes() == b"previous-complete-result"
    assert not list(tmp_path.glob(".*.tmp-*"))
