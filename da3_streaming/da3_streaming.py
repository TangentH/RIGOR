# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long)

import argparse
import csv
import gc
import glob
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from loop_utils.alignment_torch import (
    apply_sim3_direct_torch,
    depth_to_point_cloud_optimized_torch,
)
from loop_utils.config_utils import load_config
try:
    from loop_utils.loop_detector import LoopDetector
except ImportError:
    LoopDetector = None
try:
    from loop_utils.sim3loop import Sim3LoopOptimizer
except ImportError:
    Sim3LoopOptimizer = None
from loop_utils.sim3utils import (
    accumulate_sim3_transforms,
    compute_sim3_ab,
    merge_ply_files,
    precompute_scale_chunks_with_depth,
    process_loop_list,
    save_confident_pointcloud_batch,
    warmup_numba,
    weighted_align_point_maps,
)
from loop_utils.geometry_verification import (
    empty_cross_side_diagnostics,
    evaluate_cross_side_dense_geometry,
    evaluate_loop_geometry,
)
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tools.hilti_workflow.postprocess.yaw4_rig_consistency import analyze_chunk

from depth_anything_3.api import DepthAnything3

matplotlib.use("Agg")


def depth_to_point_cloud_vectorized(depth, intrinsics, extrinsics, device=None):
    """
    depth: [N, H, W] numpy array or torch tensor
    intrinsics: [N, 3, 3] numpy array or torch tensor
    extrinsics: [N, 3, 4] (w2c) numpy array or torch tensor
    Returns: point_cloud_world: [N, H, W, 3] same type as input
    """
    input_is_numpy = False
    if isinstance(depth, np.ndarray):
        input_is_numpy = True

        depth_tensor = torch.tensor(depth, dtype=torch.float32)
        intrinsics_tensor = torch.tensor(intrinsics, dtype=torch.float32)
        extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

        if device is not None:
            depth_tensor = depth_tensor.to(device)
            intrinsics_tensor = intrinsics_tensor.to(device)
            extrinsics_tensor = extrinsics_tensor.to(device)
    else:
        depth_tensor = depth
        intrinsics_tensor = intrinsics
        extrinsics_tensor = extrinsics

    if device is not None:
        depth_tensor = depth_tensor.to(device)
        intrinsics_tensor = intrinsics_tensor.to(device)
        extrinsics_tensor = extrinsics_tensor.to(device)

    # main logic

    N, H, W = depth_tensor.shape

    device = depth_tensor.device

    u = torch.arange(W, device=device).float().view(1, 1, W, 1).expand(N, H, W, 1)
    v = torch.arange(H, device=device).float().view(1, H, 1, 1).expand(N, H, W, 1)
    ones = torch.ones((N, H, W, 1), device=device)
    pixel_coords = torch.cat([u, v, ones], dim=-1)

    intrinsics_inv = torch.inverse(intrinsics_tensor)  # [N, 3, 3]
    camera_coords = torch.einsum("nij,nhwj->nhwi", intrinsics_inv, pixel_coords)
    camera_coords = camera_coords * depth_tensor.unsqueeze(-1)
    camera_coords_homo = torch.cat([camera_coords, ones], dim=-1)

    extrinsics_4x4 = torch.zeros(N, 4, 4, device=device)
    extrinsics_4x4[:, :3, :4] = extrinsics_tensor
    extrinsics_4x4[:, 3, 3] = 1.0

    c2w = torch.inverse(extrinsics_4x4)
    world_coords_homo = torch.einsum("nij,nhwj->nhwi", c2w, camera_coords_homo)
    point_cloud_world = world_coords_homo[..., :3]

    if input_is_numpy:
        point_cloud_world = point_cloud_world.cpu().numpy()

    return point_cloud_world


def remove_duplicates(data_list, event_radius_chunks=0):
    """
    data_list: [(67, (3386, 3406), 48, (2435, 2455)), ...]
    """
    event_radius_chunks = max(0, int(event_radius_chunks))
    seen = set()
    kept_keys = []
    result = []

    for item in data_list:
        if item[0] == item[2]:
            continue

        key = tuple(sorted((item[0], item[2])))

        if key in seen:
            continue
        if any(
            abs(key[0] - kept[0]) <= event_radius_chunks
            and abs(key[1] - kept[1]) <= event_radius_chunks
            for kept in kept_keys
        ):
            continue

        seen.add(key)
        kept_keys.append(key)
        result.append(item)

    return result


def owned_local_frame_slice(
    chunk_idx,
    chunk_indices,
    overlap_s,
    overlap_e,
    *,
    enabled=True,
):
    """Return the local frames owned by one overlapping inference chunk.

    Canonical ownership follows the same convention used by pose/depth output:
    non-first chunks discard ``overlap_s`` leading frames, non-final chunks
    discard ``overlap_e`` trailing frames, and the final chunk retains its
    tail.  With ownership disabled, every local frame is returned to preserve
    the historical point-cloud export behavior.
    """
    if not chunk_indices:
        raise ValueError("chunk_indices must not be empty")
    if chunk_idx < 0 or chunk_idx >= len(chunk_indices):
        raise IndexError(
            f"chunk_idx {chunk_idx} is outside [0, {len(chunk_indices)})"
        )

    chunk_start, chunk_end = chunk_indices[chunk_idx]
    chunk_length = int(chunk_end) - int(chunk_start)
    overlap_s = int(overlap_s)
    overlap_e = int(overlap_e)
    if chunk_length < 0:
        raise ValueError(f"invalid chunk range: {(chunk_start, chunk_end)}")
    if overlap_s < 0 or overlap_e < 0:
        raise ValueError("overlap_s and overlap_e must be non-negative")
    if not enabled:
        return slice(0, chunk_length)

    local_start = 0 if chunk_idx == 0 else overlap_s
    local_end = chunk_length if chunk_idx == len(chunk_indices) - 1 else (
        chunk_length - overlap_e
    )
    if local_start > local_end or local_end < 0 or local_start > chunk_length:
        raise ValueError(
            "overlap ownership leaves an invalid local range: "
            f"chunk_idx={chunk_idx}, chunk_length={chunk_length}, "
            f"overlap_s={overlap_s}, overlap_e={overlap_e}"
        )
    return slice(local_start, local_end)


def canonical_overlap_ownership_enabled(config):
    """Resolve the opt-in point-cloud ownership switch (legacy-safe by default)."""
    return bool(
        config.get("Model", {})
        .get("Pointcloud_Save", {})
        .get("canonical_overlap_ownership", False)
    )


class DA3_Streaming:
    def __init__(self, image_dir, save_dir, config):
        self.config = config

        self.chunk_size = self.config["Model"]["chunk_size"]
        self.overlap = self.config["Model"]["overlap"]
        self.overlap_s = 0
        self.overlap_e = self.overlap - self.overlap_s
        self.pointcloud_canonical_overlap_ownership = (
            canonical_overlap_ownership_enabled(self.config)
        )
        self.conf_threshold = 1.5
        self.seed = 42
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = (
            torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        )

        self.img_dir = image_dir
        self.img_list = None
        self.output_dir = save_dir
        self.confidence_zero_mask_dir = (
            self.config["Model"].get("confidence_zero_mask_dir", None)
            or self.config["Model"].get("dynamic_mask_dir", None)
            or self.config["Model"].get("person_mask_dir", None)
        )
        self.confidence_zero_mask_enabled = bool(self.confidence_zero_mask_dir)
        self.confidence_zero_mask_strict = bool(
            self.config["Model"].get(
                "confidence_zero_mask_strict",
                self.config["Model"].get("person_mask_strict", False),
            )
        )
        self.confidence_zero_mask_missing = 0
        self.confidence_zero_mask_resized = 0
        self.confidence_zero_mask_zeroed_pixels = 0
        pose_filter_cfg = self.config["Model"].get("Yaw4_Pose_Filter", {})
        self.yaw4_pose_filter_enabled = bool(pose_filter_cfg.get("enabled", False))
        self.yaw4_pose_filter_group_size = int(pose_filter_cfg.get("yaw_group_size", 4))
        self.yaw4_pose_filter_max_frame_group_dist = float(
            pose_filter_cfg.get("max_frame_group_dist", 0.1)
        )
        self.yaw4_pose_filter_max_median_group_max = float(
            pose_filter_cfg.get("max_median_group_max", 0.1)
        )
        self.yaw4_rig_repair_before_alignment = bool(
            pose_filter_cfg.get("rig_repair_before_alignment", False)
        )
        self.yaw4_rig_repair_after_alignment = bool(
            pose_filter_cfg.get("rig_repair_after_alignment", False)
        )
        if (
            self.yaw4_rig_repair_before_alignment
            and self.yaw4_rig_repair_after_alignment
        ):
            raise ValueError(
                "rig_repair_before_alignment and rig_repair_after_alignment "
                "are mutually exclusive"
            )
        self.yaw4_rig_repair_enabled = (
            self.yaw4_rig_repair_before_alignment
            or self.yaw4_rig_repair_after_alignment
        )
        self.yaw4_rig_repair_stage = (
            "pre_alignment"
            if self.yaw4_rig_repair_before_alignment
            else "post_alignment"
        )
        self.yaw4_rig_repair_loop_enabled = bool(
            pose_filter_cfg.get(
                "rig_repair_loop_predictions",
                self.yaw4_rig_repair_before_alignment,
            )
        )
        # Ordinary chunks may be repaired after their sequential Sim3 has been
        # estimated while loop windows are repaired at inference time. The two
        # switches address different predictions and are allowed together.
        self.yaw4_rig_paired_baseline_export = bool(
            pose_filter_cfg.get("rig_paired_baseline_export", False)
        )
        if (
            self.yaw4_rig_paired_baseline_export
            and not self.yaw4_rig_repair_after_alignment
        ):
            raise ValueError(
                "rig_paired_baseline_export requires post-alignment repair"
            )
        self.yaw4_rig_rotation_threshold_deg = float(
            pose_filter_cfg.get("rig_rotation_threshold_deg", 5.0)
        )
        center_depth_ratio = pose_filter_cfg.get("rig_center_depth_ratio")
        self.yaw4_rig_center_depth_ratio = (
            float(center_depth_ratio) if center_depth_ratio is not None else None
        )
        self.yaw4_rig_min_inliers = int(pose_filter_cfg.get("rig_min_inliers", 3))
        self.yaw4_rig_validate_overlap = bool(
            pose_filter_cfg.get("rig_validate_overlap", False)
        )
        self.yaw4_rig_overlap_pixel_stride = int(
            pose_filter_cfg.get("rig_overlap_pixel_stride", 4)
        )
        self.yaw4_rig_overlap_ray_tolerance_pixels = float(
            pose_filter_cfg.get("rig_overlap_ray_tolerance_pixels", 1.5)
        )
        self.yaw4_rig_overlap_confidence_quantile = float(
            pose_filter_cfg.get("rig_overlap_confidence_quantile", 0.25)
        )
        self.yaw4_rig_overlap_max_depth = float(
            pose_filter_cfg.get("rig_overlap_max_depth", 15.0)
        )
        self.yaw4_rig_overlap_min_matched_rays = int(
            pose_filter_cfg.get("rig_overlap_min_matched_rays", 100)
        )
        self.yaw4_rig_overlap_max_median_ratio = float(
            pose_filter_cfg.get("rig_overlap_max_median_ratio", 1.0)
        )
        self.yaw4_rig_overlap_max_p90_ratio = float(
            pose_filter_cfg.get("rig_overlap_max_p90_ratio", 1.0)
        )
        self.yaw4_rig_reject_unrecoverable_group = bool(
            pose_filter_cfg.get("rig_reject_unrecoverable_group", False)
        )
        self.yaw4_rig_check_orientation = bool(
            pose_filter_cfg.get("rig_check_orientation", True)
        )
        self.yaw4_rig_outlier_action = str(
            pose_filter_cfg.get("rig_outlier_action", "repair")
        ).lower()
        if self.yaw4_rig_outlier_action not in {"repair", "remove", "detect"}:
            raise ValueError(
                "rig_outlier_action must be one of: repair, remove, detect"
            )
        self.yaw4_pose_filter_zeroed_frames = 0
        self.yaw4_pose_filter_rejected_chunks = 0
        self.yaw4_pose_filter_rows = []
        self.yaw4_rig_repair_rows = []
        self.yaw4_rig_repair_group_rows = []

        self.result_unaligned_dir = os.path.join(save_dir, "_tmp_results_unaligned")
        self.result_aligned_dir = os.path.join(save_dir, "_tmp_results_aligned")
        self.result_loop_dir = os.path.join(save_dir, "_tmp_results_loop")
        self.result_output_dir = os.path.join(save_dir, "results_output")
        self.pcd_dir = os.path.join(save_dir, "pcd")
        self.pcd_baseline_dir = os.path.join(save_dir, "pcd_baseline")
        os.makedirs(self.result_unaligned_dir, exist_ok=True)
        os.makedirs(self.result_aligned_dir, exist_ok=True)
        os.makedirs(self.result_loop_dir, exist_ok=True)
        os.makedirs(self.pcd_dir, exist_ok=True)
        if self.yaw4_rig_paired_baseline_export:
            os.makedirs(self.pcd_baseline_dir, exist_ok=True)

        self.all_camera_poses = []
        self.all_camera_poses_baseline = None
        self.all_camera_intrinsics = []

        self.delete_temp_files = self.config["Model"]["delete_temp_files"]

        print("Loading model...")

        with open(self.config["Weights"]["DA3_CONFIG"]) as f:
            config = json.load(f)
        self.model = DepthAnything3(**config)
        weight = load_file(self.config["Weights"]["DA3"])
        self.model.load_state_dict(weight, strict=False)

        self.model.eval()
        self.model = self.model.to(self.device)

        self.skyseg_session = None

        self.chunk_indices = None  # [(begin_idx, end_idx), ...]

        self.loop_list = []  # e.g. [(1584, 139), ...]

        self.loop_optimizer = Sim3LoopOptimizer(self.config) if Sim3LoopOptimizer is not None else None
        self.sim3_list = []  # [(s [1,], R [3,3], T [3,]), ...]
        self.pre_loop_sim3_list = None

        self.loop_sim3_list = []  # [(chunk_idx_a, chunk_idx_b, s [1,], R [3,3], T [3,]), ...]

        self.loop_predict_list = []

        geometry_cfg = self.config.get("Loop", {}).get("GeometryVerification", {})
        self.loop_geometry_enabled = bool(geometry_cfg.get("enabled", False))
        self.loop_geometry_verification_only = bool(
            geometry_cfg.get("verification_only", False)
        )
        self.loop_geometry_max_error = float(
            geometry_cfg.get("max_side_alignment_error", 0.25)
        )
        self.loop_geometry_min_scale = float(geometry_cfg.get("min_scale", 0.8))
        self.loop_geometry_max_scale = float(geometry_cfg.get("max_scale", 1.25))
        self.loop_geometry_min_rig_acceptance = float(
            geometry_cfg.get("min_rig_acceptance", 0.8)
        )
        cross_side_cfg = geometry_cfg.get("cross_side_dense", {})
        self.loop_cross_side_enabled = bool(cross_side_cfg.get("enabled", False))
        self.loop_cross_side_record_diagnostics = bool(
            cross_side_cfg.get("record_diagnostics", False)
        )
        self.loop_cross_side_diagnostic_params = {
            "confidence_quantile": float(
                cross_side_cfg.get("confidence_quantile", 0.75)
            ),
            "min_confidence": float(cross_side_cfg.get("min_confidence", 0.0)),
            "voxel_size": float(cross_side_cfg.get("voxel_size", 0.10)),
            "max_points_per_side": int(
                cross_side_cfg.get("max_points_per_side", 30_000)
            ),
            "preselection_multiplier": int(
                cross_side_cfg.get("preselection_multiplier", 8)
            ),
            "trim_quantile": float(cross_side_cfg.get("trim_quantile", 0.90)),
            "overlap_distance": float(
                cross_side_cfg.get("overlap_distance", 0.25)
            ),
            "min_points_per_side": int(
                cross_side_cfg.get("min_points_per_side", 100)
            ),
        }
        self.loop_cross_side_gate_params = {
            "cross_side_min_mutual_count": int(
                cross_side_cfg.get("min_mutual_count", 100)
            ),
            "cross_side_max_mutual_trimmed_rmse": float(
                cross_side_cfg.get("max_mutual_trimmed_rmse", 0.25)
            ),
            "cross_side_max_mutual_median": float(
                cross_side_cfg.get("max_mutual_median", 0.15)
            ),
            "cross_side_min_overlap_a_to_b": float(
                cross_side_cfg.get("min_overlap_a_to_b", 0.35)
            ),
            "cross_side_min_overlap_b_to_a": float(
                cross_side_cfg.get("min_overlap_b_to_a", 0.35)
            ),
        }
        self.loop_geometry_rows = []
        self._last_alignment_error = float("nan")

        self.loop_enable = self.config["Model"]["loop_enable"]

        if self.loop_enable:
            loop_info_save_path = os.path.join(save_dir, "loop_closures.txt")
            self.loop_detector = LoopDetector(
                image_dir=image_dir, output=loop_info_save_path, config=self.config
            )

        if self.confidence_zero_mask_enabled:
            if not os.path.isdir(self.confidence_zero_mask_dir):
                raise FileNotFoundError(
                    f"confidence_zero_mask_dir does not exist: {self.confidence_zero_mask_dir}"
                )
            print(
                "Confidence-zero mask suppression enabled: "
                f"{self.confidence_zero_mask_dir}"
            )
        if self.yaw4_pose_filter_enabled:
            print(
                "Yaw4 pose filtering enabled: "
                f"group_size={self.yaw4_pose_filter_group_size}, "
                f"max_frame_group_dist={self.yaw4_pose_filter_max_frame_group_dist}, "
                f"max_median_group_max={self.yaw4_pose_filter_max_median_group_max}"
            )
        if self.yaw4_rig_repair_enabled:
            print(
                f"Yaw4 rig repair enabled at {self.yaw4_rig_repair_stage}: "
                f"rotation_threshold={self.yaw4_rig_rotation_threshold_deg}, "
                f"center_depth_ratio={self.yaw4_rig_center_depth_ratio}, "
                f"min_inliers={self.yaw4_rig_min_inliers}, "
                f"validate_overlap={self.yaw4_rig_validate_overlap}, "
                f"reject_unrecoverable={self.yaw4_rig_reject_unrecoverable_group}, "
                f"repair_loop_predictions={self.yaw4_rig_repair_loop_enabled}, "
                f"paired_baseline_export={self.yaw4_rig_paired_baseline_export}, "
                f"check_orientation={self.yaw4_rig_check_orientation}, "
                f"outlier_action={self.yaw4_rig_outlier_action}"
            )

        print("init done.")

    def camera_centers_from_extrinsics(self, extrinsics):
        centers = []
        for ext in extrinsics:
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :4] = ext
            c2w = np.linalg.inv(w2c)
            centers.append(c2w[:3, 3])
        return np.asarray(centers, dtype=np.float64)

    def apply_yaw4_rig_repair(
        self,
        predictions,
        image_paths,
        chunk_idx,
        chunk_range,
        *,
        scope="chunk",
        prediction_offset=0,
        loop_pair="",
    ):
        if not self.yaw4_rig_repair_enabled:
            return

        names = np.asarray([os.path.basename(path) for path in image_paths])
        frame_rows, group_rows, _, corrected, _ = analyze_chunk(
            predictions.extrinsics[prediction_offset : prediction_offset + len(image_paths)],
            names,
            center_threshold=self.yaw4_pose_filter_max_frame_group_dist,
            depth=predictions.depth[
                prediction_offset : prediction_offset + len(image_paths)
            ],
            confidence=predictions.conf[
                prediction_offset : prediction_offset + len(image_paths)
            ],
            intrinsics=predictions.intrinsics[
                prediction_offset : prediction_offset + len(image_paths)
            ],
            center_depth_ratio=self.yaw4_rig_center_depth_ratio,
            rotation_threshold_deg=self.yaw4_rig_rotation_threshold_deg,
            min_inliers=self.yaw4_rig_min_inliers,
            rotate180=True,
            check_orientation=self.yaw4_rig_check_orientation,
            validate_repair_overlap=self.yaw4_rig_validate_overlap,
            overlap_pixel_stride=self.yaw4_rig_overlap_pixel_stride,
            overlap_ray_tolerance_pixels=(
                self.yaw4_rig_overlap_ray_tolerance_pixels
            ),
            overlap_confidence_quantile=(
                self.yaw4_rig_overlap_confidence_quantile
            ),
            overlap_max_depth=self.yaw4_rig_overlap_max_depth,
            overlap_min_matched_rays=self.yaw4_rig_overlap_min_matched_rays,
            overlap_max_median_ratio=self.yaw4_rig_overlap_max_median_ratio,
            overlap_max_p90_ratio=self.yaw4_rig_overlap_max_p90_ratio,
        )
        prediction_end = prediction_offset + len(image_paths)
        if self.yaw4_rig_outlier_action == "repair":
            predictions.extrinsics[prediction_offset:prediction_end] = corrected.astype(
                predictions.extrinsics.dtype, copy=False
            )

        conf = np.array(predictions.conf, copy=True)
        if self.yaw4_rig_outlier_action == "remove":
            for group in group_rows:
                if not bool(group.get("repair_applied", False)):
                    continue
                indices = [int(value) for value in str(group["local_indices"]).split(",")]
                inliers = {
                    int(value)
                    for value in str(group["inliers"]).split(",")
                    if value != ""
                }
                outliers = [index for offset, index in enumerate(indices) if offset not in inliers]
                if outliers:
                    conf[np.asarray(outliers, dtype=np.int64) + prediction_offset] = 0
        if self.yaw4_rig_reject_unrecoverable_group:
            for group in group_rows:
                if not bool(group.get("eligible", True)) or bool(group["accepted"]):
                    continue
                indices = [int(value) for value in str(group["local_indices"]).split(",")]
                conf[np.asarray(indices, dtype=np.int64) + prediction_offset] = 0

        predictions.conf = conf
        accepted_groups = 0
        repaired_frames = 0
        removed_frames = 0
        unrecoverable_groups = 0
        rejected_groups = 0
        for group in group_rows:
            group_id = int(group["group_id"])
            accepted = bool(group["accepted"])
            eligible = bool(group.get("eligible", True))
            inliers = str(group["inliers"])
            num_inliers = int(group["num_inliers"])
            repair_applied = bool(group.get("repair_applied", False))
            if accepted:
                accepted_groups += 1
                if repair_applied:
                    if self.yaw4_rig_outlier_action == "repair":
                        repaired_frames += 1
                    elif self.yaw4_rig_outlier_action == "remove":
                        removed_frames += 1
            elif eligible:
                unrecoverable_groups += 1
                if self.yaw4_rig_reject_unrecoverable_group:
                    rejected_groups += 1
            self.yaw4_rig_repair_group_rows.append(
                {
                    "scope": scope,
                    "loop_pair": loop_pair,
                    "chunk_idx": chunk_idx,
                    "chunk_start": chunk_range[0],
                    "chunk_end": chunk_range[1],
                    "group_id": group_id,
                    "capture_key": group.get("capture_key", ""),
                    "eligible": eligible,
                    "accepted": accepted,
                    "num_inliers": num_inliers,
                    "inliers": inliers,
                    "repair_candidate": group.get("repair_candidate", False),
                    "repair_validated": group.get("repair_validated"),
                    "repair_applied": repair_applied,
                    "overlap_matched_rays": group.get("overlap_matched_rays"),
                    "overlap_raw_normalized_median": group.get(
                        "overlap_raw_normalized_median"
                    ),
                    "overlap_repaired_normalized_median": group.get(
                        "overlap_repaired_normalized_median"
                    ),
                    "overlap_raw_normalized_p90": group.get(
                        "overlap_raw_normalized_p90"
                    ),
                    "overlap_repaired_normalized_p90": group.get(
                        "overlap_repaired_normalized_p90"
                    ),
                    "center_threshold": group["center_threshold"],
                    "median_predicted_depth": group["median_predicted_depth"],
                    "center_depth_ratio": group["center_depth_ratio"],
                    "max_center_residual": group["max_center_residual"],
                    "max_rotation_residual_deg": group["max_rotation_residual_deg"],
                    "orientation_check": self.yaw4_rig_check_orientation,
                    "outlier_action": self.yaw4_rig_outlier_action,
                    "repair_stage": self.yaw4_rig_repair_stage,
                    "rejected_for_alignment": (
                        self.yaw4_rig_repair_before_alignment
                        and eligible
                        and not accepted
                        and self.yaw4_rig_reject_unrecoverable_group
                    ),
                    "rejected_for_fusion": (
                        eligible
                        and not accepted
                        and self.yaw4_rig_reject_unrecoverable_group
                    ),
                }
            )

        for row in frame_rows:
            local_idx = int(row["local_idx"])
            group = group_rows[int(row["group_id"])]
            eligible = bool(group.get("eligible", True))
            is_outlier = bool(group.get("repair_applied", False)) and not bool(
                row["inlier"]
            )
            repaired = is_outlier and self.yaw4_rig_outlier_action == "repair"
            removed = is_outlier and self.yaw4_rig_outlier_action == "remove"
            rejected = (
                eligible
                and not bool(group["accepted"])
                and self.yaw4_rig_reject_unrecoverable_group
            )
            self.yaw4_rig_repair_rows.append(
                {
                    "scope": scope,
                    "loop_pair": loop_pair,
                    "chunk_idx": chunk_idx,
                    "chunk_start": chunk_range[0],
                    "chunk_end": chunk_range[1],
                    "local_idx": local_idx,
                    "prediction_idx": prediction_offset + local_idx,
                    "global_idx": chunk_range[0] + local_idx,
                    "group_id": int(row["group_id"]),
                    "capture_key": row.get("capture_key", ""),
                    "eligible": eligible,
                    "yaw_deg": row["yaw_deg"],
                    "center_residual": row["center_residual"],
                    "rotation_residual_deg": row["rotation_residual_deg"],
                    "inlier": bool(row["inlier"]),
                    "pose_repaired": repaired,
                    "removed_for_alignment": (
                        removed and self.yaw4_rig_repair_before_alignment
                    ),
                    "removed_for_fusion": removed,
                    "orientation_check": self.yaw4_rig_check_orientation,
                    "outlier_action": self.yaw4_rig_outlier_action,
                    "repair_stage": self.yaw4_rig_repair_stage,
                    "rejected_for_alignment": (
                        rejected and self.yaw4_rig_repair_before_alignment
                    ),
                    "rejected_for_fusion": rejected,
                    "name": row["name"],
                }
            )

        print(
            f"[yaw4_rig_repair] {scope} chunk {chunk_idx}: "
            f"accepted_groups={accepted_groups}, repaired_frames={repaired_frames}, "
            f"removed_frames={removed_frames}, "
            f"unrecoverable_groups={unrecoverable_groups}, "
            f"rejected_groups={rejected_groups}"
        )
        eligible_groups = sum(bool(group.get("eligible", True)) for group in group_rows)
        return {
            "accepted_groups": accepted_groups,
            "eligible_groups": eligible_groups,
            "unrecoverable_groups": unrecoverable_groups,
            "acceptance_ratio": accepted_groups / max(eligible_groups, 1),
        }

    def apply_yaw4_pose_filter(self, chunk_idx, chunk_range, extrinsics, confs):
        if not self.yaw4_pose_filter_enabled:
            return confs, False

        filtered_confs = np.array(confs, copy=True)
        centers = self.camera_centers_from_extrinsics(extrinsics)
        group_size = self.yaw4_pose_filter_group_size
        frame_reject = np.zeros(len(centers), dtype=bool)
        group_max_dists = []

        for group_id, start in enumerate(range(0, len(centers), group_size)):
            end = min(start + group_size, len(centers))
            pts = centers[start:end]
            if len(pts) < 2:
                continue
            group_center = np.median(pts, axis=0)
            dists = np.linalg.norm(pts - group_center, axis=1)
            group_max_dists.append(float(np.max(dists)))
            local_reject = dists > self.yaw4_pose_filter_max_frame_group_dist
            frame_reject[start:end] = local_reject

            for offset, dist in enumerate(dists):
                local_idx = start + offset
                self.yaw4_pose_filter_rows.append(
                    {
                        "chunk_idx": chunk_idx,
                        "chunk_start": chunk_range[0],
                        "chunk_end": chunk_range[1],
                        "local_idx": local_idx,
                        "global_idx": chunk_range[0] + local_idx,
                        "group_id": group_id,
                        "dist_to_group_median": float(dist),
                        "frame_reject": bool(local_reject[offset]),
                        "chunk_reject": False,
                    }
                )

        median_group_max = (
            float(np.median(group_max_dists)) if len(group_max_dists) > 0 else 0.0
        )
        reject_chunk = median_group_max > self.yaw4_pose_filter_max_median_group_max
        rejected_local_indices = np.flatnonzero(frame_reject)

        if reject_chunk:
            filtered_confs[...] = 0
            self.yaw4_pose_filter_rejected_chunks += 1
            for row in self.yaw4_pose_filter_rows:
                if row["chunk_idx"] == chunk_idx:
                    row["chunk_reject"] = True
            print(
                f"[yaw4_pose_filter] reject chunk {chunk_idx}: "
                f"median_group_max={median_group_max:.6f}, "
                f"frame_rejects={rejected_local_indices.tolist()}"
            )
        else:
            for local_idx in rejected_local_indices:
                filtered_confs[local_idx] = 0
            self.yaw4_pose_filter_zeroed_frames += len(rejected_local_indices)
            if len(rejected_local_indices) > 0:
                print(
                    f"[yaw4_pose_filter] zeroed frames in chunk {chunk_idx}: "
                    f"{rejected_local_indices.tolist()} "
                    f"(median_group_max={median_group_max:.6f})"
                )

        return filtered_confs, reject_chunk

    def save_yaw4_pose_filter_report(self):
        if not self.yaw4_pose_filter_enabled:
            return
        path = os.path.join(self.output_dir, "chunk_pose_frame_filter.csv")
        fieldnames = [
            "chunk_idx",
            "chunk_start",
            "chunk_end",
            "local_idx",
            "global_idx",
            "group_id",
            "dist_to_group_median",
            "frame_reject",
            "chunk_reject",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.yaw4_pose_filter_rows)
        print(
            f"[yaw4_pose_filter] report saved to {path}; "
            f"zeroed_frames={self.yaw4_pose_filter_zeroed_frames}, "
            f"rejected_chunks={self.yaw4_pose_filter_rejected_chunks}"
        )

    def save_yaw4_rig_repair_report(self):
        if not self.yaw4_rig_repair_enabled:
            return
        frame_path = os.path.join(self.output_dir, "yaw4_rig_repair_frames.csv")
        group_path = os.path.join(self.output_dir, "yaw4_rig_repair_groups.csv")
        frame_fields = [
            "scope", "loop_pair", "chunk_idx", "chunk_start", "chunk_end",
            "local_idx", "prediction_idx", "global_idx", "group_id", "capture_key",
            "eligible", "yaw_deg", "center_residual", "rotation_residual_deg",
            "inlier", "pose_repaired", "removed_for_alignment", "removed_for_fusion",
            "orientation_check", "outlier_action", "repair_stage",
            "rejected_for_alignment", "rejected_for_fusion", "name",
        ]
        group_fields = [
            "scope", "loop_pair", "chunk_idx", "chunk_start", "chunk_end",
            "group_id", "capture_key", "eligible", "accepted",
            "num_inliers", "inliers", "max_center_residual",
            "max_rotation_residual_deg", "orientation_check", "outlier_action",
            "repair_stage", "rejected_for_alignment", "rejected_for_fusion",
        ]
        for path, fields, rows in (
            (frame_path, frame_fields, self.yaw4_rig_repair_rows),
            (group_path, group_fields, self.yaw4_rig_repair_group_rows),
        ):
            fields = list(
                dict.fromkeys([*fields, *(key for row in rows for key in row)])
            )
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        print(
            f"[yaw4_rig_repair] reports saved to {frame_path} and {group_path}"
        )

    def get_loop_pairs(self):
        salad_cfg = self.config.get("Loop", {}).get("SALAD", {})
        external_path = salad_cfg.get("external_pairs_file")
        external_mode = str(salad_cfg.get("external_pairs_mode", "replace")).lower()
        if external_mode not in {"replace", "append"}:
            raise ValueError("external_pairs_mode must be replace or append")

        loop_list = []
        if not external_path or external_mode == "append":
            self.loop_detector.run()
            loop_list.extend(self.loop_detector.get_loop_list())

        if external_path:
            external_path = os.path.expanduser(str(external_path))
            with open(external_path, newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                required = {"image_index_a", "image_index_b"}
                if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                    raise ValueError(
                        f"External loop CSV must contain {sorted(required)}: {external_path}"
                    )
                external_pairs = [
                    (int(row["image_index_a"]), int(row["image_index_b"]))
                    for row in reader
                ]
            max_external_pairs = int(salad_cfg.get("external_pairs_max", 0))
            if max_external_pairs > 0:
                external_pairs = external_pairs[:max_external_pairs]
            loop_list.extend(external_pairs)
            print(
                f"Loaded {len(external_pairs)} external loop pair(s) "
                f"in {external_mode} mode from {external_path}"
            )

        # Keep later-to-earlier ordering and remove duplicate constraints.
        return sorted({(max(a, b), min(a, b)) for a, b in loop_list}, reverse=True)

    def load_confidence_zero_mask(self, image_path, target_shape):
        if not self.confidence_zero_mask_enabled:
            return None

        mask_path = os.path.join(
            self.confidence_zero_mask_dir,
            f"{os.path.splitext(os.path.basename(image_path))[0]}.npy",
        )
        if not os.path.exists(mask_path):
            self.confidence_zero_mask_missing += 1
            message = (
                f"[confidence_zero_mask] missing mask for "
                f"{os.path.basename(image_path)}: {mask_path}"
            )
            if self.confidence_zero_mask_strict:
                raise FileNotFoundError(message)
            print(message)
            return None

        mask = np.load(mask_path, mmap_mode="r")
        if mask.ndim != 2:
            raise ValueError(f"confidence-zero mask must be 2D, got {mask.shape}: {mask_path}")
        mask = np.asarray(mask, dtype=bool)

        if mask.shape != target_shape:
            self.confidence_zero_mask_resized += 1
            resample = Image.Resampling.NEAREST if hasattr(Image, "Resampling") else Image.NEAREST
            mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
            mask = np.asarray(mask_img.resize((target_shape[1], target_shape[0]), resample=resample)) > 0

        return mask

    def apply_confidence_zero_masks(self, predictions, image_paths):
        if not self.confidence_zero_mask_enabled:
            return

        conf = predictions.conf
        if conf.ndim == 2:
            conf = conf[None, ...]

        zeroed = 0
        target_shape = conf.shape[-2:]
        for local_idx, image_path in enumerate(image_paths[: conf.shape[0]]):
            mask = self.load_confidence_zero_mask(image_path, target_shape)
            if mask is None:
                continue

            mask_pixels = int(mask.sum())
            if mask_pixels == 0:
                continue

            if isinstance(conf, torch.Tensor):
                mask_t = torch.as_tensor(mask, dtype=torch.bool, device=conf.device)
                conf[local_idx][mask_t] = 0
            else:
                conf[local_idx][mask] = 0
            zeroed += mask_pixels

        predictions.conf = conf
        self.confidence_zero_mask_zeroed_pixels += zeroed
        print(
            f"[confidence_zero_mask] zeroed {zeroed} confidence pixels in this chunk "
            f"(total={self.confidence_zero_mask_zeroed_pixels}, "
            f"missing={self.confidence_zero_mask_missing}, "
            f"resized={self.confidence_zero_mask_resized})"
        )

    def save_depth_conf_result(self, predictions, chunk_idx, s, R, T):
        if not self.config["Model"]["save_depth_conf_result"]:
            return
        os.makedirs(self.result_output_dir, exist_ok=True)

        chunk_start, _ = self.chunk_indices[chunk_idx]
        save_slice = owned_local_frame_slice(
            chunk_idx,
            self.chunk_indices,
            self.overlap_s,
            self.overlap_e,
        )
        save_indices = range(save_slice.start, save_slice.stop)

        print("[save_depth_conf_result] save_indices:")

        for local_idx in save_indices:
            global_idx = chunk_start + local_idx
            print(f"{global_idx}, ", end="")

            image = predictions.processed_images[local_idx]  # [H, W, 3] uint8
            depth = predictions.depth[local_idx]  # [H, W] float32
            conf = predictions.conf[local_idx]  # [H, W] float32
            intrinsics = predictions.intrinsics[local_idx]  # [3, 3] float32

            filename = f"frame_{global_idx}.npz"
            filepath = os.path.join(self.result_output_dir, filename)

            if self.config["Model"]["save_debug_info"]:
                np.savez_compressed(
                    filepath,
                    image=image,
                    depth=depth,
                    conf=conf,
                    intrinsics=intrinsics,
                    extrinsics=predictions.extrinsics[local_idx],
                    s=s,
                    R=R,
                    T=T,
                )
            else:
                np.savez_compressed(
                    filepath, image=image, depth=depth, conf=conf, intrinsics=intrinsics
                )
        print("")

    def _pointcloud_export_payload(
        self,
        chunk_idx,
        points,
        colors,
        confs,
        *,
        flatten,
    ):
        # Keep the historical full-chunk confidence threshold.  Canonical
        # ownership only removes duplicate frame inputs; it must not change
        # whether a pixel in an owned frame is confidence-eligible.  Sampling
        # still runs over the owned pool with the unchanged configured ratio;
        # stochastic samples are therefore not claimed to be a literal subset
        # of a legacy full-chunk sample.
        threshold = (
            float(np.mean(confs))
            * self.config["Model"]["Pointcloud_Save"]["conf_threshold_coef"]
        )
        local_slice = owned_local_frame_slice(
            chunk_idx,
            self.chunk_indices,
            self.overlap_s,
            self.overlap_e,
            enabled=self.pointcloud_canonical_overlap_ownership,
        )
        points = points[local_slice]
        colors = colors[local_slice]
        confs = confs[local_slice]
        if np.size(confs) == 0:
            raise ValueError(
                f"point-cloud ownership selected no pixels for chunk {chunk_idx}"
            )

        if flatten:
            points = points.reshape(-1, 3)
            colors = colors.reshape(-1, 3).astype(np.uint8)
            confs = confs.reshape(-1)

        return points, colors, confs, threshold

    def _save_pointcloud_chunk(
        self,
        chunk_idx,
        points,
        colors,
        confs,
        output_path,
        *,
        flatten,
    ):
        points, colors, confs, threshold = self._pointcloud_export_payload(
            chunk_idx,
            points,
            colors,
            confs,
            flatten=flatten,
        )
        save_confident_pointcloud_batch(
            points=points,
            colors=colors,
            confs=confs,
            output_path=output_path,
            conf_threshold=threshold,
            sample_ratio=self.config["Model"]["Pointcloud_Save"]["sample_ratio"],
        )

    def process_single_chunk(
        self, range_1, chunk_idx=None, range_2=None, is_loop=False, loop_context=None
    ):
        if (
            not is_loop
            and range_2 is None
            and chunk_idx is not None
            and self.config["Model"].get("reuse_existing_unaligned_chunks", False)
        ):
            cached_path = os.path.join(
                self.result_unaligned_dir, f"chunk_{chunk_idx}.npy"
            )
            if os.path.exists(cached_path):
                predictions = np.load(cached_path, allow_pickle=True).item()
                chunk_range = self.chunk_indices[chunk_idx]
                self.all_camera_poses.append(
                    (chunk_range, np.array(predictions.extrinsics, copy=True))
                )
                self.all_camera_intrinsics.append(
                    (chunk_range, np.array(predictions.intrinsics, copy=True))
                )
                print(f"[resume] loaded cached unaligned chunk: {cached_path}")
                return predictions

        start_idx, end_idx = range_1
        chunk_image_paths = self.img_list[start_idx:end_idx]
        if range_2 is not None:
            start_idx, end_idx = range_2
            chunk_image_paths += self.img_list[start_idx:end_idx]

        # images = load_and_preprocess_images(chunk_image_paths).to(self.device)
        print(f"Loaded {len(chunk_image_paths)} images")

        ref_view_strategy = self.config["Model"][
            "ref_view_strategy" if not is_loop else "ref_view_strategy_loop"
        ]

        torch.cuda.empty_cache()
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images = chunk_image_paths
                # images: ['xxx.png', 'xxx.png', ...]

                inference_kwargs = {"ref_view_strategy": ref_view_strategy}
                process_res = self.config["Model"].get("process_res")
                process_res_method = self.config["Model"].get("process_res_method")
                if process_res is not None:
                    inference_kwargs["process_res"] = int(process_res)
                if process_res_method is not None:
                    inference_kwargs["process_res_method"] = process_res_method
                predictions = self.model.inference(images, **inference_kwargs)

                predictions.depth = np.squeeze(predictions.depth)
                predictions.conf -= 1.0
                self.apply_confidence_zero_masks(predictions, chunk_image_paths)
                if not is_loop and self.yaw4_rig_repair_before_alignment:
                    self.apply_yaw4_rig_repair(
                        predictions,
                        chunk_image_paths,
                        chunk_idx,
                        self.chunk_indices[chunk_idx],
                    )
                elif is_loop and self.yaw4_rig_repair_loop_enabled:
                    if range_2 is None or loop_context is None:
                        raise ValueError("Loop rig repair requires range_2 and loop_context")
                    first_length = range_1[1] - range_1[0]
                    pair_label = (
                        f"{range_1[0]}:{range_1[1]}-{range_2[0]}:{range_2[1]}"
                    )
                    rig_a = self.apply_yaw4_rig_repair(
                        predictions,
                        chunk_image_paths[:first_length],
                        loop_context[0],
                        range_1,
                        scope="loop_a",
                        prediction_offset=0,
                        loop_pair=pair_label,
                    )
                    rig_b = self.apply_yaw4_rig_repair(
                        predictions,
                        chunk_image_paths[first_length:],
                        loop_context[2],
                        range_2,
                        scope="loop_b",
                        prediction_offset=first_length,
                        loop_pair=pair_label,
                    )
                    predictions.loop_rig_validation = {"a": rig_a, "b": rig_b}

                print(predictions.processed_images.shape)  # [N, H, W, 3] uint8
                print(predictions.depth.shape)  # [N, H, W] float32
                print(predictions.conf.shape)  # [N, H, W] float32
                print(predictions.extrinsics.shape)  # [N, 3, 4] float32 (w2c)
                print(predictions.intrinsics.shape)  # [N, 3, 3] float32
        torch.cuda.empty_cache()

        # Save predictions to disk instead of keeping in memory
        if is_loop:
            save_dir = self.result_loop_dir
            filename = f"loop_{range_1[0]}_{range_1[1]}_{range_2[0]}_{range_2[1]}.npy"
        else:
            if chunk_idx is None:
                raise ValueError("chunk_idx must be provided when is_loop is False")
            save_dir = self.result_unaligned_dir
            filename = f"chunk_{chunk_idx}.npy"

        save_path = os.path.join(save_dir, filename)

        if not is_loop and range_2 is None:
            extrinsics = predictions.extrinsics
            intrinsics = predictions.intrinsics
            chunk_range = self.chunk_indices[chunk_idx]
            self.all_camera_poses.append((chunk_range, extrinsics))
            self.all_camera_intrinsics.append((chunk_range, intrinsics))

        np.save(save_path, predictions)

        return predictions

    def get_chunk_indices(self):
        if len(self.img_list) <= self.chunk_size:
            num_chunks = 1
            chunk_indices = [(0, len(self.img_list))]
        else:
            step = self.chunk_size - self.overlap
            num_chunks = (len(self.img_list) - self.overlap + step - 1) // step
            chunk_indices = []
            for i in range(num_chunks):
                start_idx = i * step
                end_idx = min(start_idx + self.chunk_size, len(self.img_list))
                chunk_indices.append((start_idx, end_idx))
        return chunk_indices, num_chunks

    def align_2pcds(
        self,
        point_map1,
        conf1,
        point_map2,
        conf2,
        chunk1_depth,
        chunk2_depth,
        chunk1_depth_conf,
        chunk2_depth_conf,
    ):

        conf_threshold = min(np.median(conf1), np.median(conf2)) * 0.1

        scale_factor = None
        if self.config["Model"]["align_method"] == "scale+se3":
            scale_factor_return, quality_score, method_used = precompute_scale_chunks_with_depth(
                chunk1_depth,
                chunk1_depth_conf,
                chunk2_depth,
                chunk2_depth_conf,
                method=self.config["Model"]["scale_compute_method"],
            )
            print(
                f"[Depth Scale Precompute] scale: {scale_factor_return}, \
                    quality_score: {quality_score}, method_used: {method_used}"
            )
            scale_factor = scale_factor_return

        s, R, t, alignment_error = weighted_align_point_maps(
            point_map1,
            conf1,
            point_map2,
            conf2,
            conf_threshold=conf_threshold,
            config=self.config,
            precompute_scale=scale_factor,
            return_error=True,
        )
        self._last_alignment_error = alignment_error
        print("Estimated Scale:", s)
        print("Estimated Rotation:\n", R)
        print("Estimated Translation:", t)

        return s, R, t

    def get_loop_sim3_from_loop_predict(self, loop_predict_list):
        loop_sim3_list = []
        for item in loop_predict_list:
            chunk_idx_a = item[0][0]
            chunk_idx_b = item[0][2]
            chunk_a_range = item[0][1]
            chunk_b_range = item[0][3]

            point_map_loop_org = depth_to_point_cloud_vectorized(
                item[1].depth, item[1].intrinsics, item[1].extrinsics
            )

            chunk_a_s = 0
            chunk_a_e = chunk_a_len = chunk_a_range[1] - chunk_a_range[0]
            chunk_b_s = -chunk_b_range[1] + chunk_b_range[0]
            chunk_b_e = point_map_loop_org.shape[0]
            chunk_b_len = chunk_b_range[1] - chunk_b_range[0]

            chunk_a_rela_begin = chunk_a_range[0] - self.chunk_indices[chunk_idx_a][0]
            chunk_a_rela_end = chunk_a_rela_begin + chunk_a_len
            chunk_b_rela_begin = chunk_b_range[0] - self.chunk_indices[chunk_idx_b][0]
            chunk_b_rela_end = chunk_b_rela_begin + chunk_b_len

            print("chunk_a align")

            point_map_loop_a = point_map_loop_org[chunk_a_s:chunk_a_e]
            conf_loop_a = item[1].conf[chunk_a_s:chunk_a_e]
            print(self.chunk_indices[chunk_idx_a])
            print(chunk_a_range)
            print(chunk_a_rela_begin, chunk_a_rela_end)
            chunk_data_a = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_a}.npy"),
                allow_pickle=True,
            ).item()

            point_map_a = depth_to_point_cloud_vectorized(
                chunk_data_a.depth, chunk_data_a.intrinsics, chunk_data_a.extrinsics
            )
            point_map_a = point_map_a[chunk_a_rela_begin:chunk_a_rela_end]
            conf_a = chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_a_depth = np.squeeze(chunk_data_a.depth[chunk_a_rela_begin:chunk_a_rela_end])
                chunk_a_depth_conf = np.squeeze(
                    chunk_data_a.conf[chunk_a_rela_begin:chunk_a_rela_end]
                )
                chunk_a_loop_depth = np.squeeze(item[1].depth[chunk_a_s:chunk_a_e])
                chunk_a_loop_depth_conf = np.squeeze(item[1].conf[chunk_a_s:chunk_a_e])
            else:
                chunk_a_depth = None
                chunk_a_loop_depth = None
                chunk_a_depth_conf = None
                chunk_a_loop_depth_conf = None

            s_a, R_a, t_a = self.align_2pcds(
                point_map_a,
                conf_a,
                point_map_loop_a,
                conf_loop_a,
                chunk_a_depth,
                chunk_a_loop_depth,
                chunk_a_depth_conf,
                chunk_a_loop_depth_conf,
            )
            alignment_error_a = self._last_alignment_error

            print("chunk_b align")

            point_map_loop_b = point_map_loop_org[chunk_b_s:chunk_b_e]
            conf_loop_b = item[1].conf[chunk_b_s:chunk_b_e]
            print(self.chunk_indices[chunk_idx_b])
            print(chunk_b_range)
            print(chunk_b_rela_begin, chunk_b_rela_end)
            chunk_data_b = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx_b}.npy"),
                allow_pickle=True,
            ).item()

            point_map_b = depth_to_point_cloud_vectorized(
                chunk_data_b.depth, chunk_data_b.intrinsics, chunk_data_b.extrinsics
            )
            point_map_b = point_map_b[chunk_b_rela_begin:chunk_b_rela_end]
            conf_b = chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]

            if self.config["Model"]["align_method"] == "scale+se3":
                chunk_b_depth = np.squeeze(chunk_data_b.depth[chunk_b_rela_begin:chunk_b_rela_end])
                chunk_b_depth_conf = np.squeeze(
                    chunk_data_b.conf[chunk_b_rela_begin:chunk_b_rela_end]
                )
                chunk_b_loop_depth = np.squeeze(item[1].depth[chunk_b_s:chunk_b_e])
                chunk_b_loop_depth_conf = np.squeeze(item[1].conf[chunk_b_s:chunk_b_e])
            else:
                chunk_b_depth = None
                chunk_b_loop_depth = None
                chunk_b_depth_conf = None
                chunk_b_loop_depth_conf = None

            s_b, R_b, t_b = self.align_2pcds(
                point_map_b,
                conf_b,
                point_map_loop_b,
                conf_loop_b,
                chunk_b_depth,
                chunk_b_loop_depth,
                chunk_b_depth_conf,
                chunk_b_loop_depth_conf,
            )
            alignment_error_b = self._last_alignment_error

            print("a -> b SIM 3")
            s_ab, R_ab, t_ab = compute_sim3_ab((s_a, R_a, t_a), (s_b, R_b, t_b))
            print("Estimated Scale:", s_ab)
            print("Estimated Rotation:\n", R_ab)
            print("Estimated Translation:", t_ab)

            rig = getattr(item[1], "loop_rig_validation", {})
            rig_a = rig.get("a") or {}
            rig_b = rig.get("b") or {}
            rig_acceptance_a = float(rig_a.get("acceptance_ratio", float("nan")))
            rig_acceptance_b = float(rig_b.get("acceptance_ratio", float("nan")))
            scale = float(np.asarray(s_ab).reshape(-1)[0])
            cross_side = empty_cross_side_diagnostics()
            if (
                self.loop_cross_side_enabled
                or self.loop_cross_side_record_diagnostics
            ):
                cross_side = evaluate_cross_side_dense_geometry(
                    point_map_loop_a,
                    conf_loop_a,
                    point_map_loop_b,
                    conf_loop_b,
                    **self.loop_cross_side_diagnostic_params,
                )
            cross_side_gate_active = (
                self.loop_geometry_enabled and self.loop_cross_side_enabled
            )
            accepted, reasons = evaluate_loop_geometry(
                alignment_error_a=alignment_error_a,
                alignment_error_b=alignment_error_b,
                scale=scale,
                rig_acceptance_a=rig_acceptance_a,
                rig_acceptance_b=rig_acceptance_b,
                max_side_alignment_error=self.loop_geometry_max_error,
                min_scale=self.loop_geometry_min_scale,
                max_scale=self.loop_geometry_max_scale,
                min_rig_acceptance=self.loop_geometry_min_rig_acceptance,
                cross_side_enabled=cross_side_gate_active,
                cross_side_status=str(cross_side["cross_side_status"]),
                cross_side_mutual_count=int(
                    cross_side["cross_side_mutual_count"]
                ),
                cross_side_mutual_trimmed_rmse=float(
                    cross_side["cross_side_mutual_trimmed_rmse"]
                ),
                cross_side_mutual_median=float(
                    cross_side["cross_side_mutual_median"]
                ),
                cross_side_overlap_a_to_b=float(
                    cross_side["cross_side_overlap_a_to_b"]
                ),
                cross_side_overlap_b_to_a=float(
                    cross_side["cross_side_overlap_b_to_a"]
                ),
                **self.loop_cross_side_gate_params,
            )
            if not self.loop_geometry_enabled:
                accepted, reasons = True, []
            self.loop_geometry_rows.append(
                {
                    "chunk_a": chunk_idx_a,
                    "chunk_b": chunk_idx_b,
                    "range_a": f"{chunk_a_range[0]}:{chunk_a_range[1]}",
                    "range_b": f"{chunk_b_range[0]}:{chunk_b_range[1]}",
                    "alignment_error_a": alignment_error_a,
                    "alignment_error_b": alignment_error_b,
                    "scale": scale,
                    "rig_acceptance_a": rig_acceptance_a,
                    "rig_acceptance_b": rig_acceptance_b,
                    "geometry_gate_enabled": self.loop_geometry_enabled,
                    "cross_side_gate_enabled": self.loop_cross_side_enabled,
                    "cross_side_gate_active": cross_side_gate_active,
                    "cross_side_record_diagnostics": (
                        self.loop_cross_side_record_diagnostics
                    ),
                    **{
                        f"cross_side_{key}": value
                        for key, value in self.loop_cross_side_diagnostic_params.items()
                    },
                    **self.loop_cross_side_gate_params,
                    **cross_side,
                    "accepted": accepted,
                    "reasons": ";".join(reasons),
                }
            )
            print(
                "[loop_geometry] "
                f"accepted={accepted}, reasons={reasons}, "
                f"cross_side_status={cross_side['cross_side_status']}, "
                "cross_side_mutual_rmse="
                f"{float(cross_side['cross_side_mutual_trimmed_rmse']):.6f}, "
                "cross_side_overlap="
                f"({float(cross_side['cross_side_overlap_a_to_b']):.6f}, "
                f"{float(cross_side['cross_side_overlap_b_to_a']):.6f})"
            )
            if accepted:
                loop_sim3_list.append((chunk_idx_a, chunk_idx_b, (s_ab, R_ab, t_ab)))

        if self.loop_geometry_rows:
            path = os.path.join(self.output_dir, "loop_geometry_verification.csv")
            with open(path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=list(self.loop_geometry_rows[0])
                )
                writer.writeheader()
                writer.writerows(self.loop_geometry_rows)
            print(f"[loop_geometry] report saved to {path}")
        return loop_sim3_list

    def plot_loop_closure(
        self, input_abs_poses, optimized_abs_poses, save_name="sim3_opt_result.png"
    ):
        def extract_xyz(pose_tensor):
            poses = pose_tensor.cpu().numpy()
            return poses[:, 0], poses[:, 1], poses[:, 2]

        x0, _, y0 = extract_xyz(input_abs_poses)
        x1, _, y1 = extract_xyz(optimized_abs_poses)

        # Visual in png format
        plt.figure(figsize=(8, 6))
        plt.plot(x0, y0, "o--", alpha=0.45, label="Before Optimization")
        plt.plot(x1, y1, "o-", label="After Optimization")
        for i, j, _ in self.loop_sim3_list:
            plt.plot(
                [x0[i], x0[j]],
                [y0[i], y0[j]],
                "r--",
                alpha=0.25,
                label="Loop (Before)" if i == 5 else "",
            )
            plt.plot(
                [x1[i], x1[j]],
                [y1[i], y1[j]],
                "g-",
                alpha=0.25,
                label="Loop (After)" if i == 5 else "",
            )
        plt.gca().set_aspect("equal")
        plt.title("Sim3 Loop Closure Optimization")
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def process_long_sequence(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"[SETTING ERROR] Overlap ({self.overlap}) \
                    must be less than chunk size ({self.chunk_size})"
            )

        self.chunk_indices, num_chunks = self.get_chunk_indices()

        print(
            f"Processing {len(self.img_list)} images in {num_chunks} \
                chunks of size {self.chunk_size} with {self.overlap} overlap"
        )

        pre_predictions = None
        for chunk_idx in range(len(self.chunk_indices)):
            print(f"[Progress]: {chunk_idx}/{len(self.chunk_indices)}")
            cur_predictions = self.process_single_chunk(
                self.chunk_indices[chunk_idx], chunk_idx=chunk_idx
            )
            torch.cuda.empty_cache()

            if chunk_idx > 0:
                print(
                    f"Aligning {chunk_idx-1} and {chunk_idx} (Total {len(self.chunk_indices)-1})"
                )
                chunk_data1 = pre_predictions
                chunk_data2 = cur_predictions

                point_map1 = depth_to_point_cloud_vectorized(
                    chunk_data1.depth, chunk_data1.intrinsics, chunk_data1.extrinsics
                )
                point_map2 = depth_to_point_cloud_vectorized(
                    chunk_data2.depth, chunk_data2.intrinsics, chunk_data2.extrinsics
                )

                point_map1 = point_map1[-self.overlap :]
                point_map2 = point_map2[: self.overlap]
                conf1 = chunk_data1.conf[-self.overlap :]
                conf2 = chunk_data2.conf[: self.overlap]

                if self.config["Model"]["align_method"] == "scale+se3":
                    chunk1_depth = np.squeeze(chunk_data1.depth[-self.overlap :])
                    chunk2_depth = np.squeeze(chunk_data2.depth[: self.overlap])
                    chunk1_depth_conf = np.squeeze(chunk_data1.conf[-self.overlap :])
                    chunk2_depth_conf = np.squeeze(chunk_data2.conf[: self.overlap])
                else:
                    chunk1_depth = None
                    chunk2_depth = None
                    chunk1_depth_conf = None
                    chunk2_depth_conf = None

                s, R, t = self.align_2pcds(
                    point_map1,
                    conf1,
                    point_map2,
                    conf2,
                    chunk1_depth,
                    chunk2_depth,
                    chunk1_depth_conf,
                    chunk2_depth_conf,
                )
                self.sim3_list.append((s, R, t))

            pre_predictions = cur_predictions

        if self.loop_enable:
            self.loop_list = self.get_loop_pairs()
            del self.loop_detector  # Save GPU Memory

            torch.cuda.empty_cache()

            print("Loop SIM(3) estimating...")
            loop_results = process_loop_list(
                self.chunk_indices,
                self.loop_list,
                half_window=int(self.config["Model"]["loop_chunk_size"] / 2),
            )
            event_radius_chunks = self.config["Loop"].get(
                "event_nms_chunk_radius", 0
            )
            loop_results = remove_duplicates(
                loop_results, event_radius_chunks=event_radius_chunks
            )
            print(loop_results)
            # return e.g. (31, (1574, 1594), 2, (129, 149))
            for item in loop_results:
                single_chunk_predictions = self.process_single_chunk(
                    item[1], range_2=item[3], is_loop=True, loop_context=item
                )

                self.loop_predict_list.append((item, single_chunk_predictions))
                print(item)

            self.loop_sim3_list = self.get_loop_sim3_from_loop_predict(self.loop_predict_list)

            if self.loop_geometry_verification_only:
                self.save_yaw4_pose_filter_report()
                self.save_yaw4_rig_repair_report()
                print(
                    "Geometry-verification-only mode complete; "
                    "skipping loop optimization and point-cloud export"
                )
                return

            if self.loop_sim3_list:
                self.pre_loop_sim3_list = [
                    (s, np.array(R, copy=True), np.array(t, copy=True))
                    for s, R, t in self.sim3_list
                ]
                input_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                    self.sim3_list
                )
                proposed_sim3_list = self.loop_optimizer.optimize(
                    self.sim3_list, self.loop_sim3_list
                )
                self.sim3_list = proposed_sim3_list
                selected_abs_poses = self.loop_optimizer.sequential_to_absolute_poses(
                    self.sim3_list
                )
                self.plot_loop_closure(
                    input_abs_poses,
                    selected_abs_poses,
                    save_name="sim3_opt_result.png",
                )
            else:
                print("No loop candidate passed geometry verification; skipping optimization")

        print("Apply alignment")
        self.sim3_list = accumulate_sim3_transforms(self.sim3_list)
        if self.pre_loop_sim3_list is not None:
            self.pre_loop_sim3_list = accumulate_sim3_transforms(
                self.pre_loop_sim3_list
            )
        if self.yaw4_rig_paired_baseline_export:
            self.all_camera_poses_baseline = [
                (chunk_range, np.array(extrinsics, copy=True))
                for chunk_range, extrinsics in self.all_camera_poses
            ]
        for chunk_idx in range(len(self.chunk_indices) - 1):
            print(f"Applying {chunk_idx+1} -> {chunk_idx} (Total {len(self.chunk_indices)-1})")
            s, R, t = self.sim3_list[chunk_idx]

            chunk_data = np.load(
                os.path.join(self.result_unaligned_dir, f"chunk_{chunk_idx+1}.npy"),
                allow_pickle=True,
            ).item()

            aligned_chunk_data = {}

            baseline_world_points = None
            baseline_conf = None
            if self.yaw4_rig_paired_baseline_export:
                baseline_world_points = depth_to_point_cloud_optimized_torch(
                    chunk_data.depth,
                    chunk_data.intrinsics,
                    chunk_data.extrinsics,
                )
                baseline_world_points = apply_sim3_direct_torch(
                    baseline_world_points, s, R, t
                )
                baseline_conf = np.array(chunk_data.conf, copy=True)

            if self.yaw4_rig_repair_after_alignment:
                self.apply_yaw4_rig_repair(
                    chunk_data,
                    self.img_list[
                        self.chunk_indices[chunk_idx + 1][0]
                        : self.chunk_indices[chunk_idx + 1][1]
                    ],
                    chunk_idx + 1,
                    self.chunk_indices[chunk_idx + 1],
                )
                self.all_camera_poses[chunk_idx + 1] = (
                    self.chunk_indices[chunk_idx + 1],
                    np.array(chunk_data.extrinsics, copy=True),
                )

            aligned_chunk_data["world_points"] = depth_to_point_cloud_optimized_torch(
                chunk_data.depth, chunk_data.intrinsics, chunk_data.extrinsics
            )
            aligned_chunk_data["world_points"] = apply_sim3_direct_torch(
                aligned_chunk_data["world_points"], s, R, t
            )

            filtered_conf, reject_chunk = self.apply_yaw4_pose_filter(
                chunk_idx + 1,
                self.chunk_indices[chunk_idx + 1],
                chunk_data.extrinsics,
                chunk_data.conf,
            )
            aligned_chunk_data["conf"] = filtered_conf
            aligned_chunk_data["images"] = chunk_data.processed_images

            aligned_path = os.path.join(self.result_aligned_dir, f"chunk_{chunk_idx+1}.npy")
            # Aligned chunks are consumed below while still in memory. Keep the
            # historical on-disk cache by default, but allow storage-constrained
            # runs to omit this purely temporary duplicate.
            if not self.config["Model"].get("skip_aligned_cache_write", False):
                np.save(aligned_path, aligned_chunk_data)

            if chunk_idx == 0:
                chunk_data_first = np.load(
                    os.path.join(self.result_unaligned_dir, "chunk_0.npy"), allow_pickle=True
                ).item()
                baseline_points_first = None
                baseline_confs_first = None
                if self.yaw4_rig_paired_baseline_export:
                    baseline_points_first = depth_to_point_cloud_vectorized(
                        chunk_data_first.depth,
                        chunk_data_first.intrinsics,
                        chunk_data_first.extrinsics,
                    )
                    baseline_confs_first = np.array(
                        chunk_data_first.conf, copy=True
                    )
                if self.yaw4_rig_repair_after_alignment:
                    self.apply_yaw4_rig_repair(
                        chunk_data_first,
                        self.img_list[
                            self.chunk_indices[0][0] : self.chunk_indices[0][1]
                        ],
                        0,
                        self.chunk_indices[0],
                    )
                    self.all_camera_poses[0] = (
                        self.chunk_indices[0],
                        np.array(chunk_data_first.extrinsics, copy=True),
                    )
                if not self.config["Model"].get("skip_aligned_cache_write", False):
                    np.save(
                        os.path.join(self.result_aligned_dir, "chunk_0.npy"),
                        chunk_data_first,
                    )
                points_first = depth_to_point_cloud_vectorized(
                    chunk_data_first.depth,
                    chunk_data_first.intrinsics,
                    chunk_data_first.extrinsics,
                )
                colors_first = chunk_data_first.processed_images
                confs_first, reject_first_chunk = self.apply_yaw4_pose_filter(
                    0,
                    self.chunk_indices[0],
                    chunk_data_first.extrinsics,
                    chunk_data_first.conf,
                )
                ply_path_first = os.path.join(self.pcd_dir, "0_pcd.ply")
                if not reject_first_chunk:
                    self._save_pointcloud_chunk(
                        0,
                        points_first,
                        colors_first,
                        confs_first,
                        ply_path_first,
                        flatten=False,
                    )
                if (
                    self.yaw4_rig_paired_baseline_export
                    and not reject_first_chunk
                ):
                    self._save_pointcloud_chunk(
                        0,
                        baseline_points_first,
                        colors_first,
                        baseline_confs_first,
                        os.path.join(self.pcd_baseline_dir, "0_pcd.ply"),
                        flatten=False,
                    )
                if self.config["Model"]["save_depth_conf_result"]:
                    predictions = chunk_data_first
                    self.save_depth_conf_result(predictions, 0, 1, np.eye(3), np.array([0, 0, 0]))

            if not reject_chunk:
                ply_path = os.path.join(self.pcd_dir, f"{chunk_idx+1}_pcd.ply")
                self._save_pointcloud_chunk(
                    chunk_idx + 1,
                    aligned_chunk_data["world_points"],
                    aligned_chunk_data["images"],
                    aligned_chunk_data["conf"],
                    ply_path,
                    flatten=True,
                )
            if self.yaw4_rig_paired_baseline_export and not reject_chunk:
                self._save_pointcloud_chunk(
                    chunk_idx + 1,
                    baseline_world_points,
                    aligned_chunk_data["images"],
                    baseline_conf,
                    os.path.join(
                        self.pcd_baseline_dir, f"{chunk_idx+1}_pcd.ply"
                    ),
                    flatten=True,
                )

            if self.config["Model"]["save_depth_conf_result"]:
                predictions = chunk_data
                predictions.depth *= s
                self.save_depth_conf_result(predictions, chunk_idx + 1, s, R, t)

        self.save_yaw4_pose_filter_report()
        self.save_yaw4_rig_repair_report()
        self.save_camera_poses()
        if self.pre_loop_sim3_list is not None:
            self.save_camera_poses(
                poses_filename="camera_poses_pre_loop.txt",
                ply_filename="camera_poses_pre_loop.ply",
                save_intrinsics=False,
                sim3_transforms=self.pre_loop_sim3_list,
            )
        if self.all_camera_poses_baseline is not None:
            self.save_camera_poses(
                camera_poses=self.all_camera_poses_baseline,
                poses_filename="camera_poses_baseline.txt",
                ply_filename="camera_poses_baseline.ply",
                save_intrinsics=False,
            )

        print("Done.")

    def run(self):
        print(f"Loading images from {self.img_dir}...")
        self.img_list = sorted(
            glob.glob(os.path.join(self.img_dir, "*.jpg"))
            + glob.glob(os.path.join(self.img_dir, "*.png"))
        )
        # print(self.img_list)
        if len(self.img_list) == 0:
            raise ValueError(f"[DIR EMPTY] No images found in {self.img_dir}!")
        print(f"Found {len(self.img_list)} images")

        self.process_long_sequence()

    def save_camera_poses(
        self,
        camera_poses=None,
        poses_filename="camera_poses.txt",
        ply_filename="camera_poses.ply",
        save_intrinsics=True,
        sim3_transforms=None,
    ):
        """
        Save camera poses from all chunks to txt and ply files
        - txt file: Each line contains a 4x4 C2W matrix flattened into 16 numbers
        - ply file: Camera poses visualized as points with different colors for each chunk
        """
        chunk_colors = [
            [255, 0, 0],  # Red
            [0, 255, 0],  # Green
            [0, 0, 255],  # Blue
            [255, 255, 0],  # Yellow
            [255, 0, 255],  # Magenta
            [0, 255, 255],  # Cyan
            [128, 0, 0],  # Dark Red
            [0, 128, 0],  # Dark Green
            [0, 0, 128],  # Dark Blue
            [128, 128, 0],  # Olive
        ]
        print("Saving all camera poses to txt file...")

        camera_poses = self.all_camera_poses if camera_poses is None else camera_poses
        sim3_transforms = self.sim3_list if sim3_transforms is None else sim3_transforms

        all_poses = [None] * len(self.img_list)
        all_intrinsics = [None] * len(self.img_list)

        first_chunk_range, first_chunk_extrinsics = camera_poses[0]
        _, first_chunk_intrinsics = self.all_camera_intrinsics[0]

        first_owned = owned_local_frame_slice(
            0,
            [chunk_range for chunk_range, _ in camera_poses],
            self.overlap_s,
            self.overlap_e,
        )
        for local_idx in range(first_owned.start, first_owned.stop):
            idx = first_chunk_range[0] + local_idx
            w2c = np.eye(4)
            w2c[:3, :] = first_chunk_extrinsics[local_idx]
            c2w = np.linalg.inv(w2c)
            all_poses[idx] = c2w
            all_intrinsics[idx] = first_chunk_intrinsics[local_idx]

        for chunk_idx in range(1, len(camera_poses)):
            chunk_range, chunk_extrinsics = camera_poses[chunk_idx]
            _, chunk_intrinsics = self.all_camera_intrinsics[chunk_idx]
            s, R, t = sim3_transforms[
                chunk_idx - 1
            ]  # When call self.save_camera_poses(), all the sim3 are aligned to the first chunk.

            S = np.eye(4)
            S[:3, :3] = s * R
            S[:3, 3] = t

            owned = owned_local_frame_slice(
                chunk_idx,
                [item[0] for item in camera_poses],
                self.overlap_s,
                self.overlap_e,
            )

            for local_idx in range(owned.start, owned.stop):
                idx = chunk_range[0] + local_idx
                w2c = np.eye(4)
                w2c[:3, :] = chunk_extrinsics[local_idx]
                c2w = np.linalg.inv(w2c)

                transformed_c2w = S @ c2w  # Be aware of the left multiplication!
                transformed_c2w[:3, :3] /= s  # Normalize rotation

                all_poses[idx] = transformed_c2w
                all_intrinsics[idx] = chunk_intrinsics[local_idx]

        poses_path = os.path.join(self.output_dir, poses_filename)
        with open(poses_path, "w") as f:
            for pose in all_poses:
                flat_pose = pose.flatten()
                f.write(" ".join([str(x) for x in flat_pose]) + "\n")

        print(f"Camera poses saved to {poses_path}")

        if save_intrinsics:
            intrinsics_path = os.path.join(self.output_dir, "intrinsic.txt")
            with open(intrinsics_path, "w") as f:
                for intrinsic in all_intrinsics:
                    fx = intrinsic[0, 0]
                    fy = intrinsic[1, 1]
                    cx = intrinsic[0, 2]
                    cy = intrinsic[1, 2]
                    f.write(f"{fx} {fy} {cx} {cy}\n")

            print(f"Camera intrinsics saved to {intrinsics_path}")

        ply_path = os.path.join(self.output_dir, ply_filename)
        with open(ply_path, "w") as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(all_poses)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")

            color = chunk_colors[0]
            for pose in all_poses:
                position = pose[:3, 3]
                f.write(
                    f"{position[0]} {position[1]} {position[2]} {color[0]} {color[1]} {color[2]}\n"
                )

        print(f"Camera poses visualization saved to {ply_path}")

    def close(self):
        """
        Clean up temporary files and calculate reclaimed disk space.

        This method deletes all temporary files generated during processing from three directories:
        - Unaligned results
        - Aligned results
        - Loop results

        ~50 GiB for 4500-frame KITTI 00,
        ~35 GiB for 2700-frame KITTI 05,
        or ~5 GiB for 300-frame short seq.
        """
        if not self.delete_temp_files:
            return

        total_space = 0

        print(f"Deleting the temp files under {self.result_unaligned_dir}")
        for filename in os.listdir(self.result_unaligned_dir):
            file_path = os.path.join(self.result_unaligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_aligned_dir}")
        for filename in os.listdir(self.result_aligned_dir):
            file_path = os.path.join(self.result_aligned_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)

        print(f"Deleting the temp files under {self.result_loop_dir}")
        for filename in os.listdir(self.result_loop_dir):
            file_path = os.path.join(self.result_loop_dir, filename)
            if os.path.isfile(file_path):
                total_space += os.path.getsize(file_path)
                os.remove(file_path)
        print("Deleting temp files done.")

        print(f"Saved disk space: {total_space/1024/1024/1024:.4f} GiB")


def copy_file(src_path, dst_dir):
    try:
        os.makedirs(dst_dir, exist_ok=True)

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))

        shutil.copy2(src_path, dst_path)
        print(f"config yaml file has been copied to: {dst_path}")
        return dst_path

    except FileNotFoundError:
        print("File Not Found")
    except PermissionError:
        print("Permission Error")
    except Exception as e:
        print(f"Copy Error: {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DA3-Streaming")
    parser.add_argument("--image_dir", type=str, required=True, help="Image path")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./configs/base_config.yaml",
        help="Image path",
    )
    parser.add_argument("--output_dir", type=str, required=False, default=None, help="Output path")
    parser.add_argument(
        "--confidence_zero_mask_dir",
        type=str,
        required=False,
        default=None,
        help="Directory containing per-image .npy masks whose pixels should force DA3 confidence to zero. Matching is done by image stem.",
    )
    parser.add_argument(
        "--confidence_zero_mask_strict",
        action="store_true",
        help="Raise an error if a confidence-zero mask is missing for any processed image.",
    )
    parser.add_argument(
        "--person_mask_dir",
        type=str,
        required=False,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--person_mask_strict",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    config = load_config(args.config)
    confidence_zero_mask_dir = args.confidence_zero_mask_dir or args.person_mask_dir
    if confidence_zero_mask_dir is not None:
        config["Model"]["confidence_zero_mask_dir"] = confidence_zero_mask_dir
    if args.confidence_zero_mask_strict or args.person_mask_strict:
        config["Model"]["confidence_zero_mask_strict"] = True

    image_dir = args.image_dir
    path = image_dir.split("/")

    if args.output_dir is not None:
        save_dir = args.output_dir
    else:
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        exp_dir = "./exps"
        save_dir = os.path.join(exp_dir, image_dir.replace("/", "_"), current_datetime)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"The exp will be saved under dir: {save_dir}")
        copy_file(args.config, save_dir)

    if config["Model"]["align_lib"] == "numba":
        warmup_numba()

    da3_streaming = DA3_Streaming(image_dir, save_dir, config)
    da3_streaming.run()
    da3_streaming.close()

    del da3_streaming
    torch.cuda.empty_cache()
    gc.collect()

    verification_only = config.get("Loop", {}).get(
        "GeometryVerification", {}
    ).get("verification_only", False)
    if not verification_only:
        all_ply_path = os.path.join(save_dir, "pcd/combined_pcd.ply")
        input_dir = os.path.join(save_dir, "pcd")
        print("Saving all the point clouds")
        merge_ply_files(input_dir, all_ply_path)
    if not verification_only and config["Model"].get("Yaw4_Pose_Filter", {}).get(
        "rig_paired_baseline_export", False
    ):
        baseline_input_dir = os.path.join(save_dir, "pcd_baseline")
        baseline_ply_path = os.path.join(
            baseline_input_dir, "combined_pcd.ply"
        )
        print("Saving the paired unmodified baseline point clouds")
        merge_ply_files(baseline_input_dir, baseline_ply_path)
    print("DA3-Streaming done.")
    sys.exit()
