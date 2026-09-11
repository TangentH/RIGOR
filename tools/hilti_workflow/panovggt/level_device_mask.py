#!/usr/bin/env python3
"""Warp the frozen raw-ERP device mask into each IMU-levelled ERP frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hilti_workflow.view_generation.imu_level_equirect_to_equirect import (
    build_erp_remap,
    nearest_index,
)
from tools.hilti_workflow.view_generation.imu_level_equirect_to_pinhole import (
    extract_frame_timestamp,
    level_rotation_from_gravity,
    load_cam0_from_yaml,
    load_imu_series,
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--raw-erp-dir", type=Path, required=True)
    parser.add_argument("--levelled-erp-dir", type=Path, required=True)
    parser.add_argument("--yaml", type=Path, required=True)
    parser.add_argument("--static-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--imu-tau", type=float, default=2.0)
    parser.add_argument("--accel-gate-sigma", type=float, default=0.2)
    parser.add_argument("--dilate-pixels", type=int, default=3)
    args = parser.parse_args()

    raw_images = sorted(args.raw_erp_dir.glob("*.jpg"))
    levelled = sorted(args.levelled_erp_dir.glob("*.jpg"))
    if not raw_images or not levelled:
        raise ValueError("Raw and levelled ERP directories must contain JPG frames")
    if [p.name for p in raw_images] != [p.name for p in levelled]:
        raise ValueError("Raw and levelled ERP filenames do not match exactly")
    sample = cv2.imread(str(raw_images[0]), cv2.IMREAD_COLOR)
    target = cv2.imread(str(levelled[0]), cv2.IMREAD_COLOR)
    static = cv2.imread(str(args.static_mask), cv2.IMREAD_GRAYSCALE)
    if sample is None or target is None or static is None:
        raise RuntimeError("Could not decode input images or static mask")
    in_h, in_w = sample.shape[:2]
    out_h, out_w = target.shape[:2]
    if static.shape != (in_h, in_w):
        raise ValueError(f"Static mask shape {static.shape} != raw ERP {(in_h, in_w)}")
    static = static >= 128

    r_cam0_imu, _ = load_cam0_from_yaml(str(args.yaml))
    imu_ts, imu_g = load_imu_series(
        str(args.bag),
        tau_s=args.imu_tau,
        method="complementary",
        accel_gate_sigma=args.accel_gate_sigma,
    )
    kernel = None
    if args.dilate_pixels > 0:
        size = 2 * args.dilate_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    masks_dir = args.output_dir / "masks_npy"
    overlays_dir = args.output_dir / "overlays_selected"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    selected = {0, 50, 100, 150, 200, 250, 300, 350, len(raw_images) - 1}
    previous = np.eye(3, dtype=np.float32)
    rows = []
    for index, (raw, level_path) in enumerate(zip(raw_images, levelled)):
        imu_index = nearest_index(imu_ts, extract_frame_timestamp(raw))
        rotation = level_rotation_from_gravity(r_cam0_imu @ imu_g[imu_index])
        if not np.isfinite(rotation).all():
            rotation = previous
        previous = rotation
        map_x, map_y = build_erp_remap(out_w, out_h, rotation, in_w, in_h)
        projected = cv2.remap(
            static.astype(np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_WRAP,
        )
        # The frozen Floor1 levelled ERP protocol uses rotate180=True.
        projected = cv2.rotate(projected, cv2.ROTATE_180).astype(bool)
        if kernel is not None:
            projected = cv2.dilate(projected.astype(np.uint8), kernel, iterations=1).astype(bool)
        mask_path = masks_dir / f"{raw.stem}.npy"
        np.save(mask_path, projected)
        rows.append({"stem": raw.stem, "invalid_fraction": float(projected.mean())})
        if index in selected:
            image = cv2.imread(str(level_path), cv2.IMREAD_COLOR)
            overlay = image.copy()
            overlay[projected] = (
                0.55 * overlay[projected] + 0.45 * np.array([0, 0, 255])
            ).astype(np.uint8)
            cv2.imwrite(str(overlays_dir / f"{raw.stem}.jpg"), overlay)

    payload = {
        "operation": "exact static raw-ERP mask warped by the frozen IMU-levelled ERP rotation",
        "bag": str(args.bag),
        "bag_sha256": digest(args.bag),
        "static_mask": str(args.static_mask),
        "static_mask_sha256": digest(args.static_mask),
        "calibration": str(args.yaml),
        "calibration_sha256": digest(args.yaml),
        "raw_erp_dir": str(args.raw_erp_dir),
        "levelled_erp_dir": str(args.levelled_erp_dir),
        "output_dir": str(args.output_dir),
        "frames": len(rows),
        "output_shape": [out_h, out_w],
        "imu_method": "complementary",
        "imu_tau": args.imu_tau,
        "accel_gate_sigma": args.accel_gate_sigma,
        "rotate180": True,
        "dilate_pixels": args.dilate_pixels,
        "invalid_fraction_mean": float(np.mean([row["invalid_fraction"] for row in rows])),
        "per_frame": rows,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("frames", "output_shape", "invalid_fraction_mean")}, indent=2))


if __name__ == "__main__":
    main()
