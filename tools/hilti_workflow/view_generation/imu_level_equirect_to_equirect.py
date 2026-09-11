#!/usr/bin/env python3
"""Apply the HILTI IMU gravity-leveling rotation to ERP panoramas.

This is the spherical counterpart of ``imu_level_equirect_pose_sequence.py``.
It uses the same camera/IMU calibration, complementary gravity estimate, frame
timestamp association, and rotation convention, but samples a complete 2:1
equirectangular output instead of perspective crops.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from .imu_level_equirect_to_pinhole import (
        extract_frame_timestamp,
        level_rotation_from_gravity,
        load_cam0_from_yaml,
        load_imu_series,
    )
except ImportError:  # Direct script execution.
    from imu_level_equirect_to_pinhole import (
        extract_frame_timestamp,
        level_rotation_from_gravity,
        load_cam0_from_yaml,
        load_imu_series,
    )


def build_erp_remap(out_w: int, out_h: int, rot: np.ndarray, in_w: int, in_h: int):
    xs, ys = np.meshgrid(
        np.arange(out_w, dtype=np.float32),
        np.arange(out_h, dtype=np.float32),
    )
    lon = (xs + 0.5) / out_w * (2.0 * math.pi) - math.pi
    lat = (ys + 0.5) / out_h * math.pi - math.pi / 2.0
    cos_lat = np.cos(lat)
    dirs = np.stack(
        [np.sin(lon) * cos_lat, np.sin(lat), np.cos(lon) * cos_lat], axis=-1
    )
    dirs_input = dirs @ rot.T
    src_lon = np.arctan2(dirs_input[..., 0], dirs_input[..., 2])
    src_lat = np.arcsin(np.clip(dirs_input[..., 1], -1.0, 1.0))
    map_x = np.mod((src_lon + math.pi) / (2.0 * math.pi) * in_w - 0.5, in_w)
    map_y = np.clip((src_lat + math.pi / 2.0) / math.pi * in_h - 0.5, 0, in_h - 1)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def nearest_index(sorted_values: np.ndarray, value: int) -> int:
    idx = int(np.searchsorted(sorted_values, value))
    if idx >= len(sorted_values):
        return len(sorted_values) - 1
    if idx > 0 and abs(int(sorted_values[idx - 1]) - value) <= abs(int(sorted_values[idx]) - value):
        return idx - 1
    return idx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=1036)
    parser.add_argument("--height", type=int, default=518)
    parser.add_argument("--imu-tau", type=float, default=2.0)
    parser.add_argument("--imu-method", choices=("causal_accel", "complementary"), default="complementary")
    parser.add_argument("--accel-gate-sigma", type=float, default=0.2)
    parser.add_argument("--time-offset-ns", type=int, default=0)
    parser.add_argument("--use-yaml-timeshift", action="store_true")
    parser.add_argument("--rotate180", action="store_true")
    parser.add_argument("--quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0 or args.width != 2 * args.height:
        raise ValueError("ERP output must have positive 2:1 dimensions")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not images:
        raise FileNotFoundError(f"No panorama images under {input_dir}")

    sample = cv2.imread(str(images[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError(f"Could not decode {images[0]}")
    in_h, in_w = sample.shape[:2]
    r_cam0_imu, yaml_timeshift_s = load_cam0_from_yaml(args.yaml)
    imu_ts, imu_g = load_imu_series(
        args.bag,
        tau_s=args.imu_tau,
        method=args.imu_method,
        accel_gate_sigma=args.accel_gate_sigma,
    )
    offset = int(args.time_offset_ns)
    if args.use_yaml_timeshift:
        offset += int(round(yaml_timeshift_s * 1e9))

    previous = np.eye(3, dtype=np.float32)
    started = time.time()
    for frame_index, path in enumerate(images):
        timestamp = extract_frame_timestamp(path) + offset
        imu_index = nearest_index(imu_ts, timestamp)
        rotation = level_rotation_from_gravity(r_cam0_imu @ imu_g[imu_index])
        if not np.isfinite(rotation).all():
            rotation = previous
        previous = rotation
        map_x, map_y = build_erp_remap(args.width, args.height, rotation, in_w, in_h)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode {path}")
        output = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        if args.rotate180:
            output = cv2.rotate(output, cv2.ROTATE_180)
        destination = output_dir / path.name
        if not cv2.imwrite(str(destination), output, [cv2.IMWRITE_JPEG_QUALITY, args.quality]):
            raise RuntimeError(f"Could not write {destination}")
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == len(images):
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[level-erp] {frame_index + 1}/{len(images)} "
                f"({(frame_index + 1) / elapsed:.2f} frames/s) -> {output_dir}",
                flush=True,
            )


if __name__ == "__main__":
    main()
