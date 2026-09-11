#!/usr/bin/env python3
"""
Render multiple leveled yaw views from each equirectangular panorama using IMU gravity.

The gravity vector is used to level roll/pitch for every frame. Around that leveled
front-view basis, the script renders a list of yaw offsets (for example 0/90/180/270)
and writes them out in frame-major order so they can be consumed as a continuous image
sequence by DA3-Streaming.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from imu_level_equirect_to_pinhole import (
    build_remap_from_rotation,
    extract_frame_timestamp,
    level_rotation_from_gravity,
    load_cam0_from_yaml,
    load_imu_series,
)


def parse_yaws(text: str) -> list[float]:
    values = [chunk.strip() for chunk in text.split(",")]
    yaws = [float(v) for v in values if v]
    if not yaws:
        raise ValueError("No yaw offsets parsed")
    return yaws


def yaw_rotation_matrix(yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float32,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render leveled multi-yaw pinhole images from equirect panoramas using IMU"
    )
    parser.add_argument("--bag", required=True, help="Path to rosbag.db3")
    parser.add_argument("--input_dir", required=True, help="Directory containing equirectangular frames")
    parser.add_argument("--yaml", required=True, help="Path to kalibr_imucam_chain.yaml")
    parser.add_argument("--output_dir", required=True, help="Directory to save the pinhole sequence")
    parser.add_argument("--width", type=int, default=768, help="Output width")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--fov_deg", type=float, default=90.0, help="Horizontal FOV in degrees")
    parser.add_argument("--yaws", default="0,90,180,270", help="Comma-separated yaw offsets in degrees")
    parser.add_argument("--imu_tau", type=float, default=0.25, help="Low-pass time constant for accelerometer smoothing")
    parser.add_argument("--imu-method", choices=("causal_accel", "complementary"), default="causal_accel")
    parser.add_argument("--accel-gate-sigma", type=float, default=0.2)
    parser.add_argument("--time_offset_ns", type=int, default=0, help="Optional additional frame-to-IMU time offset in nanoseconds")
    parser.add_argument("--use_yaml_timeshift", action="store_true", help="Apply cam0 timeshift_cam_imu from the calibration file")
    parser.add_argument("--rotate180", action="store_true", help="Rotate the generated pinhole image by 180 degrees after projection")
    parser.add_argument("--progress_interval_s", type=float, default=5.0, help="Print progress at least this often in seconds; 0 disables time-based progress")
    parser.add_argument("--extensions", default="jpg,jpeg,png,webp", help="Comma-separated input image extensions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    yaws = parse_yaws(args.yaws)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extensions = {f".{ext.strip().lower()}" for ext in args.extensions.split(",") if ext.strip()}
    image_paths = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in extensions)
    if not image_paths:
        raise FileNotFoundError(f"No input images found under {input_dir}")

    sample = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError(f"Failed to read sample image: {image_paths[0]}")
    in_h, in_w = sample.shape[:2]

    r_cam0_imu, yaml_timeshift_s = load_cam0_from_yaml(args.yaml)
    imu_ts, imu_g_lp = load_imu_series(
        args.bag,
        tau_s=args.imu_tau,
        method=args.imu_method,
        accel_gate_sigma=args.accel_gate_sigma,
    )

    total_offset_ns = int(args.time_offset_ns)
    if args.use_yaml_timeshift:
        total_offset_ns += int(round(yaml_timeshift_s * 1e9))

    print(f"Input resolution : {in_w}x{in_h}")
    print(f"Output resolution: {args.width}x{args.height}")
    print(f"Frames           : {len(image_paths)}")
    print(f"IMU samples      : {len(imu_ts)}")
    print(f"FOV              : {args.fov_deg}")
    print(f"Yaw sequence     : {yaws}")
    print(f"LP tau (s)       : {args.imu_tau}")
    print(f"IMU method       : {args.imu_method}")
    print(f"Accel gate sigma : {args.accel_gate_sigma}")
    print(f"Total offset (ns): {total_offset_ns}")
    print(f"Rotate 180       : {args.rotate180}", flush=True)

    prev_rot = np.eye(3, dtype=np.float32)
    started_at = time.time()
    last_progress_at = started_at
    total_views = len(image_paths) * len(yaws)
    written_views = 0

    for frame_idx, image_path in enumerate(image_paths):
        frame_ts = extract_frame_timestamp(image_path) + total_offset_ns
        imu_idx = int(np.searchsorted(imu_ts, frame_ts))
        if imu_idx >= len(imu_ts):
            imu_idx = len(imu_ts) - 1
        elif imu_idx > 0:
            left_dt = abs(int(imu_ts[imu_idx - 1]) - frame_ts)
            right_dt = abs(int(imu_ts[imu_idx]) - frame_ts)
            if left_dt <= right_dt:
                imu_idx -= 1

        g_imu = imu_g_lp[imu_idx]
        g_cam0 = r_cam0_imu @ g_imu
        base_rot = level_rotation_from_gravity(g_cam0)
        if not np.isfinite(base_rot).all():
            base_rot = prev_rot
        prev_rot = base_rot

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        stem = image_path.stem
        for view_idx, yaw_deg in enumerate(yaws):
            rot = base_rot @ yaw_rotation_matrix(yaw_deg)
            map_x, map_y = build_remap_from_rotation(
                out_w=args.width,
                out_h=args.height,
                fov_deg=args.fov_deg,
                rot=rot,
                in_w=in_w,
                in_h=in_h,
            )

            pinhole = cv2.remap(
                image,
                map_x,
                map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            if args.rotate180:
                pinhole = cv2.rotate(pinhole, cv2.ROTATE_180)

            out_path = output_dir / (
                f"{frame_idx:05d}_v{view_idx:02d}_{stem}_yaw{int(round(yaw_deg))%360:03d}.jpg"
            )
            cv2.imwrite(str(out_path), pinhole, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written_views += 1

        now = time.time()
        should_print_time = args.progress_interval_s > 0 and (now - last_progress_at) >= args.progress_interval_s
        if should_print_time or (frame_idx + 1) % 25 == 0 or frame_idx + 1 == len(image_paths):
            elapsed = max(now - started_at, 1e-6)
            frame_rate = (frame_idx + 1) / elapsed
            view_rate = written_views / elapsed
            pct = 100.0 * (frame_idx + 1) / len(image_paths)
            print(
                f"[pinhole] frames={frame_idx + 1}/{len(image_paths)} "
                f"views={written_views}/{total_views} ({pct:.1f}%) "
                f"frame_rate={frame_rate:.2f}/s view_rate={view_rate:.2f}/s "
                f"elapsed={elapsed:.1f}s -> {output_dir}",
                flush=True,
            )
            last_progress_at = now


if __name__ == "__main__":
    main()
