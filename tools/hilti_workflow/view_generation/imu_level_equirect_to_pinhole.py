#!/usr/bin/env python3
"""
Render level front-view pinhole images from equirectangular panoramas using IMU gravity.

This script assumes:
- input images are already stitched equirectangular panoramas
- panorama coordinates are expressed in cam0 coordinates
- IMU topic is sensor_msgs/msg/Imu in a ROS2 SQLite bag

The output view keeps the camera's forward direction, but removes roll/pitch
using the gravity vector estimated from low-pass filtered accelerometer data.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
import struct
from pathlib import Path

import cv2
import numpy as np
import yaml


FRAME_TS_RE = re.compile(r".*_([0-9]+)\.[^.]+$")


def _read_yaml(path: str) -> str:
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin1")
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("%YAML"):
        lines = lines[1:]
    return "\n".join(lines)


def load_cam0_from_yaml(path: str) -> tuple[np.ndarray, float]:
    data = yaml.safe_load(_read_yaml(path))
    cam0 = data["cam0"]
    t_cam_imu = np.array(cam0["T_cam_imu"], dtype=np.float64)
    timeshift = float(cam0.get("timeshift_cam_imu", 0.0))
    return t_cam_imu[:3, :3], timeshift


def parse_imu_packet(data: bytes) -> tuple[int, np.ndarray, np.ndarray]:
    if isinstance(data, memoryview):
        data = data.tobytes()

    sec, nsec = struct.unpack_from("<iI", data, 4)
    strlen = struct.unpack_from("<I", data, 12)[0]
    pos = 16 + strlen
    pos = (pos + 3) & ~3

    _orientation = struct.unpack_from("<4d", data, pos)
    pos += 32
    _orientation_cov = struct.unpack_from("<9d", data, pos)
    pos += 72
    angular_velocity = np.array(struct.unpack_from("<3d", data, pos), dtype=np.float64)
    pos += 24
    _angular_cov = struct.unpack_from("<9d", data, pos)
    pos += 72
    linear_acceleration = np.array(struct.unpack_from("<3d", data, pos), dtype=np.float64)

    timestamp_ns = int(sec) * 1_000_000_000 + int(nsec)
    return timestamp_ns, angular_velocity, linear_acceleration


def load_imu_series(
    bag_path: str,
    tau_s: float,
    *,
    method: str = "causal_accel",
    accel_gate_sigma: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    conn = sqlite3.connect(bag_path)
    topic = conn.execute(
        "SELECT id FROM topics WHERE name='/imu/data_raw' AND type='sensor_msgs/msg/Imu'"
    ).fetchone()
    if topic is None:
        raise RuntimeError("No /imu/data_raw sensor_msgs/msg/Imu topic found in the bag")

    rows = conn.execute(
        "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (topic[0],)
    )

    ts_list: list[int] = []
    g_lp_list: list[np.ndarray] = []
    g_lp: np.ndarray | None = None
    gravity_norm: float | None = None
    last_ts: int | None = None

    for (blob,) in rows:
        ts_ns, gyro, accel = parse_imu_packet(blob)

        if g_lp is None or last_ts is None:
            g_lp = accel.astype(np.float64)
            gravity_norm = float(np.linalg.norm(accel))
        else:
            dt = max((ts_ns - last_ts) * 1e-9, 1e-6)
            alpha = 1.0 - math.exp(-dt / tau_s)
            if method == "causal_accel":
                g_lp = (1.0 - alpha) * g_lp + alpha * accel
            elif method == "complementary":
                assert gravity_norm is not None
                rotation, _ = cv2.Rodrigues((-gyro * min(dt, 0.1)).astype(np.float64))
                predicted = np.matmul(rotation, normalize(g_lp))
                accel_norm = float(np.linalg.norm(accel))
                norm_error = (accel_norm - gravity_norm) / max(
                    accel_gate_sigma * gravity_norm, 1e-9
                )
                reliability = math.exp(-0.5 * norm_error * norm_error)
                corrected = normalize(
                    (1.0 - alpha * reliability) * predicted
                    + alpha * reliability * normalize(accel)
                )
                g_lp = corrected * gravity_norm
            else:
                raise ValueError(f"Unknown IMU gravity method: {method}")

        ts_list.append(ts_ns)
        g_lp_list.append(g_lp.copy())
        last_ts = ts_ns

    if not ts_list:
        raise RuntimeError("No IMU messages found in /imu/data_raw")

    return np.asarray(ts_list, dtype=np.int64), np.asarray(g_lp_list, dtype=np.float64)


def extract_frame_timestamp(path: Path) -> int:
    match = FRAME_TS_RE.match(path.name)
    if not match:
        raise ValueError(f"Could not parse timestamp from filename: {path.name}")
    return int(match.group(1))


def build_remap_from_rotation(
    out_w: int,
    out_h: int,
    fov_deg: float,
    rot: np.ndarray,
    in_w: int,
    in_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    fov = math.radians(fov_deg)
    fx = out_w / (2.0 * math.tan(fov / 2.0))
    fy = fx
    cx = (out_w - 1) / 2.0
    cy = (out_h - 1) / 2.0

    xs, ys = np.meshgrid(
        np.arange(out_w, dtype=np.float32),
        np.arange(out_h, dtype=np.float32),
    )
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    z = np.ones_like(x, dtype=np.float32)

    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    dirs_world = dirs @ rot.T

    lon = np.arctan2(dirs_world[..., 0], dirs_world[..., 2])
    lat = np.arcsin(np.clip(dirs_world[..., 1], -1.0, 1.0))

    map_x = ((lon + math.pi) / (2.0 * math.pi) * in_w).astype(np.float32)
    map_y = ((lat + math.pi / 2.0) / math.pi * in_h).astype(np.float32)
    map_x = np.mod(map_x, in_w).astype(np.float32)
    map_y = np.clip(map_y, 0, in_h - 1).astype(np.float32)
    return map_x, map_y


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy()
    return v / n


def level_rotation_from_gravity(g_cam: np.ndarray) -> np.ndarray:
    down = normalize(g_cam)
    if np.linalg.norm(down) < 1e-6:
        return np.eye(3, dtype=np.float32)

    nominal_forward = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    forward = nominal_forward - np.dot(nominal_forward, down) * down
    if np.linalg.norm(forward) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        forward = forward - np.dot(forward, down) * down
    forward = normalize(forward)
    right = normalize(np.cross(down, forward))

    rot = np.stack([right, down, forward], axis=1)
    return rot.astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert equirectangular panoramas to horizon-leveled front pinhole images using IMU"
    )
    parser.add_argument("--bag", required=True, help="Path to rosbag.db3")
    parser.add_argument("--input_dir", required=True, help="Directory containing equirectangular frames")
    parser.add_argument("--yaml", required=True, help="Path to kalibr_imucam_chain.yaml")
    parser.add_argument("--output_dir", required=True, help="Directory to save pinhole frames")
    parser.add_argument("--width", type=int, default=768, help="Output width")
    parser.add_argument("--height", type=int, default=512, help="Output height")
    parser.add_argument("--fov_deg", type=float, default=90.0, help="Horizontal FOV in degrees")
    parser.add_argument("--imu_tau", type=float, default=0.25, help="Low-pass time constant for accelerometer smoothing")
    parser.add_argument("--time_offset_ns", type=int, default=0, help="Optional additional frame-to-IMU time offset in nanoseconds")
    parser.add_argument("--use_yaml_timeshift", action="store_true", help="Apply cam0 timeshift_cam_imu from the calibration file")
    parser.add_argument("--rotate180", action="store_true", help="Rotate the generated pinhole image by 180 degrees after projection")
    parser.add_argument("--extensions", default="jpg,jpeg,png,webp", help="Comma-separated input image extensions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    imu_ts, imu_g_lp = load_imu_series(args.bag, tau_s=args.imu_tau)

    total_offset_ns = int(args.time_offset_ns)
    if args.use_yaml_timeshift:
        total_offset_ns += int(round(yaml_timeshift_s * 1e9))

    print(f"Input resolution : {in_w}x{in_h}")
    print(f"Output resolution: {args.width}x{args.height}")
    print(f"Frames           : {len(image_paths)}")
    print(f"IMU samples      : {len(imu_ts)}")
    print(f"FOV              : {args.fov_deg}")
    print(f"LP tau (s)       : {args.imu_tau}")
    print(f"Total offset (ns): {total_offset_ns}")

    prev_rot = np.eye(3, dtype=np.float32)

    for idx, image_path in enumerate(image_paths, 1):
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
        rot = level_rotation_from_gravity(g_cam0)
        if not np.isfinite(rot).all():
            rot = prev_rot
        prev_rot = rot

        map_x, map_y = build_remap_from_rotation(
            out_w=args.width,
            out_h=args.height,
            fov_deg=args.fov_deg,
            rot=rot,
            in_w=in_w,
            in_h=in_h,
        )

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        pinhole = cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        if args.rotate180:
            pinhole = cv2.rotate(pinhole, cv2.ROTATE_180)

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), pinhole, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"Saved {idx}/{len(image_paths)} frames -> {output_dir}")


if __name__ == "__main__":
    main()
