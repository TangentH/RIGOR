#!/usr/bin/env python3
"""
Convert equirectangular panorama frames to fixed-view pinhole images.

Example:
    python equirect_to_pinhole.py \
        --input_dir /path/to/hilti_data/raw \
        --output_dir /path/to/hilti_data/derived/pinhole/pinhole_front_fov90 \
        --width 768 \
        --height 512 \
        --fov_deg 90 \
        --yaw_deg 0 \
        --pitch_deg 0 \
        --roll_deg 0
"""

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
    r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
    return r_yaw @ r_pitch @ r_roll


def build_remap(
    out_w: int,
    out_h: int,
    fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    in_w: int,
    in_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    fov = math.radians(fov_deg)
    fx = out_w / (2.0 * math.tan(fov / 2.0))
    fy = fx
    cx = (out_w - 1) / 2.0
    cy = (out_h - 1) / 2.0

    xs, ys = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    x = (xs - cx) / fx
    y = (ys - cy) / fy
    z = np.ones_like(x, dtype=np.float32)

    dirs = np.stack([x, y, z], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    rot = rotation_matrix(yaw_deg, pitch_deg, roll_deg)
    dirs_world = dirs @ rot.T

    lon = np.arctan2(dirs_world[..., 0], dirs_world[..., 2])
    lat = np.arcsin(np.clip(dirs_world[..., 1], -1.0, 1.0))

    map_x = ((lon + math.pi) / (2.0 * math.pi) * in_w).astype(np.float32)
    map_y = ((lat + math.pi / 2.0) / math.pi * in_h).astype(np.float32)

    map_x = np.mod(map_x, in_w).astype(np.float32)
    map_y = np.clip(map_y, 0, in_h - 1).astype(np.float32)
    return map_x, map_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert equirectangular images to pinhole views")
    parser.add_argument("--input_dir", required=True, help="Directory containing equirectangular frames")
    parser.add_argument("--output_dir", required=True, help="Directory to save pinhole frames")
    parser.add_argument("--width", type=int, default=768, help="Output pinhole width")
    parser.add_argument("--height", type=int, default=512, help="Output pinhole height")
    parser.add_argument("--fov_deg", type=float, default=90.0, help="Horizontal field of view in degrees")
    parser.add_argument("--yaw_deg", type=float, default=0.0, help="Yaw angle of the virtual camera")
    parser.add_argument("--pitch_deg", type=float, default=0.0, help="Pitch angle of the virtual camera")
    parser.add_argument("--roll_deg", type=float, default=0.0, help="Roll angle of the virtual camera")
    parser.add_argument(
        "--rotate180",
        action="store_true",
        help="Rotate the generated pinhole image by 180 degrees after projection",
    )
    parser.add_argument(
        "--extensions",
        default="jpg,jpeg,png,webp",
        help="Comma-separated input image extensions",
    )
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

    map_x, map_y = build_remap(
        out_w=args.width,
        out_h=args.height,
        fov_deg=args.fov_deg,
        yaw_deg=args.yaw_deg,
        pitch_deg=args.pitch_deg,
        roll_deg=args.roll_deg,
        in_w=in_w,
        in_h=in_h,
    )

    print(f"Input resolution : {in_w}x{in_h}")
    print(f"Output resolution: {args.width}x{args.height}")
    print(
        f"View parameters  : fov={args.fov_deg} yaw={args.yaw_deg} "
        f"pitch={args.pitch_deg} roll={args.roll_deg}"
    )
    print(f"Frames           : {len(image_paths)}")

    for idx, image_path in enumerate(image_paths, 1):
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
