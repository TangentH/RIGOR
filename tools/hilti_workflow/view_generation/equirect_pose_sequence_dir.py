#!/usr/bin/env python3
"""Render an ordered pose sequence for every image in a directory."""

import argparse
from pathlib import Path
import cv2
from equirect_to_pinhole import build_remap


def parse_pose_list(text: str):
    poses = []
    for idx, chunk in enumerate(text.split(';')):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [x.strip() for x in chunk.split(',')]
        if len(parts) != 3:
            raise ValueError(f'Pose {idx} must have yaw,pitch,roll: {chunk}')
        poses.append(tuple(float(x) for x in parts))
    if not poses:
        raise ValueError('No poses parsed')
    return poses


def parse_args():
    p = argparse.ArgumentParser(description='Render an ordered pose sequence for every equirect image in a directory')
    p.add_argument('--input_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--fov_deg', type=float, default=100.0)
    p.add_argument('--poses', required=True)
    p.add_argument('--rotate180', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    poses = parse_pose_list(args.poses)
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])
    if not image_paths:
        raise RuntimeError(f'No images found in {in_dir}')

    sample = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError(f'Failed to read sample image: {image_paths[0]}')
    in_h, in_w = sample.shape[:2]

    remaps = []
    for yaw, pitch, roll in poses:
        remaps.append((yaw, pitch, roll, *build_remap(args.width, args.height, args.fov_deg, yaw, pitch, roll, in_w, in_h)))

    for frame_idx, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f'Skipping unreadable image: {image_path}')
            continue
        stem = image_path.stem
        for view_idx, (yaw, pitch, roll, map_x, map_y) in enumerate(remaps):
            pinhole = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            if args.rotate180:
                pinhole = cv2.rotate(pinhole, cv2.ROTATE_180)
            out_path = out_dir / f'{frame_idx:05d}_v{view_idx:02d}_{stem}_yaw{int(round(yaw)):03d}_pitch{int(round(pitch)):03d}_roll{int(round(roll)):03d}.jpg'
            cv2.imwrite(str(out_path), pinhole, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if (frame_idx + 1) % 25 == 0 or frame_idx + 1 == len(image_paths):
            print(f'Rendered {frame_idx + 1}/{len(image_paths)} panorama frames -> {out_dir}')


if __name__ == '__main__':
    main()
