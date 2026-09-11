#!/usr/bin/env python3
"""Render a custom ordered sequence of pinhole views from one equirectangular image.

Pose format:
    yaw,pitch,roll;yaw,pitch,roll;...
Example:
    --poses '0,15,0;60,15,0;120,0,0;180,-15,0;240,-15,0;300,0,0'
"""

import argparse
from pathlib import Path

import cv2

from equirect_to_pinhole import build_remap


def parse_args():
    p = argparse.ArgumentParser(description='Render an ordered pose sequence from one equirect image')
    p.add_argument('--input_image', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--width', type=int, default=768)
    p.add_argument('--height', type=int, default=512)
    p.add_argument('--fov_deg', type=float, default=100.0)
    p.add_argument('--poses', required=True, help='Semicolon-separated yaw,pitch,roll triplets')
    p.add_argument('--rotate180', action='store_true')
    return p.parse_args()


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


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(args.input_image, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Failed to read image: {args.input_image}')
    in_h, in_w = image.shape[:2]
    stem = Path(args.input_image).stem
    poses = parse_pose_list(args.poses)

    for idx, (yaw, pitch, roll) in enumerate(poses):
        map_x, map_y = build_remap(
            out_w=args.width,
            out_h=args.height,
            fov_deg=args.fov_deg,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
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
        out_path = out / f'{idx:02d}_{stem}_yaw{int(round(yaw)):03d}_pitch{int(round(pitch)):03d}_roll{int(round(roll)):03d}.jpg'
        cv2.imwrite(str(out_path), pinhole, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(out_path)


if __name__ == '__main__':
    main()
