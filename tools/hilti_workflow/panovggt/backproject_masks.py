#!/usr/bin/env python3
"""Backproject arbitrary frozen perspective masks to matching ERP frames."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


VIEW_RE = re.compile(
    r"(?:^|_)(?P<stem>frame_\d+_\d+)_yaw(?P<yaw>-?\d{2,3})"
    r"(?:_pitch(?P<pitch>-?\d{2,3})_roll(?P<roll>-?\d{2,3}))?$"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotation_matrix(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw, pitch, roll = map(math.radians, (yaw_deg, pitch_deg, roll_deg))
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    r_yaw = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
    r_pitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], np.float32)
    r_roll = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], np.float32)
    return r_yaw @ r_pitch @ r_roll


def build_remap(out_w: int, out_h: int, fov_deg: float, yaw: float, pitch: float,
                roll: float, in_w: int, in_h: int):
    focal = out_w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx, cy = (out_w - 1) / 2.0, (out_h - 1) / 2.0
    xs, ys = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    directions = np.stack([(xs - cx) / focal, (ys - cy) / focal, np.ones_like(xs)], axis=-1)
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    world = directions @ rotation_matrix(yaw, pitch, roll).T
    longitude = np.arctan2(world[..., 0], world[..., 2])
    latitude = np.arcsin(np.clip(world[..., 1], -1.0, 1.0))
    map_x = np.mod((longitude + math.pi) / (2 * math.pi) * in_w, in_w).astype(np.float32)
    map_y = np.clip((latitude + math.pi / 2) / math.pi * in_h, 0, in_h - 1).astype(np.float32)
    return map_x, map_y


def build_inverse_remap(
    out_w: int,
    out_h: int,
    fov_deg: float,
    yaw: float,
    pitch: float,
    roll: float,
    in_w: int,
    in_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map every ERP output pixel back into one perspective input view.

    Pixels outside the continuous pinhole footprint map to ``-1`` so OpenCV's
    constant border produces an empty mask.  Unlike forward nearest scatter,
    this gather mapping cannot leave sampling holes inside a covered footprint.
    """
    focal = in_w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx, cy = (in_w - 1) / 2.0, (in_h - 1) / 2.0
    erp_x, erp_y = np.meshgrid(
        np.arange(out_w, dtype=np.float32),
        np.arange(out_h, dtype=np.float32),
    )
    longitude = erp_x / out_w * (2.0 * math.pi) - math.pi
    latitude = erp_y / out_h * math.pi - math.pi / 2.0
    cos_latitude = np.cos(latitude)
    world = np.stack(
        [
            cos_latitude * np.sin(longitude),
            np.sin(latitude),
            cos_latitude * np.cos(longitude),
        ],
        axis=-1,
    )
    local = world @ rotation_matrix(yaw, pitch, roll)
    visible = local[..., 2] > 0.0
    map_x = np.full((out_h, out_w), -1.0, dtype=np.float32)
    map_y = np.full((out_h, out_w), -1.0, dtype=np.float32)
    map_x[visible] = focal * (
        local[..., 0][visible] / local[..., 2][visible]
    ) + cx
    map_y[visible] = focal * (
        local[..., 1][visible] / local[..., 2][visible]
    ) + cy
    inside = (
        visible
        & (map_x >= 0.0)
        & (map_x <= in_w - 1)
        & (map_y >= 0.0)
        & (map_y <= in_h - 1)
    )
    map_x[~inside] = -1.0
    map_y[~inside] = -1.0
    return map_x, map_y


def project_view_mask_to_erp(
    mask: np.ndarray,
    *,
    yaw: float,
    pitch: float,
    roll: float,
    fov_deg: float,
    erp_width: int,
    erp_height: int,
    mapping: str,
    cached_maps: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Project one bool perspective mask with an explicit mapping policy."""
    mask = np.asarray(mask)
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("view mask must be a two-dimensional bool array")
    if mapping == "nearest-scatter":
        map_x, map_y = (
            cached_maps
            if cached_maps is not None
            else build_remap(
                mask.shape[1], mask.shape[0], fov_deg, yaw, pitch, roll,
                erp_width, erp_height,
            )
        )
        destination_x = np.rint(map_x).astype(np.int32) % erp_width
        destination_y = np.clip(
            np.rint(map_y).astype(np.int32), 0, erp_height - 1
        )
        output = np.zeros((erp_height, erp_width), dtype=np.uint8)
        np.maximum.at(output, (destination_y[mask], destination_x[mask]), 1)
        return output.astype(bool)
    if mapping == "inverse-gather":
        map_x, map_y = (
            cached_maps
            if cached_maps is not None
            else build_inverse_remap(
                erp_width, erp_height, fov_deg, yaw, pitch, roll,
                mask.shape[1], mask.shape[0],
            )
        )
        return cv2.remap(
            mask.astype(np.uint8),
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)
    raise ValueError(f"Unsupported mapping policy: {mapping}")


def parse_view(path: Path):
    match = VIEW_RE.search(path.stem)
    if not match:
        return None
    return (
        match.group("stem"),
        int(match.group("yaw")),
        int(match.group("pitch") or 0),
        int(match.group("roll") or 0),
    )


def load_packed(path: Path):
    bundle = np.load(path, allow_pickle=False)
    shape = tuple(int(item) for item in bundle["shape"])
    masks = np.unpackbits(bundle["masks"], axis=1, count=int(np.prod(shape[1:])))
    masks = masks.reshape(shape).astype(bool)
    return masks, [str(item) for item in bundle["stems"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erp-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fov", type=float, default=95.0)
    parser.add_argument("--view-width", type=int, default=768)
    parser.add_argument("--view-height", type=int, default=768)
    parser.add_argument("--rotate180", action="store_true")
    parser.add_argument("--min-view-votes", type=int, default=1)
    parser.add_argument("--max-view-mask-fraction", type=float, default=1.0)
    parser.add_argument(
        "--mapping",
        choices=("nearest-scatter", "inverse-gather"),
        default="nearest-scatter",
        help="Projection policy; the default preserves the production protocol.",
    )
    parser.add_argument("--union-packed", type=Path, action="append", default=[])
    args = parser.parse_args()

    erp_paths = sorted(path for path in args.erp_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not erp_paths:
        raise ValueError("No ERP images")
    sample = cv2.imread(str(erp_paths[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise ValueError(f"Cannot decode {erp_paths[0]}")
    in_h, in_w = sample.shape[:2]

    masks_by_stem = defaultdict(list)
    source_counts = {}
    view_keys = set()
    for directory in args.mask_dir:
        count = 0
        for path in sorted(directory.glob("*.npy")):
            parsed = parse_view(path)
            if parsed is None:
                continue
            stem, yaw, pitch, roll = parsed
            masks_by_stem[stem].append((path, yaw, pitch, roll))
            view_keys.add((yaw, pitch, roll))
            count += 1
        source_counts[str(directory)] = count
    missing = [path.stem for path in erp_paths if path.stem not in masks_by_stem]
    if missing:
        raise ValueError(f"Missing all perspective masks for {len(missing)} ERP frames; first={missing[0]}")

    packed = np.zeros((len(erp_paths), in_h, in_w), dtype=bool)
    kernel = np.ones((3, 3), np.uint8)
    map_cache = {}
    per_frame = []
    rejected_views = []
    for frame, erp_path in enumerate(erp_paths):
        votes = np.zeros((in_h, in_w), dtype=np.uint8)
        for path, yaw, pitch, roll in masks_by_stem[erp_path.stem]:
            key = (yaw, pitch, roll)
            if key not in map_cache:
                if args.mapping == "nearest-scatter":
                    map_cache[key] = build_remap(
                        args.view_width, args.view_height, args.fov,
                        yaw, pitch, roll, in_w, in_h,
                    )
                else:
                    map_cache[key] = build_inverse_remap(
                        in_w, in_h, args.fov, yaw, pitch, roll,
                        args.view_width, args.view_height,
                    )
            mask = np.load(path, allow_pickle=False).astype(bool)
            if mask.shape != (args.view_height, args.view_width):
                raise ValueError(f"Unexpected mask shape {mask.shape}: {path}")
            mask_fraction = float(mask.mean())
            if mask_fraction > args.max_view_mask_fraction:
                rejected_views.append({"path": str(path), "fraction": mask_fraction})
                continue
            if args.rotate180:
                mask = np.rot90(mask, 2)
            view_support = project_view_mask_to_erp(
                mask,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                fov_deg=args.fov,
                erp_width=in_w,
                erp_height=in_h,
                mapping=args.mapping,
                cached_maps=map_cache[key],
            )
            votes += view_support.astype(np.uint8)
        erp_mask = cv2.morphologyEx(
            (votes >= args.min_view_votes).astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        packed[frame] = erp_mask
        per_frame.append({"index": frame, "stem": erp_path.stem, "semantic_fraction": float(erp_mask.mean()),
                          "views": len(masks_by_stem[erp_path.stem])})

    union_inputs = []
    for path in args.union_packed:
        other, stems = load_packed(path)
        if stems != [item.stem for item in erp_paths] or other.shape != packed.shape:
            raise ValueError(f"Packed-mask input does not match ERP order/shape: {path}")
        before = int(np.count_nonzero(packed))
        packed |= other
        union_inputs.append({"path": str(path), "pixels_added": int(np.count_nonzero(packed)) - before})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        masks=np.packbits(packed.reshape(len(packed), -1), axis=1),
        shape=np.asarray(packed.shape, np.int32),
        stems=np.asarray([path.stem for path in erp_paths]),
    )
    report = {
        "protocol": {"erp_dir": str(args.erp_dir), "mask_dirs": [str(path) for path in args.mask_dir],
                     "fov_deg": args.fov, "view_resolution": [args.view_width, args.view_height],
                     "erp_resolution": [in_w, in_h], "rotate180": args.rotate180,
                     "min_view_votes": args.min_view_votes,
                     "max_view_mask_fraction": args.max_view_mask_fraction,
                     "mapping": args.mapping,
                     "fusion": "all-view vote threshold, 3x3 closing"},
        "source_counts": source_counts,
        "unique_views": [{"yaw": y, "pitch": p, "roll": r} for y, p, r in sorted(view_keys)],
        "union_packed": union_inputs,
        "rejected_degenerate_views": rejected_views,
        "aggregate": {"frames": len(packed), "views_per_frame_min": min(row["views"] for row in per_frame),
                      "views_per_frame_max": max(row["views"] for row in per_frame),
                      "masked_fraction_mean": float(packed.mean()),
                      "masked_fraction_max": float(np.max(packed.mean(axis=(1, 2))))},
        "per_frame": per_frame,
        "output": str(args.output),
    }
    report["sha256"] = sha256(args.output)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": report["aggregate"], "views": report["unique_views"],
                      "sha256": report["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
