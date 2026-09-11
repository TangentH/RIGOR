#!/usr/bin/env python3
"""Project a hand-drawn equirectangular static mask to HILTI yaw4 pinhole masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from imu_level_equirect_to_pinhole import (
    build_remap_from_rotation,
    extract_frame_timestamp,
    level_rotation_from_gravity,
    load_cam0_from_yaml,
    load_imu_series,
)
from imu_level_equirect_pose_sequence import parse_yaws, yaw_rotation_matrix


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    before = resolved.stat()
    digest = file_sha256(resolved)
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise RuntimeError(f"Input changed while hashing: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def content_tree_signature(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    resolved_root = root.resolve()
    relative_paths: list[tuple[str, Path]] = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative_name = resolved.relative_to(resolved_root).as_posix()
        except ValueError as error:
            raise RuntimeError(
                f"Content-tree member is outside root {resolved_root}: {resolved}"
            ) from error
        relative_paths.append((relative_name, resolved))
    relative_paths.sort(key=lambda item: item[0])
    if not relative_paths:
        raise RuntimeError(f"Content tree must be nonempty: {resolved_root}")
    files: list[dict[str, Any]] = []
    for relative_name, path in relative_paths:
        identity = stable_file_identity(path)
        files.append(
            {
                "relative_path": relative_name,
                "size_bytes": identity["size_bytes"],
                "sha256": identity["sha256"],
            }
        )
    return {
        "root": str(resolved_root),
        "count": len(files),
        "total_size_bytes": sum(int(item["size_bytes"]) for item in files),
        "tree_sha256": canonical_json_sha256(files),
        "files": files,
    }


def content_tree_identity(tree: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": int(tree["count"]),
        "total_size_bytes": int(tree["total_size_bytes"]),
        "tree_sha256": str(tree["tree_sha256"]),
        "files": tree["files"],
    }


def assert_file_identity_unchanged(
    label: str, path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    actual = stable_file_identity(path)
    expected_pair = (int(expected["size_bytes"]), str(expected["sha256"]))
    actual_pair = (int(actual["size_bytes"]), str(actual["sha256"]))
    if actual_pair != expected_pair:
        raise RuntimeError(
            f"{label} changed during static-mask projection: "
            f"expected_size_sha={expected_pair} actual_size_sha={actual_pair}"
        )
    return actual


def assert_content_tree_unchanged(
    label: str, expected: dict[str, Any], actual: dict[str, Any]
) -> None:
    if content_tree_identity(expected) != content_tree_identity(actual):
        raise RuntimeError(
            f"{label} content tree changed during static-mask projection: "
            f"expected={expected['tree_sha256']} actual={actual['tree_sha256']}"
        )


def filename_set_from_names(names: Iterable[str]) -> dict[str, Any]:
    ordered = sorted(str(name) for name in names)
    if not ordered:
        raise RuntimeError("Filename set must be nonempty")
    return {
        "count": len(ordered),
        "filename_set_sha256": canonical_json_sha256(ordered),
        "first_name": ordered[0],
        "last_name": ordered[-1],
    }


def filename_set_signature(directory: Path) -> dict[str, Any]:
    paths = sorted(path for path in directory.glob("*.npy") if path.is_file())
    empty = [path.name for path in paths if path.stat().st_size <= 0]
    if not paths or empty:
        raise RuntimeError(
            f"Static-mask outputs must be nonempty: count={len(paths)} "
            f"empty={empty[:5]}"
        )
    names = [path.name for path in paths]
    return {
        **filename_set_from_names(names),
        "total_size_bytes": sum(path.stat().st_size for path in paths),
        "all_nonempty": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Take one equirectangular binary mask and generate per-yaw4 .npy masks "
            "with the same IMU leveling/projection used for pinhole image generation."
        )
    )
    parser.add_argument("--bag", required=True, help="Path to rosbag.db3")
    parser.add_argument("--equirect-dir", required=True, help="Directory containing equirect frames")
    parser.add_argument("--pinhole-dir", required=True, help="Directory containing generated yaw4 images")
    parser.add_argument("--yaml", required=True, help="Path to kalibr_imucam_chain.yaml")
    parser.add_argument("--mask", required=True, help="Hand-drawn equirect mask PNG; white/positive means suppress")
    parser.add_argument("--output-dir", required=True, help="Output directory; writes masks_npy/ and optionally overlays/")
    parser.add_argument("--width", type=int, default=768, help="Pinhole width")
    parser.add_argument("--height", type=int, default=768, help="Pinhole height")
    parser.add_argument("--fov-deg", type=float, default=90.0, help="Horizontal FOV in degrees")
    parser.add_argument("--yaws", default="0,90,180,270", help="Comma-separated yaw offsets")
    parser.add_argument("--imu-tau", type=float, default=0.25)
    parser.add_argument(
        "--imu-method",
        choices=("causal_accel", "complementary"),
        default="causal_accel",
        help="IMU gravity estimator; must match the RGB yaw-view generator.",
    )
    parser.add_argument(
        "--accel-gate-sigma",
        type=float,
        default=0.2,
        help="Complementary-filter accelerometer gate; ignored by causal_accel.",
    )
    parser.add_argument("--time-offset-ns", type=int, default=0)
    parser.add_argument("--use-yaml-timeshift", action="store_true")
    parser.add_argument("--rotate180", action="store_true")
    parser.add_argument("--threshold", type=int, default=128, help="Mask binarization threshold")
    parser.add_argument("--dilate-pixels", type=int, default=0, help="Optional dilation after projection")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N equirect frames, 0 means all")
    parser.add_argument("--save-overlays", action="store_true", help="Write red overlays for visual checking")
    parser.add_argument("--progress-interval-s", type=float, default=5.0)
    return parser.parse_args()


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def save_overlay(image_path: Path, mask: np.ndarray, out_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    if image.shape[:2] != mask.shape:
        mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * red[mask]).astype(np.uint8)
    cv2.imwrite(str(out_path), overlay, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    args = parse_args()
    equirect_dir = Path(args.equirect_dir)
    pinhole_dir = Path(args.pinhole_dir)
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "masks_npy"
    overlays_dir = output_dir / "overlays"
    manifest_path = output_dir / "manifest.json"
    summary_path = output_dir / "summary.json"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)
    # An old manifest must never remain published while outputs are rewritten.
    manifest_path.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)

    equirect_paths = list_images(equirect_dir)
    if args.limit > 0:
        equirect_paths = equirect_paths[: args.limit]
    pinhole_paths = list_images(pinhole_dir)
    pinhole_by_name = {p.name: p for p in pinhole_paths}
    if not equirect_paths:
        raise FileNotFoundError(f"No equirect images found: {equirect_dir}")
    if not pinhole_paths:
        raise FileNotFoundError(f"No pinhole images found: {pinhole_dir}")

    yaws = parse_yaws(args.yaws)
    expected_mask_names = {
        (
            f"{frame_idx:05d}_v{view_idx:02d}_{equirect_path.stem}_"
            f"yaw{int(round(yaw_deg)) % 360:03d}.npy"
        )
        for frame_idx, equirect_path in enumerate(equirect_paths)
        for view_idx, yaw_deg in enumerate(yaws)
    }
    if len(expected_mask_names) != len(equirect_paths) * len(yaws):
        raise RuntimeError("Yaw/view configuration produces duplicate mask names")
    pinhole_mask_names = {f"{path.stem}.npy" for path in pinhole_paths}
    if args.limit > 0:
        pinhole_parity_ok = expected_mask_names <= pinhole_mask_names
    else:
        pinhole_parity_ok = expected_mask_names == pinhole_mask_names
    if not pinhole_parity_ok:
        raise RuntimeError(
            "Static-mask/RGB requested filename parity failed: "
            f"expected={len(expected_mask_names)} pinhole={len(pinhole_mask_names)} "
            f"missing={sorted(expected_mask_names - pinhole_mask_names)[:5]} "
            f"extra={sorted(pinhole_mask_names - expected_mask_names)[:5]}"
        )

    bag_path = Path(args.bag)
    yaml_path = Path(args.yaml)
    static_mask_path = Path(args.mask)
    bag_before = stable_file_identity(bag_path)
    yaml_before = stable_file_identity(yaml_path)
    static_mask_before = stable_file_identity(static_mask_path)
    equirect_tree_before = content_tree_signature(
        equirect_dir, equirect_paths
    )
    pinhole_tree_before = content_tree_signature(pinhole_dir, pinhole_paths)

    mask_img = cv2.imread(str(static_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise RuntimeError(f"Failed to read mask: {static_mask_path}")
    mask_bool = mask_img >= args.threshold
    in_h, in_w = mask_bool.shape

    r_cam0_imu, yaml_timeshift_s = load_cam0_from_yaml(str(yaml_path))
    imu_ts, imu_g_lp = load_imu_series(
        str(bag_path),
        tau_s=args.imu_tau,
        method=args.imu_method,
        accel_gate_sigma=args.accel_gate_sigma,
    )
    total_offset_ns = int(args.time_offset_ns)
    if args.use_yaml_timeshift:
        total_offset_ns += int(round(yaml_timeshift_s * 1e9))

    settings = {
        "width": int(args.width),
        "height": int(args.height),
        "fov_deg": float(args.fov_deg),
        "yaws": str(args.yaws),
        "imu_tau": float(args.imu_tau),
        "imu_method": str(args.imu_method),
        "accel_gate_sigma": float(args.accel_gate_sigma),
        "time_offset_ns": int(args.time_offset_ns),
        "use_yaml_timeshift": bool(args.use_yaml_timeshift),
        "yaml_timeshift_s": float(yaml_timeshift_s),
        "effective_time_offset_ns": int(total_offset_ns),
        "rotate180": bool(args.rotate180),
        "threshold": int(args.threshold),
        "dilate_pixels": int(args.dilate_pixels),
        "limit": int(args.limit),
    }
    request = {
        "request_schema_version": 1,
        "operation": "equirect_static_mask_to_yaw4_masks",
        "settings": settings,
        "inputs": {
            "bag": bag_before,
            "equirect_image_tree": content_tree_identity(
                equirect_tree_before
            ),
            "pinhole_image_tree": content_tree_identity(
                pinhole_tree_before
            ),
            "kalibr_yaml": yaml_before,
            "static_mask": static_mask_before,
        },
        "expected_outputs": {
            "mask_filename_set": filename_set_from_names(
                expected_mask_names
            )
        },
    }
    request_sha256 = canonical_json_sha256(request)

    kernel = None
    if args.dilate_pixels > 0:
        k = int(args.dilate_pixels) * 2 + 1
        kernel = np.ones((k, k), dtype=np.uint8)

    print(f"equirect_frames={len(equirect_paths)}")
    print(f"pinhole_images={len(pinhole_paths)}")
    print(f"mask_resolution={in_w}x{in_h} suppress_ratio={mask_bool.mean():.6f}")
    print(f"output_masks={masks_dir}", flush=True)

    prev_rot = np.eye(3, dtype=np.float32)
    started = time.time()
    last_progress = started
    written = 0
    missing_pinhole = 0

    for frame_idx, equirect_path in enumerate(equirect_paths):
        frame_ts = extract_frame_timestamp(equirect_path) + total_offset_ns
        imu_idx = int(np.searchsorted(imu_ts, frame_ts))
        if imu_idx >= len(imu_ts):
            imu_idx = len(imu_ts) - 1
        elif imu_idx > 0:
            left_dt = abs(int(imu_ts[imu_idx - 1]) - frame_ts)
            right_dt = abs(int(imu_ts[imu_idx]) - frame_ts)
            if left_dt <= right_dt:
                imu_idx -= 1

        g_cam0 = r_cam0_imu @ imu_g_lp[imu_idx]
        base_rot = level_rotation_from_gravity(g_cam0)
        if not np.isfinite(base_rot).all():
            base_rot = prev_rot
        prev_rot = base_rot

        stem = equirect_path.stem
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
            projected = cv2.remap(
                mask_bool.astype(np.uint8),
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_WRAP,
            ).astype(bool)
            if args.rotate180:
                projected = cv2.rotate(projected.astype(np.uint8), cv2.ROTATE_180).astype(bool)
            if kernel is not None:
                projected = cv2.dilate(projected.astype(np.uint8), kernel, iterations=1).astype(bool)

            out_stem = f"{frame_idx:05d}_v{view_idx:02d}_{stem}_yaw{int(round(yaw_deg))%360:03d}"
            np.save(masks_dir / f"{out_stem}.npy", projected)
            written += 1

            if args.save_overlays:
                pinhole_path = pinhole_by_name.get(f"{out_stem}.jpg") or pinhole_by_name.get(f"{out_stem}.png")
                if pinhole_path is None:
                    missing_pinhole += 1
                else:
                    save_overlay(pinhole_path, projected, overlays_dir / f"{out_stem}_static_mask.jpg")

        now = time.time()
        if (
            (args.progress_interval_s > 0 and now - last_progress >= args.progress_interval_s)
            or frame_idx + 1 == len(equirect_paths)
            or (frame_idx + 1) % 25 == 0
        ):
            pct = 100.0 * (frame_idx + 1) / len(equirect_paths)
            print(
                f"[static-mask] frames={frame_idx + 1}/{len(equirect_paths)} "
                f"views={written} ({pct:.1f}%) elapsed={now - started:.1f}s",
                flush=True,
            )
            last_progress = now

    print(f"[done] wrote {written} masks to {masks_dir}")
    if args.save_overlays and missing_pinhole:
        print(f"[warn] missing pinhole images for {missing_pinhole} overlays")
    actual_mask_names = {path.name for path in masks_dir.glob("*.npy")}
    if actual_mask_names != expected_mask_names:
        raise RuntimeError(
            "Static-mask output filename parity failed: "
            f"expected={len(expected_mask_names)} actual={len(actual_mask_names)} "
            f"missing={sorted(expected_mask_names - actual_mask_names)[:5]} "
            f"extra={sorted(actual_mask_names - expected_mask_names)[:5]}"
        )

    assert_file_identity_unchanged("rosbag", bag_path, bag_before)
    assert_file_identity_unchanged("kalibr yaml", yaml_path, yaml_before)
    assert_file_identity_unchanged(
        "static mask", static_mask_path, static_mask_before
    )
    equirect_paths_after = list_images(equirect_dir)
    if args.limit > 0:
        equirect_paths_after = equirect_paths_after[: args.limit]
    pinhole_paths_after = list_images(pinhole_dir)
    equirect_tree_after = content_tree_signature(
        equirect_dir, equirect_paths_after
    )
    pinhole_tree_after = content_tree_signature(
        pinhole_dir, pinhole_paths_after
    )
    assert_content_tree_unchanged(
        "equirect image", equirect_tree_before, equirect_tree_after
    )
    assert_content_tree_unchanged(
        "pinhole image", pinhole_tree_before, pinhole_tree_after
    )

    output_signature = filename_set_signature(masks_dir)
    output_tree = content_tree_signature(
        masks_dir,
        sorted(masks_dir.glob("*.npy")),
    )
    if int(output_tree["count"]) != len(expected_mask_names):
        raise RuntimeError(
            "Static-mask output content tree count mismatch: "
            f"expected={len(expected_mask_names)} "
            f"actual={output_tree['count']}"
        )
    manifest = {
        "schema_version": 2,
        "created_at_unix": time.time(),
        "elapsed_s": time.time() - started,
        "request": request,
        "request_sha256": request_sha256,
        "settings": settings,
        "inputs": {
            "bag": str(bag_path.resolve()),
            "bag_size_bytes": int(bag_before["size_bytes"]),
            "bag_sha256": str(bag_before["sha256"]),
            "bag_identity": bag_before,
            "equirect_dir": str(equirect_dir.resolve()),
            "equirect_frame_count": len(equirect_paths),
            "equirect_image_tree": equirect_tree_before,
            "pinhole_dir": str(pinhole_dir.resolve()),
            "pinhole_image_count": len(pinhole_paths),
            "pinhole_image_tree": pinhole_tree_before,
            "kalibr_yaml": str(yaml_path.resolve()),
            "kalibr_yaml_size_bytes": int(yaml_before["size_bytes"]),
            "kalibr_yaml_sha256": str(yaml_before["sha256"]),
            "static_mask": str(static_mask_path.resolve()),
            "static_mask_size_bytes": int(
                static_mask_before["size_bytes"]
            ),
            "static_mask_sha256": str(static_mask_before["sha256"]),
            "static_mask_resolution": [int(in_w), int(in_h)],
            "static_mask_suppress_ratio": float(mask_bool.mean()),
        },
        "outputs": {
            "masks_dir": str(masks_dir.resolve()),
            "mask_count": int(written),
            "expected_mask_count": int(len(equirect_paths) * len(yaws)),
            "filename_set": output_signature,
            "mask_content_tree": output_tree,
            "overlays_dir": str(overlays_dir.resolve())
            if args.save_overlays
            else None,
            "missing_pinhole_for_overlays": int(missing_pinhole),
        },
        "validation": {
            "request_sha256_recomputes": (
                canonical_json_sha256(request) == request_sha256
            ),
            "inputs_unchanged_during_run": True,
            "pinhole_filename_parity": True,
            "output_filename_parity": True,
            "filename_set_role": "parity_only_not_content_identity",
            "content_identity": "sorted relative_path + size_bytes + sha256",
        },
    }
    atomic_write_json(
        summary_path,
        {
            "schema_version": 2,
            "mask_count": int(written),
            "expected_mask_count": int(len(equirect_paths) * len(yaws)),
            "equirect_frame_count": len(equirect_paths),
            "view_count": len(yaws),
            "filename_set": output_signature,
            "mask_content_tree": output_tree,
            "request_sha256": request_sha256,
            "manifest": str(manifest_path.resolve()),
        },
    )
    # The manifest is the final publication step after every output check.
    atomic_write_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
