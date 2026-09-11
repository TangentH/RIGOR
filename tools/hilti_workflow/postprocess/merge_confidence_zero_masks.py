#!/usr/bin/env python3
"""Merge confidence-zero masks and optionally visualize the merged result."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", required=True, type=Path)
    parser.add_argument("--secondary-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--visualization-dir", type=Path)
    parser.add_argument("--max-visualizations", type=int, default=24)
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser.parse_args()


def mask_files(mask_dir: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(mask_dir.glob("*.npy"))}


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_file_entry(relative_path: str, path: Path) -> dict[str, Any]:
    before = path.stat()
    digest = file_sha256(path)
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return {
        "relative_path": relative_path,
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def content_tree_signature(
    root: Path, files: dict[str, Path]
) -> dict[str, Any]:
    names = sorted(files)
    if not names:
        raise RuntimeError(f"Mask content tree must be nonempty: {root}")
    entries = [
        stable_file_entry(name, files[name])
        for name in names
    ]
    return {
        "root": str(root.resolve()),
        "count": len(entries),
        "total_size_bytes": sum(
            int(entry["size_bytes"]) for entry in entries
        ),
        "tree_sha256": canonical_json_sha256(entries),
        "files": entries,
    }


def content_tree_identity(tree: dict[str, Any]) -> dict[str, Any]:
    return {
        "count": int(tree["count"]),
        "total_size_bytes": int(tree["total_size_bytes"]),
        "tree_sha256": str(tree["tree_sha256"]),
        "files": tree["files"],
    }


def assert_content_tree_unchanged(
    label: str, expected: dict[str, Any], actual: dict[str, Any]
) -> None:
    if content_tree_identity(expected) != content_tree_identity(actual):
        raise RuntimeError(
            f"{label} input content tree changed during mask merge: "
            f"expected={expected['tree_sha256']} "
            f"actual={actual['tree_sha256']}"
        )


def load_npy_verified(
    path: Path, expected_entry: dict[str, Any]
) -> np.ndarray:
    payload = path.read_bytes()
    actual_size = len(payload)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if (
        actual_size != int(expected_entry["size_bytes"])
        or actual_sha256 != str(expected_entry["sha256"])
    ):
        raise RuntimeError(
            f"Mask input changed before verified read: {path}; "
            f"expected_size_sha="
            f"({expected_entry['size_bytes']}, {expected_entry['sha256']}) "
            f"actual_size_sha=({actual_size}, {actual_sha256})"
        )
    return np.load(io.BytesIO(payload), allow_pickle=False)


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("wb") as stream:
            np.save(stream, array)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def filename_set_signature(files: dict[str, Path]) -> dict[str, Any]:
    names = sorted(files)
    empty = [name for name in names if files[name].stat().st_size <= 0]
    if not names or empty:
        raise RuntimeError(
            f"Mask filename set must be nonempty: count={len(names)} "
            f"empty={empty[:5]}"
        )
    return {
        "count": len(names),
        "filename_set_sha256": canonical_json_sha256(names),
        "total_size_bytes": sum(files[name].stat().st_size for name in names),
        "all_nonempty": True,
        "first_name": names[0],
        "last_name": names[-1],
    }


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


def mask_ratio(mask: np.ndarray) -> float:
    return float(np.asarray(mask, dtype=bool).mean())


def overlay_image(image_path: Path, mask: np.ndarray, alpha: float) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    mask_bool = np.asarray(mask, dtype=bool)
    if mask_bool.shape != (image.height, image.width):
        raise ValueError(f"mask/image size mismatch for {image_path.name}: {mask_bool.shape} vs {image.size}")

    red = Image.new("RGB", image.size, (255, 0, 0))
    mask_img = Image.fromarray((mask_bool.astype(np.uint8) * int(255 * alpha)), mode="L")
    out = Image.composite(red, image, mask_img)
    return Image.blend(image, out, alpha)


def make_contact_sheet(tiles: list[Image.Image], labels: list[str], out_path: Path, cols: int = 4) -> None:
    if not tiles:
        return
    thumb_w = 320
    thumb_h = int(thumb_w * tiles[0].height / tiles[0].width)
    label_h = 24
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        thumb = tile.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y + label_h))
        draw.text((x + 4, y + 4), label[:64], fill=(0, 0, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def main() -> int:
    args = parse_args()
    if args.output_dir.resolve() in {
        args.primary_dir.resolve(),
        args.secondary_dir.resolve(),
    }:
        raise RuntimeError("Mask merge output directory must differ from inputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir.parent / "merge_summary.json"
    # Never leave a stale successful manifest published while rewriting masks.
    manifest_path.unlink(missing_ok=True)
    primary = mask_files(args.primary_dir)
    secondary = mask_files(args.secondary_dir)
    common_names = sorted(primary.keys() & secondary.keys())
    if not common_names:
        raise RuntimeError("no common .npy mask names")
    missing_primary = sorted(secondary.keys() - primary.keys())
    missing_secondary = sorted(primary.keys() - secondary.keys())
    if missing_primary or missing_secondary:
        raise RuntimeError(
            "Mask merge requires exact primary/secondary filename parity: "
            f"missing_primary={missing_primary[:5]} "
            f"missing_secondary={missing_secondary[:5]}"
        )

    primary_tree_before = content_tree_signature(
        args.primary_dir, primary
    )
    secondary_tree_before = content_tree_signature(
        args.secondary_dir, secondary
    )
    primary_entries = {
        str(entry["relative_path"]): entry
        for entry in primary_tree_before["files"]
    }
    secondary_entries = {
        str(entry["relative_path"]): entry
        for entry in secondary_tree_before["files"]
    }
    primary_filename_set = filename_set_signature(primary)
    secondary_filename_set = filename_set_signature(secondary)
    expected_filename_set = {
        key: primary_filename_set[key]
        for key in (
            "count",
            "filename_set_sha256",
            "first_name",
            "last_name",
        )
    }
    request = {
        "request_schema_version": 1,
        "operation": "logical_or_boolean_masks",
        "inputs": {
            "primary_content_tree": primary_tree_before,
            "secondary_content_tree": secondary_tree_before,
        },
        "expected_outputs": {
            "mask_filename_set": expected_filename_set,
        },
        "diagnostics": {
            "image_dir": str(args.image_dir.resolve())
            if args.image_dir
            else None,
            "visualization_dir": str(args.visualization_dir.resolve())
            if args.visualization_dir
            else None,
            "max_visualizations": int(args.max_visualizations),
            "alpha": float(args.alpha),
        },
    }
    request_sha256 = canonical_json_sha256(request)

    summary = {
        "schema_version": 3,
        "request": request,
        "request_sha256": request_sha256,
        "primary_dir": str(args.primary_dir),
        "secondary_dir": str(args.secondary_dir),
        "output_dir": str(args.output_dir),
        "num_primary": len(primary),
        "num_secondary": len(secondary),
        "num_common": len(common_names),
        "missing_primary": missing_primary[:20],
        "missing_secondary": missing_secondary[:20],
        "primary_ratio_mean": 0.0,
        "secondary_ratio_mean": 0.0,
        "merged_ratio_mean": 0.0,
        "merged_ratio_max": 0.0,
    }

    primary_ratios: list[float] = []
    secondary_ratios: list[float] = []
    merged_ratios: list[float] = []
    vis_tiles: list[Image.Image] = []
    vis_labels: list[str] = []
    vis_stride = max(1, len(common_names) // max(1, args.max_visualizations))

    for idx, name in enumerate(common_names):
        a = load_npy_verified(
            primary[name], primary_entries[name]
        ).astype(bool)
        b = load_npy_verified(
            secondary[name], secondary_entries[name]
        ).astype(bool)
        if a.shape != b.shape:
            raise ValueError(f"mask shape mismatch for {name}: {a.shape} vs {b.shape}")
        merged = np.logical_or(a, b)
        atomic_save_npy(args.output_dir / name, merged)
        primary_ratios.append(mask_ratio(a))
        secondary_ratios.append(mask_ratio(b))
        merged_ratios.append(mask_ratio(merged))

        if args.image_dir and args.visualization_dir and idx % vis_stride == 0 and len(vis_tiles) < args.max_visualizations:
            image_path = args.image_dir / name.replace(".npy", ".jpg")
            if image_path.exists():
                vis_tiles.append(overlay_image(image_path, merged, args.alpha))
                vis_labels.append(name.replace(".npy", ""))

    summary["primary_ratio_mean"] = float(np.mean(primary_ratios))
    summary["secondary_ratio_mean"] = float(np.mean(secondary_ratios))
    summary["merged_ratio_mean"] = float(np.mean(merged_ratios))
    summary["merged_ratio_max"] = float(np.max(merged_ratios))

    if args.visualization_dir:
        make_contact_sheet(
            vis_tiles,
            vis_labels,
            args.visualization_dir
            / "merged_confidence_zero_overlay_contact_sheet.jpg",
        )

    primary_after = mask_files(args.primary_dir)
    secondary_after = mask_files(args.secondary_dir)
    primary_tree_after = content_tree_signature(
        args.primary_dir, primary_after
    )
    secondary_tree_after = content_tree_signature(
        args.secondary_dir, secondary_after
    )
    assert_content_tree_unchanged(
        "primary", primary_tree_before, primary_tree_after
    )
    assert_content_tree_unchanged(
        "secondary", secondary_tree_before, secondary_tree_after
    )

    merged_files = mask_files(args.output_dir)
    if set(merged_files) != set(primary):
        raise RuntimeError(
            "Merged-mask filename parity failed: "
            f"missing={sorted(set(primary) - set(merged_files))[:5]} "
            f"extra={sorted(set(merged_files) - set(primary))[:5]}"
        )
    merged_tree = content_tree_signature(args.output_dir, merged_files)
    summary["filename_sets"] = {
        "primary": primary_filename_set,
        "secondary": secondary_filename_set,
        "merged": filename_set_signature(merged_files),
    }
    summary["content_trees"] = {
        "primary": primary_tree_before,
        "secondary": secondary_tree_before,
        "merged": merged_tree,
    }
    summary["validation"] = {
        "request_sha256_recomputes": (
            canonical_json_sha256(request) == request_sha256
        ),
        "inputs_unchanged_during_run": True,
        "input_filename_parity": True,
        "output_filename_parity": True,
        "filename_set_role": "parity_only_not_content_identity",
        "content_identity": "sorted relative_path + size_bytes + sha256",
    }
    # Publish only after diagnostics, input re-hash, and output-tree checks.
    atomic_write_json(manifest_path, summary)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
