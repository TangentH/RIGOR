#!/usr/bin/env python3
"""Create inside-ROI and outside-ROI variants of GT-frame aligned clouds.

This is an evaluation-only operation. The input ``aligned.ply`` is never
modified; its two outputs form an exact partition of the input vertices.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from scipy.ndimage import distance_transform_edt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True, type=Path)
    parser.add_argument("--roi", required=True, type=Path)
    parser.add_argument("--masks-dir", required=True, type=Path)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--include-runs", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def safe_label(value: str) -> str:
    if not value or Path(value).name != value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in value):
        raise ValueError(f"Unsafe worker label: {value!r}")
    return value


def discover(method_root: Path) -> list[tuple[str, Path]]:
    result = []
    for aligned in method_root.glob("floor_*/*/run_*/reconstruction/aligned.ply"):
        result.append((str(aligned.parent.parent.relative_to(method_root)), aligned))
    return sorted(result)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_roi(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not key.startswith("_")}


def build_mask(path: Path, margin_px: int) -> np.ndarray:
    inside = np.asarray(Image.open(path).convert("L")) < 128
    return distance_transform_edt(~inside) <= margin_px if margin_px > 0 else inside


def roi_membership(vertices: np.ndarray, entry: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    x = np.asarray(vertices["x"], dtype=np.float64)
    y = np.asarray(vertices["y"], dtype=np.float64)
    z = np.asarray(vertices["z"], dtype=np.float64)
    matrix = np.asarray(entry.get("matrix_3x3", entry.get("matrix")), dtype=np.float64)
    pixels = np.column_stack((x, y, np.ones(len(x)))) @ matrix.T
    xp, yp = pixels[:, 0], pixels[:, 1]
    height, width = mask.shape
    finite = np.isfinite(xp) & np.isfinite(yp) & np.isfinite(z)
    in_bounds = finite & (xp >= 0) & (xp < width) & (yp >= 0) & (yp < height)
    xi = np.clip(xp.astype(np.int64), 0, width - 1)
    yi = np.clip(yp.astype(np.int64), 0, height - 1)
    in_height = (z >= float(entry["z_min"])) & (z <= float(entry["z_max"]))
    return in_bounds & mask[yi, xi] & in_height


def write_vertices(path: Path, vertices: np.ndarray) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    PlyData([PlyElement.describe(vertices, "vertex")], text=False, byte_order="<").write(str(temporary))
    os.replace(temporary, path)


def valid_outputs(reconstruction: Path, relative: str) -> bool:
    report_path = reconstruction / "roi_crop_report.json"
    inside = reconstruction / "aligned_roi.ply"
    outside = reconstruction / "aligned_outside_roi.ply"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (report_path, inside, outside)):
        return False
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        return (
            report.get("relative_path") == relative
            and report.get("status") == "complete"
            and report.get("input_points") == report.get("inside_points") + report.get("outside_points")
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def main() -> int:
    args = parse_args()
    worker = safe_label(args.worker_id)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= --shard-index < --shard-count")
    method_root = args.method_root.resolve()
    roi = load_roi(args.roi.resolve())
    includes = {value.strip().strip("/") for value in args.include_runs.split(",") if value.strip()}
    runs = discover(method_root)
    if includes:
        found = {relative for relative, _ in runs}
        missing = sorted(includes - found)
        if missing:
            raise ValueError(f"Requested runs lack aligned.ply: {missing}")
        runs = [item for item in runs if item[0] in includes]
    runs = [item for index, item in enumerate(runs) if index % args.shard_count == args.shard_index]
    status_path = method_root / "_logs" / worker / "roi_crop_status.jsonl"
    masks: dict[tuple[str, int], np.ndarray] = {}
    counts = {"complete": 0, "skipped_verified": 0, "missing_roi": 0, "failed": 0}
    for index, (relative, aligned) in enumerate(runs, 1):
        started = time.time()
        reconstruction = aligned.parent
        row: dict[str, Any] = {"relative_path": relative, "worker_id": worker}
        print(f"[{index}/{len(runs)}] {relative}", flush=True)
        if valid_outputs(reconstruction, relative):
            row["status"] = "skipped_verified"
            counts["skipped_verified"] += 1
        elif relative not in roi:
            row["status"] = "missing_roi"
            counts["missing_roi"] += 1
        elif any((reconstruction / name).exists() for name in ("aligned_roi.ply", "aligned_outside_roi.ply", "roi_crop_report.json")):
            row["status"] = "failed"
            row["error"] = "Refusing to overwrite incomplete ROI products"
            counts["failed"] += 1
        else:
            try:
                entry = roi[relative]
                margin = int(entry.get("mask_margin_px", 10))
                mask_path = args.masks_dir.resolve() / Path(entry["mask_file"]).name
                key = (str(mask_path), margin)
                if key not in masks:
                    masks[key] = build_mask(mask_path, margin)
                ply = PlyData.read(str(aligned))
                vertices = ply["vertex"].data
                keep = roi_membership(vertices, entry, masks[key])
                write_vertices(reconstruction / "aligned_roi.ply", vertices[keep])
                write_vertices(reconstruction / "aligned_outside_roi.ply", vertices[~keep])
                report = {
                    "status": "complete", "relative_path": relative,
                    "operation": "evaluation ROI partition", "gt_usage": "evaluation_only",
                    "source": str(aligned), "roi_definition": str(args.roi.resolve()),
                    "mask": str(mask_path), "mask_margin_px": margin,
                    "z_min": float(entry["z_min"]), "z_max": float(entry["z_max"]),
                    "input_points": int(len(vertices)), "inside_points": int(keep.sum()),
                    "outside_points": int((~keep).sum()), "inside_fraction": float(keep.mean()),
                    "elapsed_s": time.time() - started,
                }
                atomic_json(reconstruction / "roi_crop_report.json", report)
                row["status"] = "complete"
                row["inside_fraction"] = report["inside_fraction"]
                counts["complete"] += 1
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                counts["failed"] += 1
        row["elapsed_s"] = time.time() - started
        append_jsonl(status_path, row)
        print(json.dumps(row), flush=True)
    atomic_json(status_path.parent / "roi_crop_summary.json", {"worker_id": worker, "selected_runs": len(runs), "counts": counts})
    return 0 if counts["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
