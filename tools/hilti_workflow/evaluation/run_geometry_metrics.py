#!/usr/bin/env python3
"""Compute restart-safe geometry metrics for one aligned cloud variant.

Defaults match the evaluation protocol: full clouds, symmetric Chamfer,
bidirectional RMSE/Hausdorff, distribution statistics, and F-scores at
0.05/0.10/0.25/0.50 metres. Subsampling is opt-in and deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from plyfile import PlyData


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-root", required=True, type=Path)
    parser.add_argument("--prediction-name", required=True)
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--gt-name", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.25, 0.50])
    parser.add_argument("--subsample", type=int, default=0, help="0 evaluates every point")
    parser.add_argument("--include-runs", default="")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def safe_label(value: str) -> str:
    if not value or Path(value).name != value or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in value):
        raise ValueError(f"Unsafe label: {value!r}")
    return value


def discover(method_root: Path, prediction_name: str) -> list[tuple[str, Path]]:
    result = []
    for prediction in method_root.glob(f"floor_*/*/run_*/reconstruction/{prediction_name}"):
        result.append((str(prediction.parent.parent.relative_to(method_root)), prediction))
    return sorted(result)


def load_xyz(path: Path) -> np.ndarray:
    ply = PlyData.read(str(path), mmap="r")
    vertices = ply["vertex"].data
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError(f"No finite points: {path}")
    return points


def stable_seed(relative: str, role: str) -> int:
    digest = hashlib.sha256(f"{relative}:{role}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def subsample(points: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if limit <= 0 or len(points) <= limit:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), limit, replace=False)]


def nearest_distances(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return exact Euclidean distance from each query point to its nearest reference."""
    query_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(query))
    reference_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(reference))
    return np.asarray(query_cloud.compute_point_cloud_distance(reference_cloud))


def bidirectional_nearest_distances(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return prediction-to-target and target-to-prediction nearest distances."""
    prediction_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(prediction))
    target_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
    prediction_to_target = np.asarray(prediction_cloud.compute_point_cloud_distance(target_cloud))
    target_to_prediction = np.asarray(target_cloud.compute_point_cloud_distance(prediction_cloud))
    return prediction_to_target, target_to_prediction


def compute_metrics(prediction: np.ndarray, target: np.ndarray, thresholds: list[float]) -> dict[str, Any]:
    pred_to_gt, gt_to_pred = bidirectional_nearest_distances(prediction, target)

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()), "median": float(np.median(values)), "std": float(values.std()),
            "p90": float(np.quantile(values, 0.90)), "p95": float(np.quantile(values, 0.95)),
            "p99": float(np.quantile(values, 0.99)),
        }

    threshold_metrics: dict[str, dict[str, float]] = {}
    for threshold in thresholds:
        precision = float(np.mean(pred_to_gt < threshold))
        recall = float(np.mean(gt_to_pred < threshold))
        fscore = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        threshold_metrics[f"tau={threshold}"] = {"precision": precision, "recall": recall, "fscore": float(fscore)}
    return {
        "n_pred_points": int(len(prediction)), "n_gt_points": int(len(target)),
        "chamfer_distance": float((pred_to_gt.mean() + gt_to_pred.mean()) / 2.0),
        "hausdorff": {"pred_to_gt": float(pred_to_gt.max()), "gt_to_pred": float(gt_to_pred.max()),
                       "symmetric": float(max(pred_to_gt.max(), gt_to_pred.max()))},
        "rmse": {"pred_to_gt": float(np.sqrt(np.mean(pred_to_gt ** 2))),
                 "gt_to_pred": float(np.sqrt(np.mean(gt_to_pred ** 2)))},
        "distance_stats_pred_to_gt": stats(pred_to_gt),
        "distance_stats_gt_to_pred": stats(gt_to_pred),
        "threshold_metrics": threshold_metrics,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def valid_report(path: Path, *, relative: str, variant: str, prediction: Path, target: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        return (
            report.get("status") == "complete" and report.get("relative_path") == relative
            and report.get("variant") == variant
            and report.get("prediction", {}).get("size") == prediction.stat().st_size
            and report.get("target", {}).get("size") == target.stat().st_size
        )
    except (OSError, json.JSONDecodeError, TypeError):
        return False


def main() -> int:
    args = parse_args()
    worker = safe_label(args.worker_id)
    variant = safe_label(args.variant)
    if args.subsample < 0:
        raise ValueError("--subsample must be non-negative")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Require 0 <= --shard-index < --shard-count")
    method_root = args.method_root.resolve()
    output_root = args.output_root.resolve()
    includes = {value.strip().strip("/") for value in args.include_runs.split(",") if value.strip()}
    runs = discover(method_root, args.prediction_name)
    if includes:
        found = {relative for relative, _ in runs}
        missing = sorted(includes - found)
        if missing:
            raise ValueError(f"Requested runs lack {args.prediction_name}: {missing}")
        runs = [item for item in runs if item[0] in includes]
    runs = [item for index, item in enumerate(runs) if index % args.shard_count == args.shard_index]
    status_path = output_root / "_logs" / worker / f"metrics_{variant}_status.jsonl"
    counts = {"complete": 0, "skipped_verified": 0, "missing_gt": 0, "failed": 0}
    for index, (relative, prediction_path) in enumerate(runs, 1):
        started = time.time()
        target_path = args.gt_root.resolve() / relative / args.gt_name
        report_path = output_root / relative / "reconstruction" / f"metrics_{variant}.json"
        row: dict[str, Any] = {"relative_path": relative, "variant": variant, "worker_id": worker}
        print(f"[{index}/{len(runs)}] {variant} {relative}", flush=True)
        if not target_path.is_file():
            row["status"] = "missing_gt"
            row["error"] = str(target_path)
            counts["missing_gt"] += 1
        elif valid_report(report_path, relative=relative, variant=variant, prediction=prediction_path, target=target_path):
            row["status"] = "skipped_verified"
            counts["skipped_verified"] += 1
        elif report_path.exists():
            row["status"] = "failed"
            row["error"] = "Refusing to overwrite stale or invalid metrics report"
            counts["failed"] += 1
        else:
            try:
                prediction_source = load_xyz(prediction_path)
                target_source = load_xyz(target_path)
                prediction = subsample(prediction_source, args.subsample, stable_seed(relative, "prediction"))
                target = subsample(target_source, args.subsample, stable_seed(relative, "gt"))
                metrics = compute_metrics(prediction, target, args.thresholds)
                report = {
                    "status": "complete", "relative_path": relative, "variant": variant,
                    "protocol": "bidirectional geometry metrics",
                    "prediction": {"path": str(prediction_path), "size": prediction_path.stat().st_size,
                                   "source_points": int(len(prediction_source))},
                    "target": {"path": str(target_path), "size": target_path.stat().st_size,
                               "source_points": int(len(target_source))},
                    "parameters": {"thresholds_m": args.thresholds, "subsample": args.subsample,
                                   "subsampling": "disabled" if args.subsample == 0 else "deterministic_without_replacement"},
                    "metrics": metrics, "elapsed_s": time.time() - started,
                }
                atomic_json(report_path, report)
                row["status"] = "complete"
                row["chamfer_distance"] = metrics["chamfer_distance"]
                counts["complete"] += 1
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                counts["failed"] += 1
        row["elapsed_s"] = time.time() - started
        append_jsonl(status_path, row)
        print(json.dumps(row), flush=True)
    atomic_json(status_path.parent / f"metrics_{variant}_summary.json", {"worker_id": worker, "variant": variant, "selected_runs": len(runs), "counts": counts})
    return 0 if counts["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
