#!/usr/bin/env python3
"""Evaluate single-view versus cyclic four-view SALAD retrieval.

Ground-truth trajectories are used only after descriptor extraction to label
temporally eligible capture pairs as revisits or non-revisits.  The script is
evaluation-only and never modifies reconstruction artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNIVERSE = REPO_ROOT / "tools/hilti_workflow/manifests/hilti_all_runs.json"
TIMESTAMP_RE = re.compile(r"frame_\d+_(\d+)_yaw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=DEFAULT_UNIVERSE)
    parser.add_argument("--inventory", type=Path, required=True, help="JSON mapping each run to its GT trajectory")
    parser.add_argument("--descriptor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-capture-gap", type=int, default=80)
    parser.add_argument("--positive-radius-m", type=float, default=1.5)
    parser.add_argument("--negative-radius-m", type=float, default=3.0)
    parser.add_argument("--max-interpolation-gap-s", type=float, default=0.75)
    parser.add_argument("--single-view-threshold", type=float, default=0.85)
    parser.add_argument("--cyclic-mean-threshold", type=float, default=0.65)
    parser.add_argument("--cyclic-min-view-threshold", type=float, default=0.45)
    parser.add_argument("--cyclic-support-threshold", type=float, default=0.50)
    parser.add_argument("--cyclic-min-support", type=int, default=4)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gt_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = np.loadtxt(path, comments="#", dtype=np.float64)
    if rows.ndim == 1:
        rows = rows[None, :]
    return rows[:, 0], rows[:, 1:4]


def interpolate_positions(
    query_t: np.ndarray, gt_t: np.ndarray, gt_p: np.ndarray, max_gap: float
) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(gt_t, query_t, side="left")
    right_safe = np.clip(right, 0, len(gt_t) - 1)
    left_safe = np.clip(right - 1, 0, len(gt_t) - 1)
    exact = np.abs(gt_t[right_safe] - query_t) < 1e-9
    bracketed = (right > 0) & (right < len(gt_t))
    gap = gt_t[right_safe] - gt_t[left_safe]
    valid = exact | (bracketed & (gap <= max_gap))
    positions = np.full((len(query_t), 3), np.nan, dtype=np.float64)
    positions[exact] = gt_p[right_safe[exact]]
    interp = valid & ~exact
    if np.any(interp):
        alpha = (
            (query_t[interp] - gt_t[left_safe[interp]])
            / (gt_t[right_safe[interp]] - gt_t[left_safe[interp]])
        )
        positions[interp] = (
            gt_p[left_safe[interp]] * (1.0 - alpha[:, None])
            + gt_p[right_safe[interp]] * alpha[:, None]
        )
    return positions, valid


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = int(np.sum(ranked))
    if positives == 0:
        return None
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision[ranked]) / positives)


def recall_at_precision(labels: np.ndarray, scores: np.ndarray, target: float) -> float | None:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = int(np.sum(ranked))
    if positives == 0:
        return None
    tp = np.cumsum(ranked)
    precision = tp / np.arange(1, len(ranked) + 1)
    recall = tp / positives
    valid = precision >= target
    return float(np.max(recall[valid])) if np.any(valid) else 0.0


def threshold_metrics(labels: np.ndarray, accepted: np.ndarray) -> dict:
    tp = int(np.sum(labels & accepted))
    fp = int(np.sum(~labels & accepted))
    fn = int(np.sum(labels & ~accepted))
    return {
        "accepted": int(np.sum(accepted)),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
    }


def parse_capture_times(names: np.ndarray, capture_indices: np.ndarray) -> np.ndarray:
    unique = np.unique(capture_indices)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("capture indices must be contiguous and zero-based")
    times = np.empty(len(unique), dtype=np.float64)
    for capture in unique:
        entries = names[capture_indices == capture]
        stamps = set()
        for name in entries:
            match = TIMESTAMP_RE.search(str(name))
            if match is None:
                raise ValueError(f"timestamp not found in {name}")
            stamps.add(int(match.group(1)))
        if len(stamps) != 1:
            raise ValueError(f"inconsistent timestamps for capture {capture}")
        times[capture] = stamps.pop() * 1e-9
    return times


def score_matrices(
    descriptors: np.ndarray, support_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if descriptors.shape[0] % 4:
        raise ValueError("expected exactly four descriptors per capture")
    n = descriptors.shape[0] // 4
    d = descriptors.astype(np.float32, copy=False)
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    d = d.reshape(n, 4, -1)
    independent = np.full((n, n), -np.inf, dtype=np.float32)
    cyclic_mean = np.full((n, n), -np.inf, dtype=np.float32)
    cyclic_minimum = np.full((n, n), -np.inf, dtype=np.float32)
    cyclic_support = np.zeros((n, n), dtype=np.uint8)
    for shift in range(4):
        total = np.zeros((n, n), dtype=np.float32)
        support = np.zeros((n, n), dtype=np.uint8)
        minimum = np.full((n, n), np.inf, dtype=np.float32)
        for view in range(4):
            similarity = d[:, view] @ d[:, (view + shift) % 4].T
            independent = np.maximum(independent, similarity)
            total += similarity
            minimum = np.minimum(minimum, similarity)
            support += similarity >= support_threshold
        mean = total * 0.25
        better = mean > cyclic_mean
        cyclic_mean[better] = mean[better]
        cyclic_minimum[better] = minimum[better]
        cyclic_support[better] = support[better]
    return independent, cyclic_mean, cyclic_minimum, cyclic_support


def evaluate_run(run: str, args: argparse.Namespace, inventory: dict) -> dict:
    descriptor_path = args.descriptor_root / run / "salad_descriptors.npz"
    with np.load(descriptor_path, allow_pickle=False) as cache:
        descriptors = np.asarray(cache["descriptors"])
        names = np.asarray(cache["image_names"])
        captures = np.asarray(cache["capture_indices"], dtype=np.int64)
        views = np.asarray(cache["view_indices"], dtype=np.int64)
    order = np.lexsort((views, captures))
    descriptors, names, captures, views = (
        descriptors[order], names[order], captures[order], views[order]
    )
    if not np.array_equal(views.reshape(-1, 4), np.tile(np.arange(4), (len(views) // 4, 1))):
        raise ValueError("missing or duplicate yaw view")
    capture_times = parse_capture_times(names, captures)
    gt_path = Path(inventory["ground_truth"]["trajectory"][run]["path"])
    gt_t, gt_p = load_gt_positions(gt_path)
    positions, valid_capture = interpolate_positions(
        capture_times, gt_t, gt_p, args.max_interpolation_gap_s
    )
    independent, cyclic_mean, cyclic_minimum, cyclic_support = score_matrices(
        descriptors, args.cyclic_support_threshold
    )
    n = len(capture_times)
    i, j = np.triu_indices(n, k=args.min_capture_gap)
    pair_valid = valid_capture[i] & valid_capture[j]
    i, j = i[pair_valid], j[pair_valid]
    distances = np.linalg.norm(positions[i] - positions[j], axis=1)
    labeled = (distances <= args.positive_radius_m) | (distances >= args.negative_radius_m)
    i, j, distances = i[labeled], j[labeled], distances[labeled]
    labels = distances <= args.positive_radius_m
    single_scores = independent[i, j]
    cyclic_scores = cyclic_mean[i, j]
    cyclic_min_scores = cyclic_minimum[i, j]
    support = cyclic_support[i, j]
    frozen_accept = (
        (cyclic_scores >= args.cyclic_mean_threshold)
        & (cyclic_min_scores >= args.cyclic_min_view_threshold)
        & (support >= args.cyclic_min_support)
    )
    single_accept = single_scores >= args.single_view_threshold
    result = {
        "schema_version": 1,
        "operation": "evaluation_only_salad_retrieval_ablation",
        "run": run,
        "inputs": {
            "descriptors": f"{run}/salad_descriptors.npz",
            "descriptor_sha256": sha256_file(descriptor_path),
            "ground_truth_trajectory": gt_path.name,
            "ground_truth_sha256": sha256_file(gt_path),
        },
        "protocol": {
            "views_per_capture": 4,
            "min_capture_gap": args.min_capture_gap,
            "positive_radius_m": args.positive_radius_m,
            "negative_radius_m": args.negative_radius_m,
            "ignored_distance_interval_m": [args.positive_radius_m, args.negative_radius_m],
            "gt_role": "offline pair labels only",
            "single_view_threshold": args.single_view_threshold,
            "cyclic_mean_threshold": args.cyclic_mean_threshold,
            "cyclic_min_view_threshold": args.cyclic_min_view_threshold,
            "cyclic_support_threshold": args.cyclic_support_threshold,
            "cyclic_min_support": args.cyclic_min_support,
        },
        "counts": {
            "captures": n,
            "gt_valid_captures": int(np.sum(valid_capture)),
            "labeled_pairs": len(labels),
            "positive_pairs": int(np.sum(labels)),
            "negative_pairs": int(np.sum(~labels)),
        },
        "independent_single_view_max": {
            "average_precision": average_precision(labels, single_scores),
            "recall_at_precision_99": recall_at_precision(labels, single_scores, 0.99),
            "threshold_gate": threshold_metrics(labels, single_accept),
        },
        "common_cyclic_yaw_mean": {
            "average_precision": average_precision(labels, cyclic_scores),
            "recall_at_precision_99": recall_at_precision(labels, cyclic_scores, 0.99),
        },
        "frozen_consensus_gate": threshold_metrics(labels, frozen_accept),
    }
    return result


def finite(values: list[float | None]) -> np.ndarray:
    return np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)


def summarize(rows: list[dict]) -> dict:
    methods = {
        "independent_single_view_max": "independent_single_view_max",
        "common_cyclic_yaw_mean": "common_cyclic_yaw_mean",
    }
    aggregate: dict[str, dict] = {}
    for label, key in methods.items():
        ap = finite([row[key]["average_precision"] for row in rows])
        r99 = finite([row[key]["recall_at_precision_99"] for row in rows])
        aggregate[label] = {
            "runs": int(len(ap)),
            "macro_median_average_precision": float(np.median(ap)),
            "macro_mean_average_precision": float(np.mean(ap)),
            "macro_median_recall_at_precision_99": float(np.median(r99)),
            "macro_mean_recall_at_precision_99": float(np.mean(r99)),
        }
    ap_pairs = [
        (r["independent_single_view_max"]["average_precision"], r["common_cyclic_yaw_mean"]["average_precision"])
        for r in rows
        if r["independent_single_view_max"]["average_precision"] is not None
        and r["common_cyclic_yaw_mean"]["average_precision"] is not None
    ]
    recall_pairs = [
        (r["independent_single_view_max"]["recall_at_precision_99"], r["common_cyclic_yaw_mean"]["recall_at_precision_99"])
        for r in rows
        if r["independent_single_view_max"]["recall_at_precision_99"] is not None
        and r["common_cyclic_yaw_mean"]["recall_at_precision_99"] is not None
    ]
    single_ap = np.asarray([pair[0] for pair in ap_pairs], dtype=np.float64)
    cyclic_ap = np.asarray([pair[1] for pair in ap_pairs], dtype=np.float64)
    single_r = np.asarray([pair[0] for pair in recall_pairs], dtype=np.float64)
    cyclic_r = np.asarray([pair[1] for pair in recall_pairs], dtype=np.float64)
    aggregate["paired_cyclic_minus_single"] = {
        "average_precision_median_delta": float(np.nanmedian(cyclic_ap - single_ap)),
        "average_precision_wins_ties_losses": [
            int(np.sum(cyclic_ap > single_ap)),
            int(np.sum(cyclic_ap == single_ap)),
            int(np.sum(cyclic_ap < single_ap)),
        ],
        "recall_at_precision_99_median_delta": float(np.nanmedian(cyclic_r - single_r)),
        "recall_at_precision_99_wins_ties_losses": [
            int(np.sum(cyclic_r > single_r)),
            int(np.sum(cyclic_r == single_r)),
            int(np.sum(cyclic_r < single_r)),
        ],
    }
    gate_rows = [r["frozen_consensus_gate"] for r in rows]
    tp = sum(r["true_positive"] for r in gate_rows)
    fp = sum(r["false_positive"] for r in gate_rows)
    fn = sum(r["false_negative"] for r in gate_rows)
    aggregate["frozen_consensus_gate_micro"] = {
        "accepted": sum(r["accepted"] for r in gate_rows),
        "true_positive": tp,
        "false_positive": fp,
        "precision": float(tp / (tp + fp)) if tp + fp else None,
        "recall": float(tp / (tp + fn)) if tp + fn else None,
        "note": "No GT labels influence the frozen threshold.",
    }
    return aggregate


def main() -> int:
    args = parse_args()
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    rows = []
    started = time.time()
    runs = [entry["relative_path"] for entry in universe["runs"]]
    for run in runs:
        output = args.output_root / run / "retrieval_metrics.json"
        row = evaluate_run(run, args, inventory)
        atomic_json(output, row)
        rows.append(row)
        print(json.dumps({"run": run, "status": "complete"}), flush=True)
    aggregate = {
        "schema_version": 1,
        "operation": "aggregate_salad_retrieval_ablation",
        "runs": len(rows),
        "runtime_s": time.time() - started,
        "summary": summarize(rows),
        "per_run": rows,
    }
    atomic_json(args.output_root / "aggregate.json", aggregate)
    print(json.dumps(aggregate["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
