#!/usr/bin/env python3
"""Run sequential PanoVGGT reconstruction on an arbitrary HILTI ERP sequence.

Overlapping inference windows are joined with Sim(3) alignment. No ground
truth is consumed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hilti_workflow.panovggt import geometry as h  # noqa: E402
from tools.hilti_workflow.run_hilti_batch import write_camera_centers_ply  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--erp-mask-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--points-per-frame", type=int, default=40000)
    parser.add_argument(
        "--window-checkpoint-dir",
        type=Path,
        help=(
            "Directory for restart-safe sequential-window predictions. "
            "Defaults to <output-dir>/_window_checkpoints."
        ),
    )
    return parser.parse_args()


def load_window_checkpoint(path: Path, start: int, stop: int, stems: list[str]):
    """Load one complete prediction cache, or return None if it is unusable."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as bundle:
            cached_start = int(bundle["start"])
            cached_stop = int(bundle["stop"])
            cached_stems = [str(value) for value in bundle["stems"]]
            poses = np.asarray(bundle["camera_poses"], np.float64)
            world = np.asarray(bundle["world_points"], np.float32)
        expected = stop - start
        if (
            cached_start != start
            or cached_stop != stop
            or cached_stems != stems
            or poses.shape[0] != expected
            or world.shape[0] != expected
            or poses.ndim != 3
            or poses.shape[-2:] != (4, 4)
            or world.ndim != 4
            or world.shape[-1] != 3
            or not np.isfinite(poses).all()
            or not np.isfinite(world).all()
        ):
            return None
        return poses, world
    except (OSError, KeyError, TypeError, ValueError):
        return None


def write_window_checkpoint(
    path: Path,
    start: int,
    stop: int,
    stems: list[str],
    poses: np.ndarray,
    world: np.ndarray,
) -> None:
    """Atomically publish a complete prediction cache for one window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            start=np.asarray(start, np.int64),
            stop=np.asarray(stop, np.int64),
            stems=np.asarray(stems),
            camera_poses=np.asarray(poses, np.float64),
            world_points=np.asarray(world, np.float32),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.window_checkpoint_dir or args.output_dir / "_window_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    official = h.load_official(args.repo)
    image_paths = [Path(path) for path in official.collect_images(str(args.image_dir))]
    if len(image_paths) < 2:
        raise ValueError(f"Need at least two ERP frames, got {len(image_paths)}")
    erp_masks = h.load_packed_masks(args.erp_mask_npz, image_paths)
    model = official.load_model(str(args.repo / "training/config/default.yaml"), str(args.checkpoint), "cuda")
    model.eval().to("cuda")
    starts = h.window_starts(len(image_paths), args.chunk_size, args.stride)
    relative, sequential_rows = [], []
    samples, colours, sample_nodes = [], [], []
    local_poses = [None] * len(image_paths)
    frame_pose_nodes = np.full(len(image_paths), -1, dtype=np.int32)
    previous_poses = None
    started = time.time()
    cached_windows = 0

    for node, start in enumerate(starts):
        stop = min(start + args.chunk_size, len(image_paths))
        tick = time.time()
        stems = [path.name for path in image_paths[start:stop]]
        checkpoint_path = checkpoint_dir / f"window_{node:04d}_{start:06d}_{stop:06d}.npz"
        cached = load_window_checkpoint(checkpoint_path, start, stop, stems)
        prediction = None
        if cached is None:
            prediction = official.run_inference(
                model, [str(path) for path in image_paths[start:stop]], "cuda"
            )
            poses = np.asarray(prediction["camera_poses"], np.float64)
            world = np.asarray(
                prediction.get("world_points", prediction.get("points")), np.float32
            )
            write_window_checkpoint(checkpoint_path, start, stop, stems, poses, world)
        else:
            poses, world = cached
            cached_windows += 1
        if node == 0:
            residual = np.zeros(0)
        else:
            overlap = previous_stop - start
            transform, residual = h.estimate_pose_sim3(poses[:overlap], previous_poses[-overlap:])
            relative.append(transform)
        emit_start = start if node == 0 else start + (previous_stop - start)
        height, width = world.shape[1:3]
        for local, frame in enumerate(range(start, stop)):
            if frame < emit_start:
                continue
            ids = h.select_pixels(world[local], args.points_per_frame, 20260818 + frame, erp_masks[frame])
            colour = h.frame_colour(image_paths[frame], height, width).reshape(-1, 3)[ids]
            samples.append(world[local].reshape(-1, 3)[ids].copy())
            colours.append(colour.copy())
            sample_nodes.append(node)
            local_poses[frame] = poses[local].copy()
            frame_pose_nodes[frame] = node
        sequential_rows.append({
            "node": node, "start": start, "stop": stop,
            "overlap": 0 if node == 0 else previous_stop - start,
            "relative_scale": 1.0 if node == 0 else float(relative[-1][0]),
            "alignment_median": float(np.median(residual)) if len(residual) else 0.0,
            "alignment_p95": float(np.quantile(residual, 0.95)) if len(residual) else 0.0,
            "runtime_s": time.time() - tick,
        })
        previous_poses, previous_stop = poses, stop
        del prediction, world
        torch.cuda.empty_cache()
        source = "checkpoint" if cached is not None else "inference"
        print(
            f"[sequential {node + 1}/{len(starts)}] {start}:{stop} source={source}",
            flush=True,
        )

    if any(item is None for item in local_poses) or np.any(frame_pose_nodes < 0):
        raise RuntimeError("Some input frames were not emitted")
    del model
    torch.cuda.empty_cache()
    transforms = h.accumulate(relative)
    sequential_output = h.export_variant(
        args.output_dir,
        "sequential",
        samples,
        colours,
        sample_nodes,
        local_poses,
        frame_pose_nodes,
        transforms,
    )
    np.savez_compressed(
        args.output_dir / "chunk_sim3_sequential.npz",
        scales=np.asarray([item[0] for item in transforms]),
        rotations=np.asarray([item[1] for item in transforms]),
        translations=np.asarray([item[2] for item in transforms]),
        starts=np.asarray(starts),
    )
    with (args.output_dir / "sequential_alignment.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sequential_rows[0])); writer.writeheader(); writer.writerows(sequential_rows)
    final_ply = args.output_dir / "reconstruction.ply"
    final_poses = args.output_dir / "camera_poses_c2w.npy"
    shutil.copy2(sequential_output["ply"], final_ply)
    shutil.copy2(sequential_output["poses"], final_poses)
    final_poses_txt = args.output_dir / "camera_poses.txt"
    final_poses_ply = args.output_dir / "camera_poses.ply"
    poses = np.load(final_poses, allow_pickle=False)
    final_poses_txt.write_text(
        "\n".join(
            " ".join(f"{value:.10g}" for value in pose.reshape(-1))
            for pose in poses
        ) + "\n",
        encoding="utf-8",
    )
    write_camera_centers_ply(final_poses_txt, final_poses_ply)
    report = {
        "experiment": "PanoVGGT long-sequence reconstruction",
        "input": {
            "erp_frames": len(image_paths),
            "image_dir": str(args.image_dir),
            "mask_npz": str(args.erp_mask_npz),
            "mask_fraction_mean": float(erp_masks.mean()),
            "mask_stage": "point validity/export only; backbone unchanged",
        },
        "model": {
            "repo": str(args.repo),
            "checkpoint": str(args.checkpoint),
            "chunk_size": args.chunk_size,
            "stride": args.stride,
        },
        "density": {"points_per_frame_cap": args.points_per_frame},
        "recovery": {
            "window_checkpoint_dir": str(checkpoint_dir),
            "windows_total": len(starts),
            "windows_loaded": cached_windows,
            "cache_format": "full sequential prediction; atomic NPZ",
        },
        "outputs": {
            "sequential": sequential_output,
            "final_pointcloud": str(final_ply),
            "final_camera_poses_npy": str(final_poses),
            "final_camera_poses_txt": str(final_poses_txt),
            "final_camera_poses_ply": str(final_poses_ply),
            "selected_variant": "sequential",
        },
        "runtime_s": time.time() - started,
        "notes": ["No ground truth used."],
    }
    (args.output_dir / "workflow_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
