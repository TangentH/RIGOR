#!/usr/bin/env python3
import argparse
import gc
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as SciRot

from depth_anything_3.api import DepthAnything3
from depth_anything_3.specs import Gaussians, Prediction
from depth_anything_3.utils.gsply_helpers import save_gaussian_ply
from depth_anything_3.utils.pose_align import align_poses_umeyama


SH_C0 = 0.28209479177387814


@dataclass
class ChunkData:
    gaussians: Gaussians
    depth: np.ndarray  # [V, H, W]
    kept_views: list[int]
    global_frame_ids: list[int]


def load_c2w_poses(path: str) -> np.ndarray:
    rows = []
    with open(path, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if not vals:
                continue
            rows.append(np.array(vals, dtype=np.float32).reshape(4, 4))
    return np.stack(rows, axis=0)


def load_intrinsics(path: str) -> np.ndarray:
    mats = []
    with open(path, "r") as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if not vals:
                continue
            fx, fy, cx, cy = vals
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
            mats.append(K)
    return np.stack(mats, axis=0)


def c2w_to_w2c(c2w: np.ndarray) -> np.ndarray:
    return np.linalg.inv(c2w)


def make_chunk_indices(n: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    out = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        out.append((start, end))
        if end == n:
            break
        start = end - overlap
    return out


def kept_local_indices(chunk_idx: int, num_chunks: int, local_len: int, overlap: int) -> list[int]:
    if chunk_idx == 0:
        return list(range(0, local_len - overlap))
    if chunk_idx == num_chunks - 1:
        return list(range(overlap, local_len))
    return list(range(overlap, local_len - overlap))


def rotation_matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    quat_xyzw = SciRot.from_matrix(rot).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.stack(
        [quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3], quat_wxyz[..., 0]],
        axis=-1,
    )


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.stack(
        [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
        axis=-1,
    )


def apply_sim3_to_gaussians(gaussians: Gaussians, scale: float, rot: np.ndarray, trans: np.ndarray) -> Gaussians:
    means = gaussians.means.clone()
    scales = gaussians.scales.clone()
    rotations = gaussians.rotations.clone()
    harmonics = gaussians.harmonics.clone()
    opacities = gaussians.opacities.clone()

    device = means.device
    dtype = means.dtype

    rot_t = torch.from_numpy(rot).to(device=device, dtype=dtype)
    trans_t = torch.from_numpy(trans).to(device=device, dtype=dtype)

    means = scale * torch.einsum("ij,bgj->bgi", rot_t, means) + trans_t.view(1, 1, 3)
    scales = scales * float(scale)

    global_quat_wxyz = rotation_matrix_to_quat_wxyz(rot)
    qg_xyzw = quat_wxyz_to_xyzw(global_quat_wxyz)
    q_local_xyzw = quat_wxyz_to_xyzw(rotations.detach().cpu().numpy())
    q_out_xyzw = (SciRot.from_quat(qg_xyzw) * SciRot.from_quat(q_local_xyzw.reshape(-1, 4))).as_quat()
    q_out_wxyz = quat_xyzw_to_wxyz(q_out_xyzw).reshape(rotations.shape)
    rotations = torch.from_numpy(q_out_wxyz).to(device=device, dtype=dtype)

    return Gaussians(
        means=means,
        scales=scales,
        rotations=rotations,
        harmonics=harmonics,
        opacities=opacities,
    )


def select_views_from_chunk(chunk: ChunkData) -> tuple[Gaussians, torch.Tensor]:
    v, h, w = chunk.depth.shape
    keep = np.array(chunk.kept_views, dtype=np.int64)
    base = chunk.gaussians

    def reshape_select(t: torch.Tensor) -> torch.Tensor:
        b, n = t.shape[:2]
        assert b == 1
        reshaped = t[0].reshape(v, h, w, *t.shape[2:])
        selected = reshaped[keep]
        return selected.reshape(1, -1, *t.shape[2:])

    selected = Gaussians(
        means=reshape_select(base.means),
        scales=reshape_select(base.scales),
        rotations=reshape_select(base.rotations),
        harmonics=reshape_select(base.harmonics),
        opacities=reshape_select(base.opacities),
    )
    depth = torch.from_numpy(chunk.depth[keep]).unsqueeze(-1)
    return selected, depth


def concat_gaussians(items: list[Gaussians]) -> Gaussians:
    return Gaussians(
        means=torch.cat([x.means for x in items], dim=1),
        scales=torch.cat([x.scales for x in items], dim=1),
        rotations=torch.cat([x.rotations for x in items], dim=1),
        harmonics=torch.cat([x.harmonics for x in items], dim=1),
        opacities=torch.cat([x.opacities for x in items], dim=1),
    )


def export_preview_pointcloud(gaussians: Gaussians, save_path: str, max_points: int = 1_000_000) -> None:
    means = gaussians.means[0].detach().cpu().numpy()
    f_dc = gaussians.harmonics[0, :, :, 0].detach().cpu().numpy()
    rgb = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)
    rgb8 = (rgb * 255.0).astype(np.uint8)

    if means.shape[0] > max_points:
        idx = np.linspace(0, means.shape[0] - 1, max_points, dtype=np.int64)
        means = means[idx]
        rgb8 = rgb8[idx]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {means.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(means, rgb8):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def run_chunk(
    model: DepthAnything3,
    image_paths: list[str],
    input_exts: np.ndarray,
    input_intrs: np.ndarray,
    process_res: int,
    process_res_method: str,
    ref_view_strategy: str,
) -> Prediction:
    return model.inference(
        image=image_paths,
        extrinsics=input_exts,
        intrinsics=input_intrs,
        align_to_input_ext_scale=False,
        infer_gs=True,
        ref_view_strategy=ref_view_strategy,
        process_res=process_res,
        process_res_method=process_res_method,
        export_dir=None,
        export_format="mini_npz",
    )


def main():
    parser = argparse.ArgumentParser(description="Experimental front-view GS streaming export")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--poses_txt", required=True, help="camera_poses.txt from streaming output (c2w)")
    parser.add_argument("--intrinsic_txt", required=True, help="intrinsic.txt from streaming output")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--chunk_size", type=int, default=30)
    parser.add_argument("--overlap", type=int, default=15)
    parser.add_argument("--process_res", type=int, default=336)
    parser.add_argument("--process_res_method", default="lower_bound_resize")
    parser.add_argument("--ref_view_strategy", default="saddle_balanced")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    image_paths = sorted(
        [str(p) for p in Path(args.image_dir).glob("*.jpg")]
        + [str(p) for p in Path(args.image_dir).glob("*.png")]
    )
    if args.max_frames > 0:
        image_paths = image_paths[: args.max_frames]
    if not image_paths:
        raise ValueError(f"No images found in {args.image_dir}")

    print(f"Loaded {len(image_paths)} images from {args.image_dir}", flush=True)
    c2w = load_c2w_poses(args.poses_txt)[: len(image_paths)]
    intrs = load_intrinsics(args.intrinsic_txt)[: len(image_paths)]
    w2c = c2w_to_w2c(c2w).astype(np.float32)
    print(f"Loaded {len(c2w)} poses and {len(intrs)} intrinsics", flush=True)

    print(f"Loading model from {args.model_dir}", flush=True)
    model = DepthAnything3.from_pretrained(args.model_dir).to(args.device)
    model.eval()
    print("Model loaded", flush=True)

    chunk_ranges = make_chunk_indices(len(image_paths), args.chunk_size, args.overlap)
    print(f"Processing in {len(chunk_ranges)} chunks", flush=True)
    kept_chunks: list[ChunkData] = []
    selected_depths: list[torch.Tensor] = []
    selected_gs: list[Gaussians] = []

    os.makedirs(args.output_dir, exist_ok=True)

    for chunk_idx, (start, end) in enumerate(chunk_ranges):
        print(f"[Chunk {chunk_idx+1}/{len(chunk_ranges)}] frames {start}:{end}")
        chunk_images = image_paths[start:end]
        chunk_exts = w2c[start:end]
        chunk_intrs = intrs[start:end]

        pred = run_chunk(
            model=model,
            image_paths=chunk_images,
            input_exts=chunk_exts,
            input_intrs=chunk_intrs,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            ref_view_strategy=args.ref_view_strategy,
        )
        if pred.gaussians is None:
            raise RuntimeError("Prediction did not contain gaussians. Check model_dir and infer_gs.")

        pred_ext = pred.extrinsics
        if pred_ext.shape[-2:] == (3, 4):
            pad = np.tile(np.eye(4, dtype=np.float32)[None], (pred_ext.shape[0], 1, 1))
            pad[:, :3, :4] = pred_ext
            pred_ext = pad

        rot, trans, scale = align_poses_umeyama(
            ext_ref=chunk_exts,
            ext_est=pred_ext,
            return_aligned=False,
            ransac=len(chunk_images) >= 10,
            random_state=42,
        )
        print(f"  alignment scale={scale:.6f}")

        aligned_gs = apply_sim3_to_gaussians(pred.gaussians, scale, rot, trans)
        aligned_depth = pred.depth / scale

        keep = kept_local_indices(chunk_idx, len(chunk_ranges), end - start, args.overlap)
        chunk_data = ChunkData(
            gaussians=aligned_gs,
            depth=aligned_depth,
            kept_views=keep,
            global_frame_ids=list(range(start, end)),
        )
        kept_chunks.append(chunk_data)

        selected_chunk_gs, selected_chunk_depth = select_views_from_chunk(chunk_data)
        selected_gs.append(selected_chunk_gs)
        selected_depths.append(selected_chunk_depth)

        del pred, aligned_gs, selected_chunk_gs, selected_chunk_depth
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined_gs = concat_gaussians(selected_gs)
    combined_depth = torch.cat(selected_depths, dim=0)

    gs_path = os.path.join(args.output_dir, "combined_gs_asset.ply")
    save_gaussian_ply(
        gaussians=combined_gs,
        save_path=gs_path,
        ctx_depth=combined_depth,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=1,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )
    print(f"Saved Gaussian asset to {gs_path}")

    preview_path = os.path.join(args.output_dir, "combined_gs_preview_rgb_points.ply")
    export_preview_pointcloud(combined_gs, preview_path)
    print(f"Saved RGB preview point cloud to {preview_path}")


if __name__ == "__main__":
    main()
