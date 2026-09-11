#!/usr/bin/env python3
"""
Extract equirectangular frames from Hilti-Trimble ROS2 bag (.db3) without ROS2.

Synchronizes cam0 + cam1 fisheye images, stitches them into equirectangular
images using the Kannala-Brandt projection math from the challenge tools.

Single bag:
    python extract_hilti_frames.py \
        --bag path/to/file/rosbag.db3 \
        --yaml hilti-trimble-slam-challenge-2026/config/hilti_openvins/kalibr_imucam_chain.yaml \
        --out_dir path/to/file/equirect_frames/ \
        --mask0 hilti-trimble-slam-challenge-2026/config/hilti_openvins/mask_cam0.png \
        --mask1 hilti-trimble-slam-challenge-2026/config/hilti_openvins/mask_cam1.png \
        --stride 10 \
        --max_frames 50

Entire dataset:
    python extract_hilti_frames.py \
        --data_dir /path/to/hilti_challenge/data \
        --yaml hilti-trimble-slam-challenge-2026/config/hilti_openvins/kalibr_imucam_chain.yaml \
        --mask0 hilti-trimble-slam-challenge-2026/config/hilti_openvins/mask_cam0.png \
        --mask1 hilti-trimble-slam-challenge-2026/config/hilti_openvins/mask_cam1.png \
"""

import argparse
import math
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

JPEG_SOI = bytes([0xFF, 0xD8, 0xFF])
SYNC_TOLERANCE_NS = 5_000_000  # 5 ms


# ── Calibration helpers (ported from image_stitching.py) ─────────────────────

def _read_yaml(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin1")
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("%YAML"):
        lines = lines[1:]
    return "\n".join(lines)


def load_camera_yaml(path: str) -> Tuple[dict, dict]:
    data = yaml.safe_load(_read_yaml(path))
    return data["cam0"], data["cam1"]


def parse_intrinsics(cam: dict) -> Dict[str, Any]:
    model = str(cam.get("camera_model", "")).strip().lower()
    resolution = cam.get("resolution")
    w, h = int(resolution[0]), int(resolution[1])
    intr = cam["intrinsics"]

    if model.startswith("eucm"):
        alpha, beta, fx, fy, cx, cy = (float(x) for x in intr[:6])
        fov_deg = float(cam.get("fov_deg", 190.0))
        return dict(model="eucm", alpha=alpha, beta=beta, fx=fx, fy=fy, cx=cx, cy=cy,
                    w=w, h=h, k=np.zeros(0), max_theta=math.radians(fov_deg / 2.0))

    fx, fy, cx, cy = (float(x) for x in intr[:4])
    d = cam.get("distortion_coeffs") or []
    k = np.array(d, dtype=np.float64)
    fov_deg = float(cam.get("fov_deg", 195.0))
    return dict(model="kb", fx=fx, fy=fy, cx=cx, cy=cy, w=w, h=h, k=k,
                fov_deg=fov_deg, max_theta=math.radians(fov_deg / 2.0))


def resolve_extrinsics(cam0: dict, cam1: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return R01, t01 (cam1 relative to cam0, via IMU)."""
    T_stitch = cam1.get("T_stitch")
    if T_stitch is not None:
        T = np.array(T_stitch, dtype=np.float64)
        if T.shape == (4, 4):
            return T[:3, :3], T[:3, 3]

    def get_T(cam):
        T = cam.get("T_cam_imu")
        return np.array(T, dtype=np.float64) if T is not None else None

    T0, T1 = get_T(cam0), get_T(cam1)
    if T0 is not None and T1 is not None:
        T_rel = T1 @ np.linalg.inv(T0)
        return T_rel[:3, :3], T_rel[:3, 3]

    return np.eye(3), np.zeros(3)


# ── Projection math ───────────────────────────────────────────────────────────
def _rho_from_theta(theta: np.ndarray, k: np.ndarray) -> np.ndarray:
    rho = theta.copy()
    if k.size == 0:
        return rho
    t2 = theta * theta
    t_pow = theta * t2
    for ki in k.flat:
        rho = rho + ki * t_pow
        t_pow = t_pow * t2
    return rho


def dirs_to_pixels_kb(dirs: np.ndarray, K: dict):
    X, Y, Z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    rho_xy = np.hypot(X, Y)
    theta = np.arctan2(rho_xy, Z)
    f = 0.5 * (K["fx"] + K["fy"])
    rho_px = f * _rho_from_theta(theta, K["k"])
    nz = rho_xy > 1e-12
    ux, uy = np.zeros_like(rho_xy), np.zeros_like(rho_xy)
    ux[nz] = X[nz] / rho_xy[nz]
    uy[nz] = Y[nz] / rho_xy[nz]
    return K["cx"] + ux * rho_px, K["cy"] + uy * rho_px, theta


def dirs_to_pixels_eucm(dirs: np.ndarray, K: dict):
    X, Y, Z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    norm = np.sqrt(X**2 + Y**2 + Z**2)
    xi = float(K["alpha"])
    denom = np.where(np.abs(Z + xi * norm) < 1e-12, 1e-12, Z + xi * norm)
    theta = np.arctan2(np.hypot(X, Y), Z)
    return K["fx"] * X / denom + K["cx"], K["fy"] * Y / denom + K["cy"], theta


def build_equirect_dirs(out_w: int, out_h: int) -> np.ndarray:
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) / out_w
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) / out_h
    lon = xs[None, :] * 2 * np.pi - np.pi          # (1, W)
    lat = ys[:, None] * np.pi - (np.pi / 2)        # (H, 1)
    lon = np.broadcast_to(lon, (out_h, out_w))
    lat = np.broadcast_to(lat, (out_h, out_w))
    cos_lat = np.cos(lat)
    X = cos_lat * np.sin(lon)
    Y = np.sin(lat)
    Z = cos_lat * np.cos(lon)
    return np.stack([X, Y, Z], axis=-1).reshape(-1, 3)


def build_remap(K: dict, dirs: np.ndarray, out_h: int, out_w: int, sphere_r: float):
    pts = dirs * sphere_r
    proj = dirs_to_pixels_eucm if K["model"] == "eucm" else dirs_to_pixels_kb
    u, v, theta = proj(pts.astype(np.float64), K)
    W, H = int(K["w"]), int(K["h"])
    map_x = u.reshape(out_h, out_w).astype(np.float32)
    map_y = v.reshape(out_h, out_w).astype(np.float32)
    valid = (
        (theta <= K["max_theta"] + 1e-9).reshape(out_h, out_w)
        & (map_x >= 0) & (map_x < W - 1)
        & (map_y >= 0) & (map_y < H - 1)
    )
    return cv2.convertMaps(map_x, map_y, cv2.CV_16SC2), map_x, map_y, valid


def compute_blend_weights(
    valid0: np.ndarray, dist0: np.ndarray,
    valid1: np.ndarray, dist1: np.ndarray,
    mask0: Optional[np.ndarray], mask1: Optional[np.ndarray],
    maps0, maps1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    (m0_1, m0_2), (m1_1, m1_2) = maps0, maps1

    def apply_mask(valid, map1, map2, mask):
        if mask is None:
            return valid
        remapped = cv2.remap(mask, map1, map2, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        return valid & (remapped != 255)

    v0 = apply_mask(valid0, m0_1, m0_2, mask0)
    v1 = apply_mask(valid1, m1_1, m1_2, mask1)

    eps = 1e-6
    d0 = np.where(v0, np.maximum(dist0, eps), 0.0)
    d1 = np.where(v1, np.maximum(dist1, eps), 0.0)
    inv0 = np.divide(1.0, d0, where=d0 > 0, out=np.zeros_like(d0))
    inv1 = np.divide(1.0, d1, where=d1 > 0, out=np.zeros_like(d1))
    wsum = inv0 + inv1 + 1e-12
    w0 = (inv0 / wsum).astype(np.float32)
    w1 = (inv1 / wsum).astype(np.float32)
    mask_none = (inv0 + inv1) == 0
    return w0, w1, mask_none


def stitch(img0: np.ndarray, img1: np.ndarray,
           maps0, maps1, w0, w1, mask_none) -> np.ndarray:
    (m0_1, m0_2), (m1_1, m1_2) = maps0, maps1

    def remap(src, map1, map2):
        return cv2.remap(src, map1, map2, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    with ThreadPoolExecutor(max_workers=2) as ex:
        f0 = ex.submit(remap, img0, m0_1, m0_2)
        f1 = ex.submit(remap, img1, m1_1, m1_2)
        r0, r1 = f0.result(), f1.result()

    out = r0.astype(np.float32) * w0[..., None] + r1.astype(np.float32) * w1[..., None]
    out[mask_none] = 0.0
    return np.clip(out, 0, 255).astype(np.uint8)


# ── SQLite bag reader ─────────────────────────────────────────────────────────

def get_topic_id(conn: sqlite3.Connection, name: str) -> Optional[int]:
    row = conn.execute("SELECT id FROM topics WHERE name=?", (name,)).fetchone()
    return row[0] if row else None


def find_jpeg(data: bytes) -> Optional[bytes]:
    start = data.find(JPEG_SOI)
    return data[start:] if start != -1 else None


def extract_pairs(conn: sqlite3.Connection, tid0: int, tid1: int):
    """Yield synchronized (ts, img0_bytes, img1_bytes) pairs."""
    cur0 = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid0,))
    cur1 = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid1,))

    buf0 = cur0.fetchone()
    buf1 = cur1.fetchone()

    while buf0 and buf1:
        t0, d0 = buf0
        t1, d1 = buf1
        dt = t0 - t1

        if abs(dt) <= SYNC_TOLERANCE_NS:
            yield t0, bytes(d0) if isinstance(d0, memoryview) else d0, \
                       bytes(d1) if isinstance(d1, memoryview) else d1
            buf0 = cur0.fetchone()
            buf1 = cur1.fetchone()
        elif dt < 0:   # t0 is earlier, advance cam0
            buf0 = cur0.fetchone()
        else:           # t1 is earlier, advance cam1
            buf1 = cur1.fetchone()


# ── Per-bag extraction ────────────────────────────────────────────────────────

def process_bag(bag_path: str, out_dir: str, maps0_cv, maps1_cv,
                w0, w1, mask_none, stride: int, max_frames: int, progress_interval_s: float):
    os.makedirs(out_dir, exist_ok=True)

    conn = sqlite3.connect(bag_path)
    tid0 = get_topic_id(conn, "/cam0/image_raw/compressed")
    tid1 = get_topic_id(conn, "/cam1/image_raw/compressed")
    if tid0 is None or tid1 is None:
        print(f"  WARNING: camera topics not found in {bag_path}, skipping.")
        conn.close()
        return

    cam0_count = conn.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (tid0,)).fetchone()[0]
    cam1_count = conn.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (tid1,)).fetchone()[0]
    approx_pairs = min(int(cam0_count), int(cam1_count))
    expected_saved = approx_pairs // max(1, stride)
    if max_frames > 0:
        expected_saved = min(expected_saved, max_frames)
    print(
        f"  Camera messages: cam0={cam0_count}, cam1={cam1_count}, "
        f"approx_sync_pairs<={approx_pairs}, expected_saved~{expected_saved}",
        flush=True,
    )

    saved = total = 0
    start_time = last_report = time.monotonic()

    def report(force: bool = False) -> None:
        nonlocal last_report
        now = time.monotonic()
        if not force and progress_interval_s > 0 and now - last_report < progress_interval_s:
            return
        elapsed = max(now - start_time, 1e-6)
        pair_rate = total / elapsed
        save_rate = saved / elapsed
        pct = (100.0 * saved / expected_saved) if expected_saved > 0 else 0.0
        print(
            f"    [{os.path.basename(out_dir)}] scanned_pairs={total}/{approx_pairs} "
            f"saved={saved}/{expected_saved} ({pct:.1f}%) "
            f"pair_rate={pair_rate:.1f}/s save_rate={save_rate:.1f}/s elapsed={elapsed:.1f}s",
            flush=True,
        )
        last_report = now

    report(force=True)
    for ts, raw0, raw1 in extract_pairs(conn, tid0, tid1):
        total += 1
        report()
        if (total - 1) % stride != 0:
            continue

        jpeg0 = find_jpeg(raw0)
        jpeg1 = find_jpeg(raw1)
        if jpeg0 is None or jpeg1 is None:
            continue

        arr0 = np.frombuffer(jpeg0, np.uint8)
        arr1 = np.frombuffer(jpeg1, np.uint8)
        img0 = cv2.imdecode(arr0, cv2.IMREAD_COLOR)
        img1 = cv2.imdecode(arr1, cv2.IMREAD_COLOR)
        if img0 is None or img1 is None:
            continue

        equirect = stitch(img0, img1, maps0_cv, maps1_cv, w0, w1, mask_none)
        out_path = os.path.join(out_dir, f"frame_{saved:05d}_{ts}.jpg")
        cv2.imwrite(out_path, equirect, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1

        if saved % 10 == 0:
            report(force=True)
        if max_frames > 0 and saved >= max_frames:
            break

    conn.close()
    report(force=True)
    print(f"  Done: {saved} frames saved → {out_dir}", flush=True)
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract equirectangular frames from Hilti ROS2 bag(s)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bag", help="Path to a single rosbag.db3")
    group.add_argument("--data_dir", help="Root data directory; all rosbag.db3 files inside are processed")
    parser.add_argument("--yaml", required=True, help="kalibr_imucam_chain.yaml path")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory. Required with --bag. "
                             "With --data_dir, defaults to equirect_frames/ next to each rosbag folder.")
    parser.add_argument("--mask0", default=None)
    parser.add_argument("--mask1", default=None)
    parser.add_argument("--stride", type=int, default=1,
                        help="Save every Nth synchronized pair (default: 1 → ~30 fps)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames per bag (0 = unlimited)")
    parser.add_argument("--sphere_m", type=float, default=10.0,
                        help="Virtual sphere radius in metres (default: 10)")
    parser.add_argument("--progress_interval_s", type=float, default=5.0,
                        help="Print extraction heartbeat at least this often in seconds")
    args = parser.parse_args()

    # Load calibration
    cam0_cfg, cam1_cfg = load_camera_yaml(args.yaml)
    K0 = parse_intrinsics(cam0_cfg)
    K1 = parse_intrinsics(cam1_cfg)
    R01, t01 = resolve_extrinsics(cam0_cfg, cam1_cfg)

    # Output resolution: 2× min width, half height
    out_w = max(1024, 2 * min(int(K0["w"]), int(K1["w"])))
    out_h = out_w // 2
    print(f"Equirectangular resolution: {out_w}×{out_h}")

    # Build projection maps once (shared across all bags)
    print("Precomputing projection maps...")
    dirs = build_equirect_dirs(out_w, out_h).astype(np.float64)
    pts_cam1 = (dirs * args.sphere_m - t01) @ R01.T

    maps0_cv, mx0, my0, valid0 = build_remap(K0, dirs, out_h, out_w, args.sphere_m)
    maps1_cv, mx1, my1, valid1 = build_remap(K1, pts_cam1, out_h, out_w, args.sphere_m)

    dist0 = np.linalg.norm(dirs * args.sphere_m, axis=1).reshape(out_h, out_w)
    dist1 = np.linalg.norm(pts_cam1, axis=1).reshape(out_h, out_w)

    def load_mask(path, K):
        if path is None:
            return None
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cv2.resize(m, (int(K["w"]), int(K["h"])), interpolation=cv2.INTER_NEAREST) if m is not None else None

    mask0 = load_mask(args.mask0, K0)
    mask1 = load_mask(args.mask1, K1)

    w0, w1, mask_none = compute_blend_weights(
        valid0, dist0, valid1, dist1, mask0, mask1, maps0_cv, maps1_cv)

    # Collect bags to process
    if args.bag:
        if args.out_dir is None:
            parser.error("--out_dir is required when using --bag")
        bags: List[Tuple[str, str]] = [(args.bag, args.out_dir)]
    else:
        data_root = Path(args.data_dir)
        bags = []
        for bag_path in sorted(data_root.rglob("rosbag.db3")):
            run_dir = bag_path.parent.parent  # .../run_N/ (parent of rosbag/)
            if args.out_dir:
                rel = run_dir.relative_to(data_root)
                out_dir = str(Path(args.out_dir) / rel)
            else:
                out_dir = str(run_dir / "equirect_frames")
            bags.append((str(bag_path), out_dir))
        print(f"Found {len(bags)} bags under {args.data_dir}")

    # Process each bag
    for i, (bag_path, out_dir) in enumerate(bags, 1):
        print(f"\n[{i}/{len(bags)}] {bag_path}")
        process_bag(bag_path, out_dir, maps0_cv, maps1_cv,
                    w0, w1, mask_none, args.stride, args.max_frames, args.progress_interval_s)

    print("\nAll done.")


if __name__ == "__main__":
    main()
