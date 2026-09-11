#!/usr/bin/env python3
import argparse
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import yaml

try:
    from turbojpeg import TurboJPEG
except Exception:
    TurboJPEG = None

try:
    from rosbag2_py import (
        ConverterOptions,
        SequentialReader,
        SequentialWriter,
        StorageOptions,
        TopicMetadata,
    )
    from rclpy.serialization import deserialize_message, serialize_message
    from sensor_msgs.msg import CompressedImage
except Exception as e:
    print("ERROR: Could not import ROS2 Python modules. Run inside a ROS2 Python environment.")
    raise


def _read_text_file_flexible(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin1")
    lines = text.splitlines()
    if lines and lines[0].lstrip().startswith("%YAML"):
        lines = lines[1:]
        text = "\n".join(lines)
    return text


def load_camera_yaml(path: str) -> Tuple[dict, dict]:
    data = yaml.safe_load(_read_text_file_flexible(path))
    if not isinstance(data, dict):
        raise RuntimeError(f"YAML file '{path}' did not parse to a mapping.")
    cam0 = data.get("cam0")
    cam1 = data.get("cam1")
    if cam0 is None or cam1 is None:
        raise RuntimeError("YAML must contain top-level 'cam0' and 'cam1'.")
    return cam0, cam1


def parse_intrinsics(cam: dict) -> Dict[str, Any]:
    model = str(cam.get("camera_model", "")).strip().lower()
    resolution = cam.get("resolution", None)
    w, h = (int(resolution[0]), int(resolution[1])) if resolution is not None else (None, None)
    intr = cam.get("intrinsics", None)
    if intr is None:
        raise ValueError("intrinsics missing from YAML camera entry")

    if model.startswith("eucm"):
        if len(intr) < 6:
            raise ValueError("EUCM intrinsics must be [alpha, beta, fx, fy, cx, cy]")
        alpha = float(intr[0])
        beta = float(intr[1])
        fx = float(intr[2])
        fy = float(intr[3])
        cx = float(intr[4])
        cy = float(intr[5])
        fov_deg = float(cam.get("fov_deg", 190.0))
        return {
            "model": "eucm",
            "alpha": alpha,
            "beta": beta,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "k": np.zeros(0, dtype=np.float64),
            "max_theta": math.radians(fov_deg / 2.0),
        }

    if len(intr) < 4:
        raise ValueError("KB intrinsics must be [fx, fy, cx, cy]")
    d = cam.get("distortion_coeffs", None)
    k = np.array(d, dtype=np.float64) if d is not None else np.zeros(0, dtype=np.float64)
    fov_deg = float(cam.get("fov_deg", 195.0))
    return {
        "model": "kb",
        "fx": float(intr[0]),
        "fy": float(intr[1]),
        "cx": float(intr[2]),
        "cy": float(intr[3]),
        "w": w,
        "h": h,
        "k": k,
        "fov_deg": fov_deg,
        "max_theta": math.radians(fov_deg / 2.0),
    }


def parse_T_cam_imu(cam: dict) -> Optional[np.ndarray]:
    T = cam.get("T_cam_imu", None)
    if T is None:
        return None
    arr = np.array(T, dtype=np.float64)
    if arr.shape != (4, 4):
        return None
    return arr


def resolve_extrinsics(cam0: dict, cam1: dict) -> Tuple[np.ndarray, np.ndarray]:
    T_rel_cfg = cam1.get("T_stitch", None)
    if T_rel_cfg is not None:
        try:
            T_rel = np.array(T_rel_cfg, dtype=np.float64)
            if T_rel.shape == (4, 4):
                return T_rel[0:3, 0:3], T_rel[0:3, 3]
            raise ValueError(f"T_stitch shape is {T_rel.shape}, expected (4,4).")
        except Exception as e:
            print(f"Warning: Invalid cam1.T_stitch ({e}). Falling back to T_cam_imu.")

    T0 = parse_T_cam_imu(cam0)
    T1 = parse_T_cam_imu(cam1)
    if T0 is not None and T1 is not None:
        T_rel = T1 @ np.linalg.inv(T0)
        return T_rel[0:3, 0:3], T_rel[0:3, 3]

    print("Warning: No valid T_stitch or T_cam_imu found, using identity extrinsics.")
    return np.eye(3), np.zeros(3)


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


def dirs_to_fisheye_pixels_kb(dirs: np.ndarray, K: Dict[str, Any]):
    X = dirs[:, 0].astype(np.float64)
    Y = dirs[:, 1].astype(np.float64)
    Z = dirs[:, 2].astype(np.float64)
    rho_xy = np.hypot(X, Y)
    theta = np.arctan2(rho_xy, Z)
    f = 0.5 * (K["fx"] + K["fy"])
    rho_pixels = f * _rho_from_theta(theta, K.get("k", np.zeros(0, dtype=np.float64)))
    nz = rho_xy > 1e-12
    ux = np.zeros_like(rho_xy)
    uy = np.zeros_like(rho_xy)
    ux[nz] = X[nz] / rho_xy[nz]
    uy[nz] = Y[nz] / rho_xy[nz]
    u = K["cx"] + ux * rho_pixels
    v = K["cy"] + uy * rho_pixels
    return u, v, theta


def dirs_to_eucm_pixels(dirs: np.ndarray, K: Dict[str, Any]):
    X = dirs[:, 0].astype(np.float64)
    Y = dirs[:, 1].astype(np.float64)
    Z = dirs[:, 2].astype(np.float64)
    norm = np.sqrt(X * X + Y * Y + Z * Z)
    xi = float(K.get("alpha", 0.0))
    denom = Z + xi * norm
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    xprime = X / denom
    yprime = Y / denom
    u = K["fx"] * xprime + K["cx"]
    v = K["fy"] * yprime + K["cy"]
    theta = np.arctan2(np.hypot(X, Y), Z)
    return u, v, theta


def build_equirect_dirs(out_w: int, out_h: int) -> np.ndarray:
    xs = (np.arange(out_w).astype(np.float32) + 0.5) / float(out_w)
    ys = (np.arange(out_h).astype(np.float32) + 0.5) / float(out_h)
    lon = xs[None, :] * 2.0 * np.pi - np.pi
    lat = ys[:, None] * np.pi - (np.pi / 2.0)
    lon_grid = np.repeat(lon, out_h, axis=0)
    lat_grid = np.repeat(lat, out_w, axis=1)
    cos_lat = np.cos(lat_grid)
    X = cos_lat * np.sin(lon_grid)
    Y = np.sin(lat_grid)
    Z = cos_lat * np.cos(lon_grid)
    return np.stack((X, Y, Z), axis=-1).reshape(-1, 3)


def _prepare_static_remapped_masks(maps: Dict[str, Any], mask0_img: Optional[np.ndarray], mask1_img: Optional[np.ndarray]):
    remap_mask0 = None
    remap_mask1 = None
    if mask0_img is not None:
        remap_mask0 = cv2.remap(
            mask0_img,
            maps["map0_1"],
            maps["map0_2"],
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
    if mask1_img is not None:
        remap_mask1 = cv2.remap(
            mask1_img,
            maps["map1_1"],
            maps["map1_2"],
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
    return remap_mask0, remap_mask1


def prepare_projection_maps(
    K0: Dict[str, Any],
    K1: Dict[str, Any],
    R01: np.ndarray,
    t01: np.ndarray,
    out_w: int,
    out_h: int,
    sphere_radius_m: float,
):
    dirs = build_equirect_dirs(out_w, out_h).astype(np.float64)
    pts_world = dirs * float(sphere_radius_m)
    pts_cam0 = pts_world
    pts_cam1 = (pts_world - np.asarray(t01, dtype=np.float64).reshape(3,)) @ R01.T

    if K0.get("model", "kb") == "eucm":
        u0, v0, theta0 = dirs_to_eucm_pixels(pts_cam0, K0)
    else:
        u0, v0, theta0 = dirs_to_fisheye_pixels_kb(pts_cam0, K0)
    if K1.get("model", "kb") == "eucm":
        u1, v1, theta1 = dirs_to_eucm_pixels(pts_cam1, K1)
    else:
        u1, v1, theta1 = dirs_to_fisheye_pixels_kb(pts_cam1, K1)

    map0_x = u0.reshape(out_h, out_w).astype(np.float32)
    map0_y = v0.reshape(out_h, out_w).astype(np.float32)
    map1_x = u1.reshape(out_h, out_w).astype(np.float32)
    map1_y = v1.reshape(out_h, out_w).astype(np.float32)

    dist0 = np.linalg.norm(pts_cam0, axis=1).reshape(out_h, out_w)
    dist1 = np.linalg.norm(pts_cam1, axis=1).reshape(out_h, out_w)
    valid_theta0 = (theta0 <= (K0.get("max_theta", np.pi) + 1e-9)).reshape(out_h, out_w)
    valid_theta1 = (theta1 <= (K1.get("max_theta", np.pi) + 1e-9)).reshape(out_h, out_w)

    W0 = int(K0["w"])
    H0 = int(K0["h"])
    W1 = int(K1["w"])
    H1 = int(K1["h"])
    uv0 = (map0_x >= 0) & (map0_x < (W0 - 1)) & (map0_y >= 0) & (map0_y < (H0 - 1))
    uv1 = (map1_x >= 0) & (map1_x < (W1 - 1)) & (map1_y >= 0) & (map1_y < (H1 - 1))
    valid0 = uv0 & valid_theta0
    valid1 = uv1 & valid_theta1

    try:
        map0_1, map0_2 = cv2.convertMaps(map0_x, map0_y, cv2.CV_16SC2)
        map1_1, map1_2 = cv2.convertMaps(map1_x, map1_y, cv2.CV_16SC2)
    except Exception:
        map0_1, map0_2 = map0_x, map0_y
        map1_1, map1_2 = map1_x, map1_y

    return {
        "map0_1": map0_1,
        "map0_2": map0_2,
        "map1_1": map1_1,
        "map1_2": map1_2,
        "map0_x": map0_x,
        "map0_y": map0_y,
        "map1_x": map1_x,
        "map1_y": map1_y,
        "dist0": dist0,
        "dist1": dist1,
        "valid0": valid0,
        "valid1": valid1,
    }


def prepare_precomputed_blend_weights(
    maps: Dict[str, Any], mask0_img: Optional[np.ndarray] = None, mask1_img: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    valid0 = maps["valid0"].copy()
    valid1 = maps["valid1"].copy()
    if mask0_img is not None or mask1_img is not None:
        remap_mask0, remap_mask1 = _prepare_static_remapped_masks(maps, mask0_img, mask1_img)
        if remap_mask0 is not None:
            valid0 &= (remap_mask0 != 255)
        if remap_mask1 is not None:
            valid1 &= (remap_mask1 != 255)

    eps = 1e-6
    dist0 = np.where(valid0, np.maximum(maps["dist0"], eps), 0.0)
    dist1 = np.where(valid1, np.maximum(maps["dist1"], eps), 0.0)
    inv0 = np.zeros_like(dist0, dtype=np.float64)
    inv1 = np.zeros_like(dist1, dtype=np.float64)
    np.divide(1.0, dist0, out=inv0, where=dist0 > 0.0)
    np.divide(1.0, dist1, out=inv1, where=dist1 > 0.0)
    wsum = inv0 + inv1 + 1e-12
    w0n = (inv0 / wsum).astype(np.float32)
    w1n = (inv1 / wsum).astype(np.float32)
    mask_none = (inv0 + inv1) == 0
    return {"w0n": w0n, "w1n": w1n, "mask_none": mask_none}


def stitch_with_precomputed_maps(
    img0: np.ndarray, img1: np.ndarray, maps: Dict[str, Any], blend_weights: Dict[str, np.ndarray], ex: ThreadPoolExecutor
) -> np.ndarray:
    def _remap(src: np.ndarray, map1, map2):
        return cv2.remap(
            src,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    f0 = ex.submit(_remap, img0, maps["map0_1"], maps["map0_2"])
    f1 = ex.submit(_remap, img1, maps["map1_1"], maps["map1_2"])
    remap0 = f0.result()
    remap1 = f1.result()

    out_f = remap0.astype(np.float32) * blend_weights["w0n"][..., None] + remap1.astype(np.float32) * blend_weights["w1n"][..., None]
    out_f[blend_weights["mask_none"]] = 0.0
    return np.clip(out_f, 0, 255).astype(np.uint8)


def _normalize_grid(map_x: np.ndarray, map_y: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
    if src_w <= 1 or src_h <= 1:
        raise ValueError("Source image dimensions must be > 1 for torch grid normalization.")
    x = (map_x / float(src_w - 1)) * 2.0 - 1.0
    y = (map_y / float(src_h - 1)) * 2.0 - 1.0
    return np.stack((x, y), axis=-1).astype(np.float32)


def _build_torch_grid(map_x: np.ndarray, map_y: np.ndarray, src_w: int, src_h: int, torch_mod, device: str):
    return torch_mod.from_numpy(_normalize_grid(map_x, map_y, src_w, src_h)).to(device).unsqueeze(0)


def prepare_torch_maps_precomputed(
    maps: Dict[str, Any],
    blend_weights: Dict[str, np.ndarray],
    src0_w: int,
    src0_h: int,
    src1_w: int,
    src1_h: int,
    torch_mod,
    device: str,
    torch_dtype,
) -> Dict[str, Any]:
    grid0 = _build_torch_grid(maps["map0_x"], maps["map0_y"], src0_w, src0_h, torch_mod, device).to(dtype=torch_dtype)
    grid1 = _build_torch_grid(maps["map1_x"], maps["map1_y"], src1_w, src1_h, torch_mod, device).to(dtype=torch_dtype)

    return {
        "grid0": grid0,
        "grid1": grid1,
        "w0n": torch_mod.from_numpy(blend_weights["w0n"]).to(device=device, dtype=torch_dtype).unsqueeze(0).unsqueeze(0),
        "w1n": torch_mod.from_numpy(blend_weights["w1n"]).to(device=device, dtype=torch_dtype).unsqueeze(0).unsqueeze(0),
        "mask_none": torch_mod.from_numpy(blend_weights["mask_none"]).to(device).unsqueeze(0).unsqueeze(0),
    }


def torch_stitch(
    img0: np.ndarray, img1: np.ndarray, torch_maps: Dict[str, Any], torch_mod, torch_nn, device: str, torch_dtype
) -> np.ndarray:
    img0_c = np.ascontiguousarray(img0)
    img1_c = np.ascontiguousarray(img1)
    with torch_mod.inference_mode():
        t0 = torch_mod.from_numpy(img0_c).to(device=device, dtype=torch_dtype).permute(2, 0, 1).unsqueeze(0)
        t1 = torch_mod.from_numpy(img1_c).to(device=device, dtype=torch_dtype).permute(2, 0, 1).unsqueeze(0)
        remap0 = torch_nn.grid_sample(t0, torch_maps["grid0"], mode="bilinear", padding_mode="zeros", align_corners=True)
        remap1 = torch_nn.grid_sample(t1, torch_maps["grid1"], mode="bilinear", padding_mode="zeros", align_corners=True)
        out = remap0 * torch_maps["w0n"] + remap1 * torch_maps["w1n"]
        out = out.masked_fill(torch_maps["mask_none"], 0.0)
        out = out.clamp(0.0, 255.0).byte()
        return out.squeeze(0).permute(1, 2, 0).cpu().numpy()


def open_reader(bag_path: str, storage_id: str) -> SequentialReader:
    reader = SequentialReader()
    reader.open(StorageOptions(uri=bag_path, storage_id=storage_id), ConverterOptions("", ""))
    return reader


def open_writer(out_bag_path: str, storage_id: str) -> SequentialWriter:
    writer = SequentialWriter()
    writer.open(StorageOptions(uri=out_bag_path, storage_id=storage_id), ConverterOptions("", ""))
    return writer


def build_topic_maps(reader: SequentialReader):
    topics_and_types = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics_and_types}
    topic_key_to_name = {}
    for t in topics_and_types:
        topic_key_to_name[t.name] = t.name
        for attr in ("id", "topic_id"):
            if hasattr(t, attr):
                val = getattr(t, attr)
                if isinstance(val, (int, np.integer)):
                    topic_key_to_name[int(val)] = t.name
    return topic_types, topic_key_to_name


def make_topic_metadata(name: str, type_name: str) -> TopicMetadata:
    # Different ROS2 distros expose different TopicMetadata constructors.
    try:
        return TopicMetadata(
            name=name,
            type=type_name,
            serialization_format="cdr",
            offered_qos_profiles="",
        )
    except TypeError:
        pass

    try:
        return TopicMetadata(name=name, type=type_name, serialization_format="cdr")
    except TypeError:
        pass

    try:
        # e.g. TopicMetadata(id, name, type, serialization_format)
        return TopicMetadata(0, name, type_name, "cdr")
    except TypeError:
        pass

    # Last fallback for variants that also require qos string positionally.
    return TopicMetadata(0, name, type_name, "cdr", "")


def extract_from_bagmsg(bag_msg):
    if hasattr(bag_msg, "topic_name"):
        topic = getattr(bag_msg, "topic_name")
        ser = getattr(bag_msg, "serialized_data", None)
        t = getattr(bag_msg, "time", None)
        return topic, ser, int(t) if t is not None else None
    if hasattr(bag_msg, "topic_id"):
        topic = getattr(bag_msg, "topic_id")
        ser = getattr(bag_msg, "serialized_data", None)
        t = getattr(bag_msg, "time", None)
        return topic, ser, int(t) if t is not None else None

    if isinstance(bag_msg, (tuple, list)):
        if (
            len(bag_msg) >= 3
            and isinstance(bag_msg[0], (str, int, np.integer))
            and (
                isinstance(bag_msg[1], (bytes, bytearray, memoryview))
                or hasattr(bag_msg[1], "data")
                or hasattr(bag_msg[1], "serialized_data")
            )
            and isinstance(bag_msg[2], (int, float, np.integer))
        ):
            return bag_msg[0], bag_msg[1], int(bag_msg[2])

        topic = None
        ser = None
        t = None
        for el in bag_msg:
            if topic is None and isinstance(el, str) and el.startswith("/"):
                topic = el
            elif t is None and isinstance(el, (int, float)):
                t = int(el)
            elif ser is None and (
                isinstance(el, (bytes, bytearray))
                or hasattr(el, "data")
                or hasattr(el, "serialized_data")
            ):
                ser = el
        if topic is None and len(bag_msg) >= 1 and isinstance(bag_msg[0], str):
            topic = bag_msg[0]
        if ser is None:
            for idx in (1, 2):
                if len(bag_msg) > idx:
                    candidate = bag_msg[idx]
                    if isinstance(candidate, (bytes, bytearray)) or hasattr(candidate, "data") or hasattr(candidate, "serialized_data"):
                        ser = candidate
        if t is None and isinstance(bag_msg[-1], (int, float)):
            t = int(bag_msg[-1])
        if topic is None:
            raise RuntimeError(f"Could not determine topic from bag msg tuple: {bag_msg}")
        return topic, ser, t

    raise RuntimeError(f"Unsupported bag msg type: {type(bag_msg)}")


def resolve_topic_name(topic_key, topic_key_to_name: Dict[Any, str]) -> Optional[str]:
    if isinstance(topic_key, str):
        return topic_key
    if isinstance(topic_key, (int, np.integer)):
        return topic_key_to_name.get(int(topic_key))
    return None


def get_serialized_bytes(ser_obj):
    if ser_obj is None:
        return None
    if isinstance(ser_obj, memoryview):
        return ser_obj.tobytes()
    if isinstance(ser_obj, (bytes, bytearray)):
        return ser_obj
    if hasattr(ser_obj, "data"):
        return ser_obj.data
    if hasattr(ser_obj, "serialized_data"):
        return ser_obj.serialized_data
    return ser_obj


def decode_compressed_image_to_cv2(img_msg, turbojpeg_decoder=None) -> np.ndarray:
    data_bytes = None
    fmt = ""
    if isinstance(img_msg, CompressedImage) and hasattr(img_msg, "data"):
        data_bytes = img_msg.data
        fmt = str(getattr(img_msg, "format", "")).lower()
    elif isinstance(img_msg, (bytes, bytearray)):
        data_bytes = img_msg
    elif hasattr(img_msg, "data"):
        data_bytes = img_msg.data
    elif isinstance(img_msg, memoryview):
        data_bytes = img_msg.tobytes()
    if data_bytes is None:
        raise RuntimeError("Cannot extract bytes from CompressedImage-like object")

    if turbojpeg_decoder is not None and ("jpeg" in fmt or "jpg" in fmt or fmt == ""):
        try:
            return turbojpeg_decoder.decode(bytes(data_bytes))
        except Exception:
            pass

    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if cv_img is None:
        raise RuntimeError("cv2.imdecode returned None")
    return cv_img


def maybe_resize_to_yaml(img: np.ndarray, target_w: int, target_h: int, label: str) -> np.ndarray:
    h, w = img.shape[:2]
    if w == target_w and h == target_h:
        return img
    interp = cv2.INTER_AREA if (target_w < w or target_h < h) else cv2.INTER_LINEAR
    print(f"Warning: {label} incoming {w}x{h} != YAML {target_w}x{target_h}. Resizing.")
    return cv2.resize(img, (target_w, target_h), interpolation=interp)


def load_masks(mask0_file: Optional[str], mask1_file: Optional[str], K0: Dict[str, Any], K1: Dict[str, Any]):
    mask0_resized = None
    mask1_resized = None
    if mask0_file is not None:
        m = cv2.imread(mask0_file, cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"Warning: Failed to load mask0 file: {mask0_file}")
        else:
            mask0_resized = cv2.resize(m, (int(K0["w"]), int(K0["h"])), interpolation=cv2.INTER_NEAREST)
            print(f"Loaded mask0: {mask0_file}")
    if mask1_file is not None:
        m = cv2.imread(mask1_file, cv2.IMREAD_GRAYSCALE)
        if m is None:
            print(f"Warning: Failed to load mask1 file: {mask1_file}")
        else:
            mask1_resized = cv2.resize(m, (int(K1["w"]), int(K1["h"])), interpolation=cv2.INTER_NEAREST)
            print(f"Loaded mask1: {mask1_file}")
    return mask0_resized, mask1_resized


def encode_stitched_compressed(stitched_bgr: np.ndarray, stamp_ns: int, frame_id: str, jpeg_quality: int) -> bytes:
    ok, enc = cv2.imencode(".jpg", stitched_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        raise RuntimeError("Failed to encode stitched image as JPEG")
    msg = CompressedImage()
    msg.header.stamp.sec = int(stamp_ns // 1_000_000_000)
    msg.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    msg.data = enc.tobytes()
    return serialize_message(msg)


def main():
    parser = argparse.ArgumentParser(
        description="Stitch two compressed image topics from a ROS2 bag and write stitched CompressedImage plus all IMU messages to a new ROS2 bag."
    )
    parser.add_argument("--bag", required=True, help="Input ros2 bag URI.")
    parser.add_argument("--yaml", required=True, help="Camera YAML with cam0/cam1.")
    parser.add_argument("--out", required=True, help="Output ros2 bag URI (new directory).")
    parser.add_argument("--topic0", default="/cam0/image_raw/compressed", help="cam0 topic.")
    parser.add_argument("--topic1", default="/cam1/image_raw/compressed", help="cam1 topic.")
    parser.add_argument("--out-topic", default="/pano/image_raw/compressed", help="Output stitched CompressedImage topic.")
    parser.add_argument("--out-frame-id", default="pano", help="Header frame_id for stitched images.")
    parser.add_argument("--mask0", default=None, help="Optional static mask0 PNG (grayscale), white=masked.")
    parser.add_argument("--mask1", default=None, help="Optional static mask1 PNG (grayscale), white=masked.")
    parser.add_argument("--sync-tolerance-ms", type=float, default=5.0, help="Max delta to sync topic0/topic1.")
    parser.add_argument("--storage-id", default="sqlite3", help="Input storage id.")
    parser.add_argument("--out-storage-id", default="sqlite3", help="Output storage id.")
    parser.add_argument("--sphere-mm", type=float, default=10000.0, help="Virtual sphere radius in mm.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for stitched output [1..100].")
    parser.add_argument("--use-torch", action="store_true", help="Use torch grid_sample for remap/blend (GPU if available).")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--torch-fp16", action="store_true", help="Use fp16 tensors for torch stitching (best on CUDA).")
    parser.add_argument("--disable-turbojpeg", action="store_true", help="Disable turbojpeg decoder even if installed.")
    args = parser.parse_args()

    jpeg_quality = max(1, min(100, int(args.jpeg_quality)))
    out_parent = os.path.dirname(args.out)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    cam0, cam1 = load_camera_yaml(args.yaml)
    K0 = parse_intrinsics(cam0)
    K1 = parse_intrinsics(cam1)
    if K0.get("w") is None or K0.get("h") is None or K1.get("w") is None or K1.get("h") is None:
        raise RuntimeError("YAML cam0/cam1 must include resolution.")

    R01, t01 = resolve_extrinsics(cam0, cam1)
    target_w0 = int(K0["w"])
    target_h0 = int(K0["h"])
    target_w1 = int(K1["w"])
    target_h1 = int(K1["h"])
    out_w = max(1024, 2 * min(target_w0, target_w1))
    out_h = out_w // 2
    print(f"Stitched output resolution: {out_w}x{out_h}")

    mask0_resized, mask1_resized = load_masks(args.mask0, args.mask1, K0, K1)
    print("Precomputing projection maps...")
    maps = prepare_projection_maps(K0, K1, R01, t01, out_w, out_h, args.sphere_mm / 1000.0)
    precomputed_blend_weights = prepare_precomputed_blend_weights(maps, mask0_resized, mask1_resized)

    torch_maps = None
    torch_mod = None
    torch_nn = None
    torch_device = None
    torch_dtype = None
    if args.use_torch:
        try:
            import torch
            import torch.nn.functional as torch_nn
        except Exception as e:
            raise RuntimeError("Torch not available. Install torch or omit --use-torch.") from e
        if args.device == "auto":
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            torch_device = args.device
            if torch_device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("Torch CUDA requested but not available on this system.")
        if torch_device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        torch_dtype = torch.float16 if (args.torch_fp16 and torch_device.startswith("cuda")) else torch.float32
        torch_mod = torch
        torch_maps = prepare_torch_maps_precomputed(
            maps, precomputed_blend_weights, target_w0, target_h0, target_w1, target_h1, torch_mod, torch_device, torch_dtype
        )
        print(f"Torch device: {torch_device}, dtype: {str(torch_dtype).replace('torch.', '')}")

    turbojpeg_decoder = None
    if not args.disable_turbojpeg and TurboJPEG is not None:
        try:
            turbojpeg_decoder = TurboJPEG()
            print("Using turbojpeg decoder for CompressedImage.")
        except Exception:
            turbojpeg_decoder = None

    reader = open_reader(args.bag, args.storage_id)
    topic_types, topic_key_to_name = build_topic_maps(reader)
    if args.topic0 not in topic_types or args.topic1 not in topic_types:
        raise RuntimeError("One or both camera topics not found in input bag.")

    imu_topics = [t for t, ty in topic_types.items() if ty == "sensor_msgs/msg/Imu"]
    print(f"Found {len(imu_topics)} IMU topic(s): {imu_topics}")

    writer = open_writer(args.out, args.out_storage_id)
    writer.create_topic(make_topic_metadata(args.out_topic, "sensor_msgs/msg/CompressedImage"))
    for t in imu_topics:
        writer.create_topic(make_topic_metadata(t, topic_types[t]))

    tolerance_ns = int(args.sync_tolerance_ms * 1e6)
    buf0 = None
    buf1 = None
    stitched_count = 0
    skipped_count = 0
    imu_written = 0
    ex = ThreadPoolExecutor(max_workers=2)

    try:
        while reader.has_next():
            topic_key, ser, stamp = extract_from_bagmsg(reader.read_next())
            topic = resolve_topic_name(topic_key, topic_key_to_name)
            if topic is None:
                skipped_count += 1
                continue
            if stamp is None:
                skipped_count += 1
                continue

            raw = get_serialized_bytes(ser)
            if raw is None:
                skipped_count += 1
                continue

            if topic in imu_topics:
                writer.write(topic, raw, int(stamp))
                imu_written += 1
                continue

            if topic not in (args.topic0, args.topic1):
                continue

            try:
                msg = deserialize_message(raw, CompressedImage)
                img = decode_compressed_image_to_cv2(msg, turbojpeg_decoder=turbojpeg_decoder)
            except Exception as e:
                print(f"Warning: Failed to decode {topic}: {e}")
                skipped_count += 1
                continue

            if topic == args.topic0:
                buf0 = (int(stamp), img)
            else:
                buf1 = (int(stamp), img)

            if buf0 is None or buf1 is None:
                continue

            t0, img0 = buf0
            t1, img1 = buf1
            dt = abs(t0 - t1)
            if dt > tolerance_ns:
                if t0 < t1:
                    buf0 = None
                else:
                    buf1 = None
                continue

            buf0 = None
            buf1 = None
            img0 = maybe_resize_to_yaml(img0, target_w0, target_h0, "topic0")
            img1 = maybe_resize_to_yaml(img1, target_w1, target_h1, "topic1")
            if torch_maps is not None:
                stitched = torch_stitch(img0, img1, torch_maps, torch_mod, torch_nn, torch_device, torch_dtype)
            else:
                stitched = stitch_with_precomputed_maps(img0, img1, maps, precomputed_blend_weights, ex)

            out_stamp = int(t0)  # Required: use topic0 timestamp.
            out_ser = encode_stitched_compressed(stitched, out_stamp, args.out_frame_id, jpeg_quality)
            writer.write(args.out_topic, out_ser, out_stamp)
            stitched_count += 1

            if stitched_count % 100 == 0:
                print(f"Stitched {stitched_count} frames, IMU copied {imu_written}")
    finally:
        ex.shutdown(wait=False)

    print(f"Done. Stitched images written: {stitched_count}")
    print(f"IMU messages copied: {imu_written}")
    print(f"Skipped messages: {skipped_count}")
    if stitched_count == 0:
        print("Warning: No stitched frames produced. Check topics, sync tolerance, and YAML.")


if __name__ == "__main__":
    main()
