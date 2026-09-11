#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
import traceback

# ROS2 python APIs
try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    import rclpy
    from rclpy.serialization import deserialize_message
    # message classes
    from sensor_msgs.msg import Image, CompressedImage, Imu
    from nav_msgs.msg import Odometry
    from geometry_msgs.msg import PoseStamped
except Exception as e:
    print("ERROR: Could not import ROS2 Python modules. Run inside a ROS2 Python environment.")
    raise

# cv_bridge and OpenCV for image conversion
try:
    from cv_bridge import CvBridge
    HAVE_CV_BRIDGE = True
except Exception:
    HAVE_CV_BRIDGE = False

try:
    import cv2
    import numpy as np
except Exception:
    print("ERROR: OpenCV (cv2) and numpy are required. Install with: pip install opencv-python numpy")
    raise

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def open_reader(bag_path):
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    return reader

def topic_type_map(reader):
    topics_and_types = reader.get_all_topics_and_types()
    mapping = {}
    for t in topics_and_types:
        mapping[t.name] = t.type
    return mapping

def extract_from_bagmsg(bag_msg):
    """
    Return tuple (topic:str, serialized_bytes_or_obj, timestamp_ns:int)
    Robust to different reader.read_next() return formats.
    """
    if hasattr(bag_msg, "topic_name"):
        topic = getattr(bag_msg, "topic_name")
        ser = getattr(bag_msg, "serialized_data", None)
        t = getattr(bag_msg, "time", None)
        return topic, ser, int(t) if t is not None else None

    if isinstance(bag_msg, (tuple, list)):
        topic = None
        ser = None
        t = None
        for el in bag_msg:
            if topic is None and isinstance(el, str) and el.startswith("/"):
                topic = el
            elif t is None and isinstance(el, (int, float)):
                t = int(el)
            elif ser is None and (isinstance(el, (bytes, bytearray)) or hasattr(el, "data") or hasattr(el, "serialized_data")):
                ser = el

        if topic is None and len(bag_msg) >= 1 and isinstance(bag_msg[0], str):
            topic = bag_msg[0]
        if ser is None:
            for idx in (1, 2):
                if len(bag_msg) > idx:
                    candidate = bag_msg[idx]
                    if isinstance(candidate, (bytes, bytearray)) or hasattr(candidate, "data") or hasattr(candidate, "serialized_data"):
                        ser = candidate
        if t is None:
            last = bag_msg[-1]
            if isinstance(last, (int, float)):
                t = int(last)

        if topic is None:
            raise RuntimeError(f"Could not determine topic from bag_msg tuple: {bag_msg}")
        return topic, ser, t

    raise RuntimeError(f"Unsupported bag_msg type: {type(bag_msg)}")

def get_serialized_bytes(ser_obj):
    if ser_obj is None:
        return None
    if isinstance(ser_obj, (bytes, bytearray)):
        return ser_obj
    if hasattr(ser_obj, "data"):
        return ser_obj.data
    if hasattr(ser_obj, "serialized_data"):
        return ser_obj.serialized_data
    return ser_obj

def decode_compressed_image_to_cv2(img_msg):
    """
    Try cv_bridge.compressed_imgmsg_to_cv2 if available, otherwise decode via cv2.imdecode.
    """
    if HAVE_CV_BRIDGE and isinstance(img_msg, CompressedImage):
        bridge = CvBridge()
        try:
            return bridge.compressed_imgmsg_to_cv2(img_msg)
        except Exception:
            pass

    data_bytes = None
    if isinstance(img_msg, CompressedImage) and hasattr(img_msg, "data"):
        data_bytes = img_msg.data
    elif isinstance(img_msg, (bytes, bytearray)):
        data_bytes = img_msg
    elif hasattr(img_msg, "data"):
        data_bytes = img_msg.data
    elif isinstance(img_msg, memoryview):
        data_bytes = img_msg.tobytes()

    if data_bytes is None:
        raise RuntimeError("Cannot extract bytes from CompressedImage-like object")

    arr = np.frombuffer(data_bytes, dtype=np.uint8)
    cv_img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if cv_img is None:
        raise RuntimeError("cv2.imdecode returned None (failed to decode).")
    return cv_img

def main():
    parser = argparse.ArgumentParser(description="Convert ros2 bag to EuRoC dataset structure (robust w/ headers)")
    parser.add_argument("--bag", required=True, help="Path to ros2 bag (folder or sqlite file base)")
    parser.add_argument("--out", required=True, help="Output folder (will be created)")
    parser.add_argument("--cam-topics", nargs='+', required=True,
                        help="Camera image topics (list). Order defines cam0, cam1, ...")
    parser.add_argument("--imu-topic", required=True, help="IMU topic (sensor_msgs/Imu)")
    parser.add_argument("--gt-topic", default=None,
                        help="Ground truth topic (nav_msgs/Odometry or geometry_msgs/PoseStamped). Optional.")
    parser.add_argument("--image-encoding", default="bgr8",
                        help="cv_bridge encoding to convert images into before saving (e.g. mono8, bgr8).")
    parser.add_argument("--skip-images", action='store_true', help="Don't write images, only timestamps/IMU (faster).")
    parser.add_argument("--start", type=int, default=0, help="Start time (ns) - optional")
    parser.add_argument("--end", type=int, default=0, help="End time (ns) - optional")
    args = parser.parse_args()

    bag_path = args.bag
    out_dir = Path(args.out)
    cam_topics = args.cam_topics
    imu_topic = args.imu_topic
    gt_topic = args.gt_topic
    encoding = args.image_encoding
    save_images = not args.skip_images
    t_start = args.start if args.start > 0 else None
    t_end = args.end if args.end > 0 else None

    print("Opening bag:", bag_path)
    reader = open_reader(bag_path)
    topic_types = topic_type_map(reader)
    print("Topics found in bag:")
    for t, ty in topic_types.items():
        print(" ", t, "->", ty)

    ensure_dir(out_dir)
    mav0 = out_dir / "euroc"
    ensure_dir(mav0)

    cam_files = []
    for i, topic in enumerate(cam_topics):
        cam_dir = mav0 / f"cam{i}"
        data_dir = cam_dir / "data"
        ensure_dir(data_dir)
        cam_csv = cam_dir / "data.csv"
        fh = open(cam_csv, "w", newline='')
        # EuRoC camera CSV header
        fh.write("#timestamp [ns],filename\n")
        cam_files.append({
            "topic": topic,
            "dir": data_dir,
            "csv_path": cam_csv,
            "csv_handle": fh,
        })

    imu_dir = mav0 / "imu0"
    ensure_dir(imu_dir)
    imu_csv_path = imu_dir / "data.csv"
    imu_fh = open(imu_csv_path, "w", newline='')
    # EuRoC IMU CSV header
    imu_fh.write("#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")

    gt_fh = None
    if gt_topic:
        gt_dir = mav0 / "state_groundtruth_estimate0"
        ensure_dir(gt_dir)
        gt_csv_path = gt_dir / "data.csv"
        gt_fh = open(gt_csv_path, "w", newline='')
        # optional header for ground truth (common variant)
        gt_fh.write("#timestamp [ns],px,py,pz,qx,qy,qz,qw\n")

    cam_topic_to_index = {cf["topic"]: idx for idx, cf in enumerate(cam_files)}

    count = 0
    cam_counts = {i: 0 for i in range(len(cam_files))}
    imu_count = 0
    gt_count = 0

    print("Reading messages & exporting... (this may take a while)")

    try:
        while reader.has_next():
            bag_msg = reader.read_next()
            try:
                topic, ser_obj, t_ns = extract_from_bagmsg(bag_msg)
            except Exception as e:
                print("Warning: failed to extract fields from bag_msg; skipping this entry.")
                print("  Diagnostic:", e)
                continue

            serialized = get_serialized_bytes(ser_obj)

            if t_ns is None:
                print(f"Warning: no timestamp found for topic {topic}; skipping message.")
                continue

            if t_start and t_ns < t_start:
                continue
            if t_end and t_ns > t_end:
                continue

            if topic in cam_topic_to_index:
                cam_idx = cam_topic_to_index[topic]
                msg_type_str = topic_types.get(topic, "")
                is_compressed = 'CompressedImage' in msg_type_str or 'sensor_msgs/msg/CompressedImage' in msg_type_str
                img_msg = None
                try:
                    if is_compressed:
                        try:
                            img_msg = deserialize_message(serialized, CompressedImage)
                        except Exception:
                            img_msg = None
                    else:
                        img_msg = deserialize_message(serialized, Image)
                except Exception:
                    img_msg = None

                if save_images:
                    cv_img = None
                    try:
                        if is_compressed:
                            if img_msg is not None and HAVE_CV_BRIDGE:
                                bridge = CvBridge()
                                cv_img = bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding=encoding)
                            else:
                                try:
                                    cv_img = decode_compressed_image_to_cv2(img_msg if img_msg is not None else ser_obj)
                                except Exception:
                                    if img_msg is None:
                                        try:
                                            img_msg2 = deserialize_message(serialized, CompressedImage)
                                            cv_img = decode_compressed_image_to_cv2(img_msg2)
                                        except Exception as e2:
                                            raise RuntimeError(f"Failed to decode compressed image (fallbacks): {e2}")
                        else:
                            if img_msg is not None:
                                if HAVE_CV_BRIDGE:
                                    bridge = CvBridge()
                                    cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding=encoding)
                                else:
                                    raise RuntimeError("cv_bridge required to convert sensor_msgs/Image messages.")
                    except Exception as e:
                        print(f"Warning: failed to convert image on {topic} at {t_ns}: {e}")
                        continue

                    filename = f"{t_ns}.png"
                    outpath = cam_files[cam_idx]["dir"] / filename
                    ok = cv2.imwrite(str(outpath), cv_img)
                    if not ok:
                        print("Warning: cv2.imwrite failed for", outpath)
                        continue
                    # now write cam csv line using comma separator to match header
                    cam_files[cam_idx]["csv_handle"].write(f"{t_ns},{filename}\n")
                    cam_counts[cam_idx] += 1

            elif topic == imu_topic:
                try:
                    imu_msg = None
                    try:
                        imu_msg = deserialize_message(serialized, Imu)
                    except Exception:
                        if hasattr(ser_obj, "angular_velocity") and hasattr(ser_obj, "linear_acceleration"):
                            imu_msg = ser_obj
                        else:
                            raise

                    gx = imu_msg.angular_velocity.x
                    gy = imu_msg.angular_velocity.y
                    gz = imu_msg.angular_velocity.z
                    ax = imu_msg.linear_acceleration.x
                    ay = imu_msg.linear_acceleration.y
                    az = imu_msg.linear_acceleration.z

                    # imu_fh.write(f"{t_ns},{gx:.9e},{gy:.9e},{gz:.9e},{ax:.9e},{ay:.9e},{az:.9e}\n")
                    imu_fh.write(f"{t_ns},{gx},{gy},{gz},{ax},{ay},{az}\n")
                    imu_count += 1
                except Exception as e:
                    print("Warning: failed to deserialize/write IMU message:", e)
                    continue

            elif gt_topic and topic == gt_topic:
                wrote = False
                try:
                    odom = deserialize_message(serialized, Odometry)
                    px = odom.pose.pose.position.x
                    py = odom.pose.pose.position.y
                    pz = odom.pose.pose.position.z
                    qx = odom.pose.pose.orientation.x
                    qy = odom.pose.pose.orientation.y
                    qz = odom.pose.pose.orientation.z
                    qw = odom.pose.pose.orientation.w
                    gt_fh.write(f"{t_ns},{px:.9e},{py:.9e},{pz:.9e},{qx:.9e},{qy:.9e},{qz:.9e},{qw:.9e}\n")
                    gt_count += 1
                    wrote = True
                except Exception:
                    try:
                        ps = deserialize_message(serialized, PoseStamped)
                        px = ps.pose.position.x
                        py = ps.pose.position.y
                        pz = ps.pose.position.z
                        qx = ps.pose.orientation.x
                        qy = ps.pose.orientation.y
                        qz = ps.pose.orientation.z
                        qw = ps.pose.orientation.w
                        gt_fh.write(f"{t_ns},{px:.9e},{py:.9e},{pz:.9e},{qx:.9e},{qy:.9e},{qz:.9e},{qw:.9e}\n")
                        gt_count += 1
                        wrote = True
                    except Exception as e:
                        print("Warning: failed to deserialize groundtruth message:", e)

            count += 1
            if count % 5000 == 0:
                print(f"Processed {count} messages... (images: {sum(cam_counts.values())}, imu: {imu_count})")

    except Exception as e:
        print("ERROR reading bag:", e)
        traceback.print_exc()
    finally:
        for cf in cam_files:
            try:
                cf["csv_handle"].close()
            except Exception:
                pass
        imu_fh.close()
        if gt_fh:
            gt_fh.close()

    print("Done.")
    print("Summary:")
    for i in range(len(cam_files)):
        print(f"  cam{i}: {cam_counts[i]} images -> {cam_files[i]['csv_path']}")
    print(f"  imu: {imu_count} samples -> {imu_csv_path}")
    if gt_topic:
        print(f"  groundtruth: {gt_count} samples -> {gt_csv_path}")

if __name__ == "__main__":
    main()
