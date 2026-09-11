#!/usr/bin/env python3
"""
ros2bag_to_ros1bag.py

Convert a ROS2 sqlite bag to a ROS1 rosbag (v1) using rosbags.

Design principles:
- ROS2 reading matches rosbag2_py usage exactly.
- ROS1 writing follows the same pattern as insv_to_ros1bag.py:
  - writer.add_connection(...)
  - typestore.serialize_ros1(...)
  - Explicit message construction (Header, Image, CompressedImage, Imu)
- No raw byte forwarding.
- No implicit or fallback registration.
"""

import argparse
import math
import types
import traceback
import numpy as np

# ROS2 APIs
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image as ROS2Image
from sensor_msgs.msg import CompressedImage as ROS2CompressedImage
from sensor_msgs.msg import Imu as ROS2Imu

# ROS1 (rosbags)
from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore
from rosbags.typesys.stores.ros1_noetic import (
    builtin_interfaces__msg__Time as RosTime,
    std_msgs__msg__Header as Header,
    sensor_msgs__msg__Image as ImageMsg,
    sensor_msgs__msg__CompressedImage as CompressedImageMsg,
    sensor_msgs__msg__Imu as ImuMsg,
)

# -----------------------------------------------------------------------------
# ROS2 reader helpers
# -----------------------------------------------------------------------------

def open_reader(path: str) -> SequentialReader:
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=path, storage_id="sqlite3"),
        ConverterOptions("", "")
    )
    return reader


def topic_type_map(reader) -> dict[str, str]:
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def extract_message(bag_msg):
    if hasattr(bag_msg, "topic_name"):
        return bag_msg.topic_name, bag_msg.serialized_data, bag_msg.time
    topic, data, ts = bag_msg
    return topic, data, ts


# -----------------------------------------------------------------------------
# ROS1 helpers (same semantics as insv_to_ros1bag.py)
# -----------------------------------------------------------------------------

def ros_time_from_ns(ns: int) -> RosTime:
    sec = ns // 1_000_000_000
    nsec = ns % 1_000_000_000
    return RosTime(sec=sec, nanosec=nsec)


def make_header(ns: int, frame_id: str = "") -> Header:
    return Header(seq=0, stamp=ros_time_from_ns(ns), frame_id=frame_id)


def serialize_image(msg: ROS2Image, ns: int, typestore) -> bytes:
    hdr = make_header(ns, msg.header.frame_id)
    # Normalize data to a numpy uint8 array (serializer expects .dtype)
    # bytes(msg.data) handles array.array or bytes-like objects
    data_arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    ros1 = ImageMsg(
        hdr,
        msg.height,
        msg.width,
        msg.encoding,
        msg.is_bigendian,
        msg.step,
        data_arr
    )
    return typestore.serialize_ros1(ros1, ImageMsg.__msgtype__)


def serialize_compressed(msg: ROS2CompressedImage, ns: int, typestore) -> bytes:
    hdr = make_header(ns, msg.header.frame_id)
    # Compressed data -> numpy uint8 array
    data_arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    ros1 = CompressedImageMsg(hdr, msg.format, data_arr)
    return typestore.serialize_ros1(ros1, CompressedImageMsg.__msgtype__)


def serialize_imu(msg: ROS2Imu, ns: int, typestore) -> bytes:
    hdr = make_header(ns, msg.header.frame_id)

    orientation = types.SimpleNamespace(
        x=msg.orientation.x,
        y=msg.orientation.y,
        z=msg.orientation.z,
        w=msg.orientation.w,
    )
    angular = types.SimpleNamespace(
        x=msg.angular_velocity.x,
        y=msg.angular_velocity.y,
        z=msg.angular_velocity.z,
    )
    linear = types.SimpleNamespace(
        x=msg.linear_acceleration.x,
        y=msg.linear_acceleration.y,
        z=msg.linear_acceleration.z,
    )

    ros1 = ImuMsg(
        hdr,
        orientation,
        msg.orientation_covariance,
        angular,
        msg.angular_velocity_covariance,
        linear,
        msg.linear_acceleration_covariance,
    )

    return typestore.serialize_ros1(ros1, ImuMsg.__msgtype__)


# -----------------------------------------------------------------------------
# Main conversion
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert ROS2 bag to ROS1 bag")
    parser.add_argument("--bag", required=True, help="ROS2 bag directory")
    parser.add_argument("--out", required=True, help="Output ROS1 bag")
    args = parser.parse_args()

    reader = open_reader(args.bag)
    topic_types = topic_type_map(reader)

    image_topics = {
        t for t, ty in topic_types.items()
        if ty.endswith("/Image") and "Compressed" not in ty
    }
    compressed_topics = {
        t for t, ty in topic_types.items()
        if ty.endswith("/CompressedImage")
    }
    imu_topics = {
        t for t, ty in topic_types.items()
        if ty.endswith("/Imu")
    }

    imu_topic = next(iter(imu_topics), None)

    typestore = get_typestore(Stores.ROS1_NOETIC)

    writer = Writer(args.out)
    writer.open()

    connections = {}

    for t in image_topics:
        connections[t] = writer.add_connection(
            t, ImageMsg.__msgtype__, typestore=typestore
        )
    for t in compressed_topics:
        connections[t] = writer.add_connection(
            t, CompressedImageMsg.__msgtype__, typestore=typestore
        )
    if imu_topic:
        connections[imu_topic] = writer.add_connection(
            imu_topic, ImuMsg.__msgtype__, typestore=typestore
        )

    processed = 0

    try:
        while reader.has_next():
            topic, raw, ts = extract_message(reader.read_next())
            if topic not in connections:
                continue

            if topic in image_topics:
                msg = deserialize_message(raw, ROS2Image)
                data = serialize_image(msg, ts, typestore)
            elif topic in compressed_topics:
                msg = deserialize_message(raw, ROS2CompressedImage)
                data = serialize_compressed(msg, ts, typestore)
            elif topic == imu_topic:
                msg = deserialize_message(raw, ROS2Imu)
                data = serialize_imu(msg, ts, typestore)
            else:
                continue

            writer.write(connections[topic], ts, data)
            processed += 1

            if processed % 5000 == 0:
                print(f"Processed {processed} messages")

    except Exception:
        traceback.print_exc()
    finally:
        writer.close()

    print(f"Done. Messages written: {processed}")
    print(f"Output bag: {args.out}")


if __name__ == "__main__":
    main()
