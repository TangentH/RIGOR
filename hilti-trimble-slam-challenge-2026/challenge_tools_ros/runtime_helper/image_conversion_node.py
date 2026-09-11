#!/usr/bin/env python3
#
# Tool to decompress the CompressedImage topics
# and republish them as Image topics

import sys
from typing import List

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image


class ImageConversionNode(Node):
    def __init__(self, input_topics: List[str], output_topics: List[str], desired_encoding: str = ""):
        super().__init__('image_conversion_node')

        self._bridge = CvBridge()
        self._input_topics = input_topics
        self._output_topics = output_topics
        self._num_topics = len(input_topics)
        self._desired_encoding = desired_encoding

        self._sync_count = 0
        self._last_sync_count = 0
        # self.create_timer(1.0, self._check_sync)

        # Create publishers (avoid shadowing Node internals by using a leading underscore)
        self._publishers = [
            self.create_publisher(Image, output_topics[i], 10)
            for i in range(self._num_topics)
        ]

        # Logging
        for i in range(self._num_topics):
            self.get_logger().info(f"Input  {i}: {input_topics[i]}")
            self.get_logger().info(f"Output {i}: {output_topics[i]}")

        # If only one input -> simple subscription
        if self._num_topics == 1:
            self._sub = self.create_subscription(
                CompressedImage, input_topics[0], self._single_callback, 10
            )
        else:
            # Two inputs -> use message_filters TimeSynchronizer
            subs = [
                message_filters.Subscriber(self, CompressedImage, t)
                for t in input_topics
            ]
            # queue size 10 (tunable)
            self._sync = message_filters.TimeSynchronizer(subs, 10)
            self._sync.registerCallback(self._sync_callback)

    # ---------- SINGLE INPUT ----------
    def _single_callback(self, msg: CompressedImage):
        self._convert_and_publish(0, msg)

    # ---------- TWO INPUTS ----------
    def _sync_callback(self, msg1: CompressedImage, msg2: CompressedImage):
        self._sync_count += 1
        # order corresponds to input_topics order
        self._convert_and_publish(0, msg1)
        self._convert_and_publish(1, msg2)

    def _check_sync(self):
        if self._sync_count == self._last_sync_count and self._num_topics == 2:
            self.get_logger().warn("No synchronized messages detected.")
        self._last_sync_count = self._sync_count

    # ---------- COMMON CONVERSION ----------
    def _convert_and_publish(self, index: int, msg: CompressedImage):
        try:
            # Decode compressed bytes to OpenCV image
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            if cv_img is None:
                raise RuntimeError("cv2.imdecode returned None (invalid/unsupported compressed data)")

            # Determine encoding for cv_bridge
            if self._desired_encoding:
                encoding = self._desired_encoding
                if encoding == "mono8" and cv_img.ndim == 3:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                elif encoding == "bgr8" and cv_img.ndim == 2:
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            else:
                if cv_img.ndim == 2:
                    encoding = "mono8"
                elif cv_img.ndim == 3:
                    if cv_img.shape[2] == 3:
                        encoding = "bgr8"
                    elif cv_img.shape[2] == 4:
                        encoding = "bgra8"
                    else:
                        encoding = "bgr8"
                else:
                    encoding = "bgr8"

            img_msg = self._bridge.cv2_to_imgmsg(cv_img, encoding=encoding)
            # preserve header (timestamp + frame_id)
            img_msg.header = msg.header

            self._publishers[index].publish(img_msg)

        except Exception as e:
            self.get_logger().error(f"Error converting input[{index}] '{self._input_topics[index]}': {e}")


def main(args=None):
    rclpy.init(args=args)

    # Collect non-ROS arguments from sys.argv
    user_args = [
        a for a in sys.argv[1:]
        if not a.startswith('-') and ':=' not in a and not a.startswith('__')
    ]

    if len(user_args) not in (2, 3, 4, 5):
        print("\nUsage:")
        print("  Single input:")
        print("    ros2 run <pkg> image_conversion_node.py <in1> <out1> [encoding]")
        print("")
        print("  Two inputs:")
        print("    ros2 run <pkg> image_conversion_node.py <in1> <in2> <out1> <out2> [encoding]\n")
        rclpy.shutdown()
        return

    desired_encoding = ""
    if len(user_args) in (3, 5):
        desired_encoding = user_args[-1]

    if len(user_args) in (2, 3):
        input_topics = [user_args[0]]
        output_topics = [user_args[1]]
    else:
        input_topics = [user_args[0], user_args[1]]
        output_topics = [user_args[2], user_args[3]]

    node = ImageConversionNode(input_topics, output_topics, desired_encoding)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
