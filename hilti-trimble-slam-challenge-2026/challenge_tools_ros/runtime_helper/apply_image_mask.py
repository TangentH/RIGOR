#!/usr/bin/env python3
#
# Demonstration tool for how to read an image mask from file
# and block out pixel in the camera images which contain the operator
# tablet and wires and the operator
# 
# ros2 launch challenge_tools_ros apply_image_mask_dual.launch
#
# NOTE: the blocked pixels vary in different recordings

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge

class CompressedImageSubscriber(Node):
    def __init__(self):
        super().__init__('compressed_image_subscriber')

        self.declare_parameter('cam_topic', 'cam_topic_example')
        cam_topic = self.get_parameter('cam_topic').get_parameter_value().string_value
        self.get_logger().info(f"cam_topic: {cam_topic}")

        self.declare_parameter('mask_filename', 'cam_mask_filename_example')
        self.mask_filename = self.get_parameter('mask_filename').get_parameter_value().string_value
        self.get_logger().info(f"cam_name: {self.mask_filename}")

        self.cam_topic_out = "/masked" + cam_topic

        # Subscribe to the compressed image topic
        self.subscription = self.create_subscription(CompressedImage, cam_topic, self.listener_callback, 10)

        # Topic name and qos
        qos_profile = 10  # simple default, use rclpy.qos if you need transient_local
        self.pub = self.create_publisher(Image, self.cam_topic_out, qos_profile)
        self.bridge = CvBridge()

        self.resize_factor = 2
        self.apply_mask = True
        self.show_gui = False
        self.subscription  # prevent unused variable warning
        self.get_logger().info(f'Subscribed to {cam_topic}')

        if (self.apply_mask):
            # Read the mask as grayscale (0–255)
            self.mask = cv2.imread(self.mask_filename, cv2.IMREAD_GRAYSCALE)
            # Optional: Make sure mask is binary (0 or 255)
            _, self.mask = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)


    def listener_callback(self, msg: CompressedImage):
        self.get_logger().info(f"time: {msg.header.stamp.sec}")

        # Convert compressed image (JPEG/PNG) bytes to numpy array
        np_arr = np.frombuffer(msg.data, np.uint8)

        # Decode image using OpenCV
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().warn('Failed to decode image!')
            return

        if (self.apply_mask):

            # Ensure both have the same dimensions
            if image.shape[:2] != self.mask.shape[:2]:
                raise ValueError("Image and mask must be the same size")

            result = cv2.bitwise_and(image, image, mask=self.mask)
            image = result

        img_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        img_msg.header = msg.header
        self.pub.publish(img_msg)
    
        if (self.show_gui):
            resized = cv2.resize(image, (int(image.shape[1]/self.resize_factor), int(image.shape[0]/self.resize_factor)))
            cv2.imshow( self.cam_topic_out, resized)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)

    node = CompressedImageSubscriber()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

