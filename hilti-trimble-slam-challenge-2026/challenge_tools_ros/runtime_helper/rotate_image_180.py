#!/usr/bin/env python
# Simple tool to rotate images by 180
# This is only intended for easy visualisation of the inverted images
#
# ros2 run challenge_tools_ros rotate_image_180.py /cam0/image_raw/compressed compressed show_gui
#

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import sys

class ImageRotator180(Node):
    def __init__(self, input_topic, compressed_input, show_gui):
        super().__init__('rotate_image_180')
        self.bridge = CvBridge()
        

        if (compressed_input):
            print("Subscribe to",input_topic,"CompressedImage")
            self.subscription = self.create_subscription(CompressedImage, input_topic, self.compressed_image_callback, 10)
        else:
            print("Subscribe to",input_topic,"Image [uncompressed]")
            self.subscription = self.create_subscription(Image, input_topic, self.image_callback, 10)

        # publish uncompressed
        topic_out = input_topic + "/rotated"
        print("Publish to",input_topic,"Image [uncompressed]")
        self.publisher = self.create_publisher(Image, topic_out, 10)

        print("Will show gui") if show_gui else print("Will not show gui")

        self.get_logger().info('manual_invert_image_rotator_180 node has started.')
        self.resize_factor = 2
        
        self.do_rotation = True
        self.show_gui = show_gui
        self.publish_to_ros = True
        self.counter = 0


    def image_callback(self, msg):
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.do_work(image, msg.header)

    def compressed_image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.do_work(image, msg.header)

    def do_work(self, image, header):
        if (self.counter % 25 == 0):
            str_out = str(self.counter) + " " + str(header.stamp.sec) + " " + str(header.stamp.nanosec)
            self.get_logger().info(str_out)
        self.counter = self.counter + 1

        if (self.do_rotation):
            image = cv2.rotate(image, cv2.ROTATE_180)

        if (self.show_gui):
            resized = cv2.resize(image, (int(image.shape[1]/self.resize_factor), int(image.shape[0]/self.resize_factor)))
            cv2.imshow( "image", resized)
            cv2.waitKey(1)

        if (self.publish_to_ros):
            img_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            img_msg.header = header
            self.publisher.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)

    args = sys.argv[1:]
    num_args = len(args)
    print("Usage:")
    print("  rotate_image_180.py /input/image/topic [compressed] [gui]")
    print(" ")
    if (num_args < 1):
        print("Topic argument required.", num_args, "provided.")
        print("quitting")
        return

    compressed_input = False
    if (num_args >= 2):
        compressed_input = True

    show_gui = False
    if (num_args >= 3):
        show_gui = True

    input_topic = args[0]

    node = ImageRotator180(input_topic, compressed_input, show_gui)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
