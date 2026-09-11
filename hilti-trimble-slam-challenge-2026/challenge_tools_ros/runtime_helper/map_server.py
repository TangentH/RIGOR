#!/usr/bin/env python
# Basic tool to read a binary floor plan and publish as an occupancy map

import numpy as np
import math
import csv
import os
import time
import sys
import glob
import random
import cv2

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

class App(Node):
    def __init__(self):
        super().__init__('map_server_demo')

    def random_edit_to_map(self, data):

        indices = random.sample(range(len(data)), 10)
        for i in indices:
            data[i] = 0 # free - white grey

        indices = random.sample(range(len(data)), 10)
        for i in indices:
            data[i] = 100 # occupied - black

        # 0 free - white grey
        # 100 black
        # -1 is unknown dark grey
        return data


    def process(self, full_filename, height_offset):

        topic = "/map_" + str(int(height_offset))
        self.pub_og = self.create_publisher(OccupancyGrid, topic, 10)
        print("topic:", topic)
        self.get_logger().info(f"topic: {topic}")


        if (height_offset == 80):
            height_offset = -10.
            origin_pos = [-19.30, -16.9, height_offset]
        else:
            origin_pos = [0., 0., height_offset]

        base_resolution = 0.01
        downsample_ratio = 10
        downsampled_resolution = base_resolution*downsample_ratio

        self.get_logger().info(f"full_filename: {full_filename}")
        img = cv2.imread(full_filename, cv2.IMREAD_GRAYSCALE)
        self.get_logger().info(f"original size: {img.shape}")


        # Downsample by x10 - using INTER_AREA (best for shrinking)
        target_size = (round(img.shape[1]/downsample_ratio), round(img.shape[0]/downsample_ratio))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        print(img.shape, "smaller size")
        print(img.dtype, "data type") # 0-255 - but actual values are 0-1

        if (1==0):
            img_vis = img * 255
            cv2.imshow('Scaled Intensity', img_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        # 0, 0
        msg = OccupancyGrid()
        msg.header.frame_id = "map"
        msg.info.resolution = downsampled_resolution # 0.1
        msg.info.width = target_size[0]
        msg.info.height = target_size[1]
        msg.info.origin.position.x = origin_pos[0]# 19.30#-(msg.info.width * msg.info.resolution) / 2.
        msg.info.origin.position.y = origin_pos[1]# 16.9#-(msg.info.height * msg.info.resolution) / 2.
        msg.info.origin.position.z = origin_pos[2]# height_offset # 0.
        msg.info.origin.orientation.x = 0.
        msg.info.origin.orientation.y = 0.
        msg.info.origin.orientation.z = 0.
        msg.info.origin.orientation.w = 1.
        # empty array
        msg.data = [-1] * msg.info.width * msg.info.height

        #img_f = cv2.flip(img, 1)
        #img_f = img
        img_f = cv2.flip(img, 0)
        flat = img_f.flatten()

        for x in range(0, img.shape[0]*img.shape[1]):
            if (flat[x]==0):
                msg.data[x] = 0
            else:
                msg.data[x] = 100
        self.pub_og.publish(msg)

        while(1):
            self.get_logger().info(f"publish")
            time.sleep(1)
            self.pub_og.publish(msg)
            
        return



def main(args=None):
    rclpy.init(args=args)

    args = sys.argv[1:]
    num_args = len(args)
    print("Usage:")
    print("  map_server.py floorplan_mask.png height_offset")
    print(" ")
    if (num_args < 2):
        print("2 required arguments.", num_args, "provided.")
        print("quitting")
        return
    
    print("Using commandline configuration:")
    floorplan_filename = args[0]
    height_offset = float(args[1])
    print("  floorplan_filename:", floorplan_filename)
    print("  height_offset:", height_offset)
    print(" ")

    app = App()
    app.process(floorplan_filename, height_offset)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
