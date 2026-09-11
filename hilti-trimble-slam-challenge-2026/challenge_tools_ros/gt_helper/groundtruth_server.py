#!/usr/bin/env python
# Read a txt file of all the poses
# Publish to Rviz as a path message

import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

import numpy as np
import sys
import csv
import time
from nav_msgs.msg import Path

from challenge_tools_lib import ReactPose
from challenge_tools_lib import ReactTime
from challenge_tools_lib import ChallengeToolsLib


class GTServer(Node):
    def __init__(self):
        super().__init__('groundtruth_server')
        self.pub_path = self.create_publisher(Path, '/gt_poses', 10)
        self.fixed_frame = "map"

        self.ctl = ChallengeToolsLib()
        self.ctl.fixed_frame = self.fixed_frame


    def do_work(self, run_name):

        try:
            path = get_package_share_directory("challenge_tools_ros")
            print("share dir:", path)
        except PackageNotFoundError:
            print(f"Package challenge_tools_ros not found in the ament index.")
            return

        main_folder = os.path.join(path, "groundtruth")
        gt_filename = os.path.join(main_folder, f"{run_name}.txt")
        print("Reading: ", gt_filename)

        if not os.path.isfile(gt_filename):
            self.get_logger().error(f"Ground truth file not found: {gt_filename}")
            return

        gt_poses = self.load_tum_txt(gt_filename)

        while (1==1):
            self.get_logger().info(f"publish ground truth for: {run_name}")
            self.ctl.publish_trajectory(self.pub_path, gt_poses)
            time.sleep(1)


    def load_tum_txt(self, filename):
        poses = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue

                ts = float(parts[0])
                sec = int(ts)
                nsec = int((ts - sec) * 1e9)

                pose = ReactPose(
                    pos=[float(parts[1]), float(parts[2]), float(parts[3])],
                    quat=[float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                )
                pose_t = (ReactTime(sec, nsec), pose)
                poses.append(pose_t)

        return poses


def main(args=None):
    rclpy.init(args=args)

    args = sys.argv[1:]
    num_args = len(args)
    print("Usage:")
    print("  groundtruth_server.py run_name")
    print(" ")
    if (num_args < 1):
        print("1 required argument.", num_args, "provided.")
        print("quitting")
        return
    
    print("Using commandline configuration:")
    run_name = args[0]
    print("  run_name:", run_name)
    print(" ")

    app = GTServer()
    app.do_work(run_name)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
