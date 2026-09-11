#!/usr/bin/env python
# Tool to plot two trajectories from TUM-format files

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

import numpy as np
import sys

from challenge_tools_lib import ReactPose
from challenge_tools_lib import ReactTime
from challenge_tools_lib import ChallengeToolsLib
from nav_msgs.msg import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class GTPub(Node):
    def __init__(self):
        super().__init__('groundtruth_publisher')
        self.pub_path = self.create_publisher(Path, '/gt_poses', 10)
        self.fixed_frame = "map"

        self.ctl = ChallengeToolsLib()
        self.ctl.fixed_frame = self.fixed_frame


    def posesT_to_pos_list(self, posesT):
        gt_pos = np.empty((0, 3))
        gt_rpy = np.empty((0, 3))
        gt_time = np.empty((0, 1))
        for this_poseT in posesT:
            this_pos = this_poseT[1].pos

            this_quat = this_poseT[1].quat
            this_time = this_poseT[0]
            gt_pos = np.vstack([gt_pos, this_pos])

            this_rpy = self.ctl.quaternion_to_euler_list(this_quat)
            gt_rpy = np.vstack([gt_rpy, this_rpy])

            this_time_double = this_time.sec + this_time.nsec*1e-9
            #print(this_time_double)
            gt_time = np.vstack([gt_time, this_time_double])

        return gt_pos, gt_rpy, gt_time

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

    def do_work(self, gt_path, vo_path, floorplan_path=None):
        gt_poses = self.load_tum_txt(gt_path)
        gt_pos, gt_rpy, gt_time = self.posesT_to_pos_list(gt_poses)
        print(len(gt_pos))

        vo_poses = self.load_tum_txt(vo_path)
        vo_pos, vo_rpy, vo_time = self.posesT_to_pos_list(vo_poses)
        print(len(vo_pos))

        vo_time = vo_time -  9999.25 #10000

        rad2deg = 180./np.pi

        if floorplan_path:
            img = mpimg.imread(floorplan_path)
            plt.figure(1)
            plt.imshow(img, origin='upper')

        plt.plot( gt_pos[:,0] , gt_pos[:,1], color='green')
        plt.plot( vo_pos[:,0] , vo_pos[:,1], color='purple')

        plt.xlabel("X Metres")
        plt.ylabel("Y Metres")
        plt.title("Green - GT | Purple - VO")

        fig, axes = plt.subplots(3, 1, figsize=(6, 10))
        axes[0].plot( gt_time , gt_pos[:,0], color='green' )
        axes[0].plot( vo_time , vo_pos[:,0], color='purple' )
        axes[0].set_title("X [m]")

        axes[1].plot( gt_time , gt_pos[:,1], color='green' )
        axes[1].plot( vo_time , vo_pos[:,1], color='purple' )
        axes[1].set_title("Y [m]")

        axes[2].plot( gt_time , gt_pos[:,2], color='green' )
        axes[2].plot( vo_time , vo_pos[:,2], color='purple' )
        axes[2].set_title("Z [m]")
        plt.tight_layout()

        fig, axes = plt.subplots(3, 1, figsize=(6, 10))
        axes[0].plot( gt_time , gt_rpy[:,0]*rad2deg, color='green' )
        axes[0].plot( vo_time , vo_rpy[:,0]*rad2deg, color='purple' )
        axes[0].set_title("Roll [deg]")

        axes[1].plot( gt_time , gt_rpy[:,1]*rad2deg, color='green' )
        axes[1].plot( vo_time , vo_rpy[:,1]*rad2deg, color='purple' )
        axes[1].set_title("Pitch [deg]")

        axes[2].plot( gt_time , gt_rpy[:,2]*rad2deg, color='green' )
        axes[2].plot( vo_time , vo_rpy[:,2]*rad2deg, color='purple' )
        axes[2].set_title("Yaw [deg]")
        plt.tight_layout()

        plt.show()


def main(args=None):
    rclpy.init(args=args)

    args = sys.argv[1:]
    if len(args) < 2:
        print("Usage:")
        print("  trajectory_plotter.py <gt_tum.txt> <vo_tum.txt> [floorplan.png]")
        print(" ")
        rclpy.shutdown()
        return

    gt_path = args[0]
    vo_path = args[1]
    floorplan_path = args[2] if len(args) > 2 else None

    app = GTPub()
    app.do_work(gt_path, vo_path, floorplan_path)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
