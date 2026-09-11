#!/usr/bin/env python
# Tool to parse init_gt_poses.csv
# find the matching OpenVINS pose
# and publish to

# ros2 run challenge_tools_ros static_transform_file_publisher.py  floor_1_2025-05-05_run_1

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

import numpy as np
import sys
import os
import yaml
from nav_msgs.msg import Path

from challenge_tools_lib import ReactPose
from challenge_tools_lib import ReactTime
from challenge_tools_lib import ChallengeToolsLib

from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import PoseWithCovarianceStamped


def deg2rad(deg_in):
    return deg_in*np.pi/180.0


class StaticTransform(Node):
    def __init__(self):
        super().__init__('static_transform_node')

        self.fixed_frame = "map"

        self.ctl = ChallengeToolsLib()
        self.imu_to_cam0_mat = self.load_imu_to_cam0_mat()

        self.subscription = self.create_subscription(PoseWithCovarianceStamped, "/ov_msckf/poseimu", self.pose_callback, 10)

        self.static_initialized = False


    def load_imu_to_cam0_mat(self):
        try:
            share_dir = get_package_share_directory("challenge_tools_ros")
        except PackageNotFoundError:
            raise RuntimeError("Package challenge_tools_ros not found in the ament index.")

        yaml_path = os.path.join(share_dir, "config", "hilti_openvins", "kalibr_imucam_chain.yaml")
        try:
            with open(yaml_path, "r") as f:
                raw = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"Missing calibration file: {yaml_path}")
        # OpenCV-style YAML headers like "%YAML:1.0" are not valid for PyYAML.
        if raw.startswith("%YAML:"):
            raw = "\n".join(raw.splitlines()[1:])
        data = yaml.safe_load(raw)

        try:
            mat = np.array(data["cam0"]["T_cam_imu"], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise RuntimeError(f"Invalid cam0.T_cam_imu in {yaml_path}")

        if mat.shape != (4, 4):
            raise RuntimeError(f"cam0.T_cam_imu has invalid shape {mat.shape} in {yaml_path}")

        return mat


    def publishStaticTransformTrajectory(self, pub, these_poses, fixed_frame):
        if 'Path' not in sys.modules:
            from nav_msgs.msg import Path

        poses_ros = []
        for pose in these_poses:
            ps = self.ctl.pack_PoseStamped(pose[1][1])
            poses_ros.append(ps)

        msg = Path()
        msg.header.stamp.sec = 0 #time.sec
        msg.header.stamp.nanosec = 0# time.nsec
        msg.header.frame_id = fixed_frame # self.fixed_frame
        msg.poses = poses_ros
        pub.publish(msg)        


    def publishStaticTransform(self, pose, time):
        t = TransformStamped()
        t.header.stamp.sec = time.sec
        t.header.stamp.nanosec = time.nsec
        t.header.frame_id = 'map'
        t.child_frame_id = 'global' # 'gt0'

        t.transform.translation.x = pose.pos[0]
        t.transform.translation.y = pose.pos[1]
        t.transform.translation.z = pose.pos[2]

        q = pose.quat
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.broadcaster.sendTransform(t)
        self.get_logger().info('Static transform published')


    def publishPose(self, pub_pose, pose, time):
        msg = PoseWithCovarianceStamped()
        msg.header.stamp.sec = time.sec
        msg.header.stamp.nanosec = time.nsec
        msg.header.frame_id = self.fixed_frame
        msg.pose.pose.position.x = pose.pos[0]
        msg.pose.pose.position.y = pose.pos[1]
        msg.pose.pose.position.z = pose.pos[2] # xyzw
        msg.pose.pose.orientation.x = pose.quat[0]
        msg.pose.pose.orientation.y = pose.quat[1]
        msg.pose.pose.orientation.z = pose.quat[2]
        msg.pose.pose.orientation.w = pose.quat[3]
        pub_pose.publish(msg)


    def do_work(self, run_name):

        try:
            path = get_package_share_directory("challenge_tools_ros")
            print("share dir:", path)
        except PackageNotFoundError:
            print(f"Package challenge_tools_ros not found in the ament index.")

        csv_filename = path + "/groundtruth/init_gt_poses.csv"
        print("Reading: ", csv_filename)

        gt_poses = self.ctl.load_csv_as_poses_list_st(csv_filename)

        self.pose_gt0 = gt_poses[run_name]
        print (self.pose_gt0)

        self.broadcaster = StaticTransformBroadcaster(self)


    def pose_callback(self, msg):

        if (self.static_initialized):
            return

        current_t = ReactTime(msg.header.stamp.sec, msg.header.stamp.nanosec)
        print(current_t , " and " , self.pose_gt0[0])

        if (current_t > self.pose_gt0[0]):
            print("found match")

            p_map_cam0 = self.pose_gt0[1]

            # Unpack the OpenVINS IMU pose
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            z = msg.pose.pose.position.z
            pos = [x,y,z]
            qx = msg.pose.pose.orientation.x
            qy = msg.pose.pose.orientation.y
            qz = msg.pose.pose.orientation.z
            qw = msg.pose.pose.orientation.w
            quat = [qx,qy,qz,qw] #xyzw

            # global means global frame of openvins
            # T_a_b (4x4) or p_a_b (pos, quat): describing the pose of b expressed in frame a.
            p_global_imu = ReactPose(pos, quat)
            T_global_imu = self.ctl.pose_to_matrix(p_global_imu)

            T_cam0_imu = self.imu_to_cam0_mat
            T_imu_cam0 = np.linalg.inv(T_cam0_imu)
            T_global_cam0 = np.matmul(T_global_imu, T_imu_cam0)

            T_map_cam0 = self.ctl.pose_to_matrix(p_map_cam0)
            T_cam0_global = np.linalg.inv(T_global_cam0)
            T_map_global = np.matmul(T_map_cam0, T_cam0_global)
            p_map_global  = self.ctl.matrix_to_pose(T_map_global)

            quat = p_map_global.quat
            roll, pitch, yaw = self.ctl.quaternion_to_euler(quat[0], quat[1], quat[2], quat[3])
            print ("raw global_to_map in SLAM convention T_map_global: global->map")
            print (p_map_global.pos)
            print (roll*180.0/3.142, "roll")
            print (pitch*180.0/3.142, "pitch")
            print (yaw*180.0/3.142, "raw")

            # Inaccuracies of timing or Open VINS will mean that the pitch and roll will be non-zero. Force them to be zero.
            p_map_global_zero_pitch_roll = ReactPose(p_map_global.pos, self.ctl.euler_to_quaternion(0, 0, yaw ))
            self.publishStaticTransform(p_map_global_zero_pitch_roll, self.pose_gt0[0])

            self.static_initialized = True


def main(args=None):
    rclpy.init(args=args)

    args = sys.argv[1:]
    num_args = len(args)
    print("Usage:")
    print("  static_transform_publisher.py run_name")
    print(" ")
    if (num_args < 1):
        print("2 required arguments.", num_args, "provided.")
        print("quitting")
        return

    print("Using commandline configuration:")
    run_name = args[0]
    #run_name = "floor_1_2025-05-05_run_1"

    print("  run_name:", run_name)
    print(" ")

    app = StaticTransform()
    app.do_work(run_name)

    print("spinning")
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
