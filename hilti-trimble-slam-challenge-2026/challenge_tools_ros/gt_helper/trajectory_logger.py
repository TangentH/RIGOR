#!/usr/bin/env python
# Tool to log VO poses from TF into a TUM-format text file

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TransformStamped, PoseWithCovarianceStamped
from tf2_ros import TransformException
import tf2_ros
from rclpy.duration import Duration


import sys

from challenge_tools_lib import ReactPose
from challenge_tools_lib import ReactTime
from challenge_tools_lib import ChallengeToolsLib

class GTPub(Node):
    def __init__(self, output_filename):
        super().__init__('trajectory_logger')
        self.fixed_frame = "map"

        self.subscription2 = self.create_subscription(PoseWithCovarianceStamped, "/ov_msckf/poseimu", self.pose_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.ctl = ChallengeToolsLib()
        self.ctl.fixed_frame = self.fixed_frame

        self.poses_t = []

        self.full_filename_out = output_filename

        self.poses_file = open(self.full_filename_out, 'w')
        self.poses_file.write("# timestamp tx ty tz qx qy qz qw\n")
        self.poses_file.flush()
        self.counter = 0
        self.get_logger().info(f"Logging trajectory to: {self.full_filename_out}")


    def write_line_to_file(self, pose_t):
        this_time = pose_t[0]
        pose = pose_t[1]
        timestamp = f"{int(this_time.sec)}.{int(this_time.nsec):09d}"
        line_out = (
            f"{timestamp} "
            f"{pose.pos[0]} {pose.pos[1]} {pose.pos[2]} "
            f"{pose.quat[0]} {pose.quat[1]} {pose.quat[2]} {pose.quat[3]}\n"
        )

        self.poses_file.write(line_out)
        self.poses_file.flush()    

        self.counter = self.counter + 1


    def path_callback(self, msg):
        print("got path")

    def pose_callback(self, msg):
        print("got pose_callback - ", self.counter)

        from_frame = 'map'
        to_frame   = 'imu'

        current_time =  rclpy.time.Time.from_msg(msg.header.stamp)
        offset = Duration(seconds=1.0) # go back by one second to avoid tf look up errors
        older_time = current_time - offset

        try:
            # lookup transform
            t: TransformStamped = self.tf_buffer.lookup_transform(
                from_frame,
                to_frame,
                older_time
            )

            self.get_logger().info(
                f"Transform {from_frame} → {to_frame}: "
                f"({t.transform.translation.x:.2f}, "
                f"{t.transform.translation.y:.2f}, "
                f"{t.transform.translation.z:.2f})"
            )

            older_time_msg = older_time.to_msg()
            this_time = ReactTime(older_time_msg.sec, older_time_msg.nanosec)

            pos = [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]
            quat = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
            this_pose = ReactPose(pos, quat)
            this_pose_t = (this_time, this_pose)
            self.write_line_to_file(this_pose_t)

        except TransformException as ex:
            self.get_logger().warn(f"Could not transform {from_frame} → {to_frame}: {ex}")


        return



def main(args=None):
    rclpy.init(args=args)

    cli_args = sys.argv[1:]
    if len(cli_args) < 1:
        print("Usage: trajectory_logger.py <output_filename.txt>")
        print("  output_filename.txt: TUM-format output file (e.g., floor_1_2025-05-05_run_1.txt)")
        rclpy.shutdown()
        return

    output_filename = cli_args[0]

    app = GTPub(output_filename)
    rclpy.spin(app)
    app.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
