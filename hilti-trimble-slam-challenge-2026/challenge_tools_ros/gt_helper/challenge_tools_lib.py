import sys
import csv
import math
import numpy as np

class ReactPose:
    def __init__(self, pos=[0., 0., 0.], quat=[0., 0., 0., 1.]):
        self.pos = pos
        self.quat = quat # order is xyzw

    def __repr__(self):
        return f"Pose(pos={self.pos}, quat={self.quat})"

class ReactTime:
    def __init__(self, sec: int, nsec: int):
        self.sec = sec
        self.nsec = nsec

    def __repr__(self):
        return f"Time(sec={self.sec}, nsec={self.nsec})"

    def __eq__(self, other):
        if isinstance(other, ReactTime):
            return (self.sec, self.nsec) == (other.sec, other.nsec)
        return False  # Not a ReactTime instance

    def __gt__(self, other):
        t_this_long = self.sec*1e9 + self.nsec
        t_other_long = other.sec*1e9 + other.nsec
        return t_this_long > t_other_long

    def __lt__(self, other):
        t_this_long = self.sec*1e9 + self.nsec
        t_other_long = other.sec*1e9 + other.nsec
        return t_this_long < t_other_long

    def __ge__(self, other):
        print("ReactTime ge TODO: properly debug/test this")
        t_this_long = self.sec*1e9 + self.nsec
        t_other_long = other.sec*1e9 + other.nsec
        return t_this_long >= t_other_long

    def __le__(self, other):
        print("ReactTime le TODO: properly debug/test this")
        t_this_long = self.sec*1e9 + self.nsec
        t_other_long = other.sec*1e9 + other.nsec
        return t_this_long <= t_other_long

    def __hash__(self):
        return hash((self.sec, self.nsec))


class ChallengeToolsLib:
    def __init__(self):
        x = 0

    def convert_line_to_pose(self, line):
        xyz = [0., 0., 0.]
        quat = [0., 0., 0., 0.]
        xyz[0], xyz[1], xyz[2] = line[3], line[4], line[5]
        # order is xyzw
        quat[0], quat[1], quat[2], quat[3] = line[6], line[7], line[8], line[9]

        pose = ReactPose()
        pose.pos = xyz
        pose.quat = quat
        return pose


    def publish_trajectory(self, pub, these_poses):
        if 'Path' not in sys.modules:
            from nav_msgs.msg import Path

        poses_ros = []
        for pose in these_poses:
            ps = self.pack_PoseStamped(pose[1])
            poses_ros.append(ps)

        msg = Path()
        msg.header.stamp.sec = 0 #time.sec
        msg.header.stamp.nanosec = 0# time.nsec
        msg.header.frame_id = self.fixed_frame
        msg.poses = poses_ros
        pub.publish(msg)


    def pack_PoseStamped(self, pose):
        if 'PoseStamped' not in sys.modules:
            from geometry_msgs.msg import PoseStamped

        #print (pose)
        xyz = pose.pos# [:3] # first 3 [of 7]
        quat = pose.quat# [3:] # last 4 [of 7]

        msg = PoseStamped()
        #msg.header = msg.header
        msg.pose.position.x = xyz[0]
        msg.pose.position.y = xyz[1]
        msg.pose.position.z = xyz[2]
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        return msg


    def quaternion_to_euler(self, x, y, z, w):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


    def quaternion_to_euler_list(self, quat):
        roll, pitch, yaw = self.quaternion_to_euler(quat[0], quat[1], quat[2], quat[3])
        return [roll, pitch, yaw]
    

    def euler_to_quaternion(self, roll, pitch, yaw):
        """
        Converts euler roll, pitch, yaw to quaternion (w in last place)
        quat = [x, y, z, w]
        Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = [0] * 4
        q[0] = cy * cp * sr - sy * sp * cr
        q[1] = sy * cp * sr + cy * sp * cr
        q[2] = sy * cp * cr - cy * sp * sr
        q[3] = cy * cp * cr + sy * sp * sr # this was wrong in the source of this code - patched this

        return q
    

    def pose_to_matrix(self, pose):
        xyz = pose.pos # first 3 [of 7]
        quat = pose.quat # last 4 [of 7]
        matrix = self.quaternion_matrix(quat)
        matrix[0,3] = xyz[0]
        matrix[1,3] = xyz[1]
        matrix[2,3] = xyz[2]
        return matrix


    def matrix_to_pose(self, pose_matrix):

        quat = self.quaternion_from_matrix(pose_matrix)  # Convert to (x, y, z, w)

        xyz = [0., 0., 0.]
        xyz[0], xyz[1], xyz[2] = float(pose_matrix[0,3]), float(pose_matrix[1,3]), float(pose_matrix[2,3])
        pose = ReactPose(xyz, quat.tolist())
        return pose
    

    def quaternion_matrix(self, quaternion):
        """Return homogeneous rotation matrix from quaternion.

        >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
        >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
        True

        """
        _EPS = 2.220446049250313e-16

        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        nq = np.dot(q, q)
        if nq < _EPS:
            return np.identity(4)
        q *= math.sqrt(2.0 / nq)
        q = np.outer(q, q)
        return np.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
            (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
            (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
            (                0.0,                 0.0,                 0.0, 1.0)
            ), dtype=np.float64)


    # https://github.com/ros/geometry/blob/indigo-devel/tf/src/tf/transformations.py#L1196
    def quaternion_from_matrix(self, matrix):
        """Return quaternion from rotation matrix.

        >>> R = rotation_matrix(0.123, (1, 2, 3))
        >>> q = quaternion_from_matrix(R)
        >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
        True

        """
        q = np.empty((4, ), dtype=np.float64)
        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        t = np.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
        return q
    

    def load_csv_st(self, csv_filename):
        with open(csv_filename, mode ='r')as file:
            csvFile = csv.reader(file)

            lines = []
            for line in csvFile:
                if not line:
                    continue
                if not line[0].strip():
                    continue
                if (line[0].startswith("#")):
                    pass
                else:
                    this_line = []
                    for this_item in line[3:]:
                        this_line.append(float(this_item))

                    this_run = line[0]
                    this_time = line[2]
                    this_long_line = [this_run, this_time, this_line]
                    lines.append(this_long_line)

        return lines
    

    def convert_line_to_pose_st(self, line_in):
        line = [0,0,0] # counter, floor, time
        line.extend(line_in[2])

        xyz = [0., 0., 0.]
        quat = [0., 0., 0., 0.]
        xyz[0], xyz[1], xyz[2] = line[3], line[4], line[5]
        # order is xyzw
        quat[0], quat[1], quat[2], quat[3] = line[6], line[7], line[8], line[9]

        pose = ReactPose()
        pose.pos = xyz
        pose.quat = quat
        return pose


    def load_csv_as_poses_list_st(self, csv_filename):
        lines = self.load_csv_st(csv_filename)
        poses = {}
        for line in lines:
            pose = self.convert_line_to_pose_st(line)

            #print(line[1])
            this_time = float(line[1])
            sec = int(math.floor(this_time))
            #print (sec, "sec")
            nsec = int((this_time - sec)*1e9)
            #print (nsec, "nsec")
            this_time = ReactTime( sec, nsec)

            # counter is 0 and is lost
            poses[line[0]] = [this_time, pose]

        return poses
