#!/usr/bin/env python3

# set_initial_pose.py
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import numpy as np

class InitialPosePublisher(Node):
    def __init__(self):
        super().__init__('initial_pose_publisher')
        
        self.publisher_ = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10 
            )
        
        self.timer = self.create_timer(2.0, self.send_pose)  # send after delay

    def send_pose(self):
        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        point = JointTrajectoryPoint()

        init_pos = np.load('/home/bharath/visual_servoing/src/piper_arm/ibvs/src/Files/Initial_arm_pose.npy')

        # your desired angles
        self.get_logger().info(f"Starting with initial joint pose : {init_pos}")
        point.positions = list(init_pos)

        # point.positions = [0.0, 0.0, 0.0, 0.0]

        point.time_from_start.sec = 2
        msg.points.append(point)
        self.publisher_.publish(msg)
        
        self.timer.cancel()  # send only once

def main():
    rclpy.init()
    node = InitialPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
