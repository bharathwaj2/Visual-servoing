#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np




class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.listener_callback, 10)
        self.subscription  # prevent unused variable warning

        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.joint_sub

        self.depth_sub = self.create_subscription(
            Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.depth_sub
        
        self.joint_publisher = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        
        self.vel_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)
        
        
        self.init_positions = []
        self.target_positions = []
        self.target_depths = []

        self.cnt = 0
        self.cnt_t = 0
        self.num = 0
        # self.target = np.load(f'/home/bharath/piper_ws/src/piper_arm/ibvs/src/Files/RL_target_pose_{self.num-1}.npy')

        self.init_pos = np.load('/home/bharath/piper_ws/src/piper_arm/ibvs/src/Files/initial_pose_1.npy')
        self.get_logger().info(f"Initial pose: {self.init_pos}") 

        self.Z = np.zeros((4,1))
        self.joint_angles = None
        self.depth_image = None

        # self.timer = self.create_timer(2.0, self.send_pose)


        # Try the most common ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

        # Create detector with relaxed parameters
        parameters = cv2.aruco.DetectorParameters()

        # Make detection more permissive
        parameters.minMarkerPerimeterRate = 0.01  # Smaller markers
        parameters.maxMarkerPerimeterRate = 4.0   # Larger markers
        parameters.polygonalApproxAccuracyRate = 0.05  # Less strict polygon approximation
        parameters.minCornerDistanceRate = 0.01   # Closer corners allowed
        parameters.minDistanceToBorder = 1        # Markers can be closer to border

        # Adaptive threshold parameters
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 4
        parameters.adaptiveThreshConstant = 4

        # Create detector
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


    def joint_callback(self, msg):

        angles = msg.position
        self.joint_angles = [angles[4], angles[0], angles[1], angles[5], angles[2], angles[3]]


    def depth_callback(self, msg):

        data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        self.depth_image  = data / 1000


    def listener_callback(self, msg: Image):
        try:
            if self.joint_angles is not None and self.depth_image is not None:

                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                # Detect markers
                corners, ids, rejected = self.detector.detectMarkers(cv_image)

                # Example: Print feature points for each detected marker
                if ids is not None:

                    for marker_corners in corners:
                        
                        self.pts = marker_corners.reshape((4, 2))

                        for m in range(0,4):
                            self.Z[m] = self.depth_image[int(self.pts[m,1]),int(self.pts[m,0])]
                        
                        for (x, y) in self.pts:
                            cv2.circle(cv_image, (int(x), int(y)), 6, (0, 0, 255), -1)  # Red dot, size 6
                    
                    # # Draw target corners
                    # for i, (x, y) in enumerate(self.target):
                    #     cv2.circle(cv_image, (int(x), int(y)), 6, (0, 255, 0), -1)
                    #     cv2.putText(cv_image, f"T{i}", (int(x)+10, int(y)), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                

                # else:
                #     print("No markers detected.")

                # e.g. display the CV image:
                cv2.imshow('Camera view', cv_image)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('f'):
                    self.save_target_pose()
                elif key == ord('t'):
                    self.add_target_pose()
                elif key == ord('n'):
                    self.send_pose()
                elif key == ord('a'):
                    self.add_initial_pose()
                elif key == ord('i'):
                    self.save_initial_pose()


        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error: {e}")


    def add_initial_pose(self):
    
        self.init_positions.append(self.joint_angles)
        self.get_logger().info(f"\nInitial_pose {self.cnt+1}: {self.joint_angles}")
        self.cnt+=1

    def add_target_pose(self):
    
        self.target_positions.append(self.pts)
        self.target_depths.append(self.Z)
        self.get_logger().info(f"\nFinal_pose {self.cnt_t+1}: \n{self.pts}")
        self.cnt_t+=1

    def save_initial_pose(self):
            
        np.save(f'/home/bharath/piper_ws/src/piper_arm/ibvs/src/Files/RL_initial_pose',self.init_positions)


    def save_target_pose(self):
            
        np.save(f'/home/bharath/piper_ws/src/piper_arm/ibvs/src/Files/RL_target_pose',self.target_positions)
        np.save(f'/home/bharath/piper_ws/src/piper_arm/ibvs/src/Files/RL_target_depth',self.target_depths)
        
        # self.get_logger().info(f"final_target: {self.pts}")  
        # self.get_logger().info(f"final_depth: {self.Z}")  


    def send_pose(self):

        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        point = JointTrajectoryPoint()

        rand = np.random.uniform(-0.20,0.20,6).astype(np.float32)

        pos = self.init_pos + rand
        point.positions = pos.flatten().tolist()

        point.time_from_start.sec = 2
        msg.points.append(point)
        self.joint_publisher.publish(msg)
        


def main(args=None):
    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


