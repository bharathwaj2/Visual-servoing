#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
import tf2_ros
import numpy as np
import os

        
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # QoS profile for better synchronization
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
   
        # self.cam_info_sub

        self.subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.listener_callback, 10)
        
        # self.subscription
        
        self.depth_sub = self.create_subscription(
            Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)


        self.vel_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)

        self.n = 4

        self.num = 1
        self.dir = '/home/bharath/visual_servoing/src/piper_arm/ibvs/src/Files'
        

        self.target = np.load(f'{self.dir}/Target_pixel_pose.npy')
        self.Z_t = np.load(f'{self.dir}/Target_pixel_depth.npy')

        self.get_logger().info(f"Target: {self.target}")


        self.s_p = np.array(self.target).flatten()


        # Velocity limits
        self.vel = 0.01
        self.ang_vel = 0.025

        # Hyperparameter
        self.mu = 0.5


        self.Le = np.zeros((int(2*self.n),6), dtype=np.float32)
        self.Le_s = np.zeros((int(2*self.n),6), dtype=np.float32)
        self.Les = np.zeros((int(2*self.n),6), dtype=np.float32)
        self.Lsp = None

        self.Z = np.zeros((self.n,1))
        # self.get_logger().info(f'shape: {self.Z.shape}')

        self.camera_matrix = None
        self.depth_image = None


        self.HEM = None

        self.cam_v = []
        self.depth_data = []
        self.error_data = []

        self.not_first = True
        self.stop = False
        self.frame_id = 0

        self.goal_reached = False

        self.directory = f'{self.dir}/IBVS_{self.num}_{self.vel}_{self.ang_vel}'
        os.makedirs(self.directory, exist_ok=True)
        
    def info_callback(self, msg):

        k = msg.k
        self.camera_matrix = np.array([[k[0], k[1], k[2]],
                                      [k[3], k[4], k[5]],
                                      [0, 0, 1]], dtype=np.float32)
        

    def depth_callback(self, msg):

        data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')

        self.depth_image  = data / 1000

        u = np.unique(self.depth_image)

        

    def listener_callback(self, msg: Image):

        if self.depth_image is not None and not self.stop and self.camera_matrix is not None:

            self.calc_cam_ee_conv_matrix()
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Try the most common ArUco dictionary
            aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

            # Create detector with relaxed parameters
            parameters = cv2.aruco.DetectorParameters()

            # Make detection more permissive
            parameters.minMarkerPerimeterRate = 0.01  # Smaller markers
            parameters.maxMarkerPerimeterRate = 4.0   # Larger markers
            parameters.minMarkerDistanceRate = 0.02
            parameters.polygonalApproxAccuracyRate = 0.05  # Less strict polygon approximation
            parameters.minCornerDistanceRate = 0.01   # Closer corners allowed
            parameters.minDistanceToBorder = 1        # Markers can be closer to border

            # Adaptive threshold parameters
            parameters.adaptiveThreshWinSizeMin = 3
            parameters.adaptiveThreshWinSizeMax = 23
            parameters.adaptiveThreshWinSizeStep = 4
            parameters.adaptiveThreshConstant = 4

            # Create detector
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

            # Detect markers
            corners, ids, rejected = detector.detectMarkers(self.image)
            
            if ids is not None and len(corners) > 0 and self.HEM is not None:
                marker_corners = corners[0]
                self.pose = marker_corners.reshape((4, 2))
                
                # FIXED: Update s properly
                self.s = self.pose.flatten()

                self.e = self.s - self.s_p
                self.e_T = np.array(self.e).T.reshape((int(self.n*2),1))
                norm_err = np.linalg.norm(self.e_T)
                # self.mu = norm_err / 100

                self.get_logger().info(f"Error norm: {norm_err:.3f}")
                
                # self.step()

                # Draw detected corners
                
                for i, (x, y) in enumerate(self.pose):
                    cv2.circle(self.image, (int(x), int(y)), 6, (0, 0, 255), -1)
                    cv2.putText(self.image, str(i), (int(x)+10, int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Draw target corners
                for i, (x, y) in enumerate(self.target):
                    cv2.circle(self.image, (int(x), int(y)), 6, (0, 255, 0), -1)
                    cv2.putText(self.image, f"T{i}", (int(x)+10, int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                filename = os.path.join(self.directory, f"frame_{self.frame_id}.png")
                cv2.imwrite(filename, self.image)

                self.frame_id +=1
                    
            else:
                self.get_logger().warn("No markers detected.")

            cv2.imshow('Camera view', self.image)             
            cv2.waitKey(1)

    

    def step(self):
        
        for m in range(0,self.n):
            self.Z[m] = self.depth_image[int(self.pose[m,1]),int(self.pose[m,0])]

        self.calc_img_jacobian()


        self.v_c = -self.mu*np.dot(self.Lsp , self.e_T)     

        self.get_logger().info(f"cam_vel: \n{self.v_c}")
                

        self.v_e = np.dot(self.ce_trans , self.v_c)       

        # vel = 0.01
        self.v_e[:3,0] = np.clip(self.v_e[:3,0], -self.vel, self.vel)
        self.v_e[3:,0] = np.clip(self.v_e[3:,0], -self.ang_vel, self.ang_vel)

        self.get_logger().info(f"ee_vel: \n{self.v_e}")


        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        # msg.header.frame_id = 'end_effector_link'

        self.cam_v.append(self.v_c)
        np.save(f'{self.dir}/IBVS_cam_vel_{self.num}',self.cam_v)

        
        err_norm = np.linalg.norm(self.e_T)
        if err_norm >= 5:
        # Set linear velocities
            msg.twist.linear.x = self.v_e[0,0]
            msg.twist.linear.y = self.v_e[1,0]
            msg.twist.linear.z = self.v_e[2,0]
            
            # Set angular velocities
            msg.twist.angular.x = self.v_e[3,0]
            msg.twist.angular.y = self.v_e[4,0]
            msg.twist.angular.z = self.v_e[5,0]

        else:
            msg.twist.linear.x = 0.0
            msg.twist.linear.y = 0.0
            msg.twist.linear.z = 0.0
            
            # Set angular velocities
            msg.twist.angular.x = 0.0
            msg.twist.angular.y = 0.0
            msg.twist.angular.z = 0.0

            self.stop = True
        
        # Publish the message
        self.vel_pub.publish(msg)


    def calc_cam_ee_conv_matrix(self):

        self.HEM = self.get_CtE()

        eRc = self.HEM[:3,:3]
        etc = self.HEM[:3,3]

        # r_t = np.dot(np.transpose(eRc) , etc)
        r_t = etc

        etc_x = np.array([
                        [0, -r_t[2], r_t[1]],
                        [r_t[2], 0, -r_t[0]],
                        [-r_t[1], r_t[0], 0]
                        ])

        self.ce_trans = np.zeros([6,6])

        self.ce_trans[:3,:3] = eRc
        self.ce_trans[:3,3:] = np.dot(etc_x , eRc)
        # self.ce_trans[:3,3:] = -np.dot(eRc , etc_x)
        self.ce_trans[3:,3:] = eRc

    
    def calc_img_jacobian(self):
        
        px = self.camera_matrix[0,0]
        py = self.camera_matrix[1,1]
        u0 = self.camera_matrix[0,2]
        v0 = self.camera_matrix[1,2]

        

        zi = 1/self.Z
        # self.get_logger().info(f"depth inv: \n{zi}")
        self.depth_data.append(zi)
        np.save(f'{self.dir}/IBVS_depth_data_{self.num}',self.depth_data)
        

        self.error_data.append(self.e_T)
        np.save(f'{self.dir}/IBVS_error_data_{self.num}',self.error_data)

        for m in range(0,4):

            x = (self.s[m*2] - u0)/px
            y = (self.s[m*2+1] - v0)/py

            Zinv = 1/self.Z[m,0]

            self.Le[m*2:m*2+2,:] = np.matrix([[-Zinv, 0, x*Zinv, x*y, -(1+x**2), y] , [0, -Zinv, y*Zinv, 1+y**2, -x*y, -x]])


        for m in range(0,4):

            x = (self.s_p[m*2] - u0)/px
            y = (self.s_p[m*2+1] - v0)/py

            Zinv = 1/self.Z_t[m,0]
            # Zinv = 1/0.4

            self.Le_s[m*2:m*2+2,:] = np.matrix([[-Zinv, 0, x*Zinv, x*y, -(1+x**2), y] , [0, -Zinv, y*Zinv, 1+y**2, -x*y, -x]])


        self.L = (self.Le + self.Le_s)/2

        self.Lsp = np.linalg.pinv(self.L)
        self.not_first = False

        self.get_logger().info(f'Lsp shape: {self.Lsp.shape}')
      
    
    def get_CtE(self):
        # """Get current end-effector pose in base frame"""
        try:
            # Get transform from base to end-effector
            transform = self.tf_buffer.lookup_transform(
                'end_effector_link', 'camera_color_optical_frame', rclpy.time.Time())
            
            # Convert to 4x4 matrix
            t = transform.transform.translation
            q = transform.transform.rotation
            
            # Convert quaternion to rotation matrix
            from scipy.spatial.transform import Rotation as R
            r = R.from_quat([q.x, q.y, q.z, q.w])
            rot_matrix = r.as_matrix()
            
            pose = np.eye(4)
            pose[:3, :3] = rot_matrix
            pose[:3, 3] = [t.x, t.y, t.z]

            self.get_logger().info(f'\n{pose}')

            return pose
        
        except Exception as e:
            self.get_logger().error(f"Could not get robot pose: {e}")
            return None

def main(args=None):

    rclpy.init(args=args)
    node = ImageSubscriber()
    # node.step()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
















































        
    
        
    