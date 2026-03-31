#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TwistStamped, PoseStamped, TransformStamped
from cv_bridge import CvBridge
import cv2
import tf2_ros
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

        
class visual_servoingController(Node):
    def __init__(self):
        super().__init__('visual_servoing_hardware_controller')

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Camera info subscription
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)
   
        # Image subscription for ArUco detection
        self.subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.listener_callback, 10)
        
        # Depth subscription for 3D pose estimation
        self.depth_sub = self.create_subscription(
            Image, '/camera/camera/depth/image_rect_raw', self.depth_callback, 10)

        # Velocity publisher
        self.vel_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)

        self.n = 4  # Number of ArUco corners
        self.num = 3
        
        self.dir = f'/home/bharath/visual_servoing/src/piper_arm/visual_servoing/src/Files'

        self.target = np.load(f'{self.dir}/target_pose_5.npy')

        self.HEM = np.load(f'{self.dir}/Hand_Eye_matrix_real_arm_0_inv.npy')
        # self.HEM = np.load(f'{self.dir}/HEM_C_to_E.npy')
        # self.HEM = np.linalg.inv(hem)

        # visual_servoing control parameters using interaction matrix
        self.lamda = np.ones((6,))*4
        # self.lamda[3] = 0.0

        self.lamda = self.lamda.flatten()

        self.get_logger().info(f'lambda : {self.lamda}')
        
        self.limits = False
        self.vel_limit = 0.5
        self.ang_vel_limit = 0.75

        self.R_rel = None
        
        # Tolerance parameters for visual_servoing
        self.rotation_tolerance = 10.0
        self.translation_tolerance = 0.02 
        
        # Target offset in marker frame (z-offset)
        self.target_offset_distance = 0.20
        
        # Camera parameters
        self.camera_matrix = None
        self.depth_image = None

        # Data logging
        self.error_data = []
        self.velocity_data = []

        self.stop = False
        self.quit = False
        self.frame_id = 0
        self.cycle = 0

        # Directory for saving data
        self.directory = f'{self.dir}/visual_servoing_{self.num}_{self.lamda}'
        # os.makedirs(self.directory, exist_ok=True)
        
        # Optical frame transformation
        # self.optical_to_standard = self.create_optical_transform()
        
        # ArUco marker size (in meters)
        self.marker_size = 0.072

        self.c_star_t_o = np.array([0.0, 0.0, self.target_offset_distance])  # Desired position
        # self.theta_u_star = np.array([0, 0, 0])  # Desired angle-axis (zero rotation)


        # ArUco detection setup
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        # Relaxed detection parameters
        parameters.minMarkerPerimeterRate = 0.01
        parameters.maxMarkerPerimeterRate = 4.0
        parameters.minMarkerDistanceRate = 0.02
        parameters.polygonalApproxAccuracyRate = 0.05
        parameters.minCornerDistanceRate = 0.01
        parameters.minDistanceToBorder = 1
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 4
        parameters.adaptiveThreshConstant = 4

        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


    def info_callback(self, msg):
        """Extract camera intrinsic parameters"""
        k = msg.k
        self.camera_matrix = np.array([[k[0], k[1], k[2]],
                                      [k[3], k[4], k[5]],
                                      [0, 0, 1]], dtype=np.float32)
        

    def depth_callback(self, msg):
        """Process depth image"""
        data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        self.depth_image = data / 1000.0  # Convert to meters

        

    def listener_callback(self, msg: Image):
        """Main callback - detect ArUco and compute visual_servoing control"""
        
        if self.depth_image is not None and self.camera_matrix is not None:

            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            
            corners, ids, rejected = self.detector.detectMarkers(self.image)
            
            # Process detected markers for visual_servoing
            if ids is not None and len(corners) > 0:

                if not self.quit:

                    marker_corners = corners[0]
                    self.pose = marker_corners.reshape((4, 2))
                    
                    # Estimate 3D pose of marker
                    self.estimate_marker_pose()
                    
                    # Execute visual_servoing control step with interaction matrix
                    self.visual_servoing_control_step()

                    # Visualization
                    for i, (x, y) in enumerate(self.pose):
                        cv2.circle(self.image, (int(x), int(y)), 6, (0, 0, 255), -1)
                        cv2.putText(self.image, str(i), (int(x)+10, int(y)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        
                    filename = os.path.join(self.directory, f"frame_{self.frame_id}.png")
                    # cv2.imwrite(filename, self.image)
                    self.frame_id += 1

                    self.cycle += 1

                else:

                    marker_corners = corners[0]
                    self.pose = marker_corners.reshape((4, 2))
                    
                    # Estimate 3D pose of marker
                    self.estimate_marker_pose()
                    
            else:
                self.get_logger().warn("NO MARKER detected")

            cv2.imshow('Camera view', self.image)             
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.quit = True
            

    
    def estimate_marker_pose(self):
        """Estimate 3D pose of ArUco marker in camera frame"""
        
        corners_array = [self.pose.reshape((1, 4, 2))]
        
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners_array,
            self.marker_size,
            self.camera_matrix,
            None
        )
        
        if rvecs is not None and tvecs is not None:
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            
            # Convert rotation vector to rotation matrix
            R_curr, _ = cv2.Rodrigues(rvec)

            R_des = self.target[:3, :3] 

            # Apply 180° rotation around X-axis (ArUco to camera frame convention)
            rot_x_180 = R.from_euler('x', 180, degrees=True).as_matrix()
            R_curr = R_curr @ rot_x_180

            self.R_rel = R_des @ R_curr.T

            # Store pose in camera frame
            self.marker_rotation = R.from_matrix(self.R_rel)
            self.c_t_o = tvec  
            self.marker_euler = self.marker_rotation.as_euler('xyz', degrees=True)
            
            # Compute angle-axis representation
            self.theta_u = self.rotation_to_angle_axis(self.marker_rotation)

            # self.publish_marker_tf(rvec,tvec)
            
            # self.get_logger().info(f"Marker in camera frame - c_t_o: {self.c_t_o}, θu: {self.theta_u}")
        else:
            self.get_logger().error("Marker pose estimation failed")



    def publish_marker_tf(self, rvec, tvec):

        R_curr, _ = cv2.Rodrigues(rvec)
        r = R.from_matrix(R_curr[0:3, 0:3])
        quat = r.as_quat()

        
        tf_msg = TransformStamped()

        tf_msg.header.stamp = self.get_clock().now().to_msg()

        tf_msg.header.frame_id = 'camera_color_optical_frame'
        tf_msg.child_frame_id = 'aruco_marker'


        tf_msg.transform.translation.x = tvec[0]
        tf_msg.transform.translation.y = tvec[1]
        tf_msg.transform.translation.z = tvec[2]
        tf_msg.transform.rotation.x = quat[0]
        tf_msg.transform.rotation.y = quat[1]
        tf_msg.transform.rotation.z = quat[2]
        tf_msg.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(tf_msg)


    def rotation_to_angle_axis(self, rotation):
        """Convert rotation to angle-axis (θu) representation
        
        Args:
            rotation: scipy Rotation object
            
        Returns:
            theta_u: 3D vector representing angle-axis (θu)
        """
        # Get rotation vector (axis-angle in compact form)
        rotvec = rotation.as_rotvec()  # This gives θ*u (angle * unit_axis)
        
        # rotvec already represents θu
        return rotvec


    def angle_axis_to_rotation(self, theta_u):
        """Convert angle-axis (θu) to rotation matrix
        
        Args:
            theta_u: 3D vector representing angle-axis
            
        Returns:
            R: 3x3 rotation matrix
        """
        return R.from_rotvec(theta_u).as_matrix()


    def compute_L_theta_u(self, theta_u):
        """Compute L_θu matrix from equation (14) in the paper
        
        L_θu = I_3 - (θ/2)[u]_× + (1 - sinc(θ)/sinc²(θ/2))[u]²_×
        
        Args:
            theta_u: angle-axis vector (θu)
            
        Returns:
            L_theta_u: 3x3 interaction matrix for angle-axis
        """
        theta = np.linalg.norm(theta_u)
        
        if theta < 1e-6:
            # Near zero rotation, L_θu ≈ I
            return np.eye(3)
        
        u = theta_u / theta  # Unit axis
        u_skew = self.skew_symmetric(u)
        u_skew_squared = u_skew @ u_skew
        
        # Compute sinc functions
        sinc_theta = np.sinc(theta / np.pi)  # np.sinc(x) = sin(πx)/(πx)
        sinc_half_theta = np.sinc(theta / (2 * np.pi))
        
        # Compute L_θu
        I3 = np.eye(3)
        term1 = (theta / 2) * u_skew
        
        # Handle division carefully
        if abs(sinc_half_theta) < 1e-10:
            term2_coeff = 0
        else:
            term2_coeff = 1 - (sinc_theta / (sinc_half_theta ** 2))
        
        term2 = term2_coeff * u_skew_squared
        
        L_theta_u = I3 - term1 + term2
        
        return L_theta_u


    def compute_interaction_matrix_inverse(self, c_t_o, theta_u):
        """Compute inverse of interaction matrix L_e^-1 from equation (15)
        
        For s = (c_t_o, θu):
        
        L_e^-1 = [-I_3    [c_t_o]_× L_θu^-1]
                 [  0           L_θu^-1      ]180
        
        Args:
            c_t_o: translation vector from camera to object
            theta_u: angle-axis representation of rotation
            
        Returns:
            L_e_inv: 6x6 inverse interaction matrix
        """
        I3 = np.eye(3)
        
        # Compute L_θu and its inverse
        L_theta_u = self.compute_L_theta_u(theta_u)
        
        try:
            L_theta_u_inv = np.linalg.inv(L_theta_u)
        except np.linalg.LinAlgError:
            self.get_logger().warn("L_theta_u is singular, using pseudo-inverse")
            L_theta_u_inv = np.linalg.pinv(L_theta_u)
        
        # Compute skew-symmetric matrix of c_t_o
        c_t_o_skew = self.skew_symmetric(c_t_o)
        
        # Build L_e^-1
        L_e_inv = np.zeros((6, 6))
        L_e_inv[0:3, 0:3] = -I3
        L_e_inv[0:3, 3:6] = c_t_o_skew @ L_theta_u_inv
        L_e_inv[3:6, 0:3] = np.zeros((3, 3))
        L_e_inv[3:6, 3:6] = L_theta_u_inv
        
        return L_e_inv


    def visual_servoing_control_step(self):
        """Main visual_servoing control logic using interaction matrix
        
        Implements control law from equation (16):
        v_c = -λ((c*_t_o - c_t_o) + [c_t_o]_× θu)
        ω_c = -λ θu
        """
        
        # Compute error: e = s - s* = (c_t_o - c*_t_o, θu - θu*)
        e_translation = self.c_t_o - self.c_star_t_o
        # e_translation = np.zeros((3,))
        e_rotation = self.theta_u 
        
        # Stack error vector: e = [e_translation; e_rotation]
        e = np.concatenate([e_translation, e_rotation])
        
        
        # Check if goal is reached
        translation_error_norm = np.linalg.norm(e_translation)
        rotation_error_norm = np.linalg.norm(e_rotation)
        rotation_error_deg = np.rad2deg(rotation_error_norm)

        self.get_logger().info(f"Error \ntranslation: {round(translation_error_norm,4)}, rotation: {round(rotation_error_deg,4)}, Steps: {self.frame_id}")

        
        if (translation_error_norm < self.translation_tolerance and 
            rotation_error_deg < self.rotation_tolerance):
            # self.get_logger().info(f'TARGET REACHED - Stopping \nNo. of Steps : {self.cycle}')
            self.publish_zero_velocity()
            self.stop = True
            # return
        
        
        linear_vel, angular_vel = self.control_law_2(e)

        
        # self.get_logger().info(f"Camera velocity - linear: {linear_vel}, angular: {angular_vel}")
        
        # Log data
        self.error_data.append(e)
        # self.velocity_data.append(v_c)
        
        # Publish velocity command
        self.publish_velocity(linear_vel, angular_vel)
        
        self.get_logger().info('----------------------------------------------')


    def control_law_1(self, e):

        # Compute L_e^-1 using current c_t_o and θu
        L_e_inv = self.compute_interaction_matrix_inverse(self.c_t_o, self.theta_u)
        
        # Apply control law: v_c = -λ L_e^-1 e
        v_c = -self.lamda * (L_e_inv @ e.reshape(6, 1))
        v_c = v_c.flatten()
        
        vc = v_c[0:3]
        wc = v_c[3:6]

        return vc, wc
    

    def control_law_2(self, e):

        e_translation = e[0:3]
        c_t_o_skew = self.skew_symmetric(self.c_t_o)

        vc = -self.lamda[0:3] * (-e_translation + c_t_o_skew @ self.theta_u)
        wc = -self.lamda[3:6] * self.theta_u

        return vc, wc
    

    def control_law_3(self, e):

        vc = -self.lamda * (self.R_rel.T @ self.c_star_t_o)
        wc = -self.lamda * self.theta_u

        return vc, wc
    

    def publish_velocity(self, linear_vel, angular_vel):
        """Publish velocity command to robot in end-effector frame"""
        # Transform velocity from camera frame to end-effector frame
        vel_camera = np.concatenate([linear_vel, angular_vel])
        vel_ee = self.transform_velocity_camera_to_ee(vel_camera)
        
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'end_effector_link'
        
        msg.twist.linear.x = float(vel_ee[0])
        msg.twist.linear.y = float(vel_ee[1])
        msg.twist.linear.z = float(vel_ee[2])
        
        msg.twist.angular.x = float(vel_ee[3])
        msg.twist.angular.y = float(vel_ee[4])
        msg.twist.angular.z = float(vel_ee[5])
        
        self.vel_pub.publish(msg)
        # self.get_logger().info(f"Published vel (EE frame): linear={vel_ee[:3]}, angular={vel_ee[3:]}")


    def publish_zero_velocity(self):
        """Publish zero velocity to stop robot"""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'end_effector_link'
        
        msg.twist.linear.x = 0.0
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0
        
        self.vel_pub.publish(msg)

    
    def create_optical_transform(self):
        """Create transformation from optical frame to standard ROS frame"""
        optical_to_std = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])
        return optical_to_std
    
    
    def transform_velocity_camera_to_ee(self, vel_camera):
        """Transform velocity from camera frame to end-effector frame
        
        Uses adjoint transformation for spatial velocities
        
        Args:
            vel_camera: 6x1 array [vx, vy, vz, wx, wy, wz] in camera frame
            
        Returns:
            vel_ee: 6x1 array [vx, vy, vz, wx, wy, wz] in end-effector frame
        """
        try:

            trans_ee_to_cam = self.HEM[:3, 3]
            rot_ee_to_cam = self.HEM[:3, :3]

            t_skew = self.skew_symmetric(trans_ee_to_cam)
            vel_transform = np.zeros((6, 6))
            vel_transform[:3, :3] = rot_ee_to_cam
            vel_transform[:3, 3:] = t_skew @ rot_ee_to_cam
            vel_transform[3:, 3:] = rot_ee_to_cam
            
            # Transform velocity
            vel_camera_6d = vel_camera.reshape((6, 1))
            vel_ee = vel_transform @ vel_camera_6d

            if self.limits:
                lin_vel = np.clip(vel_ee[0:3], -self.vel_limit, self.vel_limit)
                ang_vel = np.clip(vel_ee[3:6], -self.ang_vel_limit, self.ang_vel_limit)

                vel_ee = np.concatenate([lin_vel,ang_vel])

            return vel_ee.flatten()
            
        except Exception as e:
            self.get_logger().error(f"Failed to transform velocity: {e}")
            return np.zeros(6)
    
    
    def skew_symmetric(self, v):
        """Create skew-symmetric matrix from vector for cross product
        
        [v]_× = [ 0   -v3   v2]
                [ v3   0   -v1]
                [-v2   v1   0 ]
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])



def main(args=None):
    rclpy.init(args=args)
    node = visual_servoingController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()