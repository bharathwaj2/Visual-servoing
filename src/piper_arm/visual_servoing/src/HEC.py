#!/usr/bin/env python3

import numpy as np
import tf2_ros
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
# from scipy.spatial.transform import Rotation as R


class HandEyeCalibration(Node):
    def __init__(self):
        super().__init__('hand_eye_calibration')
        
        self.bridge = CvBridge()

        # TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.sub, 10)
        
        self.subscription
        
        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera/color/camera_info', self.info_callback, 10)
        
        self.cam_info_sub
        self.camera_matrix = None
        
        self.dist_coeffs = np.zeros((4,1)) 

        self.dir = '/home/bharath/visual_servoing/src/piper_arm/visual_servoing/src/Files'

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

        self.marker_size = 0.072

        # Create detector
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        # Storage for calibration data
        self.hand_poses = []  # End-effector poses
        self.eye_poses = []   # Camera poses (corrected for optical frame)
        


    def sub(self, msg: Image):

        if self.camera_matrix is not None:
            # Read image
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            

            # Detect markers
            corners, ids, rejected = self.detector.detectMarkers(self.image)

            
            if ids is not None and len(corners) > 0:
                marker_corners = corners[0]
                pts = marker_corners.reshape((4, 2))

                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

                self.current_camera_to_marker = self.pose_from_rvec_tvec(rvecs[0], tvecs[0])
                # Draw detected corners
                
                for i, (x, y) in enumerate(pts):
                    cv2.circle(self.image, (int(x), int(y)), 6, (0, 0, 255), -1)
                    cv2.putText(self.image, str(i), (int(x)+10, int(y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow('Hand-Eye Calibration', self.image)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                self.collect_calibration_pose()
            elif key == ord('c'):
                self.result = self.solve_hand_eye_calibration()
            elif key == ord('q'):
                self.save_calibration_result(self.result)  
        
        # except Exception as e:
        #     self.get_logger().error(f"Error in image callback: {e}")


    def info_callback(self, msg):

        k = msg.k
        self.camera_matrix = np.array([[k[0], k[1], k[2]],
                                      [k[3], k[4], k[5]],
                                      [0, 0, 1]], dtype=np.float32)
        
    
    
    def get_transform_matrix(self, transform_stamped):
        """Convert TransformStamped to 4x4 homogeneous matrix"""
        trans = transform_stamped.transform.translation
        rot = transform_stamped.transform.rotation 
        
        # Create rotation matrix from quaternion
        rotation = R.from_quat([rot.x, rot.y, rot.z, rot.w])
        rot_matrix = rotation.as_matrix()
        
        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = rot_matrix
        T[:3, 3] = [trans.x, trans.y, trans.z] 
        
        return T
    
    def pose_from_rvec_tvec(self, rvec, tvec):
        """Convert rotation vector and translation vector to 4x4 pose matrix"""
        R_curr, _ = cv2.Rodrigues(rvec)

        rot_x_180 = R.from_euler('x', 180, degrees=True).as_matrix()
        R_ = R_curr @ rot_x_180
        

        pose = np.eye(4)
        pose[:3, :3] = R_
        pose[:3, 3] = tvec.flatten()

        return pose
    
    def collect_calibration_pose(self):
        """Collect a single pose pair for calibration"""
        try:
            # Get end-effector pose (hand)
            hand_transform = self.tf_buffer.lookup_transform(
                'base_link',  # target frame
                'end_effector_link',  # source frame
                rclpy.time.Time()
            )
            hand_matrix = self.get_transform_matrix(hand_transform)

            # Store the poses
            self.hand_poses.append(hand_matrix)
            self.eye_poses.append(self.current_camera_to_marker)

            return True
            
        except Exception as e:
            self.get_logger().error(f'Failed to collect poses: {e}')
            return False
    
    def solve_hand_eye_calibration(self):
        """
        Solve AX = XB hand-eye calibration problem
        A: hand poses (end-effector movement)
        B: eye poses (camera movement)  
        X: hand-eye transformation (what we want to find)
        """
        if len(self.hand_poses) < 3:
            self.get_logger().error('Need at least 3 pose pairs for calibration')
            return None
        
        # Prepare data for OpenCV's calibrateHandEye
        R_gripper2base = []  # Hand rotations
        t_gripper2base = []  # Hand translations
        R_target2cam = []    # Eye rotations
        t_target2cam = []    # Eye translations
        
        for i in range(len(self.hand_poses)):
            # Hand pose (gripper to base)
            hand_R = self.hand_poses[i][:3, :3]
            hand_t = self.hand_poses[i][:3, 3]
            R_gripper2base.append(hand_R)
            t_gripper2base.append(hand_t)
            

            eye_R = self.eye_poses[i][:3, :3]
            eye_t = self.eye_poses[i][:3, 3]
            R_target2cam.append(eye_R)
            t_target2cam.append(eye_t)
        
        # Use OpenCV for robust hand-eye calibration
        try:
            # import cv2
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # Construct homogeneous transformation matrix
            T_cam2gripper = np.eye(4)
            T_cam2gripper[:3, :3] = R_cam2gripper
            T_cam2gripper[:3, 3] = t_cam2gripper.flatten()

            self.get_logger().info(f'Hand-Eye calibration completed, proceed to save')

            
            return T_cam2gripper
            
        except ImportError:
            self.get_logger().error('OpenCV not available, using simple approach')
            return self.simple_hand_eye_solve()
    
    def simple_hand_eye_solve(self):
        """Simple hand-eye calibration without OpenCV"""
        # This is a simplified approach - for production use OpenCV
        if len(self.hand_poses) < 2:
            return None
        
        # Take relative transformations
        A = np.linalg.inv(self.hand_poses[0]) @ self.hand_poses[1]
        B = np.linalg.inv(self.eye_poses[0]) @ self.eye_poses[1]
        
        # Simple approximation: X ≈ A @ B^(-1)
        X = A @ np.linalg.inv(B)
        return X
    
    def save_calibration_result(self, T_cam2gripper):
        """Save the calibration result"""
        if T_cam2gripper is None:
            return
        
        # Extract rotation and translation
        rotation = R.from_matrix(T_cam2gripper[:3, :3])
        quat = rotation.as_quat()  # [x, y, z, w]
        trans = T_cam2gripper[:3, 3]
        
        self.get_logger().info('Hand-Eye Calibration Result:')
        self.get_logger().info(f'Translation: [{trans[0]:.4f}, {trans[1]:.4f}, {trans[2]:.4f}]')
        self.get_logger().info(f'Quaternion: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]')
        
        # Save to file or parameter server
        calibration_data = {
            'translation': trans.tolist(),
            'quaternion': quat.tolist(),
            'transformation_matrix': T_cam2gripper.tolist()
        }
        
        num = len(self.hand_poses)
        name = f'Hand_Eye_matrix_real_arm_{num}'

        
        np.save(f'{self.dir}/{name}',T_cam2gripper)

        np.save(f'{self.dir}/HEC_hand_poses_{num}',self.hand_poses)
        np.save(f'{self.dir}/HEC_eye_poses_{num}',self.eye_poses)

        import json
        with open(f'{self.dir}/{name}.json', 'w') as f:
            json.dump(calibration_data, f, indent=2)
            
        self.get_logger().info(f'Calibration saved to {name}.json')

def main(args=None):
    rclpy.init(args=args)
    
    calibrator = HandEyeCalibration()
    
    # Example usage:
    # 1. Move robot to different poses
    # 2. Collect pose pairs
    # 3. Solve calibration
    
    # Collect some poses (you'd do this interactively)
    calibrator.get_logger().info('Starting hand-eye calibration...')
    calibrator.get_logger().info('Move robot to different poses and call collect_calibration_pose()')
    
    
    rclpy.spin(calibrator)
    calibrator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

