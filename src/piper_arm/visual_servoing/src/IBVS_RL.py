#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import TwistStamped
from cv_bridge import CvBridge
import cv2
import tf2_ros
import numpy as np
from numpy import pi
import os
from gymnasium import spaces
# from gym import spaces
from DDPG.ddpg_torch import Agent

        
class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


        self.cam_info_sub = self.create_subscription(
            CameraInfo, '/camera/color/camera_info', self.info_callback, 10)
        self.cam_info_sub

        self.subscription = self.create_subscription(
            Image, '/camera/color/image_raw', self.listener_callback, 10)
        self.subscription
        
        self.depth_sub = self.create_subscription(
            Image, '/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.depth_sub

        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)
        self.joint_sub

        self.vel_pub = self.create_publisher(
            TwistStamped, '/servo_node/delta_twist_cmds', 10)
        
        self.joint_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)

        
        
        self.dir = '/home/bharath/visual_servoing/src/piper_arm/visual_servoing/src/Files'

        self.RL_init_pos = np.load(f'{self.dir}/RL_initial_pose.npy')
        
        RL_target = np.load(f'{self.dir}/RL_target_pose.npy')
        RL_depth = np.load(f'{self.dir}/RL_target_depth.npy')
        self.target = RL_target[0]
        self.Z_t = RL_depth[0]


        self.num = 1
        self.n = 4
        
        self.s_p = np.array(self.target).flatten()
        self.s_i = np.zeros_like(self.s_p)

        self.vel = 0.01
        self.ang_vel = 0.025
        self.mu = np.zeros((6,6), dtype=np.float32)

        self.Le = np.zeros((int(2*self.n),6), dtype=np.float32)
        self.Le_s = np.zeros((int(2*self.n),6), dtype=np.float32)
        self.Lsp = None

        self.Z = np.zeros((self.n,1))

        self.camera_matrix = None
        self.depth_image = None

        self.calc_cam_ee_conv_matrix()

        self.joint_angles = np.zeros((6,1), dtype=np.float32)
        self.e = np.zeros((8,1), dtype=np.float32)
        self.v_c = np.zeros((6,1), dtype=np.float32)


        self.joint_limits = np.array([
                                    [-2.618 , 2.618],
                                    [0.0, pi],
                                    [-2.967, 0.0],
                                    [-1.745, 1.745],
                                    [-1.22, 1.22],
                                    [-2.094, 2.094]
                                    ]
                                    , dtype=np.float32)


        self.N_train= 1
        self.N_eval = 1

        self.gap = int(self.N_train / 2)

        self.episode = 0
        self.train_episode = 0
        self.eval_episode = 0
        

        self.mode = 'TRAIN'
        self.First = True

        self.mu_lim = 0.5
        

        self.K_max = 500
        self.K = 0

        self.step_reward = 0.1

        self.Je = 30.0

        self.train = True  
        self.eval = False
        

        self.cam_v = []
        self.depth_data = []
        self.error_data = []

        self.q = []

        self.frame_id = 0


        self.limit_terminate = False
        self.goal_reached = False
        self.terminate = False
        self.truncate = False
        self.execute = False
        self.stop = False
        

        self.halt_counter = np.zeros((6,1))


        self.reward = 0.0
        self.truncate_reward = 1000
        self.limit_reward = 1000

        self.score = 0
        self.score_list = []
        self.avg_score = []

        self.resetting = False  
        self.reset_start_time = None 

        self.r = None
        self.reason = []
        self.success = 0.0

        self.rl_state = "RESET"  
        self.action_taken = None
        self.reward_received = None
        self.steps_since_action = 0
        self.min_steps_between_actions = 2


        self.s_ = None
        self.err_vel = []


        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()

        parameters.minMarkerPerimeterRate = 0.01  # Smaller markers
        parameters.maxMarkerPerimeterRate = 4.0   # Larger markers
        parameters.minMarkerDistanceRate = 0.02
        parameters.polygonalApproxAccuracyRate = 0.05  # Less strict polygon approximation
        parameters.minCornerDistanceRate = 0.01   # Closer corners allowed
        parameters.minDistanceToBorder = 1        # Markers can be closer to border
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 4
        parameters.adaptiveThreshConstant = 4

        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


        self.obs = np.vstack([self.e, self.v_c, self.joint_angles])
        self.obs_ = self.obs
        self.action = np.zeros((6,1), dtype=np.float32)


        high = np.array(
        [
            self.mu_lim,self.mu_lim,self.mu_lim,self.mu_lim,self.mu_lim,self.mu_lim,self.mu_lim,self.mu_lim,
            self.vel,self.vel,self.vel,self.ang_vel,self.ang_vel,self.ang_vel,
            pi,pi,pi,pi,pi,pi,

        ],dtype=np.float32)

        self.obs_space = spaces.Box(-high, high, dtype=np.float32)

        algorithm = 'DDPG'
        self.chkpt_dir = f'{self.dir}/{algorithm}/Trained_models'

        self.agent = Agent(alpha=0.0001, beta=0.001, 
                    input_dims=self.obs_space.shape, tau=0.001,
                    batch_size=64, fc1_dims=128, fc2_dims=128, 
                    n_actions=self.action.shape[0],checkpoint_dir=self.chkpt_dir,num=self.num)



    def info_callback(self, msg):

        k = msg.k
        self.camera_matrix = np.array([[k[0], k[1], k[2]],
                                      [k[3], k[4], k[5]],
                                      [0, 0, 1]], dtype=np.float32)
        

    def depth_callback(self, msg):

        data = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        self.depth_image  = data / 1000

        
    def joint_callback(self, msg):

        angles = msg.position
        self.joint_angles = [angles[4], angles[0], angles[1], angles[5], angles[2], angles[3]]
        

        
    def listener_callback(self, msg: Image):


        if self.depth_image is not None and self.camera_matrix is not None:

            if self.success <= 90:

                if self.train and not self.eval:
                    if self.train_episode < self.N_train:

                        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    
                        self.RL_loop()
                        self.reason = []

                    else:

                        cv2.imshow('Camera view', self.image)             
                        cv2.waitKey(1)

                else:
                    if self.eval_episode < self.N_eval:

                        self.mode = 'EVALUATE'
                        
                        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

                        self.RL_loop()
                        
                    else:

                        self.calc_success_rate()

                        self.eval_episode = 0

                        self.eval = False
                        self.train = True

                        self.mode = 'TRAIN'

                        cv2.imshow('Camera view', self.image)             
                        cv2.waitKey(1)

            else:

                self.get_logger().info('\nTraining successful.....saving weights of agent')
                self.agent.save_models()


    def RL_loop(self):

        if not self.execute:

            if self.resetting:

                current_time = self.get_clock().now()
                reset_duration = (current_time - self.reset_start_time).nanoseconds / 1e9
                
                if reset_duration >= 3.0:  
                    
                    self.resetting = False
                    self.execute = True
                    self.rl_state = "OBSERVE"

                else:
                    cv2.imshow('Camera view', self.image)
                    cv2.waitKey(1)
            
            else:
                    
                self.reset()

                cv2.imshow('Camera view', self.image)             
                cv2.waitKey(1)  
                
        else:  

            if not self.stop:

                if self.mode == 'TRAIN':
                    self.train_mode()
                else:
                    self.eval_mode()
                

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
                cv2.imshow('Camera view', self.image)             
                cv2.waitKey(1)
                

            else:
                
                if self.train:
                    self.train_episode += 1
                    self.episode = self.train_episode
                    self.score_list.append(self.score)
                else:
                    self.eval_episode += 1
                    self.episode = self.eval_episode
                    self.reason.append(self.r)
                
                if self.N_train > 1:
                    if self.train_episode % self.gap == 0:
                        self.train = False
                        self.eval = True

               
                self.get_logger().info(f'Episode: {self.episode}\t Score: {round(self.score,4)}\t Steps: {int(self.K)}\t Error: {round(self.norm_err,2)}')

                self.execute = False

                cv2.imshow('Camera view', self.image)             
                cv2.waitKey(1)


    def train_mode(self):

        if self.rl_state == "OBSERVE":
            init_pose, self.obs = self.get_current_obs()
            self.s_i = init_pose.flatten()

            if self.obs is not None:
                self.rl_state = "ACT"
                
        elif self.rl_state == "ACT":
            self.step()

            self.reward_received, self.action_taken, self.r
            self.rl_state = "WAIT_NEXT_OBS"
            self.steps_since_action = 0
            
        elif self.rl_state == "WAIT_NEXT_OBS":
            self.steps_since_action += 1
            
            if self.steps_since_action >= self.min_steps_between_actions:
               
                _, self.obs_ = self.get_current_obs()

                if self.obs_ is not None:
                    self.calc_reward()
                    self.rl_state = "LEARN"
                    
        elif self.rl_state == "LEARN":
            self.agent.remember(self.obs.T, self.action_taken, 
                            self.reward_received, self.obs_.T, self.stop)
            self.agent.learn()

            if not self.stop:
                self.obs = self.obs_.copy()
                self.rl_state = "ACT" 
    

    def eval_mode(self):

        if self.rl_state == "OBSERVE":
            init, self.obs = self.get_current_obs()
            self.s_i = init.flatten()

            if self.obs is not None:
                self.rl_state = "ACT"
                
        elif self.rl_state == "ACT":
            self.step()
            self.calc_reward()

            self.reward_received, self.action_taken, self.r
            self.rl_state = "WAIT_NEXT_OBS"
            self.steps_since_action = 0
            
        elif self.rl_state == "WAIT_NEXT_OBS":
            self.steps_since_action += 1
            
            if self.steps_since_action >= self.min_steps_between_actions:
                _, self.obs_ = self.get_current_obs()

                if self.obs_ is not None and not self.stop:
                    self.obs = self.obs_.copy()
                    self.rl_state = "ACT"
                    


    def reset(self):

        msg = JointTrajectory()
        msg.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        point = JointTrajectoryPoint()


        rand = np.random.randint(0,30,1)[0]
        init_pos = self.RL_init_pos[rand]
        
        point.positions = list(init_pos)

        point.time_from_start.sec = 2
        msg.points.append(point)
        self.joint_pub.publish(msg)

        self.resetting = True
        self.reset_start_time = self.get_clock().now()

        self.rl_state = "RESET"
        self.action_taken = None
        self.reward_received = None
        self.steps_since_action = 0

        self.q = []

        self.halt_counter = np.zeros((6,1))

        self.limit_terminate = False
        self.goal_reached = False
        self.terminate = False
        self.truncate = False
        self.stop = False

        self.K = 0
        self.score = 0

        
    def step(self):

        if not self.terminate:

            self.err_vel.append(self.s)
            
            # action = self.agent.choose_action(self.obs, self.mode)
            action = np.ones((6,))*0.5

            e, v_c, q = self.obs[:8] , self.obs[8:14] , self.obs[14:]
            self.q.append(q)

            self.check_joint_limits(q)

            self.action_taken = np.clip(action, -self.mu_lim, self.mu_lim)

            for i in range(0,6):
                self.mu[i,i] = self.action_taken[i]


            self.e_T = np.array(e).T.reshape((int(self.n*2),1))
            self.norm_err = np.linalg.norm(self.e_T)

            for m in range(0,self.n):
                self.Z[m] = self.depth_image[int(self.pose[m,1]),int(self.pose[m,0])]

            self.calc_img_jacobian()

            self.v_c = -self.mu @ np.dot(self.Lsp , self.e_T)     
            self.v_e = np.dot(self.ce_trans , self.v_c)

            self.v_e[:3,0] = np.clip(self.v_e[:3,0], -self.vel, self.vel)
            self.v_e[3:,0] = np.clip(self.v_e[3:,0], -self.ang_vel, self.ang_vel)

            self.cam_v.append(self.v_c)

            msg = TwistStamped()

            msg.header.stamp = self.get_clock().now().to_msg()

            if self.norm_err >= 20:

                msg.twist.linear.x = self.v_e[0,0]
                msg.twist.linear.y = self.v_e[1,0]
                msg.twist.linear.z = self.v_e[2,0]
                
                msg.twist.angular.x = self.v_e[3,0]
                msg.twist.angular.y = self.v_e[4,0]
                msg.twist.angular.z = self.v_e[5,0]

            else:
                msg.twist.linear.x = 0.0
                msg.twist.linear.y = 0.0
                msg.twist.linear.z = 0.0
                
                msg.twist.angular.x = 0.0
                msg.twist.angular.y = 0.0
                msg.twist.angular.z = 0.0

                self.goal_reached = True
                
            
            self.vel_pub.publish(msg)

            self.K += 1


    def get_current_obs(self):

        corners, ids, rejected = self.detector.detectMarkers(self.image)

        if ids is None and len(corners) == 0:

            self.terminate = True
            obs_ = self.calc_terminal_obs()
            return None, obs_
            
        else:

            marker_corners = corners[0]
            curr_pose = marker_corners.reshape((4, 2))

            self.pose = curr_pose
            self.s = curr_pose.flatten()

            err = np.array(self.s - self.s_p).reshape((int(self.n*2),1))
            q_curr = np.array(self.joint_angles).reshape((6,1))
            v_c = self.v_c
            
            obs = np.vstack([err, v_c, q_curr])

            return curr_pose, obs


    def calc_reward(self):

        if self.K >= self.K_max:
                self.truncate = True
            
        if self.goal_reached:
            reward = self.R_success()
            reason = 1
        elif self.truncate:
            reward = -self.step_reward - self.truncate_reward
            reason = 0
        elif self.terminate:
            reward = self.R_failure()
            reason = 0
        elif self.limit_terminate:
            reward = -self.step_reward - self.limit_reward
        else:
            reward = -self.step_reward - (self.norm_err/self.K_max)
            reason = None

        self.reward_received = reward
        self.r = reason

        self.stop = self.terminate or self.goal_reached or self.truncate or self.limit_terminate
        self.score += reward

        
            
    def R_success(self):
        
        T = (self.K_max - self.K)/self.K_max
        Je = np.zeros((self.K,6), dtype=np.float32)
        J = np.zeros((self.K,1))

        q = np.array(self.q).reshape((self.K,6))

        Je[0,:] = (-3*q[0,:] + 14*q[1,:] - 24*q[2,:] + 18*q[3,:] - 5*q[4,:])/2

        l = self.K
        
        Je[self.K-1,:] = (3*q[l-1,:] - 14*q[l-2,:] + 24*q[l-3,:] - 18*q[l-4,:] + 5*q[l-5,:])/2

        for i in range(2,self.K-2):
            Je[i,:] = (q[i+2,:] - 2*q[i+1,:] + 2*q[i-1,:] - q[i-2,:])/2

            J[i] = np.array(Je[i,:]).T @ Je[i,:]

        jerk = self.Je - max(J)
        n = np.maximum(0, jerk[0])/self.Je

        r_success = (T + n)*100
        # self.get_logger().info(f'{r_success}')

        return r_success
    

    def R_failure(self):

        eC = 0
        eI = 0

        c = self.camera_matrix[0,2]
        r = self.camera_matrix[1,2]

        for i in range(0,len(self.s)):
            eC += abs(self.s.T[i] - self.s_p.T[i])/(self.n*np.sqrt(c**2 + r**2))

        for i in range(0,len(self.s_i)):
            eI += abs(self.s_i.T[i] - self.s_p.T[i])/(self.n*np.sqrt(c**2 + r**2))

        r_failure = -((self.K-1)/self.K_max) - (eC/eI)

        return r_failure


    def calc_success_rate(self):

        s = 0
        for i in range(0,len(self.reason)):
            if self.reason[i] == 1:
                s += 1

        self.success = (s/self.episode) * 100


    def calc_avg_err_vel(self):
        

        N = len(self.err_vel)

        err_vel = np.array(self.err_vel).reshape((N,8))
        avg = np.zeros((8,1))

        for j in range(0,8):
            v_err = 0
            for i in range(1,N):
                v_err += err_vel[i,j] - err_vel[i-1,j]
            avg[j,:] = v_err/N-1

        return avg
    

    def calc_terminal_obs(self):

        pixel_vel = self.calc_avg_err_vel()
        s_ = self.obs.copy()
        s_[0:8,:] = self.obs[0:8,:] + pixel_vel*0.01

        return s_
    

    def check_joint_limits(self, q):

        for i in range(0,6):
            l = self.joint_limits[i,0] + 0.12
            u = self.joint_limits[i,1] - 0.12

            if q[i] <= l or q[i] >= u:
                self.halt_counter[i] += 1
                
            if self.halt_counter[i] >= 10:
                self.limit_terminate = True


    def publish_zero_vel(self):

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


    def calc_cam_ee_conv_matrix(self):

        self.HEM = self.get_CtE()

        eRc = self.HEM[:3,:3]
        eTc = self.HEM[:3,3]

        eTc_x = np.array([
                        [0, -eTc[2], eTc[1]],
                        [eTc[2], 0, -eTc[0]],
                        [-eTc[1], eTc[0], 0]
                        ])

        self.ce_trans = np.zeros([6,6])

        self.ce_trans[:3,:3] = eRc
        self.ce_trans[:3,3:] = np.dot(eTc_x , eRc)
        self.ce_trans[3:,3:] = eRc

    
    def calc_img_jacobian(self):
        
        px = self.camera_matrix[0,0]
        py = self.camera_matrix[1,1]
        u0 = self.camera_matrix[0,2]
        v0 = self.camera_matrix[1,2]


        zi = 1/self.Z
        self.depth_data.append(zi)
        # np.save(f'{self.dir}/IBVS_depth_data_{self.num}',self.depth_data)
        
        self.error_data.append(self.e_T)
        # np.save(f'{self.dir}/IBVS_error_data_{self.num}',self.error_data)

        for m in range(0,4):

            x = (self.s[m*2] - u0)/px
            y = (self.s[m*2+1] - v0)/py

            Zinv = 1/self.Z[m,0]

            self.Le[m*2:m*2+2,:] = np.matrix([[-Zinv, 0, x*Zinv, x*y, -(1+x**2), y] , [0, -Zinv, y*Zinv, 1+y**2, -x*y, -x]])


        for m in range(0,4):

            x = (self.s_p[m*2] - u0)/px
            y = (self.s_p[m*2+1] - v0)/py

            Zinv = 1/self.Z_t[m,0]

            self.Le_s[m*2:m*2+2,:] = np.matrix([[-Zinv, 0, x*Zinv, x*y, -(1+x**2), y] , [0, -Zinv, y*Zinv, 1+y**2, -x*y, -x]])


        self.L = (self.Le + self.Le_s)/2

        self.Lsp = np.linalg.pinv(self.L)



def main(args=None):

    rclpy.init(args=args)
    node = ImageSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


















































