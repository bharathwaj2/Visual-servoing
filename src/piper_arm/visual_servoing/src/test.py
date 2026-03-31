import numpy as np
from scipy.spatial.transform import Rotation as R

num = 45

dir = '/home/bharath/visual_servoing/src/piper_arm/visual_servoing/src/Files'

P = np.load(f'{dir}/Hand_Eye_matrix_real_arm_{num}.npy')

O = np.load(f'{dir}/HEM_C_to_E.npy')
T = np.load(f'{dir}/target_pose_3.npy')


matrix = np.array([
    [0.008, -1.000, -0.006, 0.016],
    [-0.171, 0.004, -0.985, 0.077],
    [0.985, 0.009, -0.170, 0.067],
    [0.000, 0.000, 0.000, 1.000]
])


P_inv = np.linalg.inv(P)
# np.save(f'{dir}/Hand_Eye_matrix_real_arm_{num}_inv.npy', P_inv)



m = np.linalg.inv(matrix)
# Save the array to a file
# np.save('{dir}/Hand_Eye_matrix_real_arm_0_inv.npy', m)

# print(m)

# R_matrix = T[:3, :3]

# # Create a Rotation object from matrix
# rot = R.from_matrix(R_matrix)

# # Convert rotation matrix to euler angles (roll, pitch, yaw) in radians
# euler_angles = rot.as_euler('xyz', degrees=False)
rot_xyz = R.from_euler('xyz', [180,180,-90], degrees=True).as_matrix()
# print(rot_xyz)
# # print("Euler angles (radians):", euler_angles)
# # print("Euler angles (degrees):", np.degrees(euler_angles))

# print('\n')


# print(P)
# print('\n')
# print(P_inv)


import cv2



Cr = np.load(f'{dir}/marker.npy')
print(Cr)
print('\n')
Cr1 = np.load(f'{dir}/marker_target.npy')
print(Cr1)
print('\n')
Cr2 = np.load(f'{dir}/marker_target_test.npy')
print(Cr2)

print('\n')

Tr =  np.eye(4)

Tr[:3,:3] = Cr[:3,:3]

Tr[:3,3] = Cr[:3,:3] @ Cr[:3,3]

# Tr = np.load(f'{dir}/target_marker_rotation.npy')
# print(Tr)

opt_to_std = np.array([
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]
        ])

tolerance = np.array([0, -0.2, -0.2, 1])

rvec = np.array([-2.65299515e+00, -1.74474412e-04, -7.20018942e-04])

desired_ee_pos_homogeneous = Cr @ tolerance
tvec_star = desired_ee_pos_homogeneous[:3]

rot_mat, _ = cv2.Rodrigues(rvec)

T_obj = np.eye(4)
T_obj[:3, :3] = rot_mat
T_obj[:3, 3] = tvec_star

name = 'marker_target'

if name=='marker':
    rot_x_180 = R.from_euler('xyz', [180,0,0], degrees=True).as_matrix()
    T_std = T_obj
else:
    rot_x_180 = R.from_euler('xyz', [-90,0,0], degrees=True).as_matrix()
    T_std = T_obj @ opt_to_std

r_obj = T_std[:3,:3]
tvec_std = T_std[:3, 3]

rot = r_obj @ rot_x_180

marker= np.eye(4)
marker[:3,3] = T_std[:3,3]
marker[:3,:3] = rot

print('\n')
# print(marker)

