import cv2
import numpy as np

# 左相机内参
left_camera_matrix = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]])
# 左相机畸变系数:[k1, k2, p1, p2, k3]
left_distortion = np.array([[0, 0, 0, 0, 0]])
# 右相机内参
right_camera_matrix = np.array([[0, 0, 0],[0, 0, 0],[0., 0., 0]])
# 右相机畸变系数:[k1, k2, p1, p2, k3]
right_distortion = np.array([[0,0, 0, 0, 0]])

# om = np.array([0, 0, 0])
# R = cv2.Rodrigues(om)[0]

# 旋转矩阵
R = np.array([[0, 0, 0],[0,0,0],[ 0, 0, 0]])
# 平移向量
T = np.array([[0], [0], [0]])

size = (1080, 720)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion,
                                                                  size, R, T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)