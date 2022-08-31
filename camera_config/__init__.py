import cv2
import numpy as np
import os

if os.path.exist("calib.npz"):
    calib = np.load("calib.npz")
    left_camera_matrix = calib["mtxL"]
    right_camera_matrix = calib["mtxR"]
    left_distortion = calib["distL"]
    right_distortion = calib["distR"]
    R = (calib["rvecsL"] + calib["rvecsR"])/2
    T = (calib["tvecsL"] + calib["tvecsR"])/2
else:
    # 左相机内参
    left_camera_matrix = np.array([[0, 0, 0],[0, 0, 0],[0, 0, 0]],dtype=float)
    # 左相机畸变系数:[k1, k2, p1, p2, k3]
    left_distortion = np.array([[0, 0, 0, 0, 0]],dtype=float)
    # 右相机内参
    right_camera_matrix = np.array([[0, 0, 0],[0, 0, 0],[0., 0., 0]],dtype=float)
    # 右相机畸变系数:[k1, k2, p1, p2, k3]
    right_distortion = np.array([[0,0, 0, 0, 0]],dtype=float)

    # om = np.array([0, 0, 0])
    # R = cv2.Rodrigues(om)[0]

    # 旋转矩阵
    R = np.array([[0, 0, 0],[0,0,0],[ 0, 0, 0]],dtype=float)
    # 平移向量
    T = np.array([[0], [0], [0]],dtype=float)

size = (1080, 720)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion,
                                                                  size, R, T)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)