import cv2
import numpy as np
import os

if os.path.exists("calib.npz"):
    calib = np.load("calib.npz")
else:
    exit(-1)

if os.path.exists("stereo_calib.npz"):
    stereo_calib = np.load("stereo_calib.npz")
else:
    exit(-1)


left_camera_matrix = calib["mtxL"]
right_camera_matrix = calib["mtxR"]
left_distortion = calib["distL"]
right_distortion = calib["distR"]
rvecsL = calib["rvecsL"]
rvecsR = calib["rvecsR"]
tvecsL = calib["tvecsL"]
tvecsR = calib["tvecsR"]
K1 = stereo_calib["K1"]
K2 = stereo_calib["K2"]
D1 = stereo_calib["D1"]
D2 = stereo_calib["D2"]
R = stereo_calib["R"]
T = stereo_calib["T"]

size = (1280, 720)
print(D1, D2)
print( K1, K2)
print(size, T.T*T, R)
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1,D1,
                                                                  K2,D2,
                                                                  size,
                                                                  T,R,)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)