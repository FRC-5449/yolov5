import cv2
import numpy as np
import os, sys

assert os.path.exists("calib.npz"), "calib.npz not exist"
assert os.path.exists("stereo_calib.npz"), "stero_calib.npz not exist"


calib = np.load("calib.npz")
stereo_calib = np.load("stereo_calib.npz")



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

stereo_param = {
    "SGBM_blockSize":9,
    "num_disp":64,
    "UniquenessRatio":5,
    "SpeckleWindowSize":60,
    "SpeckleRange":2,
    "Disp12MaxDiff":206,
    "P1":600,
    "P2":2400,
}

SGBM_stereo = cv2.StereoSGBM_create(P1=stereo_param["P1"],P2=stereo_param["P2"])
SGBM_stereo.setBlockSize(stereo_param["SGBM_blockSize"])
SGBM_stereo.setNumDisparities(stereo_param["num_disp"])
SGBM_stereo.setUniquenessRatio(stereo_param["UniquenessRatio"])
SGBM_stereo.setSpeckleWindowSize(stereo_param["SpeckleWindowSize"])
SGBM_stereo.setSpeckleRange(stereo_param["SpeckleRange"])
SGBM_stereo.setDisp12MaxDiff(stereo_param["Disp12MaxDiff"])

def compute(imgL, imgR):
    disp = SGBM_stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    threeD = cv2.reprojectImageTo3D(disp, Q)
    depth = threeD[:, :, 2]
    depth[0 > depth] = 0
    depth[500 < depth] = 500
    threeD[:,:,2] = depth
    return threeD

size = (1280, 720)

R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(K1,D1,
                                                                  K2,D2,
                                                                  size,
                                                                  R,T,)

left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
newcameramatrix1, roi1 = cv2.getOptimalNewCameraMatrix(left_camera_matrix, left_distortion, (720,1280), 1, (720,1280))
newcameramatrix2, roi2 = cv2.getOptimalNewCameraMatrix(left_camera_matrix, left_distortion, (720,1280), 1, (720,1280))

def check(*args):
    for i in args:
        if type(i) is np.ndarray:
            print(i.shape)
        else:
            print(i)


def remap(img1, img2):
    h1,  w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    print(h1,w1)
    # undistort
    dst1 = cv2.remap(img1, left_map1, left_map2, cv2.INTER_LINEAR)
    dst2 = cv2.remap(img2, right_map1, right_map2, cv2.INTER_LINEAR)
    """# crop the image
    x, y, w, h = roi1
    dst1 = dst1[y:y+h, x:x+w]
    x, y, w, h = roi2
    dst2 = dst2[y:y + h, x:x + w]"""
    return dst1, dst2

if __name__ == '__main__':
    img = cv2.imread('pic/1.png')
    import matplotlib.pyplot as plt
    # plt.imshow(unsort(img))
    il, ir = remap(img[:, :1280],img[:, 1280:])
    plt.imshow(il)
    plt.show()
    plt.imshow(ir)
    plt.show()