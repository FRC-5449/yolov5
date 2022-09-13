import cv2
import numpy as np
frame1 = cv2.imread("left.png")
frame2 = cv2.imread("right.png")


imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)




# http://wiki.ros.org/stereo_image_proc/Tutorials/ChoosingGoodStereoParameters

# BM
numberOfDisparities = ((720 // 8) + 15) & -16  # 720对应是分辨率的宽

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)  # 立体匹配
stereo.setROI1(camera_config.validPixROI1)
stereo.setROI2(camera_config.validPixROI2)
stereo.setPreFilterCap(31)
stereo.setBlockSize(15)
stereo.setMinDisparity(0)
stereo.setNumDisparities(numberOfDisparities)
stereo.setTextureThreshold(10)
stereo.setUniquenessRatio(15)
stereo.setSpeckleWindowSize(100)
stereo.setSpeckleRange(32)
stereo.setDisp12MaxDiff(1)

disparity = stereo.compute(img1_rectified, img2_rectified)  # 计算视差
if normalize:
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)  # 归一化函数算法

threeD = cv2.reprojectImageTo3D(disparity, camera_config.Q, handleMissingValues=True)  # 计算三维坐标数据值
threeD = threeD * 16

# threeD[y][x] x:0~1080; y:0~720;   !!!!!!!!!!
pictureDepth = {"depth": threeD, "det": det}