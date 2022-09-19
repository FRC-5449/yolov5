import cv2
import numpy as np

import camera_config
img = cv2.imread("pic/100.png")
frame1 = img[:,:1280]
frame2 = img[:,1280:]
frame1, frame2 = camera_config.remap(frame1, frame2)


imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)




# http://wiki.ros.org/stereo_image_proc/Tutorials/ChoosingGoodStereoParameters

# BM
numberOfDisparities = ((720 // 8) + 15) & -16  # 720对应是分辨率的宽
numberOfDisparities = 96

stereo = cv2.StereoSGBM_create(numDisparities=96, blockSize=5) #,P1=400,P2=2400)
# stereo.setPreFilterCap(31)
# stereo.setBlockSize(5)
# stereo.setMinDisparity(0)
# stereo.setNumDisparities(numberOfDisparities)
# stereo.setTextureThreshold(10)
# stereo.setUniquenessRatio(5)
# stereo.setSpeckleRange(2)
# stereo.setSpeckleWindowSize(60)
# stereo.setDisp12MaxDiff(200)


disparity = stereo.compute(imgL, imgR)  # 计算视差
normalize = False
if normalize:
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)  # 归一化函数算法

threeD = cv2.reprojectImageTo3D(disparity, camera_config.Q, handleMissingValues=False)  # 计算三维坐标数据值
threeD = threeD * 16

# threeD[y][x] x:0~1080; y:0~720;   !!!!!!!!!!
pictureDepth = {"depth": threeD}
print(threeD.shape)
print(threeD)

import matplotlib.pyplot as plt
plt.imshow(threeD[:,:,2], cmap='gray')
plt.show()
plt.boxplot(threeD[:,:,2].reshape(-1),sym='')
plt.show()

counter = 0
# ignore threeD[:,:,2] 's > 100 or < 10 value
for x in range(720):
    for y in range(1280):
        i = threeD[x,y,2]
        if i > 1000 or i < 0:
            threeD[x,y,2] = np.NAN
            counter += 1
print(counter/(720*1280/100))
plt.imshow(threeD[:,:,2])
plt.show()
plt.boxplot(threeD[:,:,2].reshape(-1)[~np.isnan(threeD[:,:,2].reshape(-1))],sym='')
plt.show()

print("End")