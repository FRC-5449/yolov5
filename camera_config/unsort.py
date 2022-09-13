import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ret, mtx, dist, rvecs, tvecs
single_camera_test_args = np.load("single_camer_test_args.npz")
mtx = single_camera_test_args["mtx"]
dist = single_camera_test_args["dist"]
img = cv.imread('../pic/1.png')[:,1280:]
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
plt.imshow(dst)
plt.show()

cv.imwrite("right.png", dst)