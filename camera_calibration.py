# https://github.com/jagracar/OpenCV-python-tests/blob/master/OpenCV-tutorials/cameraCalibration/cameraCalibration.py
'''
 Based on the following tutorial:
   http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_calib3d/py_calibration/py_calibration.html
'''

import numpy as np
import cv2
import glob

# Define the chess board rows and columns
rows = 7
cols = 6

# Set the termination criteria for the corner sub-pixel algorithm
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

# Prepare the object points: (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0). They are the same for all images
objectPoints = np.zeros((rows * cols, 3), np.float32)
objectPoints[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

# Create the arrays to store the object points and the image points
objectPointsArrayL = []
objectPointsArrayR = []
imgPointsArrayL = []
imgPointsArrayR = []

# Loop over the image files
for path in glob.glob('../data/left[0-9][0-9].jpg'):
    # Load the image and convert it to gray scale
    img = cv2.imread(path)
    imgL = img[0:720][0:1080]
    imgR = img[0:720][1080:2560]
    
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv2.findChessboardCorners(grayL, (rows, cols), None)
    retR, cornersR = cv2.findChessboardCorners(grayR, (rows, cols), None)
    
    # Make sure the chess board pattern was found in the image
    if retL:
        # Refine the corner position
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)

        # Add the object points and the image points to the arrays
        objectPointsArrayL.append(objectPoints)
        imgPointsArrayL.append(cornersL)

        # Draw the corners on the image
        cv2.drawChessboardCorners(imgL, (rows, cols), cornersL, retL)
    # Make sure the chess board pattern was found in the image
    if retR:
        # Refine the corner position
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        # Add the object points and the image points to the arrays
        objectPointsArrayR.append(objectPoints)
        imgPointsArrayR.append(cornersR)

        # Draw the corners on the image
        cv2.drawChessboardCorners(imgR, (rows, cols), cornersR, retR)

    # Display the image
    cv2.imshow('chess board(L)', imgL)
    cv2.imshow('chess board(R)', imgR)
    cv2.waitKey(500)

# Calibrate the camera and save the results
retL, mtxL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objectPointsArrayL, imgPointsArrayL, gray.shape[::-1], None, None)
retR, mtxR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objectPointsArrayR, imgPointsArrayR, gray.shape[::-1], None, None)
np.savez('camera_config/calib.npz', mtxL=mtxL,mtxR=mtxR, distL=distL,distR=distR, rvecsL=rvecsL,rvecsR=rvecsR, tvecsL=tvecsL,tvecsR=tvecsR)

from pprint import pprint
pprint(retL, retR)
pprint(mtxL, mtxR)
pprint(distL, distR)
pprint(rvecsL, rvecsR)
pprint(tvecsL, tvecsR)

# Print the camera calibration error
error = 0

for i in range(len(objectPointsArrayL)):
    imgPointsL, _ = cv2.projectPoints(objectPointsArrayL[i], rvecsL[i], tvecsL[i], mtxL, distL)
    error += cv2.norm(imgPointsArrayL[i], imgPointsL, cv2.NORM_L2) / len(imgPointsL)
for i in range(len(objectPointsArrayR)):
    imgPointsR, _ = cv2.projectPoints(objectPointsArrayR[i], rvecsR[i], tvecsR[i], mtxR, distR)
    error += cv2.norm(imgPointsArrayR[i], imgPointsR, cv2.NORM_L2) / len(imgPointsR)

print("Total error: ", error / (len(objectPointsArrayL)+len(objectPointsArrayR)))