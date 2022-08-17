import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class match:
    def __init__(self, MIN_MATCH_COUNT=10):
        self.MIN_MATCH_COUNT = MIN_MATCH_COUNT
        # Initiate SIFT detector
        self.sift = cv.SIFT_create()

    def match(self, img1, img2, output = False, gray = False):
        if gray:
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
            img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=50)
        search_params = dict(checks=5000)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        best = np.zeros((2,2))
        best_match = matches[0][0]
        min_distance = 1
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)
            k = m.distance / n.distance
            if k < min_distance:
                min_distance = k
                best_match = m
        if len(good) > self.MIN_MATCH_COUNT:
            print("Found {0} matches".format(len(good)))
        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
        # get points from best match
        src_pts = np.int64(kp1[best_match.queryIdx].pt)
        dst_pts = np.int64(kp2[best_match.trainIdx].pt)
        best = np.array([src_pts, dst_pts]).reshape(2,2)
        return best

if __name__ == '__main__':
    test = match()
    img1 = cv.imread('/Users/lishuyu/PycharmProjects/testcamera/test_crop.png', 0)  # trainImage
    img2 = cv.imread('/Users/lishuyu/PycharmProjects/testcamera/test.png', 0)  # queryImage
    kp1, kp2, good, match = test.match(img1, img2, output=True)
    # display all matches in 0.7 distance
    for m, n in match:
        # print(m, n, type(m))
        print( m.queryIdx, m.trainIdx, m.imgIdx, m.distance,
               n.queryIdx, n.trainIdx, n.imgIdx, n.distance)