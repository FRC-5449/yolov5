import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


class match:
    def __init__(self, MIN_MATCH_COUNT=10):
        self.MIN_MATCH_COUNT = MIN_MATCH_COUNT
        # Initiate SIFT detector
        self.sift = cv.SIFT_create()

    def match(self, img1, img2, output = False):
        # find the keypoints and descriptors with SIFT
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)

        if len(good) > self.MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, M)
            img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        else:
            print("Not enough matches are found - {}/{}".format(len(good), self.MIN_MATCH_COUNT))
            matchesMask = None
        if output:
            self.draw(matchesMask, img1, kp1, img2, kp2, good)
        return dst

    def draw(self, matchesMask, img1, kp1, img2, kp2, good):
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)
        img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        plt.imshow(img3, 'gray'), plt.show()

if __name__ == '__main__':
    test = match()
    img1 = cv.imread('/Users/lishuyu/PycharmProjects/testcamera/test_crop.png', 0)  # trainImage
    img2 = cv.imread('/Users/lishuyu/PycharmProjects/testcamera/test.png', 0)  # queryImage
    test.match(img1, img2, output=True)

