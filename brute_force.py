import cv2

from utils import *

# TODO: implement multiple image system


camMtx = retCamMtx("./cameraMatrix.pkl")
nCamMtx = retCamMtx("./newCameraMatrix.pkl")
distCoeff = retDistCoeff("./dist.pkl")

img1 = readIm('./Images/2.jpg')
# img1 = retUndistortedIm(img1, camMtx, nCamMtx, distCoeff)
img2 = readIm('./Images/3.jpg')
# img2 = retUndistortedIm(img2, camMtx, nCamMtx, distCoeff)
limiter = 20 # number of matches

kp1, des1, kp2, des2 = ORB_detector(img1, img2)
# kp1, des1, kp2, des2 = BRISK_detector(img1, img2)
kpL1 = retKpList(kp1)
kpL2 = retKpList(kp2)
numMatches = bruteForceMatcher(des1, des2)
# numMatches = bruteForceMatcherkNN(des1, des2)

essMtx, _ = retEssentialMat(kpL1, kpL2, camMtx, distCoeff)
_, R, t, mask = retPoseRecovery(essMtx, kpL1, kpL2)
pts_3d = retTriangulation(R, t, kpL1, kpL2, limiter)
print(f'{pts_3d}')

display2D(img1, kp1, img2, kp2, numMatches)
display3D(pts_3d)

# cv2.imshow('Image', cv2.drawKeypoints(img1, kp1, img1))
# cv2.imshow('Image', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

