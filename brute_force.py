import cv2

from utils import *

img1 = readIm('./Images/2.jpg')
img2 = readIm('./Images/3.jpg')
camMtx = retCamMtx("./cameraMatrix.pkl")
distCoeff = retDistCoeff("./dist.pkl")

kp1, des1, kp2, des2 = ORB_detector(img1, img2)
kpL1 = retKpList(kp1)
kpL2 = retKpList(kp2)
numMatches = bruteForceMatcher(des1, des2)[:20]
# numMatches = bruteForceMatcherkNN(des1, des2)
# TODO print(numMatches)

essMtx, _ = retEssentialMat(kpL1, kpL2, camMtx, distCoeff)
_, R, t, mask = retPoseRecovery(essMtx, kpL1, kpL2)
print(f'{R}\n, {t}')

out = cv2.drawMatches(img1, kp1, img2, kp2, numMatches, None)
cv2.imshow('Image', out)

# cv2.imshow('Image', cv2.drawKeypoints(img1, kp1, img1))
# cv2.imshow('Image', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

