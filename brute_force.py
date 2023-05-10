import cv2

from utils import *

img1 = readIm('./Images/2.jpg')
img2 = readIm('./Images/3.jpg')

kp1, des1, kp2, des2 = ORB_detector(img1, img2)
1
numMatches = bruteForceMatcher(des1, des2)[:20]

# print(numMatches[:15])

out = cv2.drawMatches(img1, kp1, img2, kp2, numMatches, None)
cv2.imshow('Image', out)

# cv2.imshow('Image', cv2.drawKeypoints(img1, kp1, img1))
# cv2.imshow('Image', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

