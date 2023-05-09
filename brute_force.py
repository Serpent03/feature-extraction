import cv2

from utils import *

img1 = readIm('./Images/2.jpg')
img2 = readIm('./Images/3.jpg')

kp1, des1, kp2, des2 = ORB_detector(img1, img2)

cv2.imshow('Image', cv2.drawKeypoints(img1, kp1, img1))
# cv2.imshow('Image', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

