import cv2
import numpy as np
import open3d as o3d

from utils import *

img1 = readIm('./Images/2.jpg')
img2 = readIm('./Images/3.jpg')

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

print(f'kp1: {kp1.shape}')


FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
# Example intrinsic parameters
K = np.array([[1000, 0, 320],
              [0, 1000, 240],
              [0, 0, 1]])
print(f"Good matches: {good_matches}")

# Example extrinsic parameters
R = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
t = np.array([0, 0, 0])

points3d = []
for match in good_matches:
    x1, y1 = kp1[match.queryIdx].pt
    x2, y2 = kp2[match.trainIdx].pt

    pt1 = np.array([x1, y1, 1])
    pt2 = np.array([x2, y2, 1])

    # Triangulate 3D point
    A = np.vstack((pt1*K[2]-K[0], pt2*K[2]-K[1]))
    B = np.hstack((R, t.reshape(-1, 1)))
    print(A)
    print(B)
    X = np.linalg.lstsq(B, -A, rcond=None)[0]
    points3d.append(X)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3d)

o3d.visualization.draw_geometries([pcd])
