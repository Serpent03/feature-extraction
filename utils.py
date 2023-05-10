import cv2
import pickle
import numpy as np

def readIm(pathToIm):
    rFac = 5
    im = cv2.imread(pathToIm)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.array(im)
    im = cv2.GaussianBlur(im, (5,5), 0)
    im =  cv2.resize(im, (
        int(im.shape[1] / rFac), # width
        int(im.shape[0] / rFac), # height
    ))
    print(im.shape)
    return im

def retCamMtx(pathToCamMtx):
    with open(f'{pathToCamMtx}', 'rb') as f:
        return pickle.load(f)

def retDistCoeff(pathToDistCoeff):
    with open(f'{pathToDistCoeff}', 'rb') as f:
        return np.array(pickle.load(f))

def ORB_detector(im1, im2):
    detect = cv2.ORB_create()
    kp1, des1 = detect.detectAndCompute(im1, None)
    kp2, des2 = detect.detectAndCompute(im2, None)
    # print(kp1[0].pt)
    return (kp1, des1, kp2, des2)

def retKpList(kp):
    kpL = []
    for i in kp:
        kpL.append(i.pt)
    return np.array(kpL)

def bruteForceMatcher(des1, des2):
    bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    numMatches = bfm.match(des1, des2)
    numMatches = sorted(numMatches,key=lambda x:x.distance)
    # print(numMatches[0].imgIdx)
    return numMatches

def bruteForceMatcherkNN(des1, des2):
    bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
    matches = bfm.knnMatch(des1, des2, k=1)
    return matches

def retGoodPoints(kps, bfMatches):
    points = np.float32([kps[m.queryIdx].pt for m in bfMatches]).reshape(-1, 1, 2)

def retEssentialMat(kpL1, kpL2, camMtx, dist):
    # print(type(kpL1))
    # kpL1 = np.array(kpL1) # convert to np.array just before operation
    # kpL2 = np.array(kpL2)
    # dist = np.array(dist)
    c =  cv2.findEssentialMat(kpL1, kpL2, camMtx, None, None, None, None)
    return c

def retPoseRecovery(essMtx, kpL1, kpL2):
    c = cv2.recoverPose(essMtx, kpL1, kpL2)
    return c

def retTriangulation(_R, _t, kpL1, kpL2):
    proj_matrix = np.hstack((_R, _t))
    print(kpL2[0])
    # BUG -> the points need to be iterated..
    pts_4d = cv2.triangulatePoints(np.eye(3, 4), proj_matrix, kpL1[-1], kpL2[-1])
    pts_3d = cv2.convertPointsFromHomogeneous(pts_4d.T)
    return pts_3d
    # return -1
    # TODO -> PROJ points(kpL) are causing issues with dimensionality.