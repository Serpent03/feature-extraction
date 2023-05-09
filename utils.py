import cv2

def readIm(pathToIm, rFac = 6):
    im = cv2.imread(pathToIm)
    im =  cv2.resize(im, (
        int(im.shape[1] / rFac), # width
        int(im.shape[0] / rFac), # height
    ))
    print(im.shape)
    return im

def ORB_detector(im1, im2):
    detect = cv2.ORB_create()
    kp1, des1 = detect.detectAndCompute(im1, None)
    kp2, des2 = detect.detectAndCompute(im2, None)
    return (kp1, des1, kp2, des2)
