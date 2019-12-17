import cv2
import math
import numpy


def PSNR(img1, img2):
    D = numpy.array(img1 - img2, dtype=numpy.int64)
    D[:, :] = D[:, :]**2
    RMSE = D.sum()/img1.size
    psnr = 10*math.log10(float(255.**2)/RMSE)
    return psnr

if __name__ == "__main__":
    # img1 = cv2.imread("", cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread("", cv2.IMREAD_GRAYSCALE)
    # psnr = PSNR(img1, img2)
    # print ("The PSNR between the two img of the two is %f" % psnr)
