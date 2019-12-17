import cv2
import math
import numpy


def PSNR(img1, img2):
    D = numpy.array(img1 - img2)
    RMSE = numpy.sum(D**2)/img1.size
    psnr = 10*math.log10(float(255.**2)/RMSE)
    return psnr

if __name__ == "__main__":
    
    name = 'BR 176 YUAN XIA F39Y_20160113_133403_image.jpg'
    oripic = cv2.imread('F:/compare/origin_2.jpg',0)
    result = cv2.imread('F:/compare/SK BR938 SHI SI MING F31Y_20160715_111151_image_resultat.jpg',0)
    psnr = PSNR(oripic, result)
    print ("The PSNR between the two img of the two is %f" % psnr)
