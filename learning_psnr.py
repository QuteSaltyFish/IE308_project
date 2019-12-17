import cv2
from model.PSNR import PSNR


img = cv2.imread('C:/Users/ZXLi/Pictures/idx_10.jpg')
print(img.shape)
img4 = img[:,256:512,:]
img2 = cv2.imread('D:/IE308_project/data/train_origin/SK BR 1350 YANG SHUI ZHEN F62Y_20161017_115508_image.jpg')
img2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_CUBIC)

psnr = PSNR(img2, img4)
print ("The PSNR between the two img of the two is %f" % psnr)

