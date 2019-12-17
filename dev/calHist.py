import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/train_origin/SK BR 959 YAN XU FANG F55Y_20160719_151147_image.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
img2 = cv2.imread('data/train_noise/SK BR 959 YAN XU FANG F55Y_20160719_151147_image.jpg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2HSV)

img_diff = np.abs(img2 - img)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img_diff],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
cv2.imwrite('data/test.jpg',img_diff)