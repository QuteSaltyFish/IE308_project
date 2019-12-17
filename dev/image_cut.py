import cv2
from model.PSNR import PSNR
# img = cv2.imread('D:\\IE308_project\\data\\uncropped_pic\\X20140522C0032_3_n.jpg')
# img1 = img[55:370,83:460,:]
# cv2.imwrite('D:\\IE308_project\\data\\train_noise\\X20140522C0032_3_crop.jpg',img1)

# img = cv2.imread('D:\\IE308_project\\data\\uncropped_pic\\X20140522C0032_3.jpg')
# img2 = img[55:370,83:460,:]
# cv2.imwrite('D:\\IE308_project\\data\\train_origin\\X20140522C0032_3_crop.jpg',img2)

# img = cv2.imread('D:\\IE308_project\\data\\uncropped_pic\\X20151026C0053_3_n.jpg')
# img3 = img[89:450,130:630,:]
# cv2.imwrite('D:\\IE308_project\\data\\train_noise\\X20151026C0053_3_crop.jpg',img3)

# img = cv2.imread('D:\\IE308_project\\data\\uncropped_pic\\X20151026C0053_3.jpg')
# img4 = img[89:450,130:630,:]
# cv2.imwrite('D:\\IE308_project\\data\\train_origin\\X20151026C0053_3_crop.jpg',img4)

img = cv2.imread('C:/Users/ZXLi/Pictures/idx_10.jpg')
print(img.shape)
img4 = img[:,256:512,:]
img2 = cv2.imread('D:/IE308_project/data/train_origin/SK BR 1350 YANG SHUI ZHEN F62Y_20161017_115508_image.jpg')
img2 = cv2.resize(img2, (256, 256), interpolation=cv2.INTER_CUBIC)

psnr = PSNR(img2, img4)
print ("The PSNR between the two img of the two is %f" % psnr)

