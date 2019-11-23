import cv2

img = cv2.imread('D:\\IE308_project\\data\\uncropped_pic\\X20140522C0032_3_n.jpg')
img1 = img[55:370,83:460,:]
cv2.imwrite('D:\\IE308_project\\data\\train_noise\\X20140522C0032_3_crop.jpg',img1)

img = cv2.imread('D:\\IE308_project\\data\\uncropped_pic\\X20140522C0032_3.jpg')
img2 = img[55:370,83:460,:]
cv2.imwrite('D:\\IE308_project\\data\\train_origin\\X20140522C0032_3_crop.jpg',img2)

img = cv2.imread('D:\\IE308_project\\data\\uncropped_pic\\X20151026C0053_3_n.jpg')
img3 = img[89:450,130:630,:]
cv2.imwrite('D:\\IE308_project\\data\\train_noise\\X20151026C0053_3_crop.jpg',img3)

img = cv2.imread('D:\\IE308_project\\data\\uncropped_pic\\X20151026C0053_3.jpg')
img4 = img[89:450,130:630,:]
cv2.imwrite('D:\\IE308_project\\data\\train_origin\\X20151026C0053_3_crop.jpg',img4)

