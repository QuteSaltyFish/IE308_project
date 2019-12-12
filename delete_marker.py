#%%
import numpy as np 
from PIL import Image
import torch as t
import torchvision as tv
import matplotlib.pyplot as plt
from model.func import canny
import cv2
import numpy as np  
from model.morphology import morphology
# %%
name = 'data/train_noise/SK BR285 TANG HUI F35Y_20160215_110159_image.jpg'
original_img = cv2.imread(name)
pic = cv2.resize(original_img, (256, 256), interpolation=cv2.INTER_CUBIC)
cv2.imwrite('result/origin.jpg', pic)
TH = 100
hsv=cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
r,g,b = hsv[...,0],hsv[...,1],hsv[...,2]
Image.fromarray(r)
#%%
data = r
data[data<TH] = 0
data[data>=TH] = 255
Image.fromarray(data)
Image.fromarray(data).save('tmp.jpg')


# %%
data = tv.transforms.ToTensor()(data)
mor = morphology(data, 'cpu')
kernel = t.ones(7,7)
closing = mor.Bi_Closing(kernel, save_img=False)
noise = closing - data
tv.transforms.ToPILImage()(closing).save('result/Closing.jpg')
tv.transforms.ToPILImage()(noise).save('result/noise.jpg')
# %%
mor = morphology(noise, 'cpu')
kernel = t.ones(3,3)/255
mask = mor.MReconstruction(kernel, save_img=False)

mor = morphology(mask, 'cpu')
kernel = t.ones(3,3)
mask = mor.Bi_Dilate(kernel, save_img=False)
tv.transforms.ToPILImage()(1-mask).save('result/mask.jpg')
img = tv.transforms.ToTensor()(Image.open(name).resize((256,256)))
tv.transforms.ToPILImage()(img*mask).save('result/mask_result.jpg')
tv.transforms.ToPILImage()(img - img*mask).save('result/img_result.jpg')


# #%%
# original_img = cv2.imread('result/result.jpg')
# pic = cv2.resize(original_img, (256, 256), interpolation=cv2.INTER_CUBIC)
# TH = 10
# hsv=cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)

# r,g,b = hsv[...,0],hsv[...,1],hsv[...,2]
# Image.fromarray(b)

# #%%

# pic[b<TH] = 0
# Image.fromarray(data)

# %%
