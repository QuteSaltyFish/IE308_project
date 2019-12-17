import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
img = Image.open('data/train_noise/SK BR 959 YAN XU FANG F55Y_20160719_151147_image.jpg')
img.show()
img = np.array(img)
print(img.max())
img[img<250] = 0
Image.fromarray(img).show()
