import numpy as np
import torch as t
import torchvision as tv
import os
from PIL import Image
import json

import Mydataloader

config = json.load(open('config.json'))

kernal = t.tensor([
    [0, 1.0, 0],
    [1, 1, 1],
    [0, 1, 0]
], device=config['DEVICE'])

weight = t.nn.Parameter(kernal, requires_grad=False)

train_data = Mydataloader.TrainingData()
for i in range(len(train_data)):
    img, label = train_data[i]

    out = t.nn.functional.conv2d(input=label, weight=weight.view(-1, 3, 3), padding=1)
    tv.transforms.ToPILImage()(out).save('result/out.jpg')
    tv.transforms.ToPILImage()(label).save('result/label.jpg')
