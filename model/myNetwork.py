import torch
from torchvision import models
from torchsummary import summary
import re
import os
import glob
import datetime
import time
import numpy as np
import torch.nn as nn
import json
from model import func
from model.Sobel import Sobel

class MyCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
        super().__init__()
        self.config = json.load(open('config.json'))
        self.DEVICE = self.config['DEVICE']
        if self.config["gray"]:
            image_channels = 1
        self.Sobel = Sobel()
        
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels * 4, out_channels=n_channels,
                                kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(
                n_channels, eps=0.0001, momentum=0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def fft2pic(self, data):
        frequency = torch.zeros_like(data)
        frequency = torch.stack([data, frequency], dim=0)
        frequency = frequency.permute(1,2,3,4,0)
        frequency = torch.fft(frequency,2)
        Re = frequency[...,0]
        Im = frequency[...,1]
        return Re, Im

    def forward(self, x):
        y = x
        addional = self.Sobel(x)
        Re, Im = self.fft2pic(x)
        # frequency = torch.functional.stft(x,2)

        x = torch.cat([x, addional, Re, Im], dim=1)
        out = self.dncnn(x)
        return y - out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                # print('init weight')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyCNN().to(device)
    summary(model, (3, 256, 256))
