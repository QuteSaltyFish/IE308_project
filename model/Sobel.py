import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv
import multiprocessing
from model import Mydataloader
from model.DnCnn import DnCNN
from model import func

import torch.nn.functional as F


class Sobel(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = json.load(open("config.json"))
        self.DEVICE = t.device(self.config["DEVICE"])
        Gx = t.tensor([
            [-1.0, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], device=self.DEVICE).unsqueeze(0).unsqueeze(0)
        Gy = t.tensor([
            [1.0, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ], device=self.DEVICE).unsqueeze(0).unsqueeze(0)

        self.Gx = t.nn.Parameter(data=Gx, requires_grad=False)
        self.Gy = t.nn.Parameter(data=Gy, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.Gx, padding=1) + F.conv2d(x1.unsqueeze(1), self.Gy, padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.Gx, padding=1) + F.conv2d(x2.unsqueeze(1), self.Gy, padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.Gx, padding=1) + F.conv2d(x3.unsqueeze(1), self.Gy, padding=1)
        x = t.cat([x1, x2, x3], dim=1)
        return x
