import torch as t
import torchvision as tv
from torchvision import transforms
from model import Mydataloader
import torch.utils.data.dataloader as DataLoader
import os
import json


def save_model(model, epoch):
    dir = 'saved_model/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    t.save(model.state_dict(), dir + '{}.pkl'.format(epoch))


def load_model(model, epoch):
    path = 'saved_model/''{}.pkl'.format(epoch)
    model.load_state_dict(t.load(path))
    return model
