import json
import time

import torch as t
import torchvision as tv
import torch.utils.data.dataloader as DataLoader
from PIL import Image
import os
from model import Mydataloader
from model.DnCnn import DnCNN
from model.func import save_model, eval_model_new_thread, eval_model, Sobel
import multiprocessing

if __name__ == '__main__':
    # Test Canny
    config = json.load(open("config.json"))
    os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]
    DEVICE = t.device(config["DEVICE"])
    LR = config['lr']
    EPOCH = config['epoch']

    train_data = Mydataloader.TrainingData()
    test_data = Mydataloader.TestingData()

    train_loader = DataLoader.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])
    for batch_idx, data in enumerate(test_loader):
        data = data.to(DEVICE)
        # print(data, data.device)
        result = Sobel(data, DEVICE)
        tv.transforms.ToPILImage()(result.cpu().squeeze()).save('canny.jpg')
