import argparse
import json
import os
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv

from model import Mydataloader
from model.DnCnn import DnCNN
from model.func import load_model

if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
    parser.add_argument("--epoch", default=199, type=int, help="The epoch to be tested")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_data = Mydataloader.TestingData()
    test_loader = DataLoader.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    criterian = t.nn.MSELoss()
    model = DnCNN(n_channels=8).to(DEVICE)
    # Test the train_loader
    model = load_model(model, args.epoch)
    model = model.eval()

    with t.no_grad():
        # Test the test_loader
        for batch_idx, data in enumerate(test_loader):
            data = data.to(DEVICE)
            out = model(data)
            # monitor the upper and lower boundary of output
            out_max = t.max(out)
            out_min = t.min(out)
            out = (out - out_min) / (out_max - out_min)
            tv.transforms.ToPILImage()(out.squeeze().cpu()).save('result/test_output.jpg')
            tv.transforms.ToPILImage()(data.squeeze().cpu()).save('result/test_input.jpg')
            print(data.shape, data.device)
