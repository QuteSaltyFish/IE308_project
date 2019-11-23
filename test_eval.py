import argparse
import json
import os
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv

from model import Mydataloader
from model.myNetwork import MyCNN
from model.func import load_model

if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=config["GPU"], type=str, help="choose which DEVICE U want to use")
    parser.add_argument("--epoch", default=0, type=int, help="The epoch to be tested")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_data = Mydataloader.TestingData()
    test_loader = DataLoader.DataLoader(test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    criterian = t.nn.MSELoss()
    model = MyCNN(n_channels=8).to(DEVICE)
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
            DIR = 'result/test_result/epoch_{}'.format(args.epoch)
            if not os.path.exists(DIR):
                os.makedirs(DIR)
            OUTPUT = t.cat([data, out], dim=3)
            tv.transforms.ToPILImage()(OUTPUT.squeeze().cpu()).save(DIR + '/idx_{}.jpg'.format(batch_idx))
