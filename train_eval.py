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
import skimage
if __name__ == "__main__":
    time_start = time.time()

    config = json.load(open("config.json"))
    DEVICE = t.device(config["DEVICE"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=1, type=int, help="The epoch to be tested")
    parser.add_argument("--gpu", default='1', type=str, help="choose which DEVICE U want to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    train_data = Mydataloader.TrainingData()
    train_loader = DataLoader.DataLoader(train_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    criterian = t.nn.MSELoss()
    model = MyCNN(n_channels=8).to(DEVICE)
    # Test the train_loader
    model = load_model(model, args.epoch)
    model = model.eval()

    with t.no_grad():
        # Test the test_loader
        for batch_idx, [data, label] in enumerate(train_loader):
            data = data.to(DEVICE)
            out = model(data)
            DIR = 'result/train_result/epoch_{}'.format(args.epoch)
            if not os.path.exists(DIR):
                os.makedirs(DIR)
            print("Before the operation:")
            print("The PSNR between the two img of the two is {}".format(skimage.measure.compare_psnr(255*data.cpu().squeeze().numpy(), 255*label.cpu().squeeze().numpy(), 255)))
            print("The SSIM between the two img of the two is {}".format(skimage.measure.compare_ssim(255*data.cpu().squeeze().permute(1,2,0).numpy(), 255*label.cpu().squeeze().permute(1,2,0).numpy(), multichannel=True)))
            print('-'*20)
            print("After the operation:")
            print("The PSNR between the two img of the two is {}".format(skimage.measure.compare_psnr(255*out.cpu().squeeze().numpy(), 255*label.cpu().squeeze().numpy(), 255)))
            print("The SSIM between the two img of the two is {}".format(skimage.measure.compare_ssim(255*out.cpu().squeeze().permute(1,2,0).numpy(), 255*label.cpu().squeeze().permute(1,2,0).numpy(), multichannel=True)))
            print("\n\n")

            OUTPUT = t.cat([data, out], dim=3)
            tv.transforms.ToPILImage()(OUTPUT.squeeze().cpu()).save(DIR + '/idx_{}.jpg'.format(batch_idx))
