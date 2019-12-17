import json
import time

import torch as t
import torch.utils.data.dataloader as DataLoader
import torchvision as tv
import multiprocessing
from model import Mydataloader
from model.DnCnn import DnCNN
from model.func import save_model, eval_model_new_thread, eval_model
from model.Sobel import Sobel
import torch.nn.functional as F
if __name__ == "__main__":

    Filter = Sobel()
    time_start = time.time()

    config = json.load(open("config.json"))
    # os.environ["CUDA_VISIBLE_DEVICES"] = config["GPU"]
    DEVICE = t.device(config["DEVICE"])
    LR = config['lr']
    EPOCH = config['epoch']
    train_data = Mydataloader.TrainingData()
    test_data = Mydataloader.TestingData()

    train_loader = DataLoader.DataLoader(
        train_data, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    test_loader = DataLoader.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=config["num_workers"])

    model = DnCNN(n_channels=8).to(DEVICE)

    # Multi GPU setting
    # model = t.nn.DataParallel(model,device_ids=[0,1])

    optimizer = t.optim.Adam(model.parameters())

    criterian = t.nn.MSELoss()

    # Test the train_loader
    model = model.train()
    multiprocess_idx = 2
    for epoch in range(EPOCH):
        for batch_idx, [data, label] in enumerate(train_loader):
            data, label = data.to(DEVICE), label.to(DEVICE)
            out = t.cat([data, data], dim=1)
            print(out.shape)
            Gx = t.tensor([
                [-1.0, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], device=DEVICE)
            Gx = t.stack([Gx, Gx, Gx], dim=0)
            Gy = t.tensor([
                [1.0, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ], device=DEVICE)
            x = Filter(data)
            # x = t.nn.functional.conv2d(data, Gx)
            print(x.shape)
            # out = Sobel(data, DEVICE)
            # tv.transforms.ToPILImage()(out.cpu().squeeze()).save('Sobel.jpg')
